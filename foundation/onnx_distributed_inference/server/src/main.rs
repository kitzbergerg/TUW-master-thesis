use std::io::Cursor;
use std::sync::Arc;
use std::{collections::HashMap, time::Duration};

use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use prost::Message as ProstMessage;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer;
use tokio::{
    sync::{RwLock, mpsc::UnboundedSender},
    time::sleep,
};
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;
use websocket_messages::websocket::computation_message::Data;
use websocket_messages::websocket::web_socket_message::Kind;
use websocket_messages::websocket::{
    Computation, ComputationMessage, ConnectedUsers, ExternalDataEntry, FinalModelOutput,
    FirstModelInput, InferenceRequest, InferenceResult, Initialize, ModelConfig, WebSocketMessage,
};

mod graph;

use crate::graph::{Graph, Node};

// Main application state
struct AppState {
    worker_clients: RwLock<HashMap<Uuid, tokio::sync::mpsc::UnboundedSender<Message>>>,
    active_requests: RwLock<HashMap<Uuid, ActiveRequest>>,
    graph: RwLock<Graph>,
    tokenizer: Tokenizer,
    vocab_size: usize,
}

// Tracking state for active inference requests
struct ActiveRequest {
    user_id: Uuid,
    input_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TmpExtEntry {
    path: String,
    data: String,
}

#[tokio::main]
async fn main() {
    // Initialize the tokenizer
    let tokenizer = Tokenizer::from_pretrained("microsoft/phi-2", None).unwrap();
    let vocab_size = 51200;

    // Load external data
    let external_data_p1: Vec<TmpExtEntry> =
        serde_json::from_str(include_str!("../data_p1.json")).unwrap();
    let external_data_p2: Vec<TmpExtEntry> =
        serde_json::from_str(include_str!("../data_p2.json")).unwrap();

    // Create graph
    let end_config = ModelConfig {
        model_uri: "http://localhost:3000/model/phi/split/p2/model.onnx".to_string(),
        external_data: external_data_p2
            .into_iter()
            .map(|entry| ExternalDataEntry {
                path: entry.path,
                data: entry.data,
            })
            .collect(),
    };

    let start_config = ModelConfig {
        model_uri: "http://localhost:3000/model/phi/split/p1/model.onnx".to_string(),
        external_data: external_data_p1
            .into_iter()
            .map(|entry| ExternalDataEntry {
                path: entry.path,
                data: entry.data,
            })
            .collect(),
    };

    let end = Node::new(vec![], end_config);
    let start = Node::new(vec![end.id], start_config);
    let graph = Graph::new(vec![start, end]);

    // Create shared state
    let state = Arc::new(AppState {
        worker_clients: RwLock::new(HashMap::new()),
        active_requests: RwLock::new(HashMap::new()),
        graph: RwLock::new(graph),
        tokenizer,
        vocab_size,
    });

    // Build our application with routes
    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .nest_service("/model", ServeDir::new("model"))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Run the server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("Server listening on http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    // Use a channel to handle sending messages
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Spawn a task to forward messages from the channel to the WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if sender.send(message).await.is_err() {
                break;
            }
        }
    });

    // Generate a unique user ID
    let user_id = Uuid::new_v4();
    println!("Client connected: {user_id}");

    // Store the client
    {
        let mut clients = state.worker_clients.write().await;
        clients.insert(user_id, tx.clone());

        // Broadcast the number of connected users
        broadcast_connected_users(&clients).await;
    }

    // Initialize the user in the graph and send initialization message if a node is assigned
    let node_id = state.graph.write().await.add_worker(user_id);
    if let Some(node_id) = node_id {
        println!("Assigning worker {user_id} to node {node_id}");
        let graph_data = state
            .graph
            .read()
            .await
            .get_node(&node_id)
            .map(|node| node.data.clone());

        if let Some(data) = graph_data {
            let message = WebSocketMessage {
                kind: Some(Kind::Initialize(Initialize {
                    message: Some(data),
                })),
            };
            let _ = tx.send(Message::Binary(proto_encode(message).into()));
        }
    }

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        // TODO: use more efficient encoding, json is bad for numbers
        match result {
            Ok(Message::Text(_)) => println!("Text message received"),
            Ok(Message::Binary(data)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    handle_message(data.to_vec(), &user_id, &node_id, &state).await
                });
            }
            Ok(Message::Close(_)) => break,
            _ => println!("Unexpected message"),
        }
    }

    // TODO: handle missing cache in case of ongoing inference request

    // Client disconnected
    println!("Client disconnected: {user_id}");

    // Remove the client
    state.graph.write().await.remove_worker(&user_id);
    {
        let mut clients = state.worker_clients.write().await;
        clients.remove(&user_id);

        broadcast_connected_users(&clients).await;
    }

    // Cancel the send task
    send_task.abort();
}

fn proto_encode<T: ProstMessage>(message: T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(message.encoded_len());
    message.encode(&mut buf).unwrap();
    buf
}

async fn handle_message(
    data: Vec<u8>,
    user_id: &Uuid,
    node_id: &Option<Uuid>,
    state: &Arc<AppState>,
) {
    let message = WebSocketMessage::decode(Cursor::new(&data)).unwrap();
    let message = message.kind.unwrap();

    match message {
        Kind::InitializeDone(_) => state
            .graph
            .write()
            .await
            .enable_worker(user_id, &node_id.unwrap()),
        Kind::InferenceRequest(InferenceRequest { message }) => {
            handle_inference_request(message, user_id, state).await;
        }
        Kind::Computation(Computation { message }) => {
            let message = message.unwrap();
            handle_computation_result(message, state).await;
        }
        _ => {
            eprintln!("Received unexpected message type from client");
        }
    }
}

async fn handle_inference_request(text: String, user_uuid: &Uuid, state: &Arc<AppState>) {
    // Tokenize input text
    let encoding = state.tokenizer.encode(text, false).unwrap();
    let input_ids = encoding.get_ids().to_vec();
    let attention_mask = encoding.get_attention_mask().to_vec();
    let position_ids = (0..input_ids.len()).map(|x| x as u32).collect::<Vec<_>>();

    println!(
        "Processing inference request with {} tokens",
        input_ids.len()
    );

    // Create model input
    let input = FirstModelInput {
        input_ids: input_ids.clone(),
        attention_mask,
        position_ids,
    };

    // Get worker for start node
    let start_node_id = state.graph.read().await.start_node_id;
    let worker_id = match get_worker(state, start_node_id).await {
        Some(id) => id,
        None => return,
    };

    // Create request
    let request_id = Uuid::new_v4();

    // Store active request
    state.active_requests.write().await.insert(
        request_id,
        ActiveRequest {
            user_id: *user_uuid,
            input_tokens: input_ids,
            generated_tokens: Vec::new(),
        },
    );

    // Send computation message to worker
    if let Some(tx) = state.worker_clients.read().await.get(&worker_id) {
        let message = WebSocketMessage {
            kind: Some(Kind::Computation(Computation {
                message: Some(ComputationMessage {
                    node_id: start_node_id.to_string(),
                    request_id: request_id.to_string(),
                    data: Some(Data::First(input)),
                }),
            })),
        };

        let _ = tx.send(Message::Binary(proto_encode(message).into()));
    }
}

async fn handle_computation_result(result: ComputationMessage, state: &Arc<AppState>) {
    let request_id = Uuid::parse_str(&result.request_id).unwrap();
    let node_id = Uuid::parse_str(&result.node_id).unwrap();
    println!("Processing for node: {node_id}, request: {request_id}");

    // Get next nodes
    let next_nodes = state.graph.read().await.get_next_nodes(&node_id).clone();

    if !next_nodes.is_empty() {
        for node_id in next_nodes {
            println!("Forwarding to next node: {node_id}");
            if let Some(worker_id) = get_worker(state, node_id).await {
                if let Some(tx) = state.worker_clients.read().await.get(&worker_id) {
                    let message = WebSocketMessage {
                        kind: Some(Kind::Computation(Computation {
                            message: Some(ComputationMessage {
                                node_id: node_id.to_string(),
                                request_id: result.request_id.clone(),
                                data: result.data.clone(),
                            }),
                        })),
                    };
                    let _ = tx.send(Message::Binary(proto_encode(message).into()));
                }
            }
        }
        return;
    }

    // Reached end node, process the logits
    let logits = match result.data {
        Some(Data::Logits(FinalModelOutput { data })) => data,
        _ => unreachable!(),
    };
    let start_idx = logits.len() - state.vocab_size;
    let logits_slice = &logits[start_idx..];
    let next_token = arg_max(logits_slice);

    println!(
        "Generated token: {}, Decoded: {}",
        next_token,
        state.tokenizer.decode(&[next_token], false).unwrap()
    );

    // Update active request
    let (user_id, should_finalize) = {
        let mut active_requests = state.active_requests.write().await;
        let active_request = active_requests.get_mut(&request_id).unwrap();
        active_request.generated_tokens.push(next_token);

        (
            active_request.user_id,
            active_request.generated_tokens.len() > 20,
        )
    };

    if should_finalize {
        // Generate final output and send to client
        let output_text = {
            let active_requests = state.active_requests.read().await;
            let active_request = active_requests.get(&request_id).unwrap();
            state
                .tokenizer
                .decode(&active_request.generated_tokens, false)
                .unwrap_or_default()
        };

        println!("Final result: {output_text}");

        if let Some(tx) = state.worker_clients.read().await.get(&user_id) {
            let message = WebSocketMessage {
                kind: Some(Kind::InferenceResult(InferenceResult {
                    message: output_text,
                })),
            };

            let _ = tx.send(Message::Binary(proto_encode(message).into()));
        }

        // Clean up active request
        state.active_requests.write().await.remove(&request_id);

        return;
    }

    // Generate next token
    let position = {
        let active_requests = state.active_requests.read().await;
        let active_request = active_requests.get(&request_id).unwrap();
        active_request.input_tokens.len() + active_request.generated_tokens.len() - 1
    };

    let input = FirstModelInput {
        input_ids: vec![next_token],
        attention_mask: vec![1],
        position_ids: vec![position as u32],
    };

    // Get worker for start node and send next computation
    let start_node_id = state.graph.read().await.start_node_id;
    let worker_id = match get_worker(state, start_node_id).await {
        Some(id) => id,
        None => return,
    };

    if let Some(tx) = state.worker_clients.read().await.get(&worker_id) {
        let message = WebSocketMessage {
            kind: Some(Kind::Computation(Computation {
                message: Some(ComputationMessage {
                    node_id: start_node_id.to_string(),
                    request_id: request_id.to_string(),
                    data: Some(Data::First(input)),
                }),
            })),
        };

        let _ = tx.send(Message::Binary(proto_encode(message).into()));
    }
}

async fn get_worker(state: &Arc<AppState>, node_id: Uuid) -> Option<Uuid> {
    for i in 1..10 {
        if let Some(worker_id) = state.graph.read().await.get_worker(&node_id) {
            return Some(worker_id);
        }
        let retry_in = 10 * i;
        println!("No worker found for node {node_id}. Retrying in {retry_in}s...");
        sleep(Duration::from_secs(retry_in)).await;
    }
    println!("No worker found for node {node_id} after retrying. Giving up...");
    None
}
fn arg_max(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

async fn broadcast_connected_users(clients: &HashMap<Uuid, UnboundedSender<Message>>) {
    let message = WebSocketMessage {
        kind: Some(Kind::ConnectedUsers(ConnectedUsers {
            message: clients.len() as u32,
        })),
    };
    let data = proto_encode(message);
    for tx in clients.values() {
        let _ = tx.send(Message::Binary(data.clone().into()));
    }
}
