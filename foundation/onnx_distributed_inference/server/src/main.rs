use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

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
use tokio::sync::{RwLock, mpsc::UnboundedSender};
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;
use websocket_messages::websocket::computation_message::Data;
use websocket_messages::websocket::web_socket_message::Kind;
use websocket_messages::websocket::{
    Computation, ComputationMessage, ConnectedUsers, ExternalDataEntry, FirstModelInput,
    InferenceRequest, InferenceResult, Initialize, IntermediateModelData, ModelConfig,
    WebSocketMessage,
};

mod graph;

use crate::graph::{Graph, Node};

// Main application state
struct AppState {
    graph: RwLock<ComputationalGraph>,
    tokenizer: Tokenizer,
    vocab_size: usize,
}

struct ComputationalGraph {
    graph: Graph,
    workers: HashMap<Uuid, UnboundedSender<Message>>,
    active_requests: HashMap<Uuid, ActiveRequest>,
    pending_requests: HashMap<Uuid, Vec<WebSocketMessage>>,
}

// Tracking state for active inference requests
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ActiveRequest {
    user_id: Uuid,
    input_token_len: usize,
    tokens: Vec<u32>,
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
        graph: RwLock::new(ComputationalGraph {
            graph,
            workers: HashMap::new(),
            active_requests: HashMap::new(),
            pending_requests: HashMap::new(),
        }),
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
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Spawn a task to forward messages from the channel to the WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if let Err(err) = sender.send(message).await {
                println!("Error sending message: {err}");
                break;
            }
        }
    });

    // Generate a unique user ID
    let user_id = Uuid::new_v4();
    println!("Client connected: {user_id}");

    // Initialize the user in the graph and send initialization message if a node is assigned
    let node_id = {
        let mut lock = state.graph.write().await;
        lock.workers.insert(user_id, tx.clone());
        let node_id = lock.graph.add_worker(user_id);

        let lock = lock.downgrade();
        broadcast_connected_users(&lock.workers).await;
        if let Some(node_id) = node_id {
            println!("Assigning worker {user_id} to node {node_id}");
            let node = lock.graph.get_node(&node_id).unwrap();

            let message = WebSocketMessage {
                kind: Some(Kind::Initialize(Initialize {
                    message: Some(node.data.clone()),
                })),
            };
            tx.send(Message::Binary(proto_encode(message).into()))
                .unwrap();
        }
        node_id
    };

    // Process incoming messages
    while let Some(result) = receiver.next().await {
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

    // Client disconnected
    println!("Client disconnected: {user_id}");

    // Remove the client
    {
        let mut lock = state.graph.write().await;
        lock.graph.remove_worker(&user_id);
        lock.workers.remove(&user_id);
        // TODO: remove from active/pending requests

        let lock = lock.downgrade();
        broadcast_connected_users(&lock.workers).await;
    }

    // Cancel the send task
    send_task.abort();
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
        Kind::InitializeDone(_) => {
            println!("Worker ready: {user_id}");
            let node_id = node_id.unwrap();
            let mut lock = state.graph.write().await;
            lock.graph.enable_worker(user_id, &node_id);

            if let Some(pending_requests) = lock.pending_requests.remove(&node_id) {
                let lock = lock.downgrade();
                let tx = lock.workers.get(user_id).unwrap();
                pending_requests.into_iter().for_each(|message| {
                    tx.send(Message::Binary(proto_encode(message).into()))
                        .unwrap();
                });
            }
        }
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
    let request_id = Uuid::new_v4();
    println!("Processing inference request {request_id} for user {user_uuid} with prompt '{text}'",);

    // Tokenize input text
    let input_ids = state
        .tokenizer
        .encode(text, false)
        .unwrap()
        .get_ids()
        .to_vec();

    // Create model input
    let input = FirstModelInput {
        input_ids: input_ids.clone(),
    };

    // Send computational request
    {
        let mut lock = state.graph.write().await;
        let node_id = lock.graph.start_node_id;
        let message = WebSocketMessage {
            kind: Some(Kind::Computation(Computation {
                message: Some(ComputationMessage {
                    node_id: node_id.to_string(),
                    request_id: request_id.to_string(),
                    data: Some(Data::First(input)),
                }),
            })),
        };
        lock.active_requests.insert(
            request_id,
            ActiveRequest {
                user_id: *user_uuid,
                input_token_len: input_ids.len(),
                tokens: input_ids.clone(),
            },
        );
        match lock.graph.get_worker(&node_id) {
            Some(worker_id) => {
                lock.downgrade()
                    .workers
                    .get(&worker_id)
                    .unwrap()
                    .send(Message::Binary(proto_encode(message).into()))
                    .unwrap();
            }
            None => match lock.pending_requests.get_mut(&node_id) {
                Some(messages) => messages.push(message),
                None => {
                    let _ = lock.pending_requests.insert(node_id, vec![message]);
                }
            },
        }
    }
}

async fn handle_computation_result(result: ComputationMessage, state: &Arc<AppState>) {
    let request_id = Uuid::parse_str(&result.request_id).unwrap();
    let node_id = Uuid::parse_str(&result.node_id).unwrap();
    println!("Processing for node: {node_id}, request: {request_id}");

    // Get next nodes
    {
        let mut lock = state.graph.write().await;
        let next_nodes = lock.graph.get_next_nodes(&node_id).clone();

        if !next_nodes.is_empty() {
            for node_id in next_nodes {
                println!("Forwarding to next node: {node_id}");
                let message = WebSocketMessage {
                    kind: Some(Kind::Computation(Computation {
                        message: Some(ComputationMessage {
                            node_id: node_id.to_string(),
                            request_id: result.request_id.clone(),
                            data: result.data.clone(),
                        }),
                    })),
                };

                match lock.graph.get_worker(&node_id) {
                    Some(worker_id) => {
                        lock.workers
                            .get(&worker_id)
                            .unwrap()
                            .send(Message::Binary(proto_encode(message).into()))
                            .unwrap();
                    }
                    None => match lock.pending_requests.get_mut(&node_id) {
                        Some(messages) => messages.push(message),
                        None => {
                            let _ = lock.pending_requests.insert(node_id, vec![message]);
                        }
                    },
                }
            }
            return;
        }
    }

    // Reached end node, process the logits
    let logits = match result.data {
        Some(Data::Intermediate(IntermediateModelData { mut map })) => {
            map.remove("logits").unwrap().data
        }
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
    {
        let mut lock = state.graph.write().await;
        let active_request = lock.active_requests.get_mut(&request_id).unwrap();
        active_request.tokens.push(next_token);

        let gen_token_len = active_request.tokens.len() - active_request.input_token_len;
        if gen_token_len <= 20 {
            // Get worker for start node and send next computation
            let active_request = active_request.clone();
            let node_id = lock.graph.start_node_id;
            let message = WebSocketMessage {
                kind: Some(Kind::Computation(Computation {
                    message: Some(ComputationMessage {
                        node_id: node_id.to_string(),
                        request_id: request_id.to_string(),
                        data: Some(Data::First(FirstModelInput {
                            input_ids: active_request.tokens,
                        })),
                    }),
                })),
            };

            match lock.graph.get_worker(&node_id) {
                Some(worker_id) => {
                    lock.downgrade()
                        .workers
                        .get(&worker_id)
                        .unwrap()
                        .send(Message::Binary(proto_encode(message).into()))
                        .unwrap();
                }
                None => match lock.pending_requests.get_mut(&node_id) {
                    Some(messages) => messages.push(message),
                    None => {
                        let _ = lock.pending_requests.insert(node_id, vec![message]);
                    }
                },
            }
            return;
        }

        // Generate final output and send to client
        let active_request = lock.active_requests.remove(&request_id).unwrap();
        let lock = lock.downgrade();

        let output_text = state
            .tokenizer
            .decode(
                &active_request.tokens[active_request.input_token_len..],
                false,
            )
            .unwrap();

        println!("Final result: {output_text}");

        let message = WebSocketMessage {
            kind: Some(Kind::InferenceResult(InferenceResult {
                message: output_text,
            })),
        };

        if let Some(tx) = lock.workers.get(&active_request.user_id) {
            tx.send(Message::Binary(proto_encode(message).into()))
                .unwrap();
        }
    }
}

async fn broadcast_connected_users(clients: &HashMap<Uuid, UnboundedSender<Message>>) {
    let message = WebSocketMessage {
        kind: Some(Kind::ConnectedUsers(ConnectedUsers {
            message: clients.len() as u32,
        })),
    };
    let data = proto_encode(message);
    for tx in clients.values() {
        tx.send(Message::Binary(data.clone().into())).unwrap();
    }
}

fn arg_max(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn proto_encode<T: ProstMessage>(message: T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(message.encoded_len());
    message.encode(&mut buf).unwrap();
    buf
}
