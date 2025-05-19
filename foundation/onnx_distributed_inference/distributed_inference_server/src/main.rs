use std::sync::Arc;
use std::{collections::HashMap, str::FromStr};

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
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;

mod graph;

use crate::graph::{Graph, Node};

struct AppState {
    worker_clients: RwLock<HashMap<Uuid, tokio::sync::mpsc::UnboundedSender<Message>>>,
    active_requests: RwLock<HashMap<Uuid, ActiveRequest>>,
    graph: RwLock<Graph>,
    tokenizer: Tokenizer,
    vocab_size: usize,
}

struct ActiveRequest {
    user_id: Uuid,
    input_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct ComputationMessage {
    node_id: Uuid,
    request_id: Uuid,
    data: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct WebsocketMessage {
    #[serde(rename = "type")]
    message_type: String,
    message: serde_json::Value,
}

#[tokio::main]
async fn main() {
    // Initialize the tokenizer
    let tokenizer =
        Tokenizer::from_pretrained("microsoft/phi-2", None).expect("Failed to load tokenizer");
    let vocab_size = 51200;

    // Load external data
    let external_data_p1: serde_json::Value = serde_json::from_str(include_str!("../data_p1.json"))
        .expect("Failed to parse data_p1.json");
    let external_data_p2: serde_json::Value = serde_json::from_str(include_str!("../data_p2.json"))
        .expect("Failed to parse data_p2.json");

    // Create graph
    let end = Node::new(
        vec![],
        serde_json::json!({
            "modelURI": "http://localhost:3000/model/phi/split/p2/model.onnx",
            "externalData": external_data_p2
        }),
    );

    let start = Node::new(
        vec![end.id],
        serde_json::json!({
            "modelURI": "http://localhost:3000/model/phi/split/p1/model.onnx",
            "externalData": external_data_p1
        }),
    );

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
    let user_uuid = Uuid::new_v4();
    println!("Client connected: {}", user_uuid);

    // Store the client
    {
        let mut clients = state.worker_clients.write().await;
        clients.insert(user_uuid.clone(), tx.clone());

        // Broadcast the number of connected users
        broadcast_connected_users(&clients).await;
    }

    // Initialize the user in the graph
    let node_id = {
        let mut graph = state.graph.write().await;
        graph.add_user(user_uuid)
    };

    // Send initialization message if a node is assigned
    if let Some(node_id) = node_id {
        let graph_data = {
            let graph = state.graph.read().await;
            graph.get_node(&node_id).map(|node| node.data.clone())
        };

        if let Some(data) = graph_data {
            let init_message = WebsocketMessage {
                message_type: "initialize".to_string(),
                message: data,
            };

            let json = serde_json::to_string(&init_message).unwrap();
            let _ = tx.send(Message::Text(json.into()));
        }
    }

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_message(text.to_string(), &user_uuid, &state).await;
            }
            Ok(Message::Binary(_)) => {
                println!("Binary message received");
            }
            Ok(Message::Close(_)) => {
                break;
            }
            _ => {
                println!("Unexpected message")
            }
        }
    }

    // Client disconnected
    println!("Client disconnected: {}", user_uuid);

    // Remove the client
    {
        let mut clients = state.worker_clients.write().await;
        clients.remove(&user_uuid);

        let mut graph = state.graph.write().await;
        graph.remove_user(&user_uuid);

        broadcast_connected_users(&clients).await;
    }

    // Cancel the send task
    send_task.abort();
}

async fn handle_message(text: String, user_uuid: &Uuid, state: &Arc<AppState>) {
    let message: WebsocketMessage = match serde_json::from_str(&text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to parse message: {}", e);
            return;
        }
    };

    match message.message_type.as_str() {
        "inferenceRequest" => {
            handle_inference_request(message.message, user_uuid, state).await;
        }
        "computationResult" => {
            handle_computation_result(message.message, state).await;
        }
        _ => {
            eprintln!("Unknown message type: {}", message.message_type);
        }
    }
}

async fn handle_inference_request(
    message: serde_json::Value,
    user_uuid: &Uuid,
    state: &Arc<AppState>,
) {
    let text = message.as_str().unwrap_or_default();

    // Tokenize input text
    let encoding = state.tokenizer.encode(text, false).unwrap();
    let input_ids = encoding.get_ids().to_vec();
    let attention_mask = encoding.get_attention_mask().to_vec();
    let position_ids: Vec<u64> = (0..input_ids.len()).map(|x| x as u64).collect();

    let input = serde_json::json!({
        "input_ids": input_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        "attention_mask": attention_mask.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        "position_ids": position_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
    });

    println!("Input: {:?}", input);

    // Get worker for start node
    let (worker_id, start_node_id) = {
        let graph = state.graph.read().await;
        (
            graph.get_worker(&graph.start_node_id),
            graph.start_node_id.clone(),
        )
    };

    // Create request
    let request_id = Uuid::new_v4();

    // Store active request
    {
        let mut active_requests = state.active_requests.write().await;
        active_requests.insert(
            request_id.clone(),
            ActiveRequest {
                user_id: *user_uuid,
                input_tokens: input_ids.clone(),
                generated_tokens: Vec::new(),
            },
        );
    }

    // Send computation message to worker
    if let Some(worker_id) = worker_id {
        let clients = state.worker_clients.read().await;
        if let Some(tx) = clients.get(&worker_id) {
            let comp_message = WebsocketMessage {
                message_type: "computation".to_string(),
                message: serde_json::json!({
                    "nodeId": start_node_id,
                    "requestId": request_id,
                    "data": input
                }),
            };

            let json = serde_json::to_string(&comp_message).unwrap();
            let _ = tx.send(Message::Text(json.into()));
        }
    }
}

async fn handle_computation_result(message: serde_json::Value, state: &Arc<AppState>) {
    let result = message.as_object().unwrap();
    let node_id = Uuid::from_str(result["nodeId"].as_str().unwrap_or_default()).unwrap();
    let request_id = Uuid::from_str(result["requestId"].as_str().unwrap_or_default()).unwrap();
    let data = &result["data"];

    println!("Got message: {node_id}");
    println!("Got message: {request_id}");

    // Get next nodes
    {
        let graph = state.graph.read().await;
        let next_nodes = graph.get_next_nodes(&node_id);

        if !next_nodes.is_empty() {
            // Forward computation to next nodes
            let clients = state.worker_clients.read().await;

            for node in next_nodes {
                println!("Sending to node: {}", node.id);
                if let Some(worker_id) = {
                    let graph = state.graph.read().await;
                    graph.get_worker(&node.id)
                } {
                    println!("Worker: {}", worker_id);
                    if let Some(tx) = clients.get(&worker_id) {
                        let comp_message = WebsocketMessage {
                            message_type: "computation".to_string(),
                            message: serde_json::json!({
                                "nodeId": node.id,
                                "requestId": request_id,
                                "data": data
                            }),
                        };
                        println!("Message to long");

                        let json = serde_json::to_string(&comp_message).unwrap();
                        let _ = tx.send(Message::Text(json.into()));
                    }
                }
            }
            return;
        }
    };

    // Reached end node, process the logits
    let logits = data["logits"]["data"].as_array().unwrap();
    let start_idx = logits.len() - state.vocab_size;
    let logits_slice = &logits[start_idx..];

    // Find max logit (argmax)
    let next_token = arg_max(logits_slice);

    let token_str = {
        let token_ids = vec![next_token];
        state
            .tokenizer
            .decode(&token_ids, false)
            .unwrap_or_default()
    };

    println!("Generated token: {}, Decoded: {}", next_token, token_str);

    // Update active request
    let (user_id, should_finalize) = {
        let mut active_requests = state.active_requests.write().await;
        let active_request = active_requests.get_mut(&request_id).unwrap();
        active_request.generated_tokens.push(next_token);

        let user_id = active_request.user_id.clone();
        let should_finalize = active_request.generated_tokens.len() > 20;

        (user_id, should_finalize)
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

        println!("Final result: {}", output_text);

        let clients = state.worker_clients.read().await;
        if let Some(tx) = clients.get(&user_id) {
            let result_message = WebsocketMessage {
                message_type: "inferenceResult".to_string(),
                message: serde_json::json!(output_text),
            };

            let json = serde_json::to_string(&result_message).unwrap();
            let _ = tx.send(Message::Text(json.into()));
        }

        return;
    }

    // Generate next token
    let position = {
        let active_requests = state.active_requests.read().await;
        let active_request = active_requests.get(&request_id).unwrap();
        active_request.input_tokens.len() + active_request.generated_tokens.len() - 1
    };

    let input = serde_json::json!({
        "input_ids": [next_token.to_string()],
        "attention_mask": [1.to_string()],
        "position_ids": [position.to_string()],
    });

    println!("Input: {:?}", input);

    // Get worker for start node and send next computation
    let (worker_id, start_node_id) = {
        let graph = state.graph.read().await;
        (
            graph.get_worker(&graph.start_node_id),
            graph.start_node_id.clone(),
        )
    };

    if let Some(worker_id) = worker_id {
        let clients = state.worker_clients.read().await;
        if let Some(tx) = clients.get(&worker_id) {
            let comp_message = WebsocketMessage {
                message_type: "computation".to_string(),
                message: serde_json::json!({
                    "nodeId": start_node_id,
                    "requestId": request_id,
                    "data": input
                }),
            };

            let json = serde_json::to_string(&comp_message).unwrap();
            let _ = tx.send(Message::Text(json.into()));
        }
    }
}

fn arg_max(array: &[serde_json::Value]) -> u32 {
    let mut max_idx = 0u32;
    let mut max_val = f64::NEG_INFINITY;

    for (i, val) in array.iter().enumerate() {
        if let Some(num) = val.as_f64() {
            if num > max_val {
                max_val = num;
                max_idx = i as u32;
            }
        }
    }

    max_idx
}

async fn broadcast_connected_users(
    clients: &HashMap<Uuid, tokio::sync::mpsc::UnboundedSender<Message>>,
) {
    let count = clients.len();
    let message = WebsocketMessage {
        message_type: "connectedUsers".to_string(),
        message: serde_json::json!(count),
    };

    let json = serde_json::to_string(&message).unwrap();

    for tx in clients.values() {
        let _ = tx.send(Message::Text(json.clone().into()));
    }
}
