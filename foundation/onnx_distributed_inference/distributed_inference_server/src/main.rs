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
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer;
use tokio::{
    sync::{RwLock, mpsc::UnboundedSender},
    time::sleep,
};
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;

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

// Model configuration for nodes
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    model_uri: String,
    external_data: serde_json::Value,
}

// Input tensors for model inference
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
enum ModelInput {
    First(FirstInput),
    Intermediate(HashMap<String, IntermediateResult>),
}
// Input tensors for model inference
#[derive(Serialize, Deserialize, Clone, Debug)]
struct FirstInput {
    input_ids: Vec<String>,
    attention_mask: Vec<String>,
    position_ids: Vec<String>,
}

// Output tensors from model inference
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
enum ModelOutput {
    Final { logits: FinalOutput },
    Intermediate(HashMap<String, IntermediateResult>),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FinalOutput {
    data: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct IntermediateResult {
    data: Vec<f64>,
    dims: Vec<u32>,
}

// Enum for different WebSocket message types
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum WebSocketMessage {
    #[serde(rename = "initialize")]
    Initialize { message: ModelConfig },
    #[serde(rename = "initializeDone")]
    InitializeDone,
    #[serde(rename = "inferenceRequest")]
    InferenceRequest { message: String },
    #[serde(rename = "inferenceResult")]
    InferenceResult { message: String },
    #[serde(rename = "computation")]
    Computation { message: ComputationMessage },
    #[serde(rename = "computationResult")]
    ComputationResult { message: ComputationResultMessage },
    #[serde(rename = "connectedUsers")]
    ConnectedUsers { message: usize },
}

// Computation message structure
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ComputationMessage {
    #[serde(rename = "nodeId")]
    node_id: Uuid,
    #[serde(rename = "requestId")]
    request_id: Uuid,
    data: ModelInput,
}

// Computation result message structure
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ComputationResultMessage {
    #[serde(rename = "nodeId")]
    node_id: Uuid,
    #[serde(rename = "requestId")]
    request_id: Uuid,
    data: ModelOutput,
}

#[tokio::main]
async fn main() {
    // Initialize the tokenizer
    let tokenizer = Tokenizer::from_pretrained("microsoft/phi-2", None).unwrap();
    let vocab_size = 51200;

    // Load external data
    let external_data_p1: serde_json::Value = serde_json::from_str(include_str!("../data_p1.json"))
        .expect("Failed to parse data_p1.json");
    let external_data_p2: serde_json::Value = serde_json::from_str(include_str!("../data_p2.json"))
        .expect("Failed to parse data_p2.json");

    // Create graph
    let end_config = ModelConfig {
        model_uri: "http://localhost:3000/model/phi/split/p2/model.onnx".to_string(),
        external_data: external_data_p2,
    };

    let start_config = ModelConfig {
        model_uri: "http://localhost:3000/model/phi/split/p1/model.onnx".to_string(),
        external_data: external_data_p1,
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
            let init_message = WebSocketMessage::Initialize { message: data };
            let json = serde_json::to_string(&init_message).unwrap();
            let _ = tx.send(Message::Text(json.into()));
        }
    }

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        match result {
            Ok(Message::Text(text)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    handle_message(text.to_string(), &user_id, &node_id, &state).await
                });
            }
            Ok(Message::Binary(_)) => println!("Binary message received"),
            Ok(Message::Close(_)) => break,
            _ => println!("Unexpected message"),
        }
    }

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

async fn handle_message(
    text: String,
    user_id: &Uuid,
    node_id: &Option<Uuid>,
    state: &Arc<AppState>,
) {
    let message: WebSocketMessage = serde_json::from_str(&text).unwrap();

    match message {
        WebSocketMessage::InitializeDone => state
            .graph
            .write()
            .await
            .enable_worker(user_id, &node_id.unwrap()),
        WebSocketMessage::InferenceRequest { message } => {
            handle_inference_request(message, user_id, state).await;
        }
        WebSocketMessage::ComputationResult { message } => {
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
    let position_ids = (0..input_ids.len()).collect::<Vec<_>>();

    // Create model input
    let input = FirstInput {
        input_ids: input_ids.iter().map(|x| x.to_string()).collect(),
        attention_mask: attention_mask.iter().map(|x| x.to_string()).collect(),
        position_ids: position_ids.iter().map(|x| x.to_string()).collect(),
    };

    println!(
        "Processing inference request with {} tokens",
        input_ids.len()
    );

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
        let comp_message = WebSocketMessage::Computation {
            message: ComputationMessage {
                node_id: start_node_id,
                request_id,
                data: ModelInput::First(input),
            },
        };

        let json = serde_json::to_string(&comp_message).unwrap();
        let _ = tx.send(Message::Text(json.into()));
    }
}

async fn handle_computation_result(result: ComputationResultMessage, state: &Arc<AppState>) {
    println!(
        "Processing computation result for node: {}, request: {}",
        result.node_id, result.request_id
    );

    // Get next nodes
    let next_nodes = state
        .graph
        .read()
        .await
        .get_next_nodes(&result.node_id)
        .clone();

    if !next_nodes.is_empty() {
        for node_id in next_nodes {
            println!("Forwarding to next node: {}", node_id);
            let worker_id = match get_worker(state, node_id).await {
                Some(id) => id,
                None => return,
            };
            let intermediate = match result.data.clone() {
                ModelOutput::Intermediate(hash_map) => hash_map,
                _ => unreachable!(),
            };

            if let Some(tx) = state.worker_clients.read().await.get(&worker_id) {
                let comp_message = WebSocketMessage::Computation {
                    message: ComputationMessage {
                        node_id: node_id,
                        request_id: result.request_id,
                        data: ModelInput::Intermediate(intermediate),
                    },
                };

                let json = serde_json::to_string(&comp_message).unwrap();
                let _ = tx.send(Message::Text(json.into()));
            }
        }
        return;
    }

    // Reached end node, process the logits
    let logits = match result.data {
        ModelOutput::Final { logits } => logits.data,
        ModelOutput::Intermediate(_) => unreachable!(),
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
        let active_request = active_requests.get_mut(&result.request_id).unwrap();
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
            let active_request = active_requests.get(&result.request_id).unwrap();
            state
                .tokenizer
                .decode(&active_request.generated_tokens, false)
                .unwrap_or_default()
        };

        println!("Final result: {output_text}");

        if let Some(tx) = state.worker_clients.read().await.get(&user_id) {
            let result_message = WebSocketMessage::InferenceResult {
                message: output_text,
            };

            let json = serde_json::to_string(&result_message).unwrap();
            let _ = tx.send(Message::Text(json.into()));
        }

        // Clean up active request
        state
            .active_requests
            .write()
            .await
            .remove(&result.request_id);

        return;
    }

    // Generate next token
    let position = {
        let active_requests = state.active_requests.read().await;
        let active_request = active_requests.get(&result.request_id).unwrap();
        active_request.input_tokens.len() + active_request.generated_tokens.len() - 1
    };

    let input = FirstInput {
        input_ids: vec![next_token.to_string()],
        attention_mask: vec!["1".to_string()],
        position_ids: vec![position.to_string()],
    };

    // Get worker for start node and send next computation
    let start_node_id = state.graph.read().await.start_node_id;
    let worker_id = match get_worker(state, start_node_id).await {
        Some(id) => id,
        None => return,
    };

    if let Some(tx) = state.worker_clients.read().await.get(&worker_id) {
        let comp_message = WebSocketMessage::Computation {
            message: ComputationMessage {
                node_id: start_node_id,
                request_id: result.request_id,
                data: ModelInput::First(input),
            },
        };

        let json = serde_json::to_string(&comp_message).unwrap();
        let _ = tx.send(Message::Text(json.into()));
    }
}

async fn get_worker(state: &Arc<AppState>, node_id: Uuid) -> Option<Uuid> {
    for i in 1..10 {
        match state.graph.read().await.get_worker(&node_id) {
            Some(worker_id) => return Some(worker_id),
            None => {}
        }
        let retry_in = 10 * i;
        println!("No worker found for node {node_id}. Retrying in {retry_in}s...");
        sleep(Duration::from_secs(retry_in)).await;
    }
    println!("No worker found for node {node_id} after retrying. Giving up...");
    None
}
fn arg_max(values: &[f64]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

async fn broadcast_connected_users(clients: &HashMap<Uuid, UnboundedSender<Message>>) {
    let message = WebSocketMessage::ConnectedUsers {
        message: clients.len(),
    };

    let json = serde_json::to_string(&message).unwrap();

    for tx in clients.values() {
        let _ = tx.send(Message::Text(json.clone().into()));
    }
}
