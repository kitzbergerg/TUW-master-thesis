use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use axum::{
    Router,
    body::Bytes,
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
use tokio::sync::{Mutex, mpsc::UnboundedSender};
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

mod graph;
mod protos;

use crate::graph::{Node, StructuralGraph};
use crate::protos::{
    Computation, ComputationMessage, ConnectedUsers, ExternalDataEntry, FirstModelInput,
    InferenceRequest, InferenceResult, Initialize, InitializeDone, IntermediateModelData,
    InvalidateCache, ModelConfig, WebSocketMessage, computation_message::Data,
    web_socket_message::Kind,
};

const SYSTEM_PROMPT: &str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions.";

// Main application state
struct AppState {
    // use Mutex instead of RwLock since basically every access is write
    graph: Mutex<ComputationalGraph>,
    tokenizer: Tokenizer,
    vocab_size: usize,
}

struct ComputationalGraph {
    structure: StructuralGraph,
    /// Maps workers ids to their assigned node and socket connection
    workers: HashMap<Uuid, Worker>,
    /// Maps nodes to their workers (inverse of workers)
    assigned_nodes: HashMap<Uuid, (Uuid, Status)>,
    /// Maps request_ids to their data
    inference_requests: HashMap<Uuid, ActiveInferenceRequest>,
    /// Maps nodes to a list of (original user, message) pairs
    pending_requests: HashMap<Uuid, Vec<ActiveComputation>>,
}

impl ComputationalGraph {
    pub async fn add_worker(&mut self, user_id: Uuid, tx: UnboundedSender<Message>) {
        self.workers.insert(
            user_id,
            Worker {
                assigned_node: None,
                sink: tx.clone(),
            },
        );

        let unassigned_node = self
            .structure
            .list_nodes()
            .find(|node| !self.assigned_nodes.contains_key(node));
        if let Some(unassigned_node) = unassigned_node {
            self.assign_worker(user_id, *unassigned_node);
        }
        self.broadcast_connected_users().await;
    }

    pub fn assign_worker(&mut self, worker_id: Uuid, node_id: Uuid) {
        let worker = self.workers.get_mut(&worker_id).unwrap();
        assert!(worker.assigned_node.is_none());
        assert!(!self.assigned_nodes.contains_key(&node_id));

        tracing::info!("Assigning worker {worker_id} to node {node_id}");
        self.assigned_nodes
            .insert(node_id, (worker_id, Status::Loading));
        worker.assigned_node = Some(node_id);

        let message = WebSocketMessage {
            kind: Some(Kind::Initialize(Initialize {
                node_id: node_id.to_string(),
                message: Some(self.structure.get_node(&node_id).unwrap().data.clone()),
            })),
        };
        worker
            .sink
            .send(Message::Binary(proto_encode(&message).into()))
            .unwrap();
    }

    pub async fn remove_worker(&mut self, user_id: Uuid) {
        self.inference_requests
            .retain(|_, v| v.original_user_id != user_id);
        self.pending_requests
            .values_mut()
            .for_each(|vec| vec.retain(|computation| computation.original_user_id != user_id));

        if let Some(assigned_node) = self.workers.remove(&user_id).unwrap().assigned_node {
            self.assigned_nodes.remove(&assigned_node).unwrap();

            // invalidate all previous requests and caches
            // necessary since workers need to rebuild kv caches
            // TODO: allow graceful shutdown of worker
            tracing::info!("Invalidating caches and rescheduling inference requests...");
            self.invalidate_cache(None);
            self.pending_requests.clear();
            let drained = self
                .inference_requests
                .drain()
                .map(|(_, request)| request)
                .collect::<Vec<_>>();
            for request in drained {
                self.send_inference_request(
                    Uuid::new_v4(),
                    request.original_user_id,
                    request.input_token_len,
                    request.tokens,
                )
                .await;
            }

            tracing::info!("Attempting to reassign node {assigned_node}...");
            match self
                .workers
                .iter()
                .find(|(_, worker)| worker.assigned_node.is_none())
            {
                Some((new_worker_id, _)) => self.assign_worker(*new_worker_id, assigned_node),
                None => tracing::info!("Currently no suitable worker to reassign node"),
            }
        }
        self.broadcast_connected_users().await;
    }

    async fn broadcast_connected_users(&self) {
        let message = WebSocketMessage {
            kind: Some(Kind::ConnectedUsers(ConnectedUsers {
                message: self.workers.len() as u32,
            })),
        };
        let data = proto_encode(&message);
        self.workers.values().for_each(|worker| {
            worker
                .sink
                .send(Message::Binary(data.clone().into()))
                .unwrap();
        })
    }

    async fn send_inference_request(
        &mut self,
        request_id: Uuid,
        original_user_id: Uuid,
        input_token_len: usize,
        input_ids: Vec<u32>,
    ) {
        let node_id = self.structure.start_node_id;
        let message = WebSocketMessage {
            kind: Some(Kind::Computation(Computation {
                message: Some(ComputationMessage {
                    node_id: node_id.to_string(),
                    request_id: request_id.to_string(),
                    data: Some(Data::First(FirstModelInput {
                        input_ids: input_ids.clone(),
                    })),
                }),
            })),
        };
        self.inference_requests.insert(
            request_id,
            ActiveInferenceRequest {
                original_user_id,
                input_token_len,
                tokens: input_ids.clone(),
            },
        );
        self.schedule_computation(
            node_id,
            ActiveComputation {
                original_user_id,
                message,
            },
        );
    }

    pub fn schedule_computation(&mut self, node_id: Uuid, computation: ActiveComputation) {
        let worker = self.assigned_nodes.get(&node_id).copied();
        match worker {
            Some((worker_id, Status::Available)) => self.send_computation(worker_id, computation),
            _ => {
                // no worker for now, handle message later
                tracing::info!("Waiting for node {node_id} to be ready...");
                self.pending_requests
                    .entry(node_id)
                    .or_default()
                    .push(computation);
            }
        }
    }

    fn send_computation(&mut self, worker_id: Uuid, computation: ActiveComputation) {
        tracing::debug!("Sending computation to worker {worker_id}");
        self.workers
            .get(&worker_id)
            .unwrap()
            .sink
            .send(Message::Binary(proto_encode(&computation.message).into()))
            .unwrap();
    }

    fn invalidate_cache(&self, request_id: Option<Uuid>) {
        match request_id {
            Some(request_id) => tracing::debug!("Invalidating cache for request {request_id}"),
            None => tracing::debug!("Invalidating all caches"),
        }
        let message = WebSocketMessage {
            kind: Some(Kind::InvalidateCache(InvalidateCache {
                request_id: request_id.map(|uuid| uuid.to_string()),
            })),
        };
        let encoded = proto_encode(&message);

        self.workers
            .values()
            .filter(|worker| worker.assigned_node.is_some())
            .for_each(|worker| worker.sink.send(encoded.clone().into()).unwrap());
    }
}

struct Worker {
    assigned_node: Option<Uuid>,
    sink: UnboundedSender<Message>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Status {
    Loading,
    Available,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ActiveInferenceRequest {
    original_user_id: Uuid,
    input_token_len: usize,
    tokens: Vec<u32>,
}
struct ActiveComputation {
    original_user_id: Uuid,
    message: WebSocketMessage,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TmpExtEntry {
    path: String,
    data: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                format!(
                    "{}=debug,tower_http=debug,axum::rejection=trace",
                    env!("CARGO_CRATE_NAME")
                )
                .into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

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
    let end = Node::new(None, end_config);
    let start = Node::new(Some(end.id), start_config);

    // Create shared state
    let state = Arc::new(AppState {
        graph: Mutex::new(ComputationalGraph {
            structure: StructuralGraph::new(vec![start, end]),
            workers: HashMap::new(),
            assigned_nodes: HashMap::new(),
            inference_requests: HashMap::new(),
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
    let server_addr = "127.0.0.1:3000";
    let listener = tokio::net::TcpListener::bind(server_addr).await.unwrap();
    tracing::info!("Server listening on http://{server_addr}");
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
                tracing::error!("Error sending message: {err}");
                break;
            }
        }
    });

    // Generate a unique user ID
    let user_id = Uuid::new_v4();
    tracing::info!("Client connected: {user_id}");
    state.graph.lock().await.add_worker(user_id, tx).await;

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        match result {
            Ok(Message::Binary(data)) => handle_message(data, user_id, &state).await,
            Ok(Message::Close(_)) => break,
            _ => tracing::warn!("Unexpected message"),
        }
    }

    // Client disconnected
    tracing::info!("Client disconnected: {user_id}");
    state.graph.lock().await.remove_worker(user_id).await;
    send_task.abort();
}

async fn handle_message(data: Bytes, user_id: Uuid, state: &Arc<AppState>) {
    let message = WebSocketMessage::decode(Cursor::new(&data)).unwrap();
    let message = message.kind.unwrap();

    match message {
        Kind::InitializeDone(InitializeDone { node_id }) => {
            tracing::info!("Worker {user_id} ready for node {node_id}");
            let node_id = Uuid::parse_str(&node_id).unwrap();
            let mut lock = state.graph.lock().await;
            lock.assigned_nodes.get_mut(&node_id).unwrap().1 = Status::Available;

            if let Some(pending_requests) = lock.pending_requests.remove(&node_id) {
                tracing::info!("Processing pending requests...");
                pending_requests.into_iter().for_each(|computation| {
                    lock.send_computation(user_id, computation);
                });
            }
        }
        Kind::InferenceRequest(InferenceRequest { message }) => {
            handle_inference_request(message, user_id, state).await;
        }
        Kind::Computation(Computation { message }) => {
            handle_computation_result(message.unwrap(), user_id, state).await
        }
        _ => {
            tracing::error!("Received unexpected message type from client");
        }
    }
}

async fn handle_inference_request(prompt: String, user_id: Uuid, state: &Arc<AppState>) {
    let request_id = Uuid::new_v4();
    tracing::info!(
        "Scheduling inference request {request_id} for user {user_id} with prompt '{prompt}'",
    );

    let message = format!("System: {SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:");

    // Tokenize input text
    let input_ids = state
        .tokenizer
        .encode(message, false)
        .unwrap()
        .get_ids()
        .to_vec();

    // Send inference request
    state
        .graph
        .lock()
        .await
        .send_inference_request(request_id, user_id, input_ids.len(), input_ids)
        .await;
}

async fn handle_computation_result(
    result: ComputationMessage,
    worker_id: Uuid,
    state: &Arc<AppState>,
) {
    let request_id = Uuid::parse_str(&result.request_id).unwrap();
    let node_id = Uuid::parse_str(&result.node_id).unwrap();
    tracing::debug!("Worker {worker_id} computed request {request_id} for node {node_id}");

    {
        let mut lock = state.graph.lock().await;

        if let Some(next_node_id) = lock.structure.get_next_node(&node_id) {
            let original_user_id = match lock.inference_requests.get(&request_id) {
                Some(active_request) => active_request.original_user_id,
                // user disconnected or request was cancelled
                None => return,
            };
            tracing::debug!("Forwarding to next node: {next_node_id}");
            let message = WebSocketMessage {
                kind: Some(Kind::Computation(Computation {
                    message: Some(ComputationMessage {
                        node_id: next_node_id.to_string(),
                        request_id: result.request_id.clone(),
                        data: result.data.clone(),
                    }),
                })),
            };

            lock.schedule_computation(
                next_node_id,
                ActiveComputation {
                    original_user_id,
                    message,
                },
            );
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
    tracing::debug!(
        "Generated token: {}, Decoded: {}",
        next_token,
        state.tokenizer.decode(&[next_token], false).unwrap()
    );

    // Update active request
    {
        let mut lock = state.graph.lock().await;
        let active_request = match lock.inference_requests.get_mut(&request_id) {
            Some(req) => req,
            // user disconnected or request was cancelled
            None => return,
        };
        active_request.tokens.push(next_token);

        // Tokens of 'User:', 'Assistant:' and 'System:'
        let should_stop_generating = matches!(
            active_request.tokens.as_slice(),
            [.., 12982, 25] | [.., 48902, 25] | [.., 11964, 25]
        );
        if should_stop_generating {
            // Generate final output and send to client
            let active_request = lock.inference_requests.remove(&request_id).unwrap();
            let output_text = state
                .tokenizer
                .decode(
                    &active_request.tokens
                        [active_request.input_token_len..active_request.tokens.len() - 2],
                    false,
                )
                .unwrap();
            tracing::info!("Inference result for request {request_id}: {output_text}");

            let message = WebSocketMessage {
                kind: Some(Kind::InferenceResult(InferenceResult {
                    message: output_text,
                })),
            };
            if let Some(original_user_id) = lock.workers.get(&active_request.original_user_id) {
                original_user_id
                    .sink
                    .send(Message::Binary(proto_encode(&message).into()))
                    .unwrap();
            }
            lock.invalidate_cache(Some(request_id));
            return;
        }

        // Send next computation
        let active_request = active_request.clone();
        let node_id = lock.structure.start_node_id;
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
        lock.schedule_computation(
            node_id,
            ActiveComputation {
                original_user_id: active_request.original_user_id,
                message,
            },
        );
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

fn proto_encode<T: ProstMessage>(message: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(message.encoded_len());
    message.encode(&mut buf).unwrap();
    buf
}
