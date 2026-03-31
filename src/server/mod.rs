//! HTTP Server with OpenAI-compatible API
//!
//! يوفر API مشابه لـ OpenAI حتى تتمكن من استخدامه مع أي عميل

use crate::engine::MosaicEngine;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// Application state shared across all requests
type AppState = Arc<MosaicEngine>;

/// Start the HTTP server
pub async fn start_server(engine: MosaicEngine, host: String, port: u16) -> anyhow::Result<()> {
    let state = Arc::new(engine);

    let app = Router::new()
        .route("/", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/health", get(health))
        .route("/stats", get(memory_stats))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    info!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
async fn health() -> &'static str {
    "Mosaic Inference is running"
}

/// List available models
async fn list_models(State(engine): State<AppState>) -> Json<serde_json::Value> {
    let info = engine.model_info();
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": info["name"],
            "object": "model",
            "owned_by": "mosaic-inference",
            "permission": [],
            "root": info["name"],
            "parent": null,
        }]
    }))
}

/// Memory stats endpoint
async fn memory_stats(State(engine): State<AppState>) -> Json<serde_json::Value> {
    let info = engine.model_info();
    Json(serde_json::json!({
        "model": info,
        "memory": {
            "strategy": "Layer Streaming + mmap + KV Disk Offload",
            "description": "Only 2 layers in RAM at a time. Weights on disk (mmap). KV cache tiered (RAM + Disk)."
        }
    }))
}

/// Chat completions request (OpenAI compatible)
#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Chat completions response
#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// POST /v1/chat/completions
async fn chat_completions(
    State(engine): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    info!("Chat completion request: {} messages", req.messages.len());

    // Build prompt from messages
    let prompt: String = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let max_tokens = req.max_tokens.unwrap_or(256);

    // Run inference
    let output = engine.complete(&prompt, max_tokens)
        .await
        .map_err(|e| {
            error!("Inference failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let completion_tokens = output.split_whitespace().count();

    Ok(Json(ChatResponse {
        id: format!("mosaic-{}", uuid_simple()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: output,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt.split_whitespace().count(),
            completion_tokens,
            total_tokens: prompt.split_whitespace().count() + completion_tokens,
        },
    }))
}

/// Regular completions request (OpenAI compatible)
#[derive(Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

/// POST /v1/completions
async fn completions(
    State(engine): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("Completion request: {} chars", req.prompt.len());

    let max_tokens = req.max_tokens.unwrap_or(256);
    let output = engine.complete(&req.prompt, max_tokens)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({
        "id": format!("mosaic-{}", uuid_simple()),
        "object": "text_completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        "model": req.model,
        "choices": [{
            "text": output,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": req.prompt.split_whitespace().count(),
            "completion_tokens": output.split_whitespace().count(),
            "total_tokens": req.prompt.split_whitespace().count() + output.split_whitespace().count()
        }
    })))
}

/// Simple UUID generator
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", t)[..12].to_string()
}

use tracing::error;
