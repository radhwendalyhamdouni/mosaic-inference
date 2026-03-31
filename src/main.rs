#![allow(dead_code)]

//! # Mosaic Inference
//!
//! تشغيل أي نموذج لغوي كبير على أجهزة ضعيفة (4GB RAM)
//! باستخدام: Layer Streaming + mmap + BitNet + KV Disk Offload
//!
//! الفكرة الجوهرية:
//! - الأوزان تبقى على القرص (mmap) ولا تُحمّل كلها في RAM
//! - الطبقات تُحمّل طبقة بطبقة (Layer Streaming) مع Prefetch
//! - KV Cache ينتقل للقرص عندما يمتلئ RAM
//! - دعم BitNet 1.58-bit لأوزان {-1, 0, +1}

mod loader;
mod engine;
mod cache;
mod server;
mod sampler;

use anyhow::Result;
use clap::Parser;
use tracing::info;

/// Mosaic Inference - Run any LLM on weak hardware
#[derive(Parser, Debug)]
#[command(name = "mosaic-inference")]
#[command(author = "Radhwen Daly Hamdouni")]
#[command(version)]
#[command(about = "Run any LLM on 4GB RAM using Layer Streaming + mmap", long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: String,

    /// Context window size (default: 8192)
    #[arg(short, long, default_value_t = 8192)]
    ctx_size: usize,

    /// Max layers to keep in RAM at once (default: 2)
    #[arg(long, default_value_t = 2)]
    ram_layers: usize,

    /// KV Cache disk path (default: .kv_cache/)
    #[arg(long, default_value = ".kv_cache")]
    kv_cache_dir: String,

    /// Server port (default: 8080)
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Server host (default: 0.0.0.0)
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let args = Args::parse();
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_target(false)
        .init();

    println!(r#"
   __  ___       _       _ __  __
  /  |/  /_____( )_____(_)\ \/ /
 / /|_/ / ___// / ___/ /  \  /
/ /  / / /__ / / /__ / /   / /
/_/  /_/\___//_/\___//_/   /_/

    Mosaic Inference v0.1.0
    Run any LLM on 4GB RAM
"#);

    info!("Model: {}", args.model);
    info!("Context size: {}", args.ctx_size);
    info!("RAM layers: {}", args.ram_layers);
    info!("KV cache dir: {}", args.kv_cache_dir);

    // Step 1: Load model metadata with mmap
    info!("Loading model metadata (mmap)...");
    let model = loader::gguf::GgufModel::load(&args.model)?;

    info!("Model: {} v{}", model.metadata.name, model.metadata.version);
    info!("Architecture: {}", model.metadata.architecture);
    info!("Vocabulary size: {}", model.metadata.vocab_size);
    info!("Total layers: {}", model.metadata.n_layers);
    info!("Hidden size: {}", model.metadata.n_embd);
    info!("Head size: {}", model.metadata.n_head);
    info!("File size: {} MB", model.file_size / 1_048_576);

    // Step 2: Calculate memory requirements
    let layer_size_bytes = estimate_layer_size(&model);
    let ram_needed = layer_size_bytes * args.ram_layers;
    let total_model_bytes = layer_size_bytes * model.metadata.n_layers;

    info!("=== Memory Estimation ===");
    info!("Single layer size: {:.1} MB", layer_size_bytes as f64 / 1_048_576.0);
    info!("Model total size: {:.1} MB", total_model_bytes as f64 / 1_048_576.0);
    info!("RAM for {} layers: {:.1} MB", args.ram_layers, ram_needed as f64 / 1_048_576.0);

    // Step 3: Create tiered memory system
    let memory_tier = cache::MemoryTier::new(
        args.ctx_size,
        model.metadata.n_layers,
        args.ram_layers,
        &args.kv_cache_dir,
    );

    info!("=== Memory Tiers Active ===");
    info!("Tier 1: RAM (hot KV cache)");
    info!("Tier 2: Disk ({})", args.kv_cache_dir);

    // Step 4: Create inference engine
    let engine = engine::MosaicEngine::new(model, memory_tier, args.ram_layers);

    info!("Engine initialized successfully!");

    // Step 5: Start HTTP server
    info!("Starting server at http://{}:{}", args.host, args.port);
    server::start_server(engine, args.host, args.port).await?;

    Ok(())
}

/// Estimate the size of a single transformer layer in bytes
fn estimate_layer_size(model: &loader::gguf::GgufModel) -> usize {
    // Each layer has: W_q + W_k + W_v + W_o + W_ffn_gate + W_ffn_up + W_ffn_down
    // Qwen2 uses GQA: n_head_kv < n_head
    let n_embd = model.metadata.n_embd;
    let n_head = model.metadata.n_head;
    let n_head_kv = model.metadata.n_head_kv.unwrap_or(n_head);
    let head_dim = n_embd / n_head;
    let n_ff = model.metadata.n_ff.unwrap_or(n_embd * 8 / 3);

    // Assuming Q4_K quantization (4.5 bits per weight effectively)
    let bytes_per_weight = 0.5625; // Q4_K average

    let q_proj = n_embd * (n_head * head_dim);
    let k_proj = n_embd * (n_head_kv * head_dim);
    let v_proj = n_embd * (n_head_kv * head_dim);
    let o_proj = (n_head * head_dim) * n_embd;
    let ffn_gate = n_embd * n_ff;
    let ffn_up = n_embd * n_ff;
    let ffn_down = n_ff * n_embd;

    let total_weights = q_proj + k_proj + v_proj + o_proj + ffn_gate + ffn_up + ffn_down;
    (total_weights as f64 * bytes_per_weight) as usize
}
