//! Mosaic Inference Engine
//!
//! المحرك الأساسي: Layer Streaming + mmap + KV Cache + Prefill
//!
//! Architecture:
//! 1. Prefill: Process ALL prompt tokens through all layers, building KV cache
//! 2. Decode: Process one token at a time, using cached KV from previous tokens
//! 3. KV cache persists within a single generation request

pub mod layer;
pub mod forward;
pub mod stream;

use crate::loader::gguf::GgufModel;
use std::sync::Arc;
use anyhow::Result;
use tracing::info;

/// Per-layer KV cache: stores (K, V) for each token position.
/// kv_cache[layer_idx][token_pos] = (K_vec, V_vec)
type KvCache = Vec<Vec<(Vec<f32>, Vec<f32>)>>;

/// The core inference engine
pub struct MosaicEngine {
    model: Arc<GgufModel>,
    streamer: stream::LayerStreamer,
}

impl MosaicEngine {
    pub fn new(model: GgufModel, memory: crate::cache::MemoryTier, ram_layers: usize) -> Self {
        let model = Arc::new(model);
        let memory = Arc::new(tokio::sync::RwLock::new(memory));

        info!("Mosaic Engine initialized");
        info!("Strategy: Layer Streaming with mmap ({} RAM layers)", ram_layers);

        Self {
            streamer: stream::LayerStreamer::new(model.clone(), memory, ram_layers),
            model,
        }
    }

    pub fn model_info(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.model.metadata.name,
            "architecture": self.model.metadata.architecture,
            "vocab_size": self.model.metadata.vocab_size,
            "n_layers": self.model.metadata.n_layers,
            "n_embd": self.model.metadata.n_embd,
            "n_head": self.model.metadata.n_head,
            "n_head_kv": self.model.metadata.n_head_kv,
            "n_ff": self.model.metadata.n_ff,
            "ram_layers": 2,
            "file_size_mb": self.model.file_size / 1_048_576,
            "has_tokenizer": self.model.tokenizer.is_some(),
        })
    }

    /// Simple completion API.
    ///
    /// This handles both prefill and decode in a single call:
    /// 1. Encode prompt → token IDs
    /// 2. Prefill: run all prompt tokens through all layers, building KV cache
    /// 3. Decode: generate tokens one at a time using the KV cache
    pub async fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokens = if let Some(ref tok) = self.model.tokenizer {
            tok.encode(prompt)
        } else {
            prompt.chars().map(|c| c as usize + 3).collect()
        };

        info!("=== Generation Start ===");
        info!("Prompt: \"{}\"", &prompt[..prompt.len().min(80)]);
        info!("Input tokens: {:?} ({} total)", &tokens[..tokens.len().min(10)], tokens.len());

        if tokens.is_empty() {
            return Ok(String::new());
        }

        let n_layers = self.model.metadata.n_layers;
        let eos_id = self.model.metadata.eos_token_id as usize;

        // Initialize per-layer KV cache (empty for each layer)
        let mut kv_cache: KvCache = vec![Vec::new(); n_layers];

        // ──────────────────────────────────────────────
        // Phase 1: PREFILL — process all prompt tokens
        // ──────────────────────────────────────────────
        info!("=== Prefill: processing {} prompt tokens ===", tokens.len());
        let mut last_hidden = Vec::new();

        for pos in 0..tokens.len() {
            let token_id = tokens[pos];
            let hidden = self.streamer.embed_token(token_id).await?;
            let mut state = hidden;

            for layer_idx in 0..n_layers {
                let layer_weights = self.streamer.load_layer(layer_idx).await?;

                state = layer::layer_forward_with_cache(
                    &state,
                    &layer_weights,
                    pos,
                    layer_idx,
                    &self.model.metadata,
                    &mut kv_cache[layer_idx],
                )?;

                // Log first and last few layers during prefill
                if pos == tokens.len() - 1 && (layer_idx < 3 || layer_idx >= n_layers - 2) {
                    let max_val = state.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let nan_count = state.iter().filter(|v| v.is_nan()).count();
                    info!("Prefill Layer {}: max={:.4}, nan_count={}", layer_idx, max_val, nan_count);
                }
            }

            last_hidden = state;

            // Log progress every 10 tokens
            if (pos + 1) % 10 == 0 || pos == tokens.len() - 1 {
                info!("Prefill progress: {}/{} tokens", pos + 1, tokens.len());
            }
        }

        info!("=== Prefill complete ===");

        // Get first generated token from the last prompt token's output
        let logits = forward::final_forward(&last_hidden, &self.model)?;
        let mut next_token = crate::sampler::sample_token(&logits, 0.8, 0.9)?;

        let first_tok_str = self.model.tokenizer.as_ref()
            .and_then(|t| t.tokens.get(next_token).cloned())
            .unwrap_or_else(|| format!("<{}>", next_token));
        info!("First generated token: {} ({:?})", next_token, first_tok_str);

        if next_token == eos_id {
            info!("EOS reached immediately after prefill");
            return Ok(String::new());
        }

        let mut output_tokens = vec![next_token];

        // ──────────────────────────────────────────────
        // Phase 2: DECODE — generate one token at a time
        // ──────────────────────────────────────────────
        info!("=== Decode: generating up to {} tokens ===", max_tokens);

        for step in 1..max_tokens {
            let pos = tokens.len() + output_tokens.len() - 1;

            // Embed only the new token
            let hidden = self.streamer.embed_token(next_token).await?;
            let mut state = hidden;

            // Process through all layers (KV cache already has previous tokens)
            for layer_idx in 0..n_layers {
                let layer_weights = self.streamer.load_layer(layer_idx).await?;

                state = layer::layer_forward_with_cache(
                    &state,
                    &layer_weights,
                    pos,
                    layer_idx,
                    &self.model.metadata,
                    &mut kv_cache[layer_idx],
                )?;
            }

            // Final norm + LM head
            let logits = forward::final_forward(&state, &self.model)?;

            next_token = crate::sampler::sample_token(&logits, 0.8, 0.9)?;

            let tok_str = self.model.tokenizer.as_ref()
                .and_then(|t| t.tokens.get(next_token).cloned())
                .unwrap_or_else(|| format!("<{}>", next_token));

            info!("Step {}/{}: token={} ({:?})", step + 1, max_tokens, next_token, tok_str);
            output_tokens.push(next_token);

            if next_token == eos_id {
                info!("EOS reached after {} generated tokens", output_tokens.len());
                break;
            }
        }

        // Decode output tokens to text
        let output_text = if let Some(ref tok) = self.model.tokenizer {
            tok.decode(&output_tokens)
        } else {
            output_tokens.iter()
                .filter_map(|&t| if t > 3 { Some((t as u8).saturating_sub(3) as char) } else { None })
                .collect()
        };

        info!("=== Generation Complete ===");
        info!("Generated {} tokens: \"{}\"", output_tokens.len(),
            if output_text.len() > 200 { format!("{}...", &output_text[..200]) } else { output_text.clone() });

        Ok(output_text)
    }
}
