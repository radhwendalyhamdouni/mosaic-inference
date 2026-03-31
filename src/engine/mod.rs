//! Mosaic Inference Engine
//!
//! المحرك الأساسي الذي يجمع كل الابتكارات:
//! 1. Layer Streaming: تحميل طبقة طبقة مع Double Buffering
//! 2. Sparse Attention: تشغيل طبقات مختارة فقط (Circuit Stealing)
//! 3. KV Cache Tiering: الذاكرة الهرمية (RAM → Disk)
//! 4. Native Compilation: رجم الأوزان لتعليمات آلية

pub mod layer;
pub mod forward;
pub mod stream;

use crate::cache::MemoryTier;
use crate::loader::gguf::GgufModel;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::info;

/// The core inference engine
/// يدير سير التنفيذ: تحميل الطبقات → الحساب → تخزين KV → التوقع
pub struct MosaicEngine {
    /// Reference to the memory-mapped model (weights on disk!)
    model: Arc<GgufModel>,
    /// Memory management (RAM + Disk tiers)
    memory: Arc<RwLock<MemoryTier>>,
    /// How many layers to keep in RAM simultaneously
    ram_layers: usize,
    /// Layer streaming pipeline
    streamer: stream::LayerStreamer,
}

impl MosaicEngine {
    /// Create a new inference engine
    pub fn new(
        model: GgufModel,
        memory: MemoryTier,
        ram_layers: usize,
    ) -> Self {
        let model = Arc::new(model);
        let memory = Arc::new(RwLock::new(memory));

        info!("Mosaic Engine initialized");
        info!("Strategy: Layer Streaming with {} RAM layers", ram_layers);

        Self {
            streamer: stream::LayerStreamer::new(
                model.clone(),
                memory.clone(),
                ram_layers,
            ),
            model,
            memory,
            ram_layers,
        }
    }

    /// Get model info for API responses
    pub fn model_info(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.model.metadata.name,
            "architecture": self.model.metadata.architecture,
            "vocab_size": self.model.metadata.vocab_size,
            "n_layers": self.model.metadata.n_layers,
            "n_embd": self.model.metadata.n_embd,
            "n_head": self.model.metadata.n_head,
            "ram_layers": self.ram_layers,
            "file_size_mb": self.model.file_size / 1_048_576,
        })
    }

    /// Simple completion API
    /// يأخذ رسالة ويعيد النص المُنتج
    pub async fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokens = self.tokenize_simple(prompt);
        info!("Input tokens: {}", tokens.len());

        let mut output_tokens = Vec::new();

        for _ in 0..max_tokens {
            let next_token = self.forward_pass(&tokens, &output_tokens).await?;
            output_tokens.push(next_token);

            // Check for EOS
            if next_token == self.model.metadata.eos_token_id as usize {
                break;
            }
        }

        // Decode tokens to text (placeholder)
        let output_text = self.decode_tokens_simple(&output_tokens);
        Ok(output_text)
    }

    /// Execute a full forward pass through selected layers
    /// التنفيذ الأمامي: يمر على كل طبقة مطلوبة
    async fn forward_pass(
        &self,
        prompt_tokens: &[usize],
        generated_tokens: &[usize],
    ) -> Result<usize> {
        let all_tokens: Vec<usize> = prompt_tokens.iter()
            .chain(generated_tokens.iter())
            .copied()
            .collect();

        let pos = all_tokens.len() - 1; // Current position

        // Layer-by-layer processing with streaming
        let mut hidden_state = self.streamer.embed_token(all_tokens[pos]).await?;

        for layer_idx in 0..self.model.metadata.n_layers {
            // Load layer weights (streaming - only ram_layers in memory at once)
            let layer_weights = self.streamer.load_layer(layer_idx).await?;

            // Forward through this layer
            hidden_state = forward::layer_forward(
                &hidden_state,
                &layer_weights,
                pos,
                layer_idx,
                &self.model.metadata,
            )?;

            // Drop layer weights from RAM (frees memory for next layer!)
            drop(layer_weights);

            // Store KV cache
            {
                let mut mem = self.memory.write().await;
                mem.store_kv(layer_idx, pos, &hidden_state)?;
            }
        }

        // Final norm + projection
        let logits = forward::final_forward(&hidden_state, &self.model)?;

        // Sample next token
        let next_token = crate::sampler::sample_token(&logits, 0.8, 1.0)?;

        Ok(next_token)
    }

    /// Simple tokenization placeholder
    /// في المرحلة الحالية: يقسم النص لكلمات بسيطة
    /// المستقبل: سنستخدم tokenizer حقيقي من GGUF
    fn tokenize_simple(&self, text: &str) -> Vec<usize> {
        // Placeholder: character-level tokenization
        // In production, we'd use the actual tokenizer from GGUF
        text.chars().map(|c| c as usize + 3).collect()
    }

    /// Simple token decoding placeholder
    fn decode_tokens_simple(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&t| {
                if t > 3 {
                    Some((t as u8 - 3) as char)
                } else {
                    None
                }
            })
            .collect()
    }
}
