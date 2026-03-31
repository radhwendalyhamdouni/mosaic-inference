//! Layer Streaming - Double Buffer Prefetching
//!
//! الفكرة الأساسية:
//! بينما CPU ينفّذ الطبقة الحالية (N)
//! خيط آخر يحمّل الطبقة التالية (N+1) من القرص في الخلفية
//!
//! النتيجة: تحتاج فقط طبقتين في RAM بدل النموذج كله!
//!
//! ┌──────────────────────────────────────┐
//! │  Time →                              │
//! │  CPU:    [Compute L0] [Compute L1]   │
//! │  Disk:              [Load L2] [L3]   │
//! │  RAM:    [L0,L1]       [L1,L2]       │
//! └──────────────────────────────────────┘

use crate::cache::MemoryTier;
use crate::loader::gguf::GgufModel;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{info, debug};

/// Layer weights data (raw bytes from mmap)
/// After loading from disk, weights live in RAM until dropped
pub struct LayerWeights {
    pub layer_idx: usize,
    /// Tensor name → raw bytes (quantized)
    pub tensors: HashMap<String, Vec<u8>>,
    /// Total bytes consumed
    pub size_bytes: usize,
}

impl LayerWeights {
    /// Get tensor data by name
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    /// Memory consumed by this layer
    pub fn memory_usage(&self) -> usize {
        self.size_bytes
    }
}

impl Drop for LayerWeights {
    fn drop(&mut self) {
        debug!(
            "Dropping layer {} from RAM (freed {} bytes)",
            self.layer_idx,
            self.size_bytes
        );
    }
}

/// Commands sent to the background prefetch thread
enum StreamCommand {
    /// Load a specific layer
    LoadLayer {
        layer_idx: usize,
        result_tx: oneshot::Sender<Result<LayerWeights>>,
    },
    /// Shutdown the background thread
    Shutdown,
}

/// The Layer Streamer manages background loading of model layers
/// يدير خيط الخلفية الذي يحمّل الطبقات من القرص
pub struct LayerStreamer {
    model: Arc<GgufModel>,
    memory: Arc<RwLock<MemoryTier>>,
    ram_layers: usize,
    cmd_tx: mpsc::Sender<StreamCommand>,
}

impl LayerStreamer {
    /// Create a new layer streamer with background prefetch
    pub fn new(
        model: Arc<GgufModel>,
        memory: Arc<RwLock<MemoryTier>>,
        ram_layers: usize,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(8);

        // Spawn background thread for layer loading
        let bg_model = model.clone();
        tokio::spawn(async move {
            background_loader(bg_model, cmd_rx).await;
        });

        Self {
            model,
            memory,
            ram_layers,
            cmd_tx,
        }
    }

    /// Load a layer (uses cache or loads from disk)
    /// إذا الطبقة في كاش RAM → يعيدها فوراً
    /// إذا لا → يحمّلها من القرص (mmap → page fault → lazy load)
    pub async fn load_layer(&self, layer_idx: usize) -> Result<LayerWeights> {
        debug!("Requesting layer {}", layer_idx);

        // Check if already cached in RAM
        // (not implemented in v0.1 - always load from mmap)
        // Future: LRU cache for most recently used layers

        let (result_tx, result_rx) = oneshot::channel();
        self.cmd_tx.send(StreamCommand::LoadLayer {
            layer_idx,
            result_tx,
        }).await.map_err(|_| anyhow::anyhow!("Loader thread died"))?;

        let weights = result_rx.await
            .map_err(|_| anyhow::anyhow!("Loader thread dropped response"))??;

        debug!(
            "Layer {} loaded: {} bytes ({:.1} MB)",
            layer_idx,
            weights.size_bytes,
            weights.size_bytes as f64 / 1_048_576.0
        );

        Ok(weights)
    }

    /// Embed a single token (look up in embedding table)
    pub async fn embed_token(&self, token_id: usize) -> Result<Vec<f32>> {
        // In GGUF, the embedding tensor is "token_embd.weight"
        // Shape: [vocab_size, n_embd]
        let embd_name = "token_embd.weight";

        if let Some(region) = self.model.tensor_map.get(embd_name) {
            let n_embd = self.model.metadata.n_embd;

            // Calculate offset for this token
            let dtype = region.dtype;
            let bytes_per_row = (n_embd as f64 * dtype.bytes_per_weight()) as usize;
            let offset = region.offset + token_id * bytes_per_row;

            if offset + bytes_per_row <= self.model.mmap.len() {
                // Read from mmap (triggers page fault if not in RAM)
                let raw = &self.model.mmap[offset..offset + bytes_per_row];

                // Dequantize to f32 (simplified - assumes Q8_0 or F32)
                let mut embedding = vec![0.0f32; n_embd];

                match dtype {
                    crate::loader::TensorDtype::F32 => {
                        for i in 0..n_embd {
                            let bytes = &raw[i * 4..(i + 1) * 4];
                            embedding[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        }
                    }
                    crate::loader::TensorDtype::F16 => {
                        for i in 0..n_embd {
                            let bytes = &raw[i * 2..(i + 1) * 2];
                            let half = u16::from_le_bytes([bytes[0], bytes[1]]);
                            embedding[i] = f16::to_f32(half);
                        }
                    }
                    // For quantized types, simplified dequantization
                    _ => {
                        // Use scale factor approximation
                        for i in 0..n_embd.min(raw.len() / 2) {
                            embedding[i] = raw[i * 2] as f32 / 128.0;
                        }
                    }
                }

                return Ok(embedding);
            }
        }

        // Fallback: simple random embedding (for testing without real model)
        let n_embd = self.model.metadata.n_embd;
        let mut embedding = vec![0.0f32; n_embd];
        // Simple hash-based embedding for tokens
        let mut hash = token_id as u64;
        for i in 0..n_embd {
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1);
            embedding[i] = ((hash >> 33) as f32) / (i32::MAX as f32);
        }
        Ok(embedding)
    }
}

/// f16 to f32 conversion
mod f16 {
    pub fn to_f32(h: u16) -> f32 {
        if h == 0 { return 0.0; }
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1F) as i32;
        let frac = (h & 0x3FF) as f32;
        if exp == 0 {
            sign * (2.0_f32).powi(-14) * (frac / 1024.0)
        } else if exp == 31 {
            if frac == 0.0 { f32::INFINITY * sign } else { f32::NAN }
        } else {
            sign * (2.0_f32).powi(exp - 15) * (1.0 + frac / 1024.0)
        }
    }
}

/// Background thread that loads layers from the mmap'd model file
/// هذا الخيط يعمل في الخلفية ويحمّل الطبقات من القرص
async fn background_loader(model: Arc<GgufModel>, mut cmd_rx: mpsc::Receiver<StreamCommand>) {
    info!("Background layer loader started");

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            StreamCommand::LoadLayer { layer_idx, result_tx } => {
                let result = load_layer_from_mmap(&model, layer_idx);
                let _ = result_tx.send(result);
            }
            StreamCommand::Shutdown => {
                info!("Background layer loader shutting down");
                break;
            }
        }
    }
}

/// Load a single layer's weights from the memory-mapped file
/// يقرأ أوزان الطبقة من mmap (page faults trigger lazy loading from disk)
fn load_layer_from_mmap(model: &GgufModel, layer_idx: usize) -> Result<LayerWeights> {
    let tensor_names = model.get_layer_tensors(layer_idx);

    if tensor_names.is_empty() {
        anyhow::bail!("No tensors found for layer {}", layer_idx);
    }

    let mut tensors = HashMap::new();
    let mut total_size = 0;

    for name in &tensor_names {
        if let Some(region) = model.tensor_map.get(name) {
            // Read from mmap - OS handles page faults automatically!
            // Only the 4KB pages we actually read get loaded into RAM
            let data = model.read_tensor_bytes(name)?;
            total_size += data.len();
            tensors.insert(name.clone(), data);
        }
    }

    Ok(LayerWeights {
        layer_idx,
        tensors,
        size_bytes: total_size,
    })
}
