//! Layer Streaming - Memory-Mapped Layer Loading

use anyhow::Result;
use crate::loader::gguf::GgufModel;
use crate::loader::TensorDtype;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{info, debug, warn};

/// Layer weights data (raw bytes from mmap)
pub struct LayerWeights {
    pub layer_idx: usize,
    pub tensors: HashMap<String, Vec<u8>>,
    pub dtypes: HashMap<String, TensorDtype>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub size_bytes: usize,
}

impl LayerWeights {
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    pub fn dtype_of(&self, name: &str) -> Option<TensorDtype> {
        self.dtypes.get(name).copied()
    }

    pub fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(|v| v.as_slice())
    }
}

impl Drop for LayerWeights {
    fn drop(&mut self) {
        debug!("Dropped layer {} (freed {} bytes)", self.layer_idx, self.size_bytes);
    }
}

enum StreamCommand {
    LoadLayer {
        layer_idx: usize,
        result_tx: oneshot::Sender<Result<LayerWeights>>,
    },
    Shutdown,
}

pub struct LayerStreamer {
    model: Arc<GgufModel>,
    cmd_tx: mpsc::Sender<StreamCommand>,
}

impl LayerStreamer {
    pub fn new(
        model: Arc<GgufModel>,
        _memory: Arc<RwLock<crate::cache::MemoryTier>>,
        _ram_layers: usize,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(8);

        let bg_model = model.clone();
        tokio::spawn(async move {
            background_loader(bg_model, cmd_rx).await;
        });

        Self { model, cmd_tx }
    }

    pub async fn load_layer(&self, layer_idx: usize) -> Result<LayerWeights> {
        let (result_tx, result_rx) = oneshot::channel();
        self.cmd_tx.send(StreamCommand::LoadLayer {
            layer_idx,
            result_tx,
        }).await.map_err(|_| anyhow::anyhow!("Loader thread died"))?;

        result_rx.await
            .map_err(|_| anyhow::anyhow!("Loader thread dropped response"))?
    }

    /// Embed a single token from the embedding table.
    ///
    /// GGUF convention for token_embd.weight:
    ///   ne = [n_embd, vocab_size]
    ///   ne[0] = n_embd (contiguous, columns)
    ///   ne[1] = vocab_size (rows)
    ///   Data layout: vocab_size rows, each with n_embd elements
    ///   embedding[token_id] = data[token_id * n_embd .. token_id * n_embd + n_embd]
    ///
    /// We only dequantize ONE row (the token's row) for efficiency.
    pub async fn embed_token(&self, token_id: usize) -> Result<Vec<f32>> {
        let embd_name = "token_embd.weight";
        let n_embd = self.model.metadata.n_embd;
        let vocab_size = self.model.metadata.vocab_size;

        if token_id >= vocab_size {
            info!("Token {} out of vocab range ({}), using hash embedding", token_id, vocab_size);
            return Ok(hash_embedding(token_id, n_embd));
        }

        let region = match self.model.tensor_map.get(embd_name) {
            Some(r) => r,
            None => {
                warn!("Embedding tensor '{}' not found, using hash embedding", embd_name);
                return Ok(hash_embedding(token_id, n_embd));
            }
        };

        let embd_raw = self.model.get_tensor_slice(embd_name)?;

        // GGUF: ne = [n_embd, vocab_size] → vocab_size rows of n_embd cols
        // Token token_id is at row token_id.
        // Only dequantize that single row (much faster than full tensor!)
        let embedding = crate::loader::dequant::dequantize_row(
            embd_raw, region.dtype, token_id, n_embd
        );

        if embedding.len() != n_embd {
            warn!("Embedding size mismatch: got {}, expected {}", embedding.len(), n_embd);
        }

        // Log embedding stats for debugging
        if token_id < 5 {
            let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
            let nonzero = embedding.iter().filter(|&&v| v.abs() > 1e-10).count();
            info!("Embedding[{}]: len={}, max={:.6}, min={:.6}, nonzero={}/{}",
                token_id, embedding.len(), max_val, min_val, nonzero, n_embd);
        }

        // Clip embedding values to prevent downstream numerical issues
        let clipped: Vec<f32> = embedding.iter()
            .map(|&v| v.clamp(-1e5_f32, 1e5_f32))
            .collect();

        Ok(clipped)
    }
}

fn hash_embedding(token_id: usize, n_embd: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; n_embd];
    let mut hash = token_id as u64;
    for i in 0..n_embd {
        hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1);
        embedding[i] = ((hash >> 33) as f32) / (i32::MAX as f32);
    }
    embedding
}

async fn background_loader(model: Arc<GgufModel>, mut cmd_rx: mpsc::Receiver<StreamCommand>) {
    info!("Background layer loader started");
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            StreamCommand::LoadLayer { layer_idx, result_tx } => {
                let result = load_layer_from_mmap(&model, layer_idx);
                let _ = result_tx.send(result);
            }
            StreamCommand::Shutdown => break,
        }
    }
}

fn load_layer_from_mmap(model: &GgufModel, layer_idx: usize) -> Result<LayerWeights> {
    let tensor_names = model.get_layer_tensors(layer_idx);
    if tensor_names.is_empty() {
        anyhow::bail!("No tensors for layer {}", layer_idx);
    }

    let mut tensors = HashMap::new();
    let mut dtypes = HashMap::new();
    let mut shapes = HashMap::new();
    let mut total_size = 0;

    for name in &tensor_names {
        if let Some(region) = model.tensor_map.get(name) {
            let data = model.read_tensor_bytes(name)?;
            total_size += data.len();
            tensors.insert(name.clone(), data);
            dtypes.insert(name.clone(), region.dtype);
            shapes.insert(name.clone(), region.shape.clone());
        }
    }

    debug!("Loaded layer {}: {} tensors, {} bytes", layer_idx, tensor_names.len(), total_size);
    Ok(LayerWeights { layer_idx, tensors, dtypes, shapes, size_bytes: total_size })
}
