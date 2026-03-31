//! Forward pass orchestration - Final RMSNorm + LM Head
//!
//! GGUF tensor convention for output.weight:
//!   ne = [n_embd, vocab_size]
//!   ne[0] = n_embd (contiguous dimension, columns)
//!   ne[1] = vocab_size (rows)
//!   Data: vocab_size rows, each with n_embd elements
//!   logits[j] = dot(row_j, hidden)
//!   row_j starts at byte offset j * bytes_per_row

use crate::loader::gguf::GgufModel;
use anyhow::Result;
use tracing::info;

/// Final forward pass: RMSNorm + LM head projection
pub fn final_forward(hidden: &[f32], model: &GgufModel) -> Result<Vec<f32>> {
    let n_embd = model.metadata.n_embd;
    let vocab_size = model.metadata.vocab_size;

    if vocab_size == 0 {
        anyhow::bail!("vocab_size is 0");
    }

    // 1. Final RMSNorm
    // Try multiple possible tensor names for the output norm
    let norm_raw = model.get_tensor_slice("output_norm.weight")
        .or_else(|_| model.get_tensor_slice("token_embd_norm.weight"))
        .or_else(|_| model.get_tensor_slice("model.norm.weight"))?;

    let norm_vec: Vec<f32> = if norm_raw.len() >= n_embd * 4 {
        norm_raw[..n_embd * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        info!("Output norm too short ({} bytes), using identity", norm_raw.len());
        vec![1.0f32; n_embd]
    };

    // Apply RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // Use f64 for numerical stability
    let sum_sq: f64 = hidden.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rms: f64 = (sum_sq / n_embd as f64 + 1e-5f64).sqrt();
    let normed: Vec<f32> = hidden.iter().zip(norm_vec.iter())
        .map(|(&xi, &wi)| {
            let n = (xi as f64) / rms;
            (n * wi as f64) as f32
        })
        .collect();

    let h_max: f32 = normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let h_min: f32 = normed.iter().cloned().fold(f32::INFINITY, f32::min);
    let h_nan = normed.iter().filter(|v| v.is_nan()).count();
    info!("After final norm: max={:.6}, min={:.6}, nan_count={}", h_max, h_min, h_nan);

    // 2. LM Head: logits[j] = dot(output_weight_row_j, normed)
    let output_name = "output.weight";
    let output_raw = model.get_tensor_slice(output_name)
        .map_err(|e| anyhow::anyhow!("Failed to get '{}': {}", output_name, e))?;

    let output_dtype = model.tensor_map
        .get(output_name)
        .map(|r| r.dtype)
        .unwrap_or(crate::loader::TensorDtype::F32);

    info!("LM Head: tensor='{}', dtype={:?}, bytes={}", output_name, output_dtype, output_raw.len());

    let logits = compute_logits(&normed, output_raw, output_dtype, n_embd, vocab_size);

    let logit_max: f32 = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let logit_min: f32 = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let logit_nonzero: usize = logits.iter().filter(|&&v| v.abs() > 1e-10).count();
    info!("Logits: max={:.4}, min={:.4}, non_zero={}/{}", logit_max, logit_min, logit_nonzero, vocab_size);

    Ok(logits)
}

/// Compute logits row-by-row from mmap'd quantized data.
///
/// GGUF convention for output.weight: ne = [n_embd, vocab_size]
/// → vocab_size rows, each with n_embd elements
/// → logits[j] = dot(dequant(row_j), hidden)
///
/// This is memory efficient: only one row in memory at a time.
fn compute_logits(
    hidden: &[f32],
    raw: &[u8],
    dtype: crate::loader::TensorDtype,
    n_embd: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let block_size = dtype.block_size();
    let block_bytes = dtype.block_bytes();
    let blocks_per_row = (n_embd + block_size - 1) / block_size;
    let bytes_per_row = blocks_per_row * block_bytes;

    if bytes_per_row == 0 {
        info!("bytes_per_row is 0, returning zero logits");
        return vec![0.0f32; vocab_size];
    }

    let effective_vocab = (raw.len() / bytes_per_row).min(vocab_size);

    if effective_vocab == 0 {
        info!("No complete rows in LM head data ({} bytes, need {} per row)",
            raw.len(), bytes_per_row);
        return vec![0.0f32; vocab_size];
    }

    info!("Computing logits: {} vocab rows, {} bytes/row, {} total bytes",
        effective_vocab, bytes_per_row, raw.len());

    let mut logits = vec![0.0f32; vocab_size];

    for j in 0..effective_vocab {
        let start = j * bytes_per_row;
        if start + bytes_per_row > raw.len() { break; }

        let row_bytes = &raw[start..start + bytes_per_row];
        let row_weights = crate::loader::dequant::dequantize(row_bytes, dtype, &[n_embd]);

        let mut dot = 0.0f32;
        for i in 0..n_embd.min(row_weights.len()).min(hidden.len()) {
            dot += row_weights[i] * hidden[i];
        }
        logits[j] = dot;
    }

    logits
}
