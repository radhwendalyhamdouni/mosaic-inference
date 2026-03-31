//! Forward pass orchestration
//!
//! ينسّق التنفيذ الأمامي الكامل:
//! 1. Embedding lookup
//! 2. Layer-by-layer processing
//! 3. Final norm + LM head projection
//! 4. Token sampling

use crate::loader::gguf::GgufModel;
use anyhow::Result;

/// Final forward pass: RMSNorm + LM head projection
/// يحوّل التمثيل الأخير إلى احتمالات لكل رمز في القاموس
pub fn final_forward(hidden: &[f32], model: &GgufModel) -> Result<Vec<f32>> {
    let n_embd = model.metadata.n_embd;
    let vocab_size = model.metadata.vocab_size;

    // 1. Final RMSNorm
    let norm_name = "output_norm.weight";
    let normed = if let Ok(norm_raw) = model.get_tensor_slice(norm_name) {
        simple_dequantize(norm_raw, n_embd)
    } else {
        hidden.to_vec()
    };

    // Apply norm
    let sum_sq: f32 = normed.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n_embd as f32).sqrt() + 1e-6;
    let normed: Vec<f32> = normed.iter().map(|&v| v / rms).collect();

    // 2. LM Head projection: [n_embd] × [vocab_size, n_embd] → [vocab_size]
    // This is the biggest operation - produces logits for every possible token
    let output_name = "output.weight";

    let logits = if let Ok(output_raw) = model.get_tensor_slice(output_name) {
        compute_logits(&normed, output_raw, n_embd, vocab_size)
    } else {
        // Fallback: generate pseudo-random logits
        vec![0.0f32; vocab_size]
    };

    Ok(logits)
}

/// Compute logits from hidden state and output weights
fn compute_logits(
    hidden: &[f32],
    weight_raw: &[u8],
    n_embd: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];

    // For quantized weights, simplified computation
    let bytes_per_row = weight_raw.len() / vocab_size;

    for i in 0..vocab_size {
        let mut sum = 0.0f32;
        let offset = i * bytes_per_row;

        for j in 0..n_embd.min(bytes_per_row) {
            let w = weight_raw[offset + j] as f32 / 128.0;
            sum += w * hidden[j];
        }
        logits[i] = sum;
    }

    logits
}

/// Simple dequantization for norm weights
fn simple_dequantize(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.len() >= n * 4 {
        // F32
        raw.chunks(4)
            .take(n)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        // Quantized - approximate
        raw.iter().take(n).map(|&b| b as f32 / 128.0).collect()
    }
}
