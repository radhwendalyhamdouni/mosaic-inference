//! Transformer Layer - Single layer forward pass with KV cache support
//!
//! هيكل الطبقة الواحدة:
//! Input → RMSNorm → Q/K/V → RoPE → Attention (with KV cache) → Output → +Residual
//!        → RMSNorm → FFN(SwiGLU) → +Residual → Output

use crate::loader::ModelMetadata;
use crate::engine::stream::LayerWeights;
use crate::loader::dequant::dequantize;
use crate::loader::TensorDtype;
use anyhow::Result;
use tracing::trace;

/// Maximum hidden value magnitude to prevent numerical overflow
const MAX_HIDDEN_VAL: f64 = 1e5;
/// Minimum RMS value to prevent division issues
const MIN_RMS: f64 = 1e-6;

/// Execute forward pass through a single transformer layer with KV cache.
pub fn layer_forward_with_cache(
    hidden: &[f32],
    weights: &LayerWeights,
    pos: usize,
    layer_idx: usize,
    meta: &ModelMetadata,
    kv_cache: &mut Vec<(Vec<f32>, Vec<f32>)>,
) -> Result<Vec<f32>> {
    let n_embd = meta.n_embd;
    let n_head = meta.n_head;
    let n_head_kv = meta.n_head_kv.unwrap_or(n_head);
    let head_dim = n_embd / n_head;
    let n_ff = meta.n_ff.unwrap_or(n_embd * 8 / 3);
    let n_rot = meta.n_rot.unwrap_or(head_dim);

    let residual_1 = hidden.to_vec();

    // === Self-Attention ===
    let normed = safe_rms_norm(hidden, &get_norm_tensor(weights, &format!("blk.{}.attn_norm.weight", layer_idx), n_embd)?);

    let q = linear(&normed, weights, &format!("blk.{}.attn_q.weight", layer_idx), n_embd, n_embd)?;
    let k_size = n_head_kv * head_dim;
    let k = linear(&normed, weights, &format!("blk.{}.attn_k.weight", layer_idx), n_embd, k_size)?;
    let v = linear(&normed, weights, &format!("blk.{}.attn_v.weight", layer_idx), n_embd, k_size)?;

    // Apply RoPE to Q and K
    let q_rope = apply_rope(&q, pos, head_dim, n_rot);
    let k_rope = apply_rope(&k, pos, head_dim, n_rot);

    // Store K, V in cache for this position
    kv_cache.push((k_rope.clone(), v.clone()));

    // Multi-head attention over ALL cached positions
    let attn_out = multi_head_attention_cached(&q_rope, kv_cache, n_head, n_head_kv, head_dim)?;

    let attn_proj = linear(&attn_out, weights, &format!("blk.{}.attn_output.weight", layer_idx), n_embd, n_embd)?;

    let mut hidden_state = vec_add(&residual_1, &attn_proj);

    // Clip to prevent numerical overflow in subsequent layers
    for val in hidden_state.iter_mut() {
        *val = val.clamp(-MAX_HIDDEN_VAL as f32, MAX_HIDDEN_VAL as f32);
    }

    // === FFN (SwiGLU) ===
    let residual_2 = hidden_state.clone();
    let normed_2 = safe_rms_norm(&hidden_state, &get_norm_tensor(weights, &format!("blk.{}.ffn_norm.weight", layer_idx), n_embd)?);

    let gate = linear(&normed_2, weights, &format!("blk.{}.ffn_gate.weight", layer_idx), n_embd, n_ff)?;
    let up = linear(&normed_2, weights, &format!("blk.{}.ffn_up.weight", layer_idx), n_embd, n_ff)?;

    let gate_up: Vec<f32> = gate.iter().zip(up.iter())
        .map(|(&a, &b)| silu(a) * b)
        .collect();

    let ffn_out = linear(&gate_up, weights, &format!("blk.{}.ffn_down.weight", layer_idx), n_ff, n_embd)?;
    hidden_state = vec_add(&residual_2, &ffn_out);

    // Clip again
    for val in hidden_state.iter_mut() {
        *val = val.clamp(-MAX_HIDDEN_VAL as f32, MAX_HIDDEN_VAL as f32);
    }

    Ok(hidden_state)
}

/// Matrix-vector multiply with quantized weights
fn linear(
    input: &[f32],
    weights: &LayerWeights,
    tensor_name: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    let raw = weights.get(tensor_name);
    if raw.is_none() {
        trace!("Tensor {} not found, returning zeros", tensor_name);
        return Ok(vec![0.0f32; out_features]);
    }
    let raw = raw.unwrap();

    let dtype = weights.dtype_of(tensor_name).unwrap_or(TensorDtype::F32);
    let w_shape: [usize; 2] = [in_features, out_features];
    let dequant = dequantize(raw, dtype, &w_shape);

    if dequant.is_empty() {
        return Ok(vec![0.0f32; out_features]);
    }

    let num_rows = (dequant.len() / in_features.max(1)).min(out_features);
    let mut output = vec![0.0f32; out_features];

    for j in 0..num_rows {
        // Use f64 intermediate to prevent overflow with large weight/input values
        let mut sum: f64 = 0.0;
        let input_len = in_features.min(input.len());
        for i in 0..input_len {
            let idx = j * in_features + i;
            if idx < dequant.len() {
                sum += (dequant[idx] as f64) * (input[i] as f64);
            }
        }
        let val = sum as f32;
        // Clamp to prevent downstream NaN/Inf propagation
        output[j] = if val.is_finite() { val.clamp(-1e5, 1e5) } else { 0.0 };
    }

    Ok(output)
}

/// Get a norm tensor as F32
fn get_norm_tensor(weights: &LayerWeights, name: &str, n: usize) -> Result<Vec<f32>> {
    let raw = weights.get(name)
        .ok_or_else(|| anyhow::anyhow!("Norm tensor {} not found", name))?;

    if raw.len() >= n * 4 {
        Ok(raw[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        Ok(vec![1.0f32; n])
    }
}

/// RMSNorm epsilon — standard value for most LLaMA-based models
const RMS_NORM_EPS: f64 = 1e-5;

/// RMSNorm with f64 intermediate computation to prevent overflow
fn safe_rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // Use f64 for the sum of squares to avoid overflow
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rms: f64 = (sum_sq / n as f64 + RMS_NORM_EPS).sqrt();

    x.iter().zip(weight.iter()).map(|(&xi, &wi)| {
        let normalized = (xi as f64) / rms;
        (normalized * wi as f64) as f32
    }).collect()
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    if x > 20.0 { return x; }
    if x < -20.0 { return 0.0; }
    x * (1.0 / (1.0 + (-x).exp()))
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// RoPE - Rotary Position Embedding
fn apply_rope(x: &[f32], pos: usize, head_dim: usize, n_rot: usize) -> Vec<f32> {
    let mut result = x.to_vec();
    let half_rot = n_rot / 2;
    let num_heads = x.len() / head_dim;

    for h in 0..num_heads {
        let ho = h * head_dim;
        for i in 0..half_rot.min(head_dim / 2) {
            let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / n_rot as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx1 = ho + i;
            let idx2 = ho + i + half_rot;
            if idx2 < result.len() {
                let (v1, v2) = (result[idx1], result[idx2]);
                // GGML/llama.cpp NEoX half-split convention:
                //   dst[ic] = src[ic]*cos + src[id]*sin
                //   dst[id] = src[id]*cos - src[ic]*sin
                result[idx1] = v1 * cos_val + v2 * sin_val;
                result[idx2] = v2 * cos_val - v1 * sin_val;
            }
        }
    }
    result
}

/// Multi-Head Attention with GQA and KV cache
fn multi_head_attention_cached(
    q: &[f32],
    kv_cache: &[(Vec<f32>, Vec<f32>)],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let n_rep = n_head / n_head_kv;
    let n_tokens = kv_cache.len();
    let mut output = vec![0.0f32; q.len()];

    if n_tokens == 0 {
        return Ok(output);
    }

    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    for head in 0..n_head {
        let kv_head = head / n_rep;
        let qs = head * head_dim;
        let ks = kv_head * head_dim;
        let vs = kv_head * head_dim;

        // Compute attention scores using f64 to prevent overflow
        let mut scores = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let k_vec = &kv_cache[t].0;
            let mut score: f64 = 0.0;
            for j in 0..head_dim {
                let qi = qs + j;
                let ki = ks + j;
                if qi < q.len() && ki < k_vec.len() {
                    score += (q[qi] as f64) * (k_vec[ki] as f64);
                }
            }
            // Scale and clamp
            let score = (score * inv_sqrt_d as f64)
                .clamp(-1e4, 1e4);
            scores.push(score);
        }

        // Softmax (numerically stable with f64 intermediate)
        let mut sum: f64 = 0.0f64;
        let max_val = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for score in scores.iter_mut() {
            let diff = *score - max_val;
            *score = if diff.is_nan() || diff.is_infinite() {
                0.0
            } else {
                diff.exp().clamp(0.0, 1e10)
            };
            sum += *score;
        }

        // Weighted sum of V
        if sum > 0.0 {
            for j in 0..head_dim {
                let oi = qs + j;
                let vi = vs + j;
                if oi >= output.len() { break; }
                let mut val: f64 = 0.0;
                for t in 0..n_tokens {
                    let v_vec = &kv_cache[t].1;
                    if vi < v_vec.len() {
                        val += scores[t] * (v_vec[vi] as f64);
                    }
                }
                output[oi] = val as f32;
            }
        }
    }

    Ok(output)
}
