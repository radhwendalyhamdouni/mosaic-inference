//! Transformer Layer - Single layer forward pass
//!
//! هيكل الطبقة الواحدة في محول Transformer:
//!
//! Input
//!   │
//!   ├── RMSNorm ───────────────────────────┐
//!   │    │                                  │ (Residual)
//!   │    ├── Q/K/V Projections             │
//!   │    ├── RoPE (Position Encoding)       │
//!   │    ├── Self-Attention                 │
//!   │    ├── Output Projection              │
//!   │    └── Add (Skip Connection) ─────────┤
//!   │                                       │
//!   ├── RMSNorm ───────────────────────────┐
//!   │    │                                  │ (Residual)
//!   │    ├── FFN Gate + SiLU              │
//!   │    ├── FFN Up                        │
//!   │    ├── Multiply (Gate × Up)         │
//!   │    ├── FFN Down                      │
//!   │    └── Add (Skip Connection) ─────────┤
//!   │                                       │
//! Output ←──────────────────────────────────┘

use crate::loader::ModelMetadata;
use crate::loader::TensorDtype;
use crate::engine::stream::LayerWeights;
use anyhow::Result;
use tracing::trace;

/// A single transformer layer configuration
#[derive(Debug)]
pub struct LayerConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_rot: usize,
}

impl LayerConfig {
    pub fn from_metadata(meta: &ModelMetadata) -> Self {
        let n_head_kv = meta.n_head_kv.unwrap_or(meta.n_head);
        Self {
            n_embd: meta.n_embd,
            n_head: meta.n_head,
            n_head_kv,
            head_dim: meta.n_embd / meta.n_head,
            n_ff: meta.n_ff.unwrap_or(meta.n_embd * 8 / 3),
            n_rot: meta.n_rot.unwrap_or(meta.n_embd / meta.n_head),
        }
    }
}

/// Execute forward pass through a single transformer layer
pub fn layer_forward(
    hidden: &[f32],
    weights: &LayerWeights,
    pos: usize,
    layer_idx: usize,
    meta: &ModelMetadata,
) -> Result<Vec<f32>> {
    let config = LayerConfig::from_metadata(meta);
    let n_embd = config.n_embd;

    trace!("Forward pass: layer {} at pos {}", layer_idx, pos);

    // Save residual (skip connection)
    let residual_1 = hidden.to_vec();

    // === Self-Attention Block ===
    // 1. RMSNorm
    let normed = rms_norm(hidden, &get_tensor_scale(weights, "blk.{}.attn_norm.weight", layer_idx, n_embd)?);

    // 2. Q, K, V projections
    let q = linear_q(&normed, weights, layer_idx, "attn_q", n_embd, config.n_head * config.head_dim)?;
    let k = linear_q(&normed, weights, layer_idx, "attn_k", n_embd, config.n_head_kv * config.head_dim)?;
    let v = linear_q(&normed, weights, layer_idx, "attn_v", n_embd, config.n_head_kv * config.head_dim)?;

    // 3. RoPE (Rotary Position Embedding)
    let q_rope = apply_rope(&q, pos, config.head_dim, config.n_rot);
    let k_rope = apply_rope(&k, pos, config.head_dim, config.n_rot);

    // 4. Self-Attention (simplified - single head)
    let attn_out = scaled_dot_product_attention(
        &q_rope,
        &k_rope,
        &v,
        config.n_head,
        config.n_head_kv,
        config.head_dim,
    )?;

    // 5. Output projection
    let attn_proj = linear_q(&attn_out, weights, layer_idx, "attn_output", n_embd, n_embd)?;

    // 6. Add residual
    let mut hidden_state = vec_add(&residual_1, &attn_proj, n_embd);

    // === Feed-Forward Block (SwiGLU) ===
    let residual_2 = hidden_state.clone();

    // 7. RMSNorm
    let normed_2 = rms_norm(&hidden_state, &get_tensor_scale(weights, "blk.{}.ffn_norm.weight", layer_idx, n_embd)?);

    // 8. FFN: gate = SiLU(norm * W_gate), up = norm * W_up, down = (gate * up) * W_down
    let gate = linear_q(&normed_2, weights, layer_idx, "ffn_gate", n_embd, config.n_ff)?;
    let up = linear_q(&normed_2, weights, layer_idx, "ffn_up", n_embd, config.n_ff)?;

    // 9. SiLU activation + element-wise multiply
    let gate_silu: Vec<f32> = gate.iter().map(|&x| silu(x)).collect();
    let gate_up: Vec<f32> = gate_silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();

    // 10. Down projection
    let ffn_out = linear_q(&gate_up, weights, layer_idx, "ffn_down", config.n_ff, n_embd)?;

    // 11. Add residual
    hidden_state = vec_add(&residual_2, &ffn_out, n_embd);

    Ok(hidden_state)
}

/// RMSNorm (Root Mean Square Normalization)
/// أكثر كفاءة من LayerNorm - لا تحتاج mean subtraction
fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32).sqrt() + 1e-6;

    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect()
}

/// SiLU activation function (Swish): x * sigmoid(x)
#[inline(always)]
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

/// Element-wise vector addition
fn vec_add(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    a.iter().zip(b.iter()).take(n).map(|(&x, &y)| x + y).collect()
}

/// Get tensor scale (normalized weight for RMSNorm)
fn get_tensor_scale(
    weights: &LayerWeights,
    pattern: &str,
    layer_idx: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    let name = pattern.replace("{}", &layer_idx.to_string());
    let raw = weights.get(&name)
        .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;

    // For F32 tensors, direct conversion
    if raw.len() == n_embd * 4 {
        Ok(raw.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        // Approximate for other types
        Ok(raw.iter().take(n_embd).map(|&b| b as f32 / 128.0).collect())
    }
}

/// Simplified linear projection with quantized weights
/// W: [out_features, in_features] × input: [in_features] → output: [out_features]
fn linear_q(
    input: &[f32],
    weights: &LayerWeights,
    layer_idx: usize,
    proj_name: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    let w_name = format!("blk.{}.{}.weight", layer_idx, proj_name);
    let bias_name = format!("blk.{}.{}.bias", layer_idx, proj_name);

    let raw_w = weights.get(&w_name);
    let raw_b = weights.get(&bias_name);

    // Simplified matmul: treat weights as f32 or dequantize
    let output = if let Some(w_raw) = raw_w {
        if w_raw.len() >= out_features * in_features * 4 {
            // F32 weights
            let mut result = vec![0.0f32; out_features];
            for i in 0..out_features {
                let mut sum = 0.0f32;
                for j in 0..in_features {
                    let w_off = (i * in_features + j) * 4;
                    let w = f32::from_le_bytes([w_raw[w_off], w_raw[w_off + 1], w_raw[w_off + 2], w_raw[w_off + 3]]);
                    sum += w * input[j];
                }
                result[i] = sum;
            }

            // Add bias if present
            if let Some(b_raw) = raw_b {
                if b_raw.len() >= out_features * 4 {
                    for i in 0..out_features {
                        let b_off = i * 4;
                        result[i] += f32::from_le_bytes([b_raw[b_off], b_raw[b_off + 1], b_raw[b_off + 2], b_raw[b_off + 3]]);
                    }
                }
            }

            result
        } else {
            // Quantized weights - simplified dequantization
            let mut result = vec![0.0f32; out_features];
            for i in 0..out_features {
                let mut sum = 0.0f32;
                for j in 0..in_features.min(w_raw.len()) {
                    sum += (w_raw[i * in_features + j] as f32 / 128.0) * input[j];
                }
                result[i] = sum;
            }
            result
        }
    } else {
        // Tensor not found - return zeros (placeholder for incomplete models)
        vec![0.0f32; out_features]
    };

    Ok(output)
}

/// Apply Rotary Position Embedding (RoPE)
/// يضيف معلومات الموقع للـ Query و Key
/// بدون RoPE: النموذج لا يعرف ترتيب الكلمات!
fn apply_rope(x: &[f32], pos: usize, head_dim: usize, n_rot: usize) -> Vec<f32> {
    let mut result = x.to_vec();
    let half_rot = n_rot / 2;

    // Precompute frequencies
    let freq_base = 10000.0f32;
    for i in 0..half_rot.min(head_dim / 2) {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / n_rot as f32);
        let angle = pos as f32 * freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        // Apply rotation to pairs of elements
        let idx1 = i;
        let idx2 = i + half_rot;

        if idx2 < result.len() {
            let v1 = result[idx1];
            let v2 = result[idx2];
            result[idx1] = v1 * cos_val - v2 * sin_val;
            result[idx2] = v1 * sin_val + v2 * cos_val;
        }
    }

    result
}

/// Scaled Dot-Product Attention (SDPA)
/// جوهر آلية الانتباه الذاتي
fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; q.len()];

    // Simplified: single token attention
    // For each head, compute attention score
    let n_rep = n_head / n_head_kv; // GQA: repeat KV heads

    for head in 0..n_head {
        let kv_head = head / n_rep; // Which KV head to use
        let q_start = head * head_dim;
        let k_start = kv_head * head_dim;
        let v_start = kv_head * head_dim;

        // Compute dot product Q·K / sqrt(d)
        let mut score = 0.0f32;
        for j in 0..head_dim {
            if q_start + j < q.len() && k_start + j < k.len() {
                score += q[q_start + j] * k[k_start + j];
            }
        }
        score /= (head_dim as f32).sqrt();

        // Softmax (simplified for single token: just sigmoid)
        let weight = 1.0 / (1.0 + (-score).exp());

        // Weighted sum of V
        for j in 0..head_dim {
            if q_start + j < output.len() && v_start + j < v.len() {
                output[q_start + j] = weight * v[v_start + j];
            }
        }
    }

    Ok(output)
}
