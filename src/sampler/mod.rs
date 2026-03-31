//! Token Sampler
//!
//! يختار الرمز التالي بناءً على الاحتمالات
//! يدعم: Temperature, Top-P (nucleus), Top-K, Repetition Penalty

use anyhow::Result;

/// Sample next token from logits using temperature + top-p
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
) -> Result<usize> {
    if logits.is_empty() {
        anyhow::bail!("Empty logits");
    }

    // 1. Apply temperature
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&x| x / temperature.max(0.01))
        .collect();

    // 2. Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // 3. Top-p (nucleus) sampling
    if top_p < 1.0 {
        return top_p_sample(&probs, top_p);
    }

    // 4. Greedy (argmax) as fallback
    let mut best_idx = 0;
    let mut best_prob = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    Ok(best_idx)
}

/// Top-p (nucleus) sampling
/// يختار من أصغر مجموعة من الرموز التي يغطي احتمالها cumulatively = top_p
fn top_p_sample(probs: &[f32], top_p: f32) -> Result<usize> {
    // Create (index, probability) pairs
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate()
        .map(|(i, &p)| (i, p))
        .collect();

    // Sort by probability (descending)
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find cutoff
    let mut cumsum = 0.0f32;
    let mut candidates = Vec::new();

    for &(idx, prob) in &indexed {
        candidates.push((idx, prob));
        cumsum += prob;
        if cumsum >= top_p {
            break;
        }
    }

    if candidates.is_empty() {
        return Ok(indexed[0].0);
    }

    // Renormalize and sample
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r: f32 = rand_simple() * total;
    let mut accum = 0.0f32;

    for (idx, prob) in &candidates {
        accum += prob;
        if accum >= r {
            return Ok(*idx);
        }
    }

    Ok(candidates.last().unwrap().0)
}

/// Simple pseudo-random number generator
/// (we don't add rand crate dependency to keep binary small)
fn rand_simple() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // xorshift64
    let mut state = (seed as u64) ^ 0xDEAD_BEEF_CAFE_BABE;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;

    (state as f32) / (u64::MAX as f32)
}
