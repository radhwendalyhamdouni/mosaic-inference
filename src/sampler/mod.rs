//! Token Sampler
//!
//! يختار الرمز التالي بناءً على الاحتمالات
//! يدعم: Temperature, Top-P (nucleus), Greedy

use anyhow::Result;

/// Sample next token from logits using temperature + top-p
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
) -> Result<usize> {
    if logits.is_empty() {
        anyhow::bail!("Empty logits - cannot sample");
    }

    // Check for all-NaN or all-zero
    let mut has_valid = false;
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l.is_finite() && !l.is_nan() {
            has_valid = true;
            if l > max_logit {
                max_logit = l;
            }
        }
    }

    if !has_valid {
        // All logits are NaN or infinite - fallback to random
        tracing::warn!("All logits invalid, returning random token");
        return Ok(rand_simple() as usize % logits.len());
    }

    // 1. Apply temperature (clamp to avoid division by zero)
    let temp = temperature.max(0.01);
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&x| {
            if x.is_nan() { 0.0 } else { x / temp }
        })
        .collect();

    // 2. Softmax with numerical stability
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter()
        .map(|&x| {
            let v = (x - max_val).exp();
            if v.is_nan() || v.is_infinite() { 0.0 } else { v }
        })
        .collect();
    let sum: f32 = exps.iter().sum();

    let probs: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|&x| x / sum).collect()
    } else {
        // Fallback: uniform distribution
        vec![1.0 / logits.len() as f32; logits.len()]
    };

    // 3. Top-p (nucleus) sampling
    if top_p < 1.0 {
        return top_p_sample(&probs, top_p);
    }

    // 4. Greedy (argmax) as fallback
    let mut best_idx = 0;
    let mut best_prob = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p.is_finite() && p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    Ok(best_idx)
}

/// Top-p (nucleus) sampling
fn top_p_sample(probs: &[f32], top_p: f32) -> Result<usize> {
    // Create (index, probability) pairs, filter out non-finite
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate()
        .filter(|(_, &p)| p.is_finite() && p > 0.0)
        .map(|(i, &p)| (i, p))
        .collect();

    if indexed.is_empty() {
        // Fallback: return the index of the max logit (greedy)
        let mut best = 0;
        let mut best_p = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_p { best_p = p; best = i; }
        }
        return Ok(best);
    }

    // Sort by probability (descending)
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
    if total <= 0.0 {
        return Ok(candidates[0].0);
    }

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

/// Simple pseudo-random number generator (xorshift64)
fn rand_simple() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let mut state = (seed as u64) ^ 0xDEAD_BEEF_CAFE_BABE;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;

    (state as f32) / (u64::MAX as f32)
}
