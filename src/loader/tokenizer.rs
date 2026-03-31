//! GGML/SentencePiece compatible BPE Tokenizer
//!
//! Tokenizer layout in GGUF:
//! - Tokens [3..259] are byte tokens: <0x00>..<0xFF> (type=6)
//! - Token [0] is <unk> (type=2)
//! - Tokens [1,2] are <s>, </s> (type=3) - BOS/EOS special tokens
//! - Remaining tokens are BPE merge results (type=1)
//!
//! Encoding: text → prepend_space → bytes → byte_token_ids → BPE merges
//! Decoding: token_ids → token_strings → replace ▁ with space → trim

use std::collections::HashMap;

/// A BPE tokenizer loaded from GGUF metadata
pub struct BpeTokenizer {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub merges: Vec<(usize, usize)>, // (left, right) ordered by priority
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    /// byte_value → token_id (byte 0→token 3, byte 1→token 4, etc.)
    byte_to_token: Vec<usize>,
    /// Precomputed token string → token_id
    token_to_id: HashMap<String, usize>,
}

impl BpeTokenizer {
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        merges_str: Vec<String>,
        token_types: Vec<i32>,
        bos_id: usize,
        eos_id: usize,
        pad_id: usize,
    ) -> Self {
        // Build byte → token mapping
        // In GGML, byte tokens are type=6 and formatted as <0xHH>
        let mut byte_to_token = vec![0usize; 256];
        let mut byte_count = 0;

        for (i, tok) in tokens.iter().enumerate() {
            let tt = token_types.get(i).copied().unwrap_or(0);
            if tt == 6 {
                // Byte token - parse <0xHH>
                if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() >= 5 {
                    let hex = &tok[3..tok.len() - 1];
                    if let Ok(byte_val) = usize::from_str_radix(hex, 16) {
                        if byte_val < 256 {
                            byte_to_token[byte_val] = i;
                            byte_count += 1;
                        }
                    }
                }
            }
        }

        // Build token string → ID lookup
        let mut token_to_id = HashMap::new();
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i);
        }

        // Parse merges
        let mut merges: Vec<(usize, usize)> = Vec::new();
        for merge_str in &merges_str {
            // Find the first space to split "left right"
            if let Some(space_pos) = merge_str.find(' ') {
                let left_str = &merge_str[..space_pos];
                let right_str = &merge_str[space_pos + 1..];

                let left_id = token_to_id.get(left_str).copied().unwrap_or(0);
                let right_id = token_to_id.get(right_str).copied().unwrap_or(0);

                // Only add if both sides exist
                if token_to_id.contains_key(left_str) && token_to_id.contains_key(right_str) {
                    merges.push((left_id, right_id));
                }
            }
        }

        tracing::info!(
            "Tokenizer: {} tokens, {} byte-tokens, {} merges, BOS={}, EOS={}",
            tokens.len(), byte_count, merges.len(), bos_id, eos_id
        );

        Self {
            tokens,
            scores,
            merges,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            pad_token_id: pad_id,
            byte_to_token,
            token_to_id,
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return vec![self.bos_token_id];
        }

        // Step 1: Prepend space (SentencePiece convention)
        let processed = format!(" {}", text);
        let bytes = processed.as_bytes();

        // Step 2: Map each byte to its token ID
        let mut word: Vec<usize> = bytes.iter()
            .map(|&b| self.byte_to_token[b as usize])
            .collect();

        // Step 3: Apply BPE merges
        // Each merge combines two adjacent tokens into one
        for &(left_id, right_id) in &self.merges {
            let mut i = 0;
            let mut new_word = Vec::with_capacity(word.len());
            let mut changed = false;

            while i < word.len() {
                if i + 1 < word.len() && word[i] == left_id && word[i + 1] == right_id {
                    // Try to find the merged token
                    let left_str = &self.tokens[left_id];
                    let right_str = &self.tokens[right_id];
                    let merged_str = format!("{}{}", left_str, right_str);

                    if let Some(&merged_id) = self.token_to_id.get(&merged_str) {
                        new_word.push(merged_id);
                        i += 2;
                        changed = true;
                        continue;
                    }
                }
                new_word.push(word[i]);
                i += 1;
            }

            if changed && new_word.len() < word.len() {
                word = new_word;
            }
        }

        // Prepend BOS
        let mut result = vec![self.bos_token_id];
        result.extend(word);
        result
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut result = String::new();

        for &tid in token_ids {
            if tid == self.bos_token_id || tid == self.eos_token_id || tid == self.pad_token_id {
                continue;
            }

            if tid < self.tokens.len() {
                let tok = &self.tokens[tid];
                // Replace ▁ with space (SentencePiece word piece marker)
                if tok == "▁" {
                    result.push(' ');
                } else if tok.starts_with('▁') {
                    result.push(' ');
                    result.push_str(&tok[3..]); // skip the ▁ UTF-8 bytes (3 bytes)
                } else {
                    result.push_str(tok);
                }
            }
        }

        // Remove leading space
        let trimmed = result.trim_start().to_string();
        // Replace multiple spaces with single
        let mut clean = String::new();
        let mut prev_space = false;
        for ch in trimmed.chars() {
            if ch == ' ' && prev_space {
                continue;
            }
            prev_space = ch == ' ';
            clean.push(ch);
        }
        clean
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}
