//! Dequantization - Converting compressed weights to f32
//!
//! Supports: F32, F16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K
//!
//! Reference: https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c

/// Dequantize raw bytes to f32 array based on dtype and shape
pub fn dequantize(bytes: &[u8], dtype: super::TensorDtype, shape: &[usize]) -> Vec<f32> {
    let total_elements: usize = shape.iter().product();
    if total_elements == 0 {
        return Vec::new();
    }

    match dtype {
        super::TensorDtype::F32 => dequant_f32(bytes, total_elements),
        super::TensorDtype::F16 => dequant_f16(bytes, total_elements),
        super::TensorDtype::Q4_0 => dequant_q4_0(bytes, total_elements),
        super::TensorDtype::Q4_1 => dequant_q4_1(bytes, total_elements),
        super::TensorDtype::Q8_0 => dequant_q8_0(bytes, total_elements),
        super::TensorDtype::Q4_K => dequant_q4_k(bytes, total_elements),
        super::TensorDtype::Q5_K => dequant_q5_k(bytes, total_elements),
        super::TensorDtype::Q6_K => dequant_q6_k(bytes, total_elements),
        _ => {
            tracing::warn!("Unsupported dtype {:?}, using zero approximation", dtype);
            vec![0.0f32; total_elements]
        }
    }
}

/// Dequantize a single row from quantized data
pub fn dequantize_row(bytes: &[u8], dtype: super::TensorDtype, row_idx: usize, cols: usize) -> Vec<f32> {
    match dtype {
        super::TensorDtype::F32 => {
            let offset = row_idx * cols * 4;
            if offset + cols * 4 > bytes.len() {
                vec![0.0f32; cols]
            } else {
                bytes[offset..offset + cols * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
        super::TensorDtype::F16 => {
            let offset = row_idx * cols * 2;
            if offset + cols * 2 > bytes.len() {
                vec![0.0f32; cols]
            } else {
                bytes[offset..offset + cols * 2]
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            }
        }
        _ => dequantize_block_row(bytes, dtype, row_idx, cols),
    }
}

fn dequantize_block_row(bytes: &[u8], dtype: super::TensorDtype, row_idx: usize, cols: usize) -> Vec<f32> {
    let block_size = dtype.block_size();
    let blocks_per_row = (cols + block_size - 1) / block_size;
    let mut result = vec![0.0f32; cols];

    for b in 0..blocks_per_row {
        let block_idx = row_idx * blocks_per_row + b;
        let start_elem = b * block_size;
        let end_elem = (start_elem + block_size).min(cols);

        let block_bytes = get_block_bytes(bytes, dtype, block_idx);
        if block_bytes.is_empty() { continue; }

        let dequant_block: Vec<f32> = match dtype {
            super::TensorDtype::Q4_0 => dequant_q4_0_block(&block_bytes),
            super::TensorDtype::Q4_1 => dequant_q4_1_block(&block_bytes),
            super::TensorDtype::Q8_0 => dequant_q8_0_block(&block_bytes),
            super::TensorDtype::Q4_K => dequant_q4_k_block(&block_bytes),
            super::TensorDtype::Q5_K => dequant_q5_k_block(&block_bytes),
            super::TensorDtype::Q6_K => dequant_q6_k_block(&block_bytes),
            _ => vec![0.0; block_size],
        };

        for (i, &val) in dequant_block.iter().enumerate().take(end_elem - start_elem) {
            result[start_elem + i] = val;
        }
    }

    result
}

fn get_block_bytes(bytes: &[u8], dtype: super::TensorDtype, block_idx: usize) -> Vec<u8> {
    let block_bytes_size = dtype.block_bytes();
    let offset = block_idx * block_bytes_size;
    if offset + block_bytes_size > bytes.len() {
        Vec::new()
    } else {
        bytes[offset..offset + block_bytes_size].to_vec()
    }
}

// ============================================================
// F32
// ============================================================

fn dequant_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    let count = (n * 4).min(bytes.len() / 4);
    bytes[..count]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ============================================================
// F16
// ============================================================

fn dequant_f16(bytes: &[u8], n: usize) -> Vec<f32> {
    let count = (n * 2).min(bytes.len() / 2);
    bytes[..count]
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Convert IEEE 754 half-precision (f16) to f32
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    if h == 0 { return 0.0; }
    if h == 0x8000 { return -0.0; }
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as f32;
    if exp == 0 { sign * 2.0_f32.powi(-14) * (frac / 1024.0) }
    else if exp == 31 { if frac == 0.0 { f32::INFINITY * sign } else { f32::NAN } }
    else { sign * 2.0_f32.powi(exp - 15) * (1.0 + frac / 1024.0) }
}

// ============================================================
// Q4_0: 4-bit quantized, block size 32
// Layout per block (18 bytes):
//   [0..2]  d: f16 scale factor
//   [2..18] qs: 32 nibbles packed (4 bits each)
//
// From ggml: element i = d * ((qs[i/2] >> (4*(i%2))) & 0xF) - 8*d
//   i even: low nibble;  i odd: high nibble
// ============================================================

fn dequant_q4_0(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    let block_size = 32;
    let num_blocks = (n + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let block_start = b * 18;
        if block_start + 18 > bytes.len() { break; }
        let block = &bytes[block_start..block_start + 18];
        let dequant = dequant_q4_0_block(block);
        result.extend_from_slice(&dequant);
    }

    result.truncate(n);
    result
}

fn dequant_q4_0_block(block: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut result = Vec::with_capacity(32);

    for i in 0..32 {
        let byte_idx = 2 + i / 2;
        // GGML convention: even index = low nibble, odd index = high nibble
        let shift = if i % 2 == 0 { 0 } else { 4 };
        let nibble = (block[byte_idx] >> shift) & 0x0F;
        let q = (nibble as i32) - 8;
        result.push(d * q as f32);
    }

    result
}

// ============================================================
// Q4_1: 4-bit quantized with min, block size 32
// Layout per block (20 bytes):
//   [0..2]  d: f16 scale
//   [2..4]  m: f16 min
//   [4..20] qs: 32 nibbles packed
//
// From ggml: element i = m + d * ((qs[i/2] >> (4*(i%2))) & 0xF)
// ============================================================

fn dequant_q4_1(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    let block_size = 32;
    let num_blocks = (n + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let block_start = b * 20;
        if block_start + 20 > bytes.len() { break; }
        let block = &bytes[block_start..block_start + 20];
        let dequant = dequant_q4_1_block(block);
        result.extend_from_slice(&dequant);
    }

    result.truncate(n);
    result
}

fn dequant_q4_1_block(block: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let mut result = Vec::with_capacity(32);

    for i in 0..32 {
        let byte_idx = 4 + i / 2;
        // GGML convention: even index = low nibble, odd index = high nibble
        let shift = if i % 2 == 0 { 0 } else { 4 };
        let q = (block[byte_idx] >> shift) & 0x0F;
        result.push(m + d * q as f32);
    }

    result
}

// ============================================================
// Q8_0: 8-bit quantized, block size 32
// Layout per block (34 bytes):
//   [0..2]  d: f16 scale
//   [2..34] qs: 32 int8 values
// ============================================================

fn dequant_q8_0(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    let block_size = 32;
    let num_blocks = (n + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let block_start = b * 34;
        if block_start + 34 > bytes.len() { break; }
        let block = &bytes[block_start..block_start + 34];
        let dequant = dequant_q8_0_block(block);
        result.extend_from_slice(&dequant);
    }

    result.truncate(n);
    result
}

fn dequant_q8_0_block(block: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut result = Vec::with_capacity(32);

    for i in 0..32 {
        result.push(d * block[2 + i] as f32);
    }

    result
}

// ============================================================
// Q4_K: K-quant 4-bit, super-block size 256
// 144 bytes per super-block
//
// Block layout (144 bytes):
//   [0..2]    d:     f16 super-block scale
//   [2..4]    dmin:  f16 super-block minimum
//   [4..16]   scales: 12 bytes encoding 8 (sc, m) pairs in 6-bit format
//   [16..144] qs:    128 bytes = 256 nibbles (4 bits each)
//
// Scales encoding (get_scale_min_k4 from ggml):
//   For j < 4:  sc = q[j] & 63,          m = q[j+4] & 63
//   For j >= 4: sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
//               m = (q[j+4] >> 4)  | ((q[j-0] >> 6) << 4)
//
// Element layout (4 groups of 64):
//   Group g (g=0..3): uses qs[g*32 .. g*32+31] (32 bytes)
//     First 32 elements:  d*sc[2g]   * (qs[l] & 0xF)          - dmin*m[2g]
//     Next  32 elements:  d*sc[2g+1] * (qs[l] >> 4)           - dmin*m[2g+1]
//
// Translated from ggml-quants.c dequantize_row_q4_K
// ============================================================

fn dequant_q4_k(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    let block_size = 256;
    let num_blocks = (n + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let block_start = b * 144;
        if block_start + 144 > bytes.len() { break; }
        let block = &bytes[block_start..block_start + 144];
        let dequant = dequant_q4_k_block(block);
        result.extend_from_slice(&dequant);
    }

    result.truncate(n);
    result
}

/// Decode Q4_K scales: returns 8 (scale, min) pairs from 12 bytes
#[inline]
fn get_scale_min_k4(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for j in 0..4 {
        scales[j] = sc[j] & 63;
        mins[j] = sc[j + 4] & 63;
    }
    for j in 4..8 {
        scales[j] = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        mins[j] = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }

    (scales, mins)
}

fn dequant_q4_k_block(block: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let sc = &block[4..16];
    let qs = &block[16..144];

    let (scales, mins) = get_scale_min_k4(sc);

    let mut result = Vec::with_capacity(256);

    // 4 groups of 64 elements each
    for g in 0..4 {
        let q_base = g * 32;

        // Sub-block pair: (sc[2g], m[2g]) and (sc[2g+1], m[2g+1])
        let d1 = d * scales[2 * g] as f32;
        let m1 = dmin * mins[2 * g] as f32;
        let d2 = d * scales[2 * g + 1] as f32;
        let m2 = dmin * mins[2 * g + 1] as f32;

        // First 32 elements: low nibble of each qs byte
        for l in 0..32 {
            let nibble = (qs[q_base + l] & 0x0F) as f32;
            result.push(d1 * nibble - m1);
        }
        // Next 32 elements: high nibble of each qs byte
        for l in 0..32 {
            let nibble = (qs[q_base + l] >> 4) as f32;
            result.push(d2 * nibble - m2);
        }
    }

    result
}

// ============================================================
// Q5_K: K-quant 5-bit, super-block size 256
// 176 bytes per super-block
//
// Block layout (176 bytes):
//   [0..2]    d:     f16 super-block scale
//   [2..4]    dmin:  f16 super-block minimum
//   [4..16]   scales: 12 bytes encoding 8 (sc, m) pairs (same as Q4_K)
//   [16..144] qs:    128 bytes = 256 nibbles (low 4 bits)
//   [144..176] qh:   32 bytes = 256 bits (high 1 bit per element)
//
// Element layout (4 groups of 64):
//   Group g (g=0..3): uses qs[g*32..g*32+31]
//     First 32 elements:  d*sc[2g]   * ((qs[l]&0xF) | (qh_bit<<4)) - dmin*m[2g]
//     Next  32 elements:  d*sc[2g+1] * ((qs[l]>>4)  | (qh_bit<<4)) - dmin*m[2g+1]
//
// qh packing: 1 bit per element, 8 bits per byte
//   qh[element / 8] >> (element % 8) & 1
//
// Translated from ggml-quants.c dequantize_row_q5_K
// ============================================================

fn dequant_q5_k(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    let block_size = 256;
    let num_blocks = (n + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let block_start = b * 176;
        if block_start + 176 > bytes.len() { break; }
        let block = &bytes[block_start..block_start + 176];
        let dequant = dequant_q5_k_block(block);
        result.extend_from_slice(&dequant);
    }

    result.truncate(n);
    result
}

fn dequant_q5_k_block(block: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let sc = &block[4..16];
    let qs = &block[16..144];
    let qh = &block[144..176]; // 32 bytes = 256 bits

    let (scales, mins) = get_scale_min_k4(sc);

    let mut result = Vec::with_capacity(256);

    // 4 groups of 64 elements each
    for g in 0..4 {
        let q_base = g * 32;
        let elem_base = g * 64;

        let d1 = d * scales[2 * g] as f32;
        let m1 = dmin * mins[2 * g] as f32;
        let d2 = d * scales[2 * g + 1] as f32;
        let m2 = dmin * mins[2 * g + 1] as f32;

        // First 32 elements: low nibble + high bit from qh
        for l in 0..32 {
            let elem_idx = elem_base + l;
            let ql = (qs[q_base + l] & 0x0F) as i32;
            let qh_bit = ((qh[elem_idx / 8] >> (elem_idx % 8)) & 1) as i32;
            let q = ql | (qh_bit << 4);
            result.push(d1 * q as f32 - m1);
        }
        // Next 32 elements: high nibble + high bit from qh
        for l in 0..32 {
            let elem_idx = elem_base + 32 + l;
            let ql = ((qs[q_base + l] >> 4) & 0x0F) as i32;
            let qh_bit = ((qh[elem_idx / 8] >> (elem_idx % 8)) & 1) as i32;
            let q = ql | (qh_bit << 4);
            result.push(d2 * q as f32 - m2);
        }
    }

    result
}

// ============================================================
// Q6_K: K-quant 6-bit, super-block size 256
// 210 bytes per super-block
//
// Block layout (210 bytes):
//   [0..128]   ql:    128 bytes - quantized low 4 bits (packed as nibble pairs)
//   [128..192] qh:    64 bytes  - quantized high 2 bits (4 values per byte)
//   [192..208] scales: 16 bytes  - signed int8 scales (one per 16-element sub-block)
//   [208..210] d:     2 bytes   - f16 super-block scale
//
// Translated from ggml-quants.c dequantize_row_q6_K:
//   Process in 2 passes of 128 elements.
//   For each pass (n=0, n=128):
//     ql_idx = n/4 + l (so pass 0: l, pass 1: 32+l)
//     qh_idx = n/128*32 + l (so pass 0: l, pass 1: 32+l)
//     sc_idx = is + n/128*8 (so pass 0: is+0..1, pass 1: is+8..9)
//     For l in 0..31:
//       is = l/16  (0 or 1)
//       q1 = (ql[ql_idx]     & 0xF) | (((qh[qh_idx] >> 0) & 3) << 4)  - 32
//       q2 = (ql[ql_idx+32]  & 0xF) | (((qh[qh_idx] >> 2) & 3) << 4)  - 32
//       q3 = (ql[ql_idx]     >> 4)  | (((qh[qh_idx] >> 4) & 3) << 4)  - 32
//       q4 = (ql[ql_idx+32]  >> 4)  | (((qh[qh_idx] >> 6) & 3) << 4)  - 32
//       y[n+l]      = d * sc[sc_idx] * q1
//       y[n+l+32]   = d * sc[sc_idx] * q2
//       y[n+l+64]   = d * sc[sc_idx+4] * q3
//       y[n+l+96]   = d * sc[sc_idx+4] * q4
//
// Translated from ggml-quants.c dequantize_row_q6_K
// ============================================================

fn dequant_q6_k(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    let block_size = 256;
    let num_blocks = (n + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let block_start = b * 210;
        if block_start + 210 > bytes.len() { break; }
        let block = &bytes[block_start..block_start + 210];
        let dequant = dequant_q6_k_block(block);
        result.extend_from_slice(&dequant);
    }

    result.truncate(n);
    result
}

fn dequant_q6_k_block(block: &[u8]) -> Vec<f32> {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mut result = vec![0.0f32; 256];

    // Two passes of 128 elements each
    for (pass, y_base) in [0usize, 128].iter().enumerate() {
        let y_base = *y_base;
        let ql_base = pass * 32;  // GGML: ql_idx = n/4 + l → pass 0: 0..31, pass 1: 32..63
        let qh_base = pass * 32;  // 0 or 32
        let sc_base = pass * 8;   // 0 or 8

        for l in 0..32usize {
            let is = l / 16;

            let qh_l = qh[qh_base + l] as i32;
            let q1 = ((ql[ql_base + l] & 0x0F) as i32
                     | (((qh_l >> 0) & 3) << 4)) - 32;
            let q2 = ((ql[ql_base + l + 32] & 0x0F) as i32
                     | (((qh_l >> 2) & 3) << 4)) - 32;
            let q3 = (((ql[ql_base + l] >> 4) & 0x0F) as i32
                     | (((qh_l >> 4) & 3) << 4)) - 32;
            let q4 = (((ql[ql_base + l + 32] >> 4) & 0x0F) as i32
                     | (((qh_l >> 6) & 3) << 4)) - 32;

            let s1 = sc[sc_base + is] as f32 * d;
            let s2 = sc[sc_base + is + 4] as f32 * d;

            result[y_base + l]      = s1 * q1 as f32;
            result[y_base + l + 32] = s1 * q2 as f32;
            result[y_base + l + 64] = s2 * q3 as f32;
            result[y_base + l + 96] = s2 * q4 as f32;
        }
    }

    result
}
