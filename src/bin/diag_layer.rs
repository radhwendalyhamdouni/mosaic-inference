use std::fs::File;
use memmap2::Mmap;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

fn main() {
    let path = "./models/tinyllama-1.1b-chat-q4_0.gguf";
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let mut c = Cursor::new(&mmap[..]);

    // Parse GGUF header
    let magic = read_str(&mut c, 4);
    assert_eq!(magic, "GGUF");
    let version = c.read_u32::<LittleEndian>().unwrap();
    let tc = c.read_u64::<LittleEndian>().unwrap();
    let kvc = c.read_u64::<LittleEndian>().unwrap();
    
    // Skip metadata
    for _ in 0..kvc {
        let _key = gguf_str(&mut c).unwrap();
        let vt = c.read_u32::<LittleEndian>().unwrap();
        skip_val(&mut c, vt);
    }

    // Data offset (alignment=32, default)
    let alignment = 32;
    let pos = c.position() as usize;
    let data_offset = (pos + alignment - 1) / alignment * alignment;
    println!("data_offset = {}", data_offset);

    // Read tensor info
    let mut tensors: Vec<(String, Vec<usize>, u32, u64)> = Vec::new();
    for _ in 0..tc {
        let name = gguf_str(&mut c).unwrap();
        let nd = c.read_u32::<LittleEndian>().unwrap();
        let mut dims = Vec::new();
        for _ in 0..nd { dims.push(c.read_u64::<LittleEndian>().unwrap() as usize); }
        let dtype = c.read_u32::<LittleEndian>().unwrap();
        let offset = c.read_u64::<LittleEndian>().unwrap();
        tensors.push((name, dims, dtype, offset));
    }

    // Find key tensors
    let get_tensor = |name: &str| -> (usize, Vec<usize>, u32) {
        for (n, d, dt, off) in &tensors {
            if n == name {
                return (*off as usize, d.clone(), *dt);
            }
        }
        panic!("Tensor not found: {}", name);
    };

    let (embd_off, embd_dims, embd_dtype) = get_tensor("token_embd.weight");
    let (norm_off, norm_dims, _norm_dtype) = get_tensor("blk.0.attn_norm.weight");
    let (q_off, q_dims, q_dtype) = get_tensor("blk.0.attn_q.weight");
    let (k_off, k_dims, k_dtype) = get_tensor("blk.0.attn_k.weight");
    let (v_off, v_dims, v_dtype) = get_tensor("blk.0.attn_v.weight");
    let (o_off, o_dims, o_dtype) = get_tensor("blk.0.attn_output.weight");

    println!("=== Tensor Info ===");
    println!("token_embd.weight: offset={}, dims={:?}, dtype={}", embd_off, embd_dims, embd_dtype);
    println!("blk.0.attn_norm.weight: offset={}, dims={:?}", norm_off, norm_dims);
    println!("blk.0.attn_q.weight: offset={}, dims={:?}, dtype={}", q_off, q_dims, q_dtype);
    println!("blk.0.attn_k.weight: offset={}, dims={:?}, dtype={}", k_off, k_dims, k_dtype);
    println!("blk.0.attn_v.weight: offset={}, dims={:?}, dtype={}", v_off, v_dims, v_dtype);
    println!("blk.0.attn_output.weight: offset={}, dims={:?}, dtype={}", o_off, o_dims, o_dtype);

    let n_embd = embd_dims[0];
    let vocab_size = embd_dims[1];
    let head_dim = 64;
    let n_head = 32;
    let n_head_kv = 4;
    let k_size = n_head_kv * head_dim; // 256

    // Embed token 1 (BOS) - dequantize one row
    println!("\n=== Embedding token 1 (BOS) ===");
    let embd_abs = data_offset + embd_off;
    let embd_data = &mmap[embd_abs..];
    
    // For Q4_K (dtype=12): block_size=256, block_bytes=144
    let block_bytes_size: usize = 144;
    let blocks_per_row = (n_embd + 255) / 256;
    
    let mut embedding = vec![0.0f32; n_embd];
    let row_start_block = 1 * blocks_per_row; // token 1 = row 1
    
    println!("blocks_per_row={}, row_start_block={}", blocks_per_row, row_start_block);
    
    for b in 0..blocks_per_row {
        let block_idx = row_start_block + b;
        let byte_off = block_idx * block_bytes_size;
        if byte_off + block_bytes_size > embd_data.len() { break; }
        
        let block = &embd_data[byte_off..byte_off + block_bytes_size];
        let vals = dequant_q4_k_block(block);
        let start = b * 256;
        let end = (start + 256).min(n_embd);
        for (i, &v) in vals.iter().enumerate().take(end - start) {
            embedding[start + i] = v;
        }
    }
    
    let emb_max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let emb_min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
    let emb_nan = embedding.iter().filter(|v| v.is_nan()).count();
    println!("Embedding: max={:.2}, min={:.2}, nan={}", emb_max, emb_min, emb_nan);

    // RMSNorm
    println!("\n=== RMSNorm ===");
    let norm_abs = data_offset + norm_off;
    let norm_data = &mmap[norm_abs..norm_abs + norm_dims[0] * 4];
    let norm_w: Vec<f32> = norm_data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    
    let sum_sq: f32 = embedding.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n_embd as f32).sqrt() + 1e-5;
    let normed: Vec<f32> = embedding.iter().zip(norm_w.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect();
    
    let norm_max = normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let norm_min = normed.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm_nan = normed.iter().filter(|v| v.is_nan()).count();
    println!("RMS={:.6}, Normed: max={:.6}, min={:.6}, nan={}", rms, norm_max, norm_min, norm_nan);

    // Q projection
    println!("\n=== Q Projection (Q4_K, dims={:?}) ===", q_dims);
    let q_abs = data_offset + q_off;
    let q_data = &mmap[q_abs..q_abs + calc_size(&q_dims, 12)];
    let q_out = linear_f32(&normed, q_data, 12, &q_dims, n_embd, n_embd);
    let q_max = q_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let q_nan = q_out.iter().filter(|v| v.is_nan()).count();
    println!("Q: max={:.4}, nan={}", q_max, q_nan);

    // K projection
    println!("\n=== K Projection (Q4_K, dims={:?}) ===", k_dims);
    let k_abs = data_offset + k_off;
    let k_data = &mmap[k_abs..k_abs + calc_size(&k_dims, 12)];
    let k_out = linear_f32(&normed, k_data, 12, &k_dims, n_embd, k_size);
    let k_max = k_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let k_nan = k_out.iter().filter(|v| v.is_nan()).count();
    println!("K: max={:.4}, nan={}", k_max, k_nan);

    // V projection
    println!("\n=== V Projection (Q6_K, dims={:?}) ===", v_dims);
    let v_abs = data_offset + v_off;
    let v_size = calc_size(&v_dims, 14); // Q6_K
    let v_data = &mmap[v_abs..v_abs + v_size];
    let v_out = linear_f32(&normed, v_data, 14, &v_dims, n_embd, k_size);
    let v_max = v_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_nan = v_out.iter().filter(|v| v.is_nan()).count();
    println!("V: max={:.4}, nan={}", v_max, v_nan);

    // Output projection
    println!("\n=== Output Projection (Q4_K, dims={:?}) ===", o_dims);
    // For single-token attention, output = V (attn weight = 1.0)
    // Then apply output projection
    let o_abs = data_offset + o_off;
    let o_data = &mmap[o_abs..o_abs + calc_size(&o_dims, 12)];
    let o_out = linear_f32(&v_out, o_data, 12, &o_dims, n_embd, n_embd);
    let o_max = o_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let o_nan = o_out.iter().filter(|v| v.is_nan()).count();
    println!("Output proj: max={:.4}, nan={}", o_max, o_nan);

    // Residual
    let mut hidden = vec_add(&embedding, &o_out);
    let h_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let h_nan = hidden.iter().filter(|v| v.is_nan()).count();
    println!("\nAfter attention residual: max={:.4}, nan={}", h_max, h_nan);
}

fn calc_size(dims: &[usize], dtype: u32) -> usize {
    let total: usize = dims.iter().product();
    let (block_size, block_bytes) = match dtype {
        0 => (1, 4),       // F32
        1 => (1, 2),       // F16
        2 => (32, 18),     // Q4_0
        12 => (256, 144),  // Q4_K
        14 => (256, 210),  // Q6_K
        _ => (256, 144),   // default Q4_K
    };
    ((total + block_size - 1) / block_size) * block_bytes
}

fn linear_f32(input: &[f32], data: &[u8], dtype: u32, dims: &[usize], in_feat: usize, out_feat: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_feat];
    let (block_size, block_bytes): (usize, usize) = match dtype {
        0 => (1, 4),
        1 => (1, 2),
        12 => (256, 144),
        14 => (256, 210),
        _ => (256, 144),
    };
    let blocks_per_row = (in_feat + block_size - 1) / block_size;
    
    for j in 0..out_feat.min(dims.get(1).copied().unwrap_or(0)) {
        let mut sum = 0.0f32;
        for i in 0..in_feat {
            let block_idx = j * blocks_per_row + i / block_size;
            let elem_in_block = i % block_size;
            
            let bo = block_idx * block_bytes;
            if bo + block_bytes > data.len() { break; }
            
            let val = match dtype {
                12 => dequant_q4_k_elem(&data[bo..bo+block_bytes], elem_in_block),
                14 => dequant_q6_k_elem(&data[bo..bo+block_bytes], elem_in_block),
                _ => 0.0f32,
            };
            sum += val * input[i];
        }
        output[j] = sum;
    }
    output
}

fn dequant_q4_k_elem(block: &[u8], elem_idx: usize) -> f32 {
    if block.len() < 144 { return 0.0; }
    let d = f16_f32(block[0], block[1]);
    let dmin = f16_f32(block[2], block[3]);
    let sc = &block[4..16];
    let qs = &block[16..144];
    
    let sub_block = elem_idx / 32;
    let elem_in_sub = elem_idx % 32;
    let is = sub_block;
    
    let (sc_val, m_val) = get_scale_min(is, sc);
    let d1 = d * sc_val;
    let m1 = dmin * m_val;
    
    let q_base = sub_block * 32;
    let nibble = if elem_in_sub < 32 {
        (qs[q_base + elem_in_sub] & 0x0F) as f32
    } else {
        ((qs[q_base + (elem_in_sub - 32)] >> 4) & 0x0F) as f32
    };
    
    d1 * nibble - m1
}

fn dequant_q6_k_elem(block: &[u8], elem_idx: usize) -> f32 {
    if block.len() < 210 { return 0.0; }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = f16_f32(block[208], block[209]);
    
    let is = elem_idx / 16;
    let ql_base = (elem_idx / 128) * 64;
    let l = elem_idx % 16;
    
    let qh_l = qh[ql_base + l] as i32;
    
    let (q1, q2) = if elem_idx % 32 < 16 {
        // First half
        let a = (ql[ql_base + l] & 0x0F) as i32 | (((qh_l >> 0) & 3) << 4);
        let b = (ql[ql_base + l + 32] & 0x0F) as i32 | (((qh_l >> 2) & 3) << 4);
        (a, b)
    } else {
        let a = ((ql[ql_base + l] >> 4) & 0x0F) as i32 | (((qh_l >> 4) & 3) << 4);
        let b = ((ql[ql_base + l + 32] >> 4) & 0x0F) as i32 | (((qh_l >> 6) & 3) << 4);
        (a, b)
    };
    
    let q = if elem_idx % 32 < 16 { q1 - 32 } else { q2 - 32 };
    let sc_base = (elem_idx / 128) * 8;
    let s = sc[sc_base + is] as f32 * d;
    
    s * q as f32
}

fn get_scale_min(j: usize, sc: &[u8]) -> (f32, f32) {
    let (s, m) = if j < 4 {
        ((sc[j] & 63) as f32, (sc[j + 4] & 63) as f32)
    } else {
        let hi: u8 = sc[j - 4] >> 6;
        let a = (sc[j + 4] & 0x0F) as f32 + hi as f32 * 16.0;
        let b = ((sc[j + 4]) >> 4) as f32 + (sc[j] >> 6) as f32 * 16.0;
        (a, b)
    };
    (s, m)
}

fn dequant_q4_k_block(block: &[u8]) -> Vec<f32> {
    let d = f16_f32(block[0], block[1]);
    let dmin = f16_f32(block[2], block[3]);
    let sc = &block[4..16];
    let qs = &block[16..144];
    
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
    
    let mut result = Vec::with_capacity(256);
    for g in 0..4 {
        let q_base = g * 32;
        let d1 = d * scales[2 * g] as f32;
        let m1 = dmin * mins[2 * g] as f32;
        let d2 = d * scales[2 * g + 1] as f32;
        let m2 = dmin * mins[2 * g + 1] as f32;
        for l in 0..32 {
            result.push(d1 * (qs[q_base + l] & 0x0F) as f32 - m1);
        }
        for l in 0..32 {
            result.push(d2 * ((qs[q_base + l] >> 4) as f32) - m2);
        }
    }
    result
}

fn f16_f32(b0: u8, b1: u8) -> f32 {
    let h = u16::from_le_bytes([b0, b1]);
    if h == 0 { return 0.0; }
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as f32;
    if exp == 0 { sign * 2.0_f32.powi(-14) * (frac / 1024.0) }
    else if exp == 31 { if frac == 0.0 { f32::INFINITY * sign } else { f32::NAN } }
    else { sign * 2.0_f32.powi(exp - 15) * (1.0 + frac / 1024.0) }
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

fn read_str(c: &mut Cursor<&[u8]>, n: usize) -> String {
    let mut b = vec![0u8; n]; c.read_exact(&mut b).unwrap();
    String::from_utf8_lossy(&b).to_string()
}

fn gguf_str(c: &mut Cursor<&[u8]>) -> std::io::Result<String> {
    let n = c.read_u64::<LittleEndian>()? as usize;
    if n == 0 { return Ok(String::new()); }
    let mut b = vec![0u8; n]; c.read_exact(&mut b)?;
    Ok(String::from_utf8(b).unwrap())
}

fn skip_val(c: &mut Cursor<&[u8]>, vt: u32) {
    match vt {
        0|1|7 => { let _ = c.read_u8(); }
        2|3 => { let _ = c.read_u16::<LittleEndian>(); }
        4|5 => { let _ = c.read_u32::<LittleEndian>(); }
        6 => { let _ = c.read_f32::<LittleEndian>(); }
        8 => { let n = c.read_u64::<LittleEndian>().unwrap() as usize; let mut b = vec![0u8; n]; c.read_exact(&mut b).unwrap(); }
        9 => { let et = c.read_u32::<LittleEndian>().unwrap(); let n = c.read_u64::<LittleEndian>().unwrap() as usize; for _ in 0..n { skip_val(c, et); } }
        10|11 => { let _ = c.read_u64::<LittleEndian>(); }
        12 => { let _ = c.read_f64::<LittleEndian>(); }
        _ => {}
    }
}
