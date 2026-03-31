//! Quick test: dequantize a small Q6_K tensor and check for NaN/Inf
use std::fs::File;
use memmap2::Mmap;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

fn main() {
    let path = "./models/tinyllama-1.1b-chat-q4_0.gguf";
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let mut c = Cursor::new(&mmap[..]);

    // Read GGUF header
    let magic = read_str(&mut c, 4);
    println!("Magic: {}", magic);
    let version = c.read_u32::<LittleEndian>().unwrap();
    println!("Version: {}", version);
    let tensor_count = c.read_u64::<LittleEndian>().unwrap();
    let kv_count = c.read_u64::<LittleEndian>().unwrap();

    // Skip metadata
    for _ in 0..kv_count {
        let key = gguf_str(&mut c).unwrap();
        let vt = c.read_u32::<LittleEndian>().unwrap();
        skip_val(&mut c, vt);
    }

    // Read tensor info
    let mut tensor_starts: Vec<(String, usize, u32, Vec<u64>, usize)> = Vec::new();
    let mut data_start = 0usize;

    for i in 0..tensor_count as usize {
        let name = gguf_str(&mut c).unwrap();
        let nd = c.read_u32::<LittleEndian>().unwrap();
        let mut dims = Vec::new();
        for _ in 0..nd { dims.push(c.read_u64::<LittleEndian>().unwrap()); }
        let dtype = c.read_u32::<LittleEndian>().unwrap();
        let offset = c.read_u64::<LittleEndian>().unwrap();
        tensor_starts.push((name.clone(), offset as usize, dtype, dims.clone(), i));
    }

    // data_start = aligned cursor position
    let alignment = 32;
    let pos = c.position() as usize;
    data_start = (pos + alignment - 1) / alignment * alignment;
    println!("Data starts at: {}", data_start);

    // Find a small Q6_K tensor to test
    for (name, offset, dtype, dims, _) in &tensor_starts {
        if *dtype == 14 && name.contains("blk.0.attn_v") {
            let total: usize = dims.iter().map(|&d| d as usize).product();
            let start = data_start + offset;
            let block_count = (total + 255) / 256;
            let expected_bytes = block_count * 210;
            println!("\n=== Testing {} ===", name);
            println!("  dims: {:?}", dims);
            println!("  total elements: {}", total);
            println!("  offset: {}", start);
            println!("  expected bytes (Q6_K): {}", expected_bytes);

            let raw = &mmap[start..start + expected_bytes.min(mmap.len() - start)];

            // Test dequantize first block
            let mut nan_count = 0;
            let mut inf_count = 0;
            let mut zero_count = 0;

            for b in 0..block_count.min(10) {
                let block = &raw[b * 210..(b + 1) * 210];
                let result = dequant_q6_k_block(block);
                for &v in &result {
                    if v.is_nan() { nan_count += 1; }
                    if v.is_infinite() { inf_count += 1; }
                    if v == 0.0 { zero_count += 1; }
                }
                println!("  Block {}: first 8 values = {:?}", b, &result[..8.min(result.len())]);
            }

            println!("  NaN: {}, Inf: {}, Zero: {} (out of {} sampled)", 
                nan_count, inf_count, zero_count, 10 * 256);
        }
    }

    // Also test Q4_K (dtype=12)
    for (name, offset, dtype, dims, _) in &tensor_starts {
        if *dtype == 12 && name.contains("blk.0.attn_q") {
            let total: usize = dims.iter().map(|&d| d as usize).product();
            let start = data_start + offset;
            let block_count = (total + 255) / 256;
            let expected_bytes = block_count * 144;
            println!("\n=== Testing {} (Q4_K) ===", name);
            println!("  dims: {:?}", dims);
            println!("  total: {}, blocks: {}, bytes: {}", total, block_count, expected_bytes);

            let raw = &mmap[start..start + expected_bytes.min(mmap.len() - start)];

            for b in 0..block_count.min(3) {
                let block = &raw[b * 144..(b + 1) * 144];
                let result = dequant_q4_k_block(block);
                let has_nan = result.iter().any(|v| v.is_nan());
                let has_inf = result.iter().any(|v| v.is_infinite());
                println!("  Block {}: first 8 = {:?}, nan={}, inf={}", b, &result[..8.min(result.len())], has_nan, has_inf);
            }
        }
    }

    // Test F32 tensor (norm weights)
    for (name, offset, dtype, dims, _) in &tensor_starts {
        if *dtype == 0 && name.contains("blk.0.attn_norm") {
            let total: usize = dims.iter().map(|&d| d as usize).product();
            let start = data_start + offset;
            let expected_bytes = total * 4;
            println!("\n=== Testing {} (F32) ===", name);
            println!("  dims: {:?}", dims);
            println!("  bytes: {}", expected_bytes);

            let raw = &mmap[start..start + expected_bytes.min(mmap.len() - start)];
            let vals: Vec<f32> = raw.chunks(4)
                .take(16)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            println!("  First 16 values: {:?}", vals);
        }
    }
}

fn dequant_q6_k_block(block: &[u8]) -> Vec<f32> {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales_raw = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut result = Vec::with_capacity(256);

    for sb in 0..16 {
        let scale = scales_raw[sb] as f32 * d;
        let base = sb * 16;
        for i in 0..16 {
            let ql_byte = ql[(base + i) / 2];
            let ql_val = if (base + i) % 2 == 0 { ql_byte & 0x0F } else { (ql_byte >> 4) & 0x0F };
            let qh_byte = qh[(base + i) / 4];
            let qh_shift = 2 * (3 - ((base + i) % 4));
            let qh_val = (qh_byte >> qh_shift) & 0x03;
            let q = (ql_val as i32) | ((qh_val as i32) << 4);
            let val = (q - 32) as f32;
            result.push(val * scale);
        }
    }
    result
}

fn dequant_q4_k_block(block: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let mut scales = Vec::with_capacity(8);
    let sc = &block[4..8];
    let m = &block[8..16];
    for i in 0..8 {
        let sc_val = ((sc[i / 2] >> (4 * (i % 2))) & 0x0F) as f32;
        let m_val = m[i] as f32;
        scales.push(sc_val + m_val / 16.0);
    }
    let mut result = Vec::with_capacity(256);
    for sb in 0..8 {
        let scale = scales[sb];
        let q_offset = 16 + sb * 16;
        for i in 0..32 {
            let byte_idx = q_offset + i / 2;
            let shift = if i % 2 == 0 { 4 } else { 0 };
            let q = ((block[byte_idx] >> shift) & 0x0F) as i32;
            result.push(dmin + d * (q - 8) as f32 * scale);
        }
    }
    result
}

fn f16_to_f32(h: u16) -> f32 {
    if h == 0 { return 0.0; }
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as f32;
    if exp == 0 { sign * 2.0_f32.powi(-14) * (frac / 1024.0) }
    else if exp == 31 { if frac == 0.0 { f32::INFINITY * sign } else { f32::NAN } }
    else { sign * 2.0_f32.powi(exp - 15) * (1.0 + frac / 1024.0) }
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
