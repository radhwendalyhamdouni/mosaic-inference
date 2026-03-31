use std::fs::File;
use memmap2::Mmap;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

fn main() {
    let path = "./models/tinyllama-1.1b-chat-q4_0.gguf";
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let mut c = Cursor::new(&mmap[..]);

    read_str(&mut c, 4);
    c.read_u32::<LittleEndian>().unwrap();
    let tc = c.read_u64::<LittleEndian>().unwrap();
    let kvc = c.read_u64::<LittleEndian>().unwrap();
    for _ in 0..kvc { let _ = gguf_str(&mut c).unwrap(); let vt = c.read_u32::<LittleEndian>().unwrap(); skip_val(&mut c, vt); }

    let alignment = 32;
    let pos = c.position() as usize;
    let data_offset = (pos + alignment - 1) / alignment * alignment;

    let mut ow_offset = 0usize;
    let mut ow_dims = Vec::new();
    let mut on_offset = 0usize;
    let mut on_dims = Vec::new();

    for _ in 0..tc {
        let name = gguf_str(&mut c).unwrap();
        let nd = c.read_u32::<LittleEndian>().unwrap();
        let mut dims = Vec::new();
        for _ in 0..nd { dims.push(c.read_u64::<LittleEndian>().unwrap() as usize); }
        let dtype = c.read_u32::<LittleEndian>().unwrap();
        let offset = c.read_u64::<LittleEndian>().unwrap() as usize;
        if name == "output.weight" { ow_offset = data_offset + offset; ow_dims = dims.clone(); }
        if name == "output_norm.weight" { on_offset = data_offset + offset; on_dims = dims; }
    }

    let n_embd = ow_dims[0];
    let vocab = ow_dims[1];
    println!("output.weight: dims={:?}, offset={}", ow_dims, ow_offset);
    println!("output_norm: dims={:?}, offset={}", on_dims, on_offset);

    // Read norm weights
    let norm_bytes = on_dims[0] * 4;
    let norm: Vec<f32> = mmap[on_offset..on_offset + norm_bytes]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("Norm first 4: {:?}", &norm[..4]);

    // Create hidden vector (all 1.0)
    let hidden = vec![1.0f32; n_embd];
    let rms: f32 = (hidden.iter().map(|&v| v * v).sum::<f32>() / n_embd as f32).sqrt() + 1e-6;
    let normed: Vec<f32> = hidden.iter().zip(norm.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect();
    println!("Normed first 4: {:?}", &normed[..4]);

    // Compute logits for first 100 vocab entries using direct Q6_K dequant
    let block_size = 256;
    let blocks_per_row = (vocab + block_size - 1) / block_size;
    let bytes_per_block = 210;
    
    println!("\nComputing logits (sample 100 of {})...", vocab);
    let mut logits = vec![0.0f32; vocab];
    let mut nan_count = 0usize;
    let mut nonzero_count = 0usize;

    for j in 0..100 {
        let mut sum = 0.0f32;
        for i in 0..n_embd {
            let block_idx = i * blocks_per_row + j / block_size;
            let elem_in_block = j % block_size;
            let sub_block = elem_in_block / 16;
            let elem_in_sub = elem_in_block % 16;
            
            let bo = block_idx * bytes_per_block;
            if bo + bytes_per_block > mmap.len() { break; }
            
            let base = sub_block * 16;
            let idx = base + elem_in_sub;
            let ql_byte = mmap[bo + idx / 2];
            let ql_val = if idx % 2 == 0 { ql_byte & 0x0F } else { (ql_byte >> 4) & 0x0F };
            let qh_byte = mmap[bo + 128 + idx / 4];
            let qh_shift = 2 * (3 - (idx % 4));
            let qh_val = (qh_byte >> qh_shift) & 0x03;
            let q = (ql_val as i32) | ((qh_val as i32) << 4);
            let val = (q - 32) as f32;
            let scale = mmap[bo + 192 + sub_block] as f32 * f16_f32(mmap[bo + 208], mmap[bo + 209]);
            sum += val * scale * normed[i];
        }
        logits[j] = sum;
        if sum.is_nan() || sum.is_infinite() { nan_count += 1; }
        else if sum != 0.0 { nonzero_count += 1; }
    }
    
    println!("NaN/Inf: {}, Non-zero: {}", nan_count, nonzero_count);
    println!("First 20 logits: {:?}", &logits[..20]);

    // Top 5
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 logits:");
    for (idx, val) in indexed.iter().take(5) {
        println!("  [{}] {:.6}", idx, val);
    }
}

fn f16_f32(b0: u8, b1: u8) -> f32 {
    f16_to_f32(u16::from_le_bytes([b0, b1]))
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
