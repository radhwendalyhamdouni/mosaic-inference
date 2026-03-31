//! Quick tool to dump GGUF metadata keys (excluding large arrays)
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 { &args[1] } else { "./models/tinyllama-1.1b-chat-q4_0.gguf" };

    let file = File::open(path).expect("Cannot open file");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap failed") };
    let mut cursor = Cursor::new(&mmap[..]);

    let magic = read_str(&mut cursor, 4);
    println!("Magic: {}", magic);
    let version = cursor.read_u32::<LittleEndian>().unwrap();
    println!("Version: {}", version);
    let tensor_count = cursor.read_u64::<LittleEndian>().unwrap();
    println!("Tensor count: {}", tensor_count);
    let kv_count = cursor.read_u64::<LittleEndian>().unwrap();
    println!("KV count: {}", kv_count);
    println!();

    for _ in 0..kv_count {
        let key = read_gguf_str(&mut cursor).unwrap();
        let vtype = cursor.read_u32::<LittleEndian>().unwrap();
        
        match vtype {
            0 => { let v = cursor.read_u8().unwrap(); println!("{} = {}", key, v); }
            1 => { let v = cursor.read_i8().unwrap(); println!("{} = {}", key, v); }
            2 => { let v = cursor.read_u16::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            3 => { let v = cursor.read_i16::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            4 => { let v = cursor.read_u32::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            5 => { let v = cursor.read_i32::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            6 => { let v = cursor.read_f32::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            7 => { let v = cursor.read_u8().unwrap(); println!("{} = {}", key, v != 0); }
            8 => { let v = read_gguf_str(&mut cursor).unwrap(); println!("{} = \"{}\"", key, v); }
            9 => {
                let etype = cursor.read_u32::<LittleEndian>().unwrap();
                let len = cursor.read_u64::<LittleEndian>().unwrap() as usize;
                // Skip array elements but print count
                for _ in 0..len {
                    skip_value(&mut cursor, etype);
                }
                println!("{} = Array(type={}, len={})", key, etype, len);
            }
            10 => { let v = cursor.read_u64::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            11 => { let v = cursor.read_i64::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            12 => { let v = cursor.read_f64::<LittleEndian>().unwrap(); println!("{} = {}", key, v); }
            _ => { println!("{} = UNKNOWN_TYPE({})", key, vtype); break; }
        }
    }

    // Print tensor names
    println!("\n=== Tensors ===");
    for i in 0..tensor_count as usize {
        let name = read_gguf_str(&mut cursor).unwrap();
        let n_dims = cursor.read_u32::<LittleEndian>().unwrap();
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(cursor.read_u64::<LittleEndian>().unwrap());
        }
        let dtype = cursor.read_u32::<LittleEndian>().unwrap();
        let offset = cursor.read_u64::<LittleEndian>().unwrap();
        println!("  [{}] {} dims={:?} dtype={} offset={}", i, name, dims, dtype, offset);
    }
}

fn read_str(cursor: &mut Cursor<&[u8]>, len: usize) -> String {
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf).unwrap();
    String::from_utf8_lossy(&buf).to_string()
}

fn read_gguf_str(cursor: &mut Cursor<&[u8]>) -> Result<String, Box<dyn std::error::Error>> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    if len == 0 { return Ok(String::new()); }
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn skip_value(cursor: &mut Cursor<&[u8]>, vtype: u32) {
    match vtype {
        0 => { let _ = cursor.read_u8(); }
        1 => { let _ = cursor.read_i8(); }
        2 => { let _ = cursor.read_u16::<LittleEndian>(); }
        3 => { let _ = cursor.read_i16::<LittleEndian>(); }
        4 => { let _ = cursor.read_u32::<LittleEndian>(); }
        5 => { let _ = cursor.read_i32::<LittleEndian>(); }
        6 => { let _ = cursor.read_f32::<LittleEndian>(); }
        7 => { let _ = cursor.read_u8(); }
        8 => {
            let len = cursor.read_u64::<LittleEndian>().unwrap() as usize;
            let mut buf = vec![0u8; len];
            cursor.read_exact(&mut buf).unwrap();
        }
        9 => {
            let etype = cursor.read_u32::<LittleEndian>().unwrap();
            let len = cursor.read_u64::<LittleEndian>().unwrap() as usize;
            for _ in 0..len {
                skip_value(cursor, etype);
            }
        }
        10 => { let _ = cursor.read_u64::<LittleEndian>(); }
        11 => { let _ = cursor.read_i64::<LittleEndian>(); }
        12 => { let _ = cursor.read_f64::<LittleEndian>(); }
        _ => {}
    }
}
