//! Debug: dump first 20 tokens, their types, and find byte tokens
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "./models/tinyllama-1.1b-chat-q4_0.gguf".to_string());
    let file = File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut c = Cursor::new(&mmap[..]);

    read_str(&mut c, 4);
    c.read_u32::<LittleEndian>().unwrap();
    c.read_u64::<LittleEndian>().unwrap();
    let kv = c.read_u64::<LittleEndian>().unwrap() as usize;

    let mut all_tokens: Vec<String> = Vec::new();
    let mut all_types: Vec<i32> = Vec::new();

    for _ in 0..kv {
        let key = gguf_str(&mut c).unwrap();
        let vt = c.read_u32::<LittleEndian>().unwrap();
        match vt {
            8 => {
                let v = gguf_str(&mut c).unwrap();
                // We're looking for tokenizer.ggml.tokens and tokenizer.ggml.token_type
            }
            9 => {
                let et = c.read_u32::<LittleEndian>().unwrap();
                let len = c.read_u64::<LittleEndian>().unwrap() as usize;
                println!("{}: Array(et={}, len={})", key, et, len);
                if et == 8 && key == "tokenizer.ggml.tokens" {
                    for i in 0..len {
                        let s = gguf_str(&mut c).unwrap();
                        all_tokens.push(s);
                    }
                    println!("  Stored {} tokens", all_tokens.len());
                    // Print first 20
                    println!("  --- First 20 tokens ---");
                    for i in 0..20.min(all_tokens.len()) {
                        println!("  [{}] {:?}", i, all_tokens[i]);
                    }
                    // Print last 10
                    println!("  --- Last 10 tokens ---");
                    for i in (all_tokens.len().saturating_sub(10))..all_tokens.len() {
                        println!("  [{}] {:?}", i, all_tokens[i]);
                    }
                } else if et == 5 && key == "tokenizer.ggml.token_type" {
                    for i in 0..len {
                        let v = c.read_i32::<LittleEndian>().unwrap();
                        all_types.push(v);
                    }
                    println!("  Stored {} token types", all_types.len());
                    // Count types
                    let mut counts = HashMap::new();
                    for &t in &all_types {
                        *counts.entry(t).or_insert(0usize) += 1;
                    }
                    println!("  Type distribution: {:?}", counts);
                    // Print first 20 types
                    println!("  --- First 20 token types ---");
                    for i in 0..20.min(all_types.len()) {
                        println!("  [{}] type={}", i, all_types[i]);
                    }
                    // Find byte tokens (type=3 or type=2)
                    let mut byte_tokens = Vec::new();
                    for i in 0..all_types.len() {
                        if all_types[i] == 3 || all_types[i] == 2 {
                            byte_tokens.push(i);
                        }
                    }
                    if !byte_tokens.is_empty() {
                        println!("  --- Byte tokens (type=3) ---");
                        for &i in byte_tokens.iter().take(10) {
                            println!("  [{}] {:?}", i, all_tokens.get(i).map(|s| s.as_str()).unwrap_or("?"));
                        }
                    } else {
                        println!("  No byte tokens found!");
                        // Check type=0 tokens that are single chars
                        println!("  --- Single-char tokens (type=0) ---");
                        let mut single_chars = Vec::new();
                        for i in 0..all_types.len() {
                            if all_types[i] == 0 && all_tokens.get(i).map(|s| s.len() == 1).unwrap_or(false) {
                                single_chars.push(i);
                            }
                        }
                        println!("  Found {} single-char tokens", single_chars.len());
                        for &i in single_chars.iter().take(20) {
                            let tok = all_tokens.get(i).map(|s| s.as_str()).unwrap_or("?");
                            let b = all_tokens[i].as_bytes()[0];
                            println!("  [{}] byte={} char={:?}", i, b, tok);
                        }
                    }
                } else {
                    for _ in 0..len { skip_val(&mut c, et); }
                }
            }
            _ => { skip_val(&mut c, vt); }
        }
    }
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

use std::collections::HashMap;
