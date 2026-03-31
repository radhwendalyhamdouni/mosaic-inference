//! GGUF File Parser with Memory-Mapped Loading
//!
//! GGUF هو تنسيق الملفات القياسي لمشروع GGML (llama.cpp)
//! الميزة الأساسية: mmap يسمح بقراءة الأوزان من القرص بدون تحميلها كلها
//!
//! هيكل ملف GGUF:
//! ┌─────────────────────────┐
//! │ Magic: "GGUF" (4 bytes) │
//! │ Version (uint32)        │
//! │ Tensor count (uint64)   │
//! │ Metadata KV count       │
//! │ Metadata (key-value)    │
//! │ Alignment padding       │
//! │ Tensor data (offsets)   │
//! └─────────────────────────┘

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};
use std::path::Path;

use super::{MappedRegion, ModelMetadata, TensorDtype};

/// A GGUF model loaded with memory-mapped I/O
/// الملف يبقى على القرص، ونقرأ منه فقط ما نحتاجه
pub struct GgufModel {
    /// Memory-mapped file data (not in RAM until accessed!)
    pub mmap: Mmap,
    pub metadata: ModelMetadata,
    /// Mapping from tensor name to its location in the file
    pub tensor_map: HashMap<String, MappedRegion>,
    /// Offset where tensor data starts
    pub data_offset: usize,
    /// Total file size in bytes
    pub file_size: usize,
    /// Tensor info: name -> (n_dims, dims, offset, type)
    tensor_infos: Vec<TensorInfo>,
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    offset: u64,
    dtype: TensorDtype,
}

impl GgufModel {
    /// Load a GGUF model using memory-mapped I/O
    /// الملف لا يُحمّل في RAM! فقط الـ metadata
    pub fn load(path: &str) -> Result<Self> {
        let path = Path::new(path);
        let file = File::open(path)
            .with_context(|| format!("Cannot open model file: {}", path.display()))?;

        let file_size = file.metadata()?.len() as usize;

        // Memory-map the entire file
        // هذا لا يستهلك RAM! فقط يربط العنوان الافتراضي بالملف
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| "Failed to memory-map model file")?;

        info!("Memory-mapped {} bytes ({:.1} MB)", file_size, file_size as f64 / 1_048_576.0);

        let mut cursor = Cursor::new(&mmap[..]);

        // Read header
        let magic = read_string(&mut cursor, 4)?;
        if magic != "GGUF" {
            anyhow::bail!("Not a GGUF file! Magic: {:?}", magic);
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        info!("GGUF version: {}", version);

        let _tensor_count = cursor.read_u64::<LittleEndian>()?;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;

        info!("Tensor count: {}", _tensor_count);
        info!("Metadata KV pairs: {}", metadata_kv_count);

        // Read metadata key-value pairs
        let mut metadata_map: HashMap<String, Value> = HashMap::new();

        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let value_type = cursor.read_u32::<LittleEndian>()?;
            let value = read_gguf_value(&mut cursor, value_type)?;
            metadata_map.insert(key, value);
        }

        // Extract model metadata from GGUF metadata
        let metadata = extract_metadata(&metadata_map);

        // Calculate alignment and data offset
        let alignment = metadata_map
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        // Pad to alignment
        let current_pos = cursor.position() as usize;
        let data_offset = (current_pos + alignment - 1) / alignment * alignment;

        info!("Data offset: {} bytes", data_offset);

        // Read tensor info
        let mut tensor_infos = Vec::new();
        for _ in 0.._tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64::<LittleEndian>()?);
            }

            let dtype_raw = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;

            let dtype = TensorDtype::from_gguf_type(dtype_raw)?;

            tensor_infos.push(TensorInfo {
                name,
                n_dims,
                dims,
                offset,
                dtype,
            });
        }

        // Build tensor map
        let mut tensor_map = HashMap::new();
        for info in &tensor_infos {
            let size = calculate_tensor_size(&info.dims, &info.dtype);
            tensor_map.insert(
                info.name.clone(),
                MappedRegion {
                    offset: (data_offset + info.offset as usize),
                    size,
                    tensor_name: info.name.clone(),
                    shape: info.dims.iter().map(|&d| d as usize).collect(),
                    dtype: info.dtype,
                },
            );
        }

        Ok(GgufModel {
            mmap,
            metadata,
            tensor_map,
            data_offset,
            file_size,
            tensor_infos,
        })
    }

    /// Read raw bytes for a specific tensor from the memory-mapped file
    /// يقرأ من القرص فقط الصفحات المطلوبة (lazy loading)
    pub fn read_tensor_bytes(&self, tensor_name: &str) -> Result<Vec<u8>> {
        let region = self.tensor_map
            .get(tensor_name)
            .with_context(|| format!("Tensor not found: {}", tensor_name))?;

        if region.offset + region.size > self.mmap.len() {
            anyhow::bail!(
                "Tensor {} out of bounds: offset={} size={} file_len={}",
                tensor_name, region.offset, region.size, self.mmap.len()
            );
        }

        // This triggers a page fault → OS loads only the needed 4KB pages
        Ok(self.mmap[region.offset..region.offset + region.size].to_vec())
    }

    /// Get a reference to tensor data in the mmap (zero-copy!)
    /// لا ينسخ البيانات! يعيد مرجع مباشر للبيانات المعيّنة في الذاكرة
    pub fn get_tensor_slice(&self, tensor_name: &str) -> Result<&[u8]> {
        let region = self.tensor_map
            .get(tensor_name)
            .with_context(|| format!("Tensor not found: {}", tensor_name))?;

        if region.offset + region.size > self.mmap.len() {
            anyhow::bail!("Tensor {} out of bounds", tensor_name);
        }

        Ok(&self.mmap[region.offset..region.offset + region.size])
    }

    /// Get all tensor names for a specific layer
    /// مثال: layer 5 → ["blk.5.attn_q.weight", "blk.5.attn_k.weight", ...]
    pub fn get_layer_tensors(&self, layer_idx: usize) -> Vec<String> {
        let prefix = format!("blk.{}.", layer_idx);
        self.tensor_map
            .keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect()
    }

    /// Get the total size of all tensors for a specific layer
    pub fn get_layer_size(&self, layer_idx: usize) -> usize {
        self.get_layer_tensors(layer_idx)
            .iter()
            .map(|name| self.tensor_map[name].size)
            .sum()
    }

    /// Get all layer indices
    pub fn get_layer_indices(&self) -> Vec<usize> {
        let mut layers = Vec::new();
        for key in self.tensor_map.keys() {
            if let Some(rest) = key.strip_prefix("blk.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        if !layers.contains(&idx) {
                            layers.push(idx);
                        }
                    }
                }
            }
        }
        layers.sort();
        layers
    }
}

/// Calculate tensor size in bytes based on shape and dtype
fn calculate_tensor_size(dims: &[u64], dtype: &TensorDtype) -> usize {
    let total_elements: usize = dims.iter().map(|&d| d as usize).product();
    (total_elements as f64 * dtype.bytes_per_weight()) as usize
}

/// GGUF value types enum
#[derive(Debug, Clone)]
enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl Value {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint64(v) => Some(*v),
            Value::Uint32(v) => Some(*v as u64),
            Value::Int32(v) => Some(*v as u64),
            Value::Uint16(v) => Some(*v as u64),
            Value::Uint8(v) => Some(*v as u64),
            _ => None,
        }
    }

    fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Read a fixed-length string from cursor
fn read_string(cursor: &mut Cursor<&[u8]>, len: usize) -> Result<String> {
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

/// Read a GGUF string (length-prefixed)
fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    if len == 0 {
        return Ok(String::new());
    }
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

/// Read a GGUF value based on type
fn read_gguf_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<Value> {
    match value_type {
        0 => Ok(Value::Uint8(cursor.read_u8()?)),
        1 => Ok(Value::Int8(cursor.read_i8()?)),
        2 => Ok(Value::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(Value::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(Value::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(Value::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(Value::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(Value::Bool(cursor.read_u8()? != 0)),
        8 => Ok(Value::String(read_gguf_string(cursor)?)),
        9 => {
            // Array
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_gguf_value(cursor, elem_type)?);
            }
            Ok(Value::Array(arr))
        }
        10 => Ok(Value::Uint64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(Value::Int64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(Value::Float64(cursor.read_f64::<LittleEndian>()?)),
        _ => anyhow::bail!("Unknown GGUF value type: {}", value_type),
    }
}

/// Extract ModelMetadata from GGUF metadata map
fn extract_metadata(map: &HashMap<String, Value>) -> ModelMetadata {
    let get_str = |key: &str| -> String {
        map.get(key)
            .and_then(|v| v.as_string())
            .unwrap_or("unknown")
            .to_string()
    };

    let get_u64 = |key: &str| -> u64 {
        map.get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };

    ModelMetadata {
        name: get_str("general.name"),
        version: get_str("general.version"),
        architecture: get_str("general.architecture"),
        vocab_size: get_u64(&format!("{}.vocab_size", get_str("general.architecture"))) as usize,
        n_embd: get_u64(&format!("{}.embedding_length", get_str("general.architecture"))) as usize,
        n_head: get_u64(&format!("{}.attention.head_count", get_str("general.architecture"))) as usize,
        n_head_kv: {
            let v = get_u64(&format!("{}.attention.head_count_kv", get_str("general.architecture")));
            if v == 0 { None } else { Some(v as usize) }
        },
        n_layers: get_u64(&format!("{}.block_count", get_str("general.architecture"))) as usize,
        n_ff: {
            let v = get_u64(&format!("{}.feed_forward_length", get_str("general.architecture")));
            if v == 0 { None } else { Some(v as usize) }
        },
        n_rot: {
            let v = get_u64(&format!("{}.rotary.dim", get_str("general.architecture")));
            if v == 0 { None } else { Some(v as usize) }
        },
        ftype: get_u64(&format!("{}.quantization_version", get_str("general.architecture"))) as u32,
        bos_token_id: get_u64("tokenizer.ggml.bos_token_id") as u32,
        eos_token_id: get_u64("tokenizer.ggml.eos_token_id") as u32,
    }
}
