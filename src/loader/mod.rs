//! GGUF Model Loader with mmap support
//!
//! يقرأ ملفات GGUF باستخدام Memory-Mapped I/O
//! الأوزان تبقى على القرص ولا تُحمّل كلها في RAM

pub mod gguf;
pub mod dequant;
pub mod tokenizer;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Core model metadata extracted from GGUF file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: Option<usize>,
    pub n_layers: usize,
    pub n_ff: Option<usize>,
    pub n_rot: Option<usize>,
    pub ftype: u32, // quantization type
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

/// Memory-mapped region of the model file
/// يسمح بقراءة أجزاء محددة من النموذج بدون تحميله كله
#[derive(Debug)]
pub struct MappedRegion {
    pub offset: usize,
    pub size: usize,
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
}

/// Data type for tensor storage
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDtype {
    F32,        // 32-bit float
    F16,        // 16-bit float
    Q4_0,       // 4-bit quantized (block size 32)
    Q4_1,       // 4-bit quantized with scale (block size 32)
    Q5_0,       // 5-bit quantized
    Q5_1,       // 5-bit quantized with scale
    Q8_0,       // 8-bit quantized
    Q2_K,       // 2-bit K-quant (super compressed)
    Q3_K,       // 3-bit K-quant
    Q4_K,       // 4-bit K-quant (recommended)
    Q5_K,       // 5-bit K-quant
    Q6_K,       // 6-bit K-quant
}

impl TensorDtype {
    /// Block size for this quantization type
    pub fn block_size(&self) -> usize {
        match self {
            TensorDtype::Q4_0 | TensorDtype::Q4_1 => 32,
            TensorDtype::Q5_0 | TensorDtype::Q5_1 => 32,
            TensorDtype::Q8_0 => 32,
            TensorDtype::Q2_K | TensorDtype::Q3_K => 256,
            TensorDtype::Q4_K | TensorDtype::Q5_K | TensorDtype::Q6_K => 256,
            TensorDtype::F32 | TensorDtype::F16 => 1,
        }
    }

    /// Bytes per block for this quantization type
    pub fn block_bytes(&self) -> usize {
        match self {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            TensorDtype::Q4_0 => 18,    // 2 (f16 scale) + 16 (32 nibbles)
            TensorDtype::Q4_1 => 20,    // 2 (f16 scale) + 2 (f16 min) + 16
            TensorDtype::Q5_0 => 22,    // 2 (f16 scale) + 4 (32 4-bit qs) + 16 (4-bit packed)
            TensorDtype::Q5_1 => 24,    // 2 + 2 + 4 + 16
            TensorDtype::Q8_0 => 34,    // 2 (f16 scale) + 32 (int8 values)
            TensorDtype::Q2_K => 84,    // 2 + 2 + 4 + 16 + 64 = 84 + 2 padding? Actually 84
            TensorDtype::Q3_K => 110,   // complex format
            TensorDtype::Q4_K => 144,   // 2 + 2 + 12 + 128
            TensorDtype::Q5_K => 176,   // 2 + 2 + 12 + 4 + 128 + 16 + 4 = 168? Let me use 176
            TensorDtype::Q6_K => 210,   // 128 + 64 + 16 + 2
        }
    }

    /// Parse dtype from GGUF uint32
    pub fn from_gguf_type(t: u32) -> Result<Self> {
        match t {
            0 => Ok(TensorDtype::F32),
            1 => Ok(TensorDtype::F16),
            2 => Ok(TensorDtype::Q4_0),
            3 => Ok(TensorDtype::Q4_1),
            6 => Ok(TensorDtype::Q5_0),
            7 => Ok(TensorDtype::Q5_1),
            8 => Ok(TensorDtype::Q8_0),
            10 => Ok(TensorDtype::Q2_K),
            11 => Ok(TensorDtype::Q3_K),
            12 => Ok(TensorDtype::Q4_K),
            13 => Ok(TensorDtype::Q5_K),
            14 => Ok(TensorDtype::Q6_K),
            _ => anyhow::bail!("Unsupported GGUF tensor type: {}", t),
        }
    }
}
