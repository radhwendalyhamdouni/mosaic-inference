//! Memory Tiering System
//!
//! نظام الذاكرة الهرمي:
//! - Tier 1: RAM (سريع، محدود)
//! - Tier 2: Disk KV Cache (أبطأ، واسع)
//!
//! KV Cache = ذاكرة السياق
//! بدونها: النموذج ينسى كل كلمة سابقة
//! معها: يتذكر السياق الكامل (ولكن تستهلك ذاكرة!)

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use tracing::{info, debug};

/// Disk-backed KV Cache for each layer
pub struct MemoryTier {
    /// Maximum tokens to keep in RAM per layer
    ram_window_size: usize,
    /// Maximum tokens to keep on disk per layer
    disk_window_size: usize,
    /// Directory for disk-backed KV cache
    cache_dir: PathBuf,
    /// Hot cache: layer_idx -> (token_pos, k_vec, v_vec)
    hot_cache: HashMap<usize, HashMap<usize, (Vec<f32>, Vec<f32>)>>,
    /// Total layers
    n_layers: usize,
    /// Disk write count (for monitoring)
    disk_writes: usize,
    /// Disk read count
    disk_reads: usize,
}

impl MemoryTier {
    /// Create a new memory tier system
    pub fn new(
        ctx_size: usize,
        n_layers: usize,
        _ram_layers: usize,
        cache_dir: &str,
    ) -> Self {
        // RAM holds the most recent tokens
        // Disk holds older tokens
        let ram_window_size = (ctx_size / 3).max(1024); // At least 1024 tokens in RAM
        let disk_window_size = ctx_size; // Full context on disk

        // Create cache directory
        let cache_path = PathBuf::from(cache_dir);
        if !cache_path.exists() {
            fs::create_dir_all(&cache_path).ok();
        }

        info!("Memory Tier initialized:");
        info!("  RAM window: {} tokens per layer", ram_window_size);
        info!("  Disk window: {} tokens per layer", disk_window_size);
        info!("  Cache dir: {}", cache_dir);

        Self {
            ram_window_size,
            disk_window_size,
            cache_dir: cache_path,
            hot_cache: HashMap::new(),
            n_layers,
            disk_writes: 0,
            disk_reads: 0,
        }
    }

    /// Store KV vectors for a specific layer and token position
    /// إذا كانت ضمن نافذة RAM → تُخزن في RAM
    /// إذا تجاوزت النافذة → تُكتب على القرص
    pub fn store_kv(&mut self, layer_idx: usize, token_pos: usize, kv: &[f32]) -> Result<()> {
        // Split kv into k and v (they have the same length)
        let half = kv.len() / 2;
        let k = kv[..half].to_vec();
        let v = kv[half..].to_vec();

        if token_pos < self.ram_window_size {
            // Store in RAM (hot cache)
            let layer_cache = self.hot_cache
                .entry(layer_idx)
                .or_insert_with(HashMap::new);
            layer_cache.insert(token_pos, (k, v));
        } else {
            // Evict from RAM if present, store on disk
            let to_evict: Vec<(usize, Vec<f32>, Vec<f32>)> = {
                if let Some(layer_cache) = self.hot_cache.get_mut(&layer_idx) {
                    let keys_to_remove: Vec<usize> = layer_cache.keys()
                        .filter(|&&pos| pos < token_pos.saturating_sub(self.ram_window_size))
                        .copied()
                        .collect();
                    keys_to_remove.into_iter()
                        .filter_map(|key| layer_cache.remove(&key).map(|(k, v)| (key, k, v)))
                        .collect()
                } else {
                    Vec::new()
                }
            };

            // Write evicted entries to disk
            for (key, k_evict, v_evict) in to_evict {
                self.write_kv_to_disk(layer_idx, key, &k_evict, &v_evict)?;
            }

            // Write current KV to disk
            self.write_kv_to_disk(layer_idx, token_pos, &k, &v)?;
        }

        Ok(())
    }

    /// Retrieve KV vectors for a specific layer and token position
    /// يبحث في RAM أولاً، ثم القرص
    pub fn get_kv(&mut self, layer_idx: usize, token_pos: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        // Check hot cache (RAM) first
        if let Some(layer_cache) = self.hot_cache.get(&layer_idx) {
            if let Some(kv) = layer_cache.get(&token_pos) {
                return Ok(kv.clone());
            }
        }

        // Not in RAM → load from disk
        let (k, v) = self.read_kv_from_disk(layer_idx, token_pos)?;
        self.disk_reads += 1;
        Ok((k, v))
    }

    /// Write KV data to disk
    fn write_kv_to_disk(&mut self, layer_idx: usize, token_pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        let dir = self.cache_dir.join(format!("layer_{}", layer_idx));
        fs::create_dir_all(&dir)?;

        let path = dir.join(format!("kv_{}.bin", token_pos));
        let mut file = fs::File::create(path)?;

        // Write k and v as raw f32 bytes
        file.write_all(&(k.len() as u32).to_le_bytes())?;
        for val in k {
            file.write_all(&val.to_le_bytes())?;
        }
        for val in v {
            file.write_all(&val.to_le_bytes())?;
        }

        self.disk_writes += 1;
        Ok(())
    }

    /// Read KV data from disk
    fn read_kv_from_disk(&self, layer_idx: usize, token_pos: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let path = self.cache_dir
            .join(format!("layer_{}", layer_idx))
            .join(format!("kv_{}.bin", token_pos));

        if !path.exists() {
            anyhow::bail!("KV cache not found for layer {} pos {}", layer_idx, token_pos);
        }

        let mut file = fs::File::open(path)?;
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        let mut buf = vec![0u8; len * 4 * 2]; // k + v
        file.read_exact(&mut buf)?;

        let k: Vec<f32> = buf[..len * 4]
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let v: Vec<f32> = buf[len * 4..]
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        debug!(
            "KV cache loaded from disk: layer={} pos={} len={}",
            layer_idx, token_pos, len
        );

        Ok((k, v))
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> MemoryStats {
        let mut ram_entries = 0;
        let mut ram_bytes = 0;

        for layer_cache in self.hot_cache.values() {
            for (k, v) in layer_cache.values() {
                ram_entries += 1;
                ram_bytes += k.len() * 4 + v.len() * 4;
            }
        }

        MemoryStats {
            ram_entries,
            ram_bytes,
            ram_mb: ram_bytes as f64 / 1_048_576.0,
            disk_writes: self.disk_writes,
            disk_reads: self.disk_reads,
            n_layers: self.n_layers,
            ram_window_size: self.ram_window_size,
        }
    }

    /// Clear all caches
    pub fn clear(&mut self) -> Result<()> {
        self.hot_cache.clear();

        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
            fs::create_dir_all(&self.cache_dir)?;
        }

        info!("Memory tier cleared");
        Ok(())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub ram_entries: usize,
    pub ram_bytes: usize,
    pub ram_mb: f64,
    pub disk_writes: usize,
    pub disk_reads: usize,
    pub n_layers: usize,
    pub ram_window_size: usize,
}
