# Mosaic Inference

> تشغيل أي نموذج لغوي كبير على أجهزة ضعيفة (4GB RAM, HDD) باستخدام Rust

## المفهوم

بدل تحميل النموذج كله في الذاكرة، نستخدم **4 ابتكارات معاً**:

1. **mmap Loading**: الأوزان تبقى على القرص، OS يحمّل فقط الصفحات المطلوبة (lazy loading)
2. **Layer Streaming**: تحميل طبقة طبقة مع Double Buffer Prefetch (طبقتين فقط في RAM!)
3. **KV Disk Offload**: سياق المحادثة ينتقل للقرص تلقائياً عندما يمتلئ RAM
4. **BitNet Ready**: هيكل جاهز لدعم أوزان 1-bit {-1, 0, +1}

## كيف يعمل؟

```
الطريقة التقليدية:
  النموذج كله (4GB) → RAM → يفشل على 4GB! ❌

Mosaic Inference:
  الطبقة 0,1 → RAM (200MB فقط) ← CPU ينفّذ
  الطبقة 2-35 → على القرص (mmap) ← تُحمّل عند الحاجة
  KV Cache القديم → على القرص ← يُقرأ عند الحاجة
  النتيجة: يعمل على 4GB! ✅
```

## Architecture

```
src/
├── main.rs              # Entry point + CLI
├── loader/
│   ├── mod.rs           # Tensor types + metadata
│   └── gguf.rs          # GGUF parser with mmap
├── engine/
│   ├── mod.rs           # Core inference engine
│   ├── layer.rs         # Transformer layer forward pass
│   ├── forward.rs       # Final projection + logits
│   └── stream.rs        # Layer streaming + double buffering
├── cache/
│   └── mod.rs           # KV Cache tiering (RAM → Disk)
├── sampler/
│   └── mod.rs           # Token sampling (temperature, top-p)
└── server/
    └── mod.rs           # OpenAI-compatible HTTP API
```

## Setup

### Prerequisites

```bash
# Install Rust (Debian)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Build

```bash
git clone https://github.com/radhwendalyhamdouni/mosaic-inference.git
cd mosaic-inference
cargo build --release
```

### Run

```bash
# Download a GGUF model (example: Qwen2.5-Coder-3B Q2_K)
# From: https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF

# Run the server
./target/release/mosaic-inference \
  --model qwen2.5-coder-3b-instruct-q2_k.gguf \
  --ctx-size 8192 \
  --ram-layers 2 \
  --port 8080 \
  --verbose
```

### API Usage (OpenAI Compatible)

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mosaic",
    "messages": [
      {"role": "user", "content": "Write a Python function to sort an array"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# List models
curl http://localhost:8080/v1/models

# Memory stats
curl http://localhost:8080/stats

# Health check
curl http://localhost:8080/health
```

## Memory Optimization

### ZRAM Setup (Recommended for 4GB systems)

```bash
# Compress RAM to effectively double your memory
sudo modprobe zram
sudo zramctl /dev/zram0 --algorithm lz4 --size 4G
sudo mkswap /dev/zram0
sudo swapon -p 200 /dev/zram0

# Add USB drive as additional swap (optional)
sudo mkswap /dev/sdb1
sudo swapon -p 100 /dev/sdb1
sudo sysctl vm.swappiness=10
```

### Expected Memory Usage

| Component | Without Mosaic | With Mosaic |
|-----------|---------------|-------------|
| Model weights (7B Q2_K) | 2.8 GB | ~150 MB (2 layers) |
| KV Cache (8K context) | 1.5 GB | ~200 MB (windowed) |
| System + OS | 1.2 GB | 1.2 GB |
| **Total** | **5.5 GB** ❌ | **~1.55 GB** ✅ |

## Supported Formats

- GGUF (llama.cpp format) - all quantization types:
  - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (K-quants)
  - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (legacy)
  - F16, F32 (full precision)

## Roadmap

- [x] Phase 1: GGUF parser with mmap
- [x] Phase 1: Layer streaming with double buffering
- [x] Phase 1: KV Cache disk offload
- [x] Phase 1: OpenAI-compatible HTTP API
- [ ] Phase 2: AVX2 SIMD optimizations
- [ ] Phase 2: Real tokenizer (SentencePiece + BPE)
- [ ] Phase 2: BitNet 1.58-bit support
- [ ] Phase 3: Pattern database for code generation
- [ ] Phase 3: Selective layer execution (Circuit Stealing)
- [ ] Phase 3: Embedding-based code search

## Technical Details

### How mmap works here

When we `mmap` the model file:
- The file stays on disk (SSD/HDD)
- No RAM is used initially
- When we access a tensor → OS triggers a page fault → loads only that 4KB page
- Unused pages can be evicted by OS automatically
- On Linux, the OS is very smart about this

### How Layer Streaming works

```
Time →
CPU:    [Execute L0] [Execute L1] [Execute L2] [Execute L3] ...
Disk:              [Load L2]   [Load L3]   [Load L4]   ...
RAM:    [L0,L1]      [L1,L2]     [L2,L3]     [L3,L4]   ...
```

While CPU processes layer N, a background thread loads layer N+1.
Only 2 layers ever in RAM, regardless of total model size.

### How KV Disk Offload works

```
Position:  0  1  2  ...  1023  1024  1025  ...  8191
Memory:    [RAM - hot window]   [Disk - cold storage]
```

Most recent tokens stay in RAM (fast access).
Older tokens move to disk automatically.
When needed, they're loaded back (slower but works).

## License

MIT
