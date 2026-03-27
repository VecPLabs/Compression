# GIHKCC Monolithic — KV Cache Compression for Standard Transformers

**VecP Labs LLC** | [vecplabs.com](https://vecplabs.com) | Patent Pending (USPTO 63/931,565)

Adaptation of Guardian-Informed Hierarchical KV Cache Compression for monolithic (non-Cerberus) transformer architectures. Implements Levels 1–3 of the recursive multi-scale fold-compress pipeline using statistical SNR as a proxy for Guardian-based keyframe placement.

## What This Does

Compresses a transformer's KV cache by exploiting inter-layer, inter-keyframe, and inter-token redundancy — the same structural redundancy that H.264/H.265 exploits in video, applied to attention state.

**Pipeline:** `KV → L1 (inter-layer fold) → L2 (inter-keyframe fold) → L3 (inter-token fold) → quantize`

### Compression Levels

| Level | Redundancy Source | Proxy for Guardian | Estimated Ratio |
|-------|------------------|--------------------|-----------------|
| L0 (Cerberus only) | Inter-head-group | N/A — structural | 1.5–2x |
| **L1** | Inter-layer (depth) | Cosine similarity | **3–5x** |
| **L2** | Inter-keyframe (deep structure) | Super-keyframe grouping | **2–3x** |
| **L3** (optional) | Inter-token (sequence coherence) | Token-stride keyframes | **1.5–2x** |

Without L0 (monolithic), practical ceiling is ~60–100x with quantized deltas on real model KV caches.

## Files

```
gihkcc.py       — Core compression engine (L1/L2/L3 + quantization)
gihkcc_hf.py    — HuggingFace transformers integration
test_gihkcc.py  — Test suite with synthetic KV cache validation
```

## Quick Start

### Standalone (synthetic validation)

```bash
pip install torch
python test_gihkcc.py
```

### With a HuggingFace model

```python
from gihkcc import GIHKCCConfig, compress_kv_cache, decompress_kv_cache
from gihkcc_hf import GIHKCCWrappedModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

config = GIHKCCConfig(
    l1_snr_threshold=0.92,   # Keyframe when similarity drops below this
    l1_max_keyframe_span=8,  # Force keyframe every N layers
    l2_enabled=True,         # Enable inter-keyframe fold
    l2_super_keyframe_interval=4,
    l3_enabled=False,        # Token fold — enable for edge deployments
    delta_quant_bits=8,      # 8-bit delta quantization
)

wrapped = GIHKCCWrappedModel(model, tokenizer, config)

# Analyze and compress KV cache for a prompt
result = wrapped.analyze_cache("The quick brown fox jumps over the lazy dog.")
print(result["compression_summary"])
print(result["memory"])
print(result["reconstruction"])

# Print SNR heatmap
wrapped.print_snr_heatmap(result["snr_profile_keys"], label="Keys")
```

### Direct API (bring your own KV tensors)

```python
import torch
from gihkcc import GIHKCCConfig, compress_kv_cache, decompress_kv_cache, estimate_memory_bytes

# keys/values: List[Tensor] per layer, shape [num_heads, seq_len, head_dim]
config = GIHKCCConfig(l1_snr_threshold=0.92, l2_enabled=True)
compressed = compress_kv_cache(keys, values, config)

print(compressed.summary())
print(estimate_memory_bytes(compressed))

# Round-trip
recon_keys, recon_values = decompress_kv_cache(compressed)
```

## Key Design Decisions (Monolithic Adaptation)

1. **Statistical SNR replaces Guardian SNR.** Cosine similarity between adjacent-layer KV states serves as the keyframe placement signal. This is noisier than Guardian-computed SNR but captures the same structural boundaries.

2. **No L0.** Inter-head-group factorization requires Cerberus's structural separation of Tongue/Brain/Guardian heads. Monolithic models interleave all functions across all heads — no shared residual basis to factor out.

3. **Uniform compression policy.** Cerberus applies different aggressiveness per head group (Guardian: lossless, Reasoning: moderate, Language: aggressive). Monolithic models get a single policy across all heads.

4. **Max keyframe span as safety net.** Without Guardian SNR's structural guarantees, the max span parameter prevents unbounded error accumulation in high-similarity regions.

## Architecture Notes

The compression is **lossless at L1/L2** (before quantization) — deltas are exact floating-point differences. Quantization introduces bounded error proportional to `delta_quant_bits`. At 8-bit quantization on typical float16 deltas, mean error is ~1e-3, well within the noise floor of most generation tasks.

For safety-critical deployments, disable L3 and use 16-bit or no quantization on keyframes. The pipeline supports graceful degradation: under memory pressure, retroactively enable L3 or widen L2 intervals rather than discarding content.

## License

Proprietary — VecP Labs LLC. Contact for licensing terms.
Free for non-commercial research. Enterprise licensing available.
