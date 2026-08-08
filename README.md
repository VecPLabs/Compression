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
gihkcc.py                    — Core L1/L2/L3 folding and quantization
gihkcc_v2.py                 — Closed-loop predictive K/V coding
gihkcc_hf.py                 — Hugging Face integration
kvtc.py                       — DCT transform coding for folded deltas
pca_layer.py                  — Per-layer PCA compression
ternary.py                    — Ternary/XNOR residual compression
turboquant.py                 — PolarQuant/QJL experiments
turboquant_paper.py           — Paper-reference TurboQuant MSE and QJL paths
test_gihkcc.py               — Synthetic GIHKCC validation
test_kvtc.py                 — Synthetic KVTC validation
test_tonight.py              — Broader synthetic experiment harness
test_*monolithic*.py         — Hugging Face model experiments
test_cerberus.py             — Cerberus checkpoint experiment
```

The files named `test_*.py` currently include both automated tests and research
experiment harnesses. CI deliberately runs only the deterministic synthetic
suites (`test_gihkcc.py` and `test_kvtc.py`); model and checkpoint experiments
are opt-in because they require external artifacts.

## Quick Start

### Standalone (synthetic validation)

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[dev]"
python -m pytest
```

The individual harnesses remain runnable with `python test_gihkcc.py` and
`python test_kvtc.py` when verbose diagnostic output is useful.

### Comparable benchmarks

Run the shared synthetic benchmark to compare standalone codecs with valid
GIHKCC delta-codec combinations using end-to-end error and encoded-size metrics:

```bash
python benchmark_compression.py
python benchmark_compression.py --layers 32 --heads 16 --tokens 512 --json results/benchmark.json
# After installation, the equivalent entry point is:
gihkcc-benchmark
```

For a real GPT-NeoX residual-to-K/V projection experiment:

```bash
python -m pip install -e ".[hf]"
python benchmark_real_residual.py --model EleutherAI/pythia-70m --tokens 256
python validate_autoregressive.py --wikitext --prefix 32 --steps 1024 --bits 4
```

GIHKCC is treated as a reversible structural transform in this benchmark. Its
deltas must be passed through a codec (currently KVTC or TurboQuant) before the
stack claims a storage reduction.

The harness also includes N-level XNOR sweeps and PCA combinations. Treat
results as tensor reconstruction benchmarks, not generation-quality evidence;
the latter requires running the model experiments on real caches.

`turboquant.py` is the earlier rotated-uniform prototype. The separate
`turboquant_paper.py` reference follows the paper's fixed Lloyd–Max codebook,
per-vector normalization, and optional one-bit QJL residual construction so
the two implementations remain directly comparable.

GIHKCC v2 forms deltas against the decoder-visible reconstructed anchor. This
closed loop prevents anchor quantization error from propagating into every
dependent layer and supports separate key/value precision budgets.

### Historical Cerberus results

The earlier 9× lossless and 16.8× at roughly 48.8 dB figures in the experiment
scripts use a different denominator: they store one shared Cerberus residual
stream and compare its size with the full projected K/V cache. K/V is regenerated
through model projection matrices. Some comparison rows were preserved as
hard-coded references, so reproduce them with the original checkpoint before
using them as current benchmark results.

On Pythia-70M at 256 tokens, using an FP16 storage baseline and one 8-bit
residual anchor, closed-loop residual deltas measured 14.55×/29.38 dB at 1 bit,
10.55×/35.34 dB at 2 bits, and 8.28×/40.96 dB at 3 bits when K/V was regenerated
through the model's real layer norms and fused Q/K/V projections. These are
projected-tensor results; generation-quality validation is still required.

Live-cache validation rebuilds a Transformers `DynamicCache` from decoded
residual history, including GPT-NeoX partial RoPE, and then performs genuine
token-by-token cached inference. On 1,024 WikiText-2 validation tokens with
Pythia-70M, 4-bit residual deltas measured 6.81× persistent-cache compression,
33.3304 baseline perplexity versus 33.7885 compressed perplexity (+1.37%),
0.017992 mean logit KL, and 90.92% top-1 agreement. The requested 2-bit point
did not preserve quality: on 256 tokens it measured 10.55× but increased
perplexity by 38.43%. Rebuilt K/V is transient scratch and the reference
validator recomputes it each step; production latency and peak-memory kernels
remain future work.

Adjacent closed-loop prediction is available with `--prediction adjacent`.
Rather than coding every layer against a shared anchor, it codes layer `l`
against decoder-visible layer `l - 1`; the next delta therefore corrects prior
reconstruction error instead of accumulating it. On the 256-token validation,
the 2-bit result improved from +38.43% to +11.88% perplexity at 10.55x. The
4-bit result measured -0.30% at 6.81x. On the longer 1,024-token validation,
adjacent 4-bit measured 33.5449 perplexity versus 33.3304 baseline (+0.64%),
0.011719 mean logit KL, and 94.04% top-1 agreement. This supersedes the shared
anchor as the current validated Pareto predictor.

Bidirectional endpoint prediction is available with `--prediction middle_out`.
It stores 8-bit anchors at the first and last layers, then decodes closed-loop
deltas inward from both ends. On six-layer Pythia-70M at 256 tokens, its 2-,
3-, and 4-bit points measured respectively 7.93x/+14.75%, 6.81x/+3.29%, and
5.96x/+0.73% compression/perplexity change. Compared with forward-adjacent
prediction, middle-out improved KL and top-1 agreement at low precision (the
2-bit KL fell from 0.164979 to 0.129186 and top-1 rose from 73.05% to 78.12%),
but the second anchor's overhead prevented a better compression/perplexity
Pareto point on this shallow six-layer model. The trade should improve as
model depth amortizes the extra anchor.

On the deeper 24-layer Pythia-410M model, a 256-token WikiText-2 validation
gave the forward-adjacent 2-bit codec 14.12x compression with 12.7408
perplexity versus 12.1769 baseline (+4.63%), 0.071094 mean logit KL, and
86.33% top-1 agreement. Middle-out measured 12.72x/+4.99%, 0.070216 KL, and
84.38% top-1 agreement. The second anchor was well amortized, but it still did
not beat forward-adjacent prediction. The validator accepts `--device`; this
run used CPU because the local PyTorch installation lacks CUDA support.

The same Pythia-410M adjacent 2-bit configuration was extended to 1,024
WikiText-2 tokens using the validator's `--incremental` path. It retained
14.12x compression with 11.9055 perplexity versus 11.1973 baseline (+6.32%),
0.064923 mean logit KL, and 87.79% top-1 agreement. Incremental validation
compresses only each new token, replaces the uncompressed cache entry with K/V
projected from its decoded residual, and performs one full-history payload
accounting pass at the end. A reference comparison matched rounded perplexity,
ratio, and top-1 results.

`benchmark_downstream.py` performs live compressed-cache scoring on standard
tasks. A quick Pythia-410M sample gave equal aggregate LAMBADA exact match on
50 examples (50.0% native and 50.0% adjacent 2-bit). On 25 HellaSwag validation
examples, normalized multiple-choice accuracy was 16% native and 20% compressed.
These small samples show no detected task regression, but the HellaSwag result
is near chance and has high sampling uncertainty; neither result should be
treated as a full benchmark score. Every continuation token after prefill is
scored with the newly appended K/V replaced by the decoded compressed entry.

A one-layer-at-a-time precision sweep is available through
`sweep_layer_bits.py`. On the 128-token diagnostic, late layers appeared safe
to downgrade, but the candidates did not generalize to 256 tokens: the best
7.06× mixed allocation still increased perplexity by 1.78%, and a single L4
downgrade increased it by 2.19%. Uniform 4-bit remains the validated Pareto
choice for Pythia-70M. This negative result suggests token-age or error-budgeted
adaptation is more promising than a static per-layer bit map.

The live validator also supports `--capture-point preprojection`, which hooks
the normalized hidden state immediately entering the fused QKV projection.
Lossless cache parity remained exact, but 4-bit independent pre-projection
encoding measured 7.93× with a 19.01% perplexity increase over 256 WikiText-2
tokens. Cross-layer folding was substantially worse. Compressing the residual
before LayerNorm is preferable because normalization attenuates reconstruction
error before it reaches the projection.

An experimental `--ln-aware-candidates` mode also tries several orthogonal
quantizer rotations and selects the one with the lowest post-LayerNorm hidden
state MSE. Rotation choices are locked after the initial prefix to avoid
temporal jitter. This proxy did not improve generation quality: on the same
64-token diagnostic, ordinary 2-bit compression increased perplexity by
49.09%, while eight-way LayerNorm-aware selection increased it by 64.59% at
the same 10.54x ratio. Local normalized-state MSE is therefore not a useful
selection objective; future optimization should score projected Q/K/V error
or teacher-logit KL directly.

### With a HuggingFace model

Install the optional integration dependencies first:

```bash
python -m pip install -e ".[hf]"
```

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
