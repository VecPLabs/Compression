# From Residual Compression to Direct Packed Attention

## A measured study of compressed autoregressive state for transformer inference

**VecP Labs LLC**  
**Technical report — August 2026**  
**Status:** Experimental research implementation; not yet a production serving system

## Abstract

Autoregressive transformer inference stores intermediate state so each newly
generated token can attend to prior tokens without recomputing the full prefix.
This state is normally represented as a floating-point key/value (K/V) cache.
We investigate two alternative representations: aggressively compressed
residual streams from which historical K/V can be regenerated, and projected
K/V tensors stored in a pagewise asymmetric quantized form. The residual path
reaches 14.12x cache compression on Pythia-410M, but repeated reconstruction,
LayerNorm, and projection make it substantially slower than native K/V caching.
The projected path therefore quantizes each token's post-RoPE K/V once and
performs attention directly over packed cold pages.

The final projected-cache design uses 32-token pages, per-channel affine key
quantization across each page, per-token/per-head symmetric value quantization,
an optional FP16 hot tail, and a Triton kernel that combines packed extraction,
query-key products, page-local softmax statistics, and weighted value
accumulation. On Pythia-410M over five non-overlapping WikiText-2 windows of
1,024 scored tokens each, K8/V4 with a 32-token hot tail achieves 2.32x
cache-only compression, median 25.77 tokens/s, median +0.115% perplexity change,
median KL divergence 0.00207, and median 97.66% top-1 agreement. K6/V4 reaches
2.69x compression, median 25.52 tokens/s, median +0.813% perplexity change,
median KL 0.00642, and median 95.70% top-1 agreement. These results are for
batch-one decoding on a desktop RTX 4070 Ti Super and should not be interpreted
as production serving throughput.

## 1. Motivation

The K/V cache trades memory for computation. Each layer stores projected keys
and values for every prior token, avoiding repeated projection during decoding.
At long context lengths and larger batch sizes, cache residency and bandwidth
can become limiting factors.

Our initial hypothesis was that the cache could be represented more compactly
upstream: transformer residual states exhibit strong cross-layer structure, so
an anchor plus decoder-visible adjacent deltas could act as a compressed source
from which K/V is regenerated. This hypothesis was correct about compression
but incomplete about serving cost. A representation is useful only when the
consumer can operate on it efficiently. Reconstructing historical residuals and
repeating LayerNorm and QKV projection at every generated token returned work
that conventional K/V caching was designed to eliminate.

This led to the central engineering transition in this study:

1. Treat residual compression as an archival or source representation.
2. Project serving K/V exactly once when a token is appended.
3. Quantize keys and values according to their different attention geometry.
4. Consume the packed representation directly inside attention.

## 2. Relationship to prior work

The asymmetric key/value geometry is not claimed as novel. KIVI reports that
keys should be quantized per channel while values should be quantized per token,
and develops a tuning-free asymmetric K/V cache quantizer [1]. KVQuant also
uses per-channel key quantization, pre-RoPE keys, non-uniform formats, outlier
handling, and custom CUDA kernels [2]. Recent work has continued to examine
TurboQuant-inspired K/V error and optimized 4-bit serving paths [3, 4].

The contribution of this report is the measured design path and implementation
around those ideas:

- closed-loop adjacent residual compression with real packed payloads;
- algebraic shared-rotation encoding and decoding;
- empirical identification of residual reprojection as the serving bottleneck;
- a page/head-parallel packed-attention kernel for the selected K/V formats;
- a reproducible quality, compression, throughput, and negative-result record.

No claim is made that pagewise asymmetric K/V quantization itself originates in
this work. Direct performance comparisons with official KIVI and KVQuant
implementations remain future work.

## 3. Residual-stream compression

![Optimization timeline from packed residuals to direct packed attention](docs/figures/optimization-timeline.svg)

### 3.1 Closed-loop adjacent prediction

Let `h_l` denote the residual state entering transformer layer `l`. The first
layer is encoded as an anchor. Subsequent layers encode a delta against the
decoder-visible reconstruction of the preceding layer:

```
d_l = h_l - h_hat_(l-1)
h_hat_l = h_hat_(l-1) + Q(d_l)
```

Using the reconstructed rather than full-precision reference makes the stream
closed-loop: encoder and decoder follow the same chain and do not silently
depend on unavailable original tensors.

Each payload is normalized, randomly rotated, scalar-quantized, and bit-packed.
With the same orthogonal rotation at every layer, reconstruction can sum the
anchor and delta codewords before applying a single inverse rotation. The
streaming encoder similarly uses

```
(h_l - h_hat_(l-1)) R^T = h_l R^T - h_hat_rotated_(l-1)
```

to batch all forward rotations and run the recurrence in rotated space.

### 3.2 Residual result and limitation

The 2-bit adjacent residual representation reaches 14.12x persistent
compression on Pythia-410M. End-to-end quality remained usable, but serving
throughput exposed the architectural cost:

| Residual implementation | Throughput | PPL change | Temporary peak |
|---|---:|---:|---:|
| Initial packed PyTorch | 3.44 tok/s | +5.49% | 39.96 MB |
| Triton decode + one rotation | 6.56 tok/s | +5.17% | 7.19 MB |
| Fused chain kernel | 10.28 tok/s | +6.12% | 11.56 MB |
| Shared-rotation streaming encode | 11.57 tok/s | +4.00% | 9.61 MB |
| FP16 fused SDPA operating point | 13.05 tok/s | +4.77% | 8.65 MB |

Compression was not the problem. Historical residuals still had to be decoded,
inverse-rotated, normalized, and projected into K/V for every layer and every
new token. Further dense projection folding added 101.15 MB of transformed
weights and reduced throughput to 9.75 tokens/s. This negative result motivated
the projected-cache path.

## 4. Projected pagewise K/V cache

![Direct packed K/V cache architecture](docs/figures/packed-kv-architecture.svg)

### 4.1 Cache organization

For each generated token and layer, the attention module produces a post-RoPE
key and a value. The controller stores these tensors once in chronological
order. The cache contains:

- completed quantized cold pages;
- an exact pending page that has not reached 32 tokens;
- an optional exact FP16 hot tail;
- the current token, included transiently during attention.

Completed pages are contiguous in memory. This avoids a Python launch per page
and permits a two-dimensional `(page, head)` Triton grid.

### 4.2 Key quantization

For a page of `P` tokens, key values have shape `[P, H, D]`. Each `(head,
channel)` coordinate is quantized across the page using an affine range:

```
scale[h,d] = (max_t K[t,h,d] - min_t K[t,h,d]) / (2^b_k - 1)
code[t,h,d] = round((K[t,h,d] - min[h,d]) / scale[h,d])
```

The minimum and scale are stored in FP16. We evaluate 8-bit, 6-bit, and 4-bit
key codes. Four-bit keys caused a clear quality regression and are not part of
the recommended frontier.

### 4.3 Value quantization

Values use a symmetric per-token/per-head scale:

```
scale[t,h] = max_d |V[t,h,d]| / (2^(b_v-1) - 1)
code[t,h,d] = round(V[t,h,d] / scale[t,h])
```

All reported projected-cache configurations use 4-bit values. Codes are truly
bit-packed; reported resident bytes count packed codes, FP16 metadata, and any
FP16 hot or pending tensors.

### 4.4 Direct packed-page attention

The final kernel avoids materializing decoded cold K/V. One Triton program is
launched for every `(page, head)` pair. It:

1. extracts packed key codes, including cross-byte fields;
2. applies page/channel key metadata;
3. computes query-key scores;
4. calculates page-local maximum and exponential denominator;
5. extracts and scales packed values;
6. accumulates the page's weighted value vector.

The kernel returns only a maximum, denominator, and value accumulator per page
and head. These small statistics are merged with exact pending, hot, and
current tokens using the online-softmax identity. Page/head parallelization
improved K8/V4 throughput from 19.79 to 24.61 tokens/s on the 256-token trace
while retaining the direct kernel's memory reduction.

## 5. Experimental setup

### 5.1 Hardware and software

- GPU: NVIDIA GeForce RTX 4070 Ti Super, 16 GB
- Model: EleutherAI Pythia-410M
- Model revision: `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`
- PyTorch: 2.11.0+cu128
- Triton: `triton-windows==3.7.1.post27`
- Dataset: WikiText-2 raw validation split
- Decode batch size: 1
- Page size: 32 tokens
- Prefix length: 32 tokens

Native and compressed timings are paired within each process. Timings are
CUDA-synchronized, but the machine is a desktop environment rather than an
isolated benchmark server. Cache ratios are cache-only ratios and do not imply
the same reduction in total model-plus-runtime memory.

### 5.2 Metrics

- **Perplexity:** teacher-forced next-token perplexity over scored tokens.
- **Relative PPL change:** `compressed_ppl / baseline_ppl - 1`.
- **KL divergence:** mean divergence between compressed and baseline next-token
  distributions.
- **Top-1 agreement:** fraction of positions with the same argmax token.
- **Resident ratio:** native FP16 K/V bytes divided by compressed cache bytes.
- **Throughput:** scored decode tokens divided by synchronized wall time.

### 5.3 Protocols

Two long-context protocols are reported:

1. Three independent process repeats on one 1,024-token scoring window, used
   primarily to characterize timing variance.
2. Five non-overlapping windows at token offsets 0, 1,056, 2,112, 3,168, and
   4,224. Each window uses a 32-token prefix and 1,024 scored tokens. This is
   the primary quality result.

## 6. Results

### 6.1 Three-repeat timing study

| Configuration | Ratio | Compressed median (range) | Native median | PPL change | Top-1 |
|---|---:|---:|---:|---:|---:|
| K8/V4 hot-32 | 2.32x | 27.29 (25.66--27.33) tok/s | 53.48 | +0.043% | 97.07% |
| K6/V4 hot-32 | 2.69x | 26.12 (23.72--26.35) tok/s | 51.60 | +1.29% | 95.51% |
| K8/V4 hot-0 | 2.42x | 25.79 (25.61--26.50) tok/s | 50.19 | +0.50% | 96.97% |

All three modes sustain approximately half of paired native throughput. The
fully compressed K8/V4 result shows that an FP16 tail is not required for
near-baseline quality, although the tail improves throughput by reducing cold
packed-attention work.

### 6.2 Five-window quality study

![Compression, perplexity, and throughput Pareto frontier](docs/figures/pareto-frontier.svg)

| Configuration | Ratio | PPL median (range) | KL median (range) | Top-1 median (range) | tok/s median (range) |
|---|---:|---:|---:|---:|---:|
| K8/V4 hot-32 | 2.32x | **+0.115%** (-0.275%--+0.232%) | **0.00207** (0.00172--0.00216) | **97.66%** (97.07%--97.85%) | **25.77** (23.17--27.29) |
| K6/V4 hot-32 | **2.69x** | +0.813% (+0.246%--+1.293%) | 0.00642 (0.00597--0.00706) | 95.70% (95.31%--95.90%) | 25.52 (23.30--26.12) |

K8/V4 remains within approximately ±0.3% relative perplexity on every window.
K6/V4 remains below +1.3% on every window and provides the stronger compression
point. Quality varies much less than desktop wall-clock throughput.

### 6.3 Aggressive key precision

On the 256-token trace, K4/V4 hot-32 reaches 2.72x compression and 16.75
tokens/s but increases perplexity by 9.05% and reduces top-1 agreement to
85.94%. At 64 scored tokens, after ensuring a completed quantized page is
actually consumed, K4/V4 increased perplexity by 5.55%. This is a clear
rejection of 4-bit keys under the current affine page format.

## 7. Negative results and lessons

The project intentionally retains negative results because they constrain the
design space:

- **Residual reconstruction is compact but compute-heavy.** A 14.12x cache
  ratio did not translate into competitive serving speed.
- **Static layer bit maps did not generalize.** Candidates selected on shorter
  traces regressed at longer validation lengths.
- **Folded dense projection lost.** Algebraically combining inverse rotation,
  LayerNorm affine parameters, and K/V projection added persistent weights and
  ran slower than separate cuBLAS operations.
- **Global and per-head rotated K/V MSE quantization lost.** Even high-bit
  controls degraded attention more than expected, demonstrating that generic
  reconstruction error is not a sufficient objective for keys.
- **A head-wide direct kernel under-occupied the GPU.** Splitting work by page
  and head was necessary for a speed improvement.
- **Short page tests can be misleading.** A 32-step evaluation with a hot-32
  tail completed its first cold page only after the final scored token. The
  protocol was corrected to ensure quantized pages were consumed.
- **Static residual repacking did not survive closed loop.** PCA and a
  projection-aware orthogonal basis reduced open-loop residual diagnostics,
  but at 10.56x cache compression their 64-token Pythia-70M perplexity changes
  were +92.16% and +89.81%, versus +523.80% for the identity allocation under
  the corrected norm-scaled Lloyd–Max codec. A
  basis fitted to raw adjacent deltas does not remain aligned with the
  decoder-visible error distribution after quantized predictions feed forward.
  Three rounds of reader-aware refitting on the reconstructed trajectory were
  also rejected at +145.67% perplexity change. Sharing one reader-aware basis
  across all six layers improved the two-bit result to +44.47% at the same
  10.56x ratio, supporting block stationarity, but still failed the quality bar.
  A three-band allocation selected 10% protected directions by calibration
  logit KL. On a separate window it reduced PPL damage from +55.63% to +41.87%
  and KL from 0.700297 to 0.332601 at 10.47x rather than 10.56x, while top-1
  agreement declined from 62.50% to 59.38%. Protecting a compact core helps,
  but the remaining absolute error rejects the current codec.
- **Message-axis coding supports the communication view but not the storage
  frontier.** Capturing attention and MLP writes separately and weighting a
  depth-axis transform by the residual prefix sums reduced two-bit PPL damage
  to +52.29%, versus +225.36% for identity message coordinates, but stored two
  writes per layer and reached only 4.00x compression. Coding their exported
  sum was worse: PCA and prefix-aware bases changed PPL by +320.11% and
  +788.97% at 5.33x. The internal writes expose useful cancellation structure;
  their sum is semantically correct but poorly matched to this scalar codec.
- **Mechanistic phase boundaries are not rate-distortion boundaries.** On
  24-layer Pythia-410M, adjacent update CKA/norm changes selected
  `0–5–9–20–24`, broadly resembling short identify/plan phases, a long
  production phase, and a short terminal phase. At the same four-anchor count,
  this layout produced +660.65% PPL at 10.61x versus +138.82% for uniform
  `0–6–12–18–24`. Adding a six-layer chain cap still produced +538.36% at
  9.79x. A large computational regime change is not automatically a safe
  point to quantize an anchor or change codebooks.
- **Whole-stream repacking improves accounting, not quality.** A single
  reader-aware block over all 24 Pythia-410M layers reached 14.11x compression
  but +199.48% PPL. A single prefix-aware transform over all 48 attention/MLP
  writes reached only 6.40x and +153.44% PPL. Larger objects amortize fixed
  overhead as expected, but the current repacking distortion remains
  incompatible with autoregressive cache reconstruction.
- **Activation-weighted `down_proj` reveals a compact model-derived message
  geometry.** On Qwen2.5-0.5B, weighting `down_proj` columns by observed gated
  coefficient variance consistently improved raw weight SVD on held-out
  messages. At rank 64 of residual width 896, intervening on MLP writes at four
  sampled layers gave -0.57% PPL change, 0.263576 KL, and 92.06% top-1 on a
  short held-out passage, versus +2.18%, 0.304945, and 84.13% for message PCA.
  The negative PPL delta is treated as variation. This is evidence for learned
  MLP-message packing geometry, but not yet persistent-state compression.

## 8. Limitations

This report does not yet establish broad model or serving generality.

1. The primary results use one model, Pythia-410M.
2. Evaluation is limited to WikiText-2 teacher-forced decoding; downstream task
   accuracy has not yet been measured for the projected-cache path.
3. Batch size is one. Compression may create more value by enabling larger
   batches, but this has not been demonstrated.
4. The desktop GPU environment introduces timing variance.
5. Results are cache-only; total application memory includes model weights,
   allocator state, kernels, and temporary buffers.
6. The prototype targets GPT-NeoX multi-head attention. GQA and other model
   adapters require separate validation.
7. No direct benchmark against official KIVI, KVQuant, FP8 cache, or production
   inference engines is included.
8. The current kernel returns page statistics to PyTorch for the final merge;
   further fusion may reduce launches and temporary allocations.

## 9. Reproducibility

The implementation and raw reports are contained in this repository.

Primary entry points:

- `validate_quantized_kv_generation.py`: paired native/compressed validation.
- `quantized_kv_generation.py`: page construction, hot/cold management, and
  patched GPT-NeoX attention.
- `triton_kernels.py`: packed extraction, residual decoding, and direct
  packed-page attention kernels.
- `aggregate_quantized_kv_results.py`: median/range aggregation.

Primary aggregate artifacts:

- `results/pythia410m_projectedkv_directpacked_n1024_repeats3_aggregate.json`
- `results/pythia410m_projectedkv_directpacked_n1024_windows5_aggregate.json`

Representative K8/V4 command:

```powershell
$env:TRITON_CACHE_DIR = Join-Path (Get-Location) '.triton-cache'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe validate_quantized_kv_generation.py `
  --model EleutherAI/pythia-410m `
  --revision 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f `
  --prefix 32 --steps 1024 --token-offset 0 `
  --scheme kivi --key-bits 8 --value-bits 4 `
  --page-size 32 --hot-window 32 --direct-packed-attention `
  --token-cache checkpoints/wikitext-validation-pythia-tokens.pt
```

At the time of this report, the repository test suite contains 33 passing
tests, including packed-index ranges, checkpoint round trips, shared-rotation
encoding, fused projection algebra, SDPA parity, page-codec order and accuracy,
and direct packed-attention parity against explicit decode plus attention.

## 10. Conclusion

The central conclusion is that compressed transformer state must be designed
with its consumer. Residual streams can provide exceptional representation
compression, but regenerating projected K/V forfeits much of the computational
advantage of caching. Projecting once, quantizing keys and values according to
their distinct geometry, and moving packed decode into attention yields a more
balanced serving result.

On Pythia-410M, K8/V4 provides a near-baseline quality point at 2.32x cache
compression and roughly half native throughput across five 1,024-token windows.
K6/V4 increases compression to 2.69x while keeping median perplexity change
below one percent. These results justify broader evaluation, not a claim of
universal superiority. The next required steps are downstream benchmarks,
larger MHA and GQA models, batch/concurrency measurements, and direct comparison
with established K/V quantization systems.

## References

1. Z. Liu et al., “KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
   Cache,” 2024. <https://arxiv.org/abs/2402.02750>
2. C. Hooper et al., “KVQuant: Towards 10 Million Context Length LLM Inference
   with KV Cache Quantization,” 2024. <https://arxiv.org/abs/2401.18079>
3. P. D'Alberto, “Statistical Inference and Quality Measures of KV Cache
   Quantisations Inspired by TurboQuant,” 2026.
   <https://arxiv.org/abs/2605.08114>
4. I. Chakrabarti et al., “UltraQuant: 4-bit KV Caching for Context-Heavy
   Agents,” 2026. <https://arxiv.org/abs/2606.20474>
