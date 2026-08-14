# GIHKCC Monolithic — KV Cache Compression for Standard Transformers

**VecP Labs LLC** | [vecplabs.com](https://vecplabs.com) | Patent Pending (USPTO 63/931,565)

Adaptation of Guardian-Informed Hierarchical KV Cache Compression for monolithic (non-Cerberus) transformer architectures. Implements Levels 1–3 of the recursive multi-scale fold-compress pipeline using statistical SNR as a proxy for Guardian-based keyframe placement.

> **Latest result:** direct packed K/V attention on Pythia-1.4B reaches 2.32×
> measured cache compression with +0.115% median perplexity change across five
> held-out windows. Read the [technical report](TECHNICAL_REPORT.md) or the
> shorter [website writeup](WEBSITE_WRITEUP.md).

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
experiment harnesses. CI runs the deterministic suites selected in
`pyproject.toml`; model and checkpoint experiments are opt-in because they
require external artifacts.

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
historical run used CPU. The project environment now uses PyTorch 2.11.0 with
CUDA 12.8, and cache reconstruction explicitly transfers captured CPU
residuals back to the model device.

The same Pythia-410M adjacent 2-bit configuration was evaluated on five
non-overlapping 1,024-token WikiText-2 windows using the validator's
`--incremental` path. It retained 14.12x compression with an equal-token
aggregate perplexity of 14.1941 versus 13.1615 baseline (+7.85%). Per-window
changes ranged from +6.70% to +8.85%; mean KL was 0.07787 and mean top-1
agreement was 85.86%. Incremental validation compresses only each new token,
replaces the uncompressed cache entry with K/V projected from its decoded
residual, and performs one full-history payload accounting pass at the end.

`benchmark_downstream.py` performs live compressed-cache scoring on standard
tasks and supports pinned model revisions, deterministic random sampling,
Wilson confidence intervals, resumable offsets, and JSON output. Paired CUDA
evaluation measured LAMBADA 493/1000 native versus 492/1000 compressed,
HellaSwag 41/100 versus 41/100, ARC-Easy 45/100 versus 44/100, and
ARC-Challenge 30/100 versus 29/100. The aggregate difference is -3 correct
decisions over 1,300 examples. See `results/README.md` and its source JSON
artifacts for protocols, confidence intervals, agreement rates, and caveats.

Cross-architecture validation on Qwen2.5-0.5B passed exact lossless cache
parity but exposed a structural limitation: its 14-query/2-KV-head GQA cache
is already much narrower than the residual stream. At 256 tokens, adjacent
GIHKCC measured 2.01x/+11.63% at 2 bits, 1.42x/+2.49% at 3 bits, and
1.09x/-0.35% at 4 bits. The method is therefore attractive on wide MHA caches
such as Pythia, but not directly on aggressively grouped-query caches.

CUDA systems measurements also separate payload size from implementation
performance. The current reference path achieved 16.18 tokens/s versus 56.69
native on Pythia-410M and 11.92 versus 30.64 on Qwen. It reconstructs a normal
FP16 K/V cache, so it does not yet deliver production resident-VRAM savings.
See `results/README.md` for the pinned measurements and limitations.

A packed blockwise-attention prototype now demonstrates actual tensor-resident
compression on Pythia-410M. At 1,024 tokens, packed indices plus FP16 norms
occupied 7.13 MB versus 100.66 MB of FP16 K/V (14.12x). Online blockwise
softmax reduced deepest-layer temporary allocation from 151.01 MB to 43.85 MB
with 256-token blocks, while taking 56.9 ms versus 31.3 ms and matching the
materialized compressed-attention output within 6.1e-5. A 64-token block cut
temporary peak to 13.40 MB at substantially higher Python prototype latency.
This validates the memory mechanism at one attention layer; it is not yet an
end-to-end fused generation kernel.

End-to-end GPT-NeoX integration now runs all attention layers directly over the
packed history without creating historical K/V. On a 32-token prefix plus 256
WikiText-2 tokens, Pythia-410M measured 14.12x resident-history compression,
12.2199 baseline versus 12.8906 packed perplexity (+5.49%), 87.89% top-1
agreement, and 39.96 MB temporary peak above the model/packed baseline. The
unfused Python path achieved 3.44 tokens/s versus 48.90 native, making kernel
fusion—not compression quality or memory representation—the immediate blocker.

The first Triton optimization fuses packed extraction, centroid lookup, and
norm scaling. Adjacent-chain linearity then reduces up to 24 inverse rotations
to one per decoded target layer. On the same 256-token trace, the combined path
reached 6.56 tokens/s, reduced temporary peak from 39.96 MB to 7.19 MB, and
measured +5.17% PPL with 89.45% top-1 agreement. Deepest-layer block-256
attention is now 18.46 ms versus 16.24 ms materialized, with exact output parity
and 2.49x lower temporary allocation. End-to-end generation remains 7.47x
slower than native, so further projection/softmax fusion is still required.

The second-stage kernel stores adjacent delta payloads as contiguous
layer-major matrices and decodes the anchor plus every required delta in one
runtime-depth Triton launch before the shared inverse rotation. A 24-layer CUDA
parity test passes. On the same 32-prefix/256-token trace it reaches 10.28
tokens/s, a 1.57x improvement over the one-rotation chain and a 2.99x
improvement over the initial packed prototype. The remaining gap to the paired
53.29-token/s native run is 5.18x. Temporary peak rises from 7.19 MB to 11.56
MB because streaming concatenation reallocates the contiguous matrices;
capacity-managed append buffers are therefore the next memory optimization.

Exact-capacity append buffers remove per-token matrix reallocation. On the
256-token trace they reduced temporary allocation above the resident base from
11.56 MB to 9.61 MB and held physical resident storage equal to the 2.00 MB
logical payload. Total peak allocation was effectively unchanged because the
final packed history is required resident state, and throughput changed only
from 10.28 to 10.32 tokens/s. This rules out append copying as the primary
runtime bottleneck.

Streaming adjacent compression now uses the shared rotation on the encoder
side. All 24 layer states are forward-rotated in one batched matmul, then the
closed-loop recurrence runs directly in rotated space because `(state -
reconstructed) R^T = state R^T - reconstructed_rotated`. On 256 tokens this
raises throughput from 10.32 to 11.57 tokens/s. Against the paired 51.64-token/s
native run, the remaining gap is 4.46x. PPL change improves from +6.12% to
+4.00%, mean KL falls from 0.07266 to 0.06802, and top-1 agreement moves
slightly from 87.89% to 87.50%.

An optional `--fused-projection` experiment folds inverse rotation, LayerNorm
affine parameters, and historical K/V projection into transformed weights. The
identity is numerically validated, but the implementation is a negative result:
it adds 101.15 MB of weights and reaches only 9.75 tokens/s on the 256-token
trace versus 11.57 tokens/s for the unfused cuBLAS path. Its +4.84% PPL change
is also worse than +4.00%. Small-matrix efficiency and normalization-correction
overhead outweigh the reduced nominal FLOPs, so this path is not the default.

For histories fitting one block, optional `--fused-attention` dispatches the
projected historical K/V plus the current token to fused FP16 scaled-dot-product
attention. The 256-token trace reaches 13.05 tokens/s, 12.8% faster than the
11.57-token/s online-FP32 path and 3.79x faster than the initial packed
prototype. This is a speed-oriented Pareto point rather than a replacement:
PPL change is +4.77% and top-1 agreement is 86.72%, versus +4.00% and 87.50%
for online FP32. Once history exceeds the block size, dispatch falls back to
online blockwise softmax. FP32 SDPA was also tested and rejected at 8.90
tokens/s on the full trace.

A projected-K/V serving prototype now stores post-RoPE keys and values once,
with an exact FP16 hot tail and packed symmetric cold stream. A no-eviction
control matches native closely (+0.55% PPL, 100% top-1), validating cache order
and attention semantics. Direct int8 stream tests also verify token order and
reconstruction error. Uniform K/V quantization is not yet viable, however:
hot-32/int8 cold measured +12.59% PPL over 16 tokens, while hot-32/4-bit cold
was substantially worse despite reaching 19.93 tokens/s. This points to
attention geometry rather than plumbing: the next iteration should use
pagewise per-channel key quantization and per-token value quantization.

That KIVI-style geometry succeeds. With 32-token pages and an FP16 hot-32
tail, the full 256-token Pythia-410M trace establishes three projected-K/V
points. K8/V4 reaches 21.05 tokens/s at 2.09x compression with -0.45% PPL,
KL 0.00175, and 96.48% top-1. K6/V4 reaches 16.53 tokens/s at 2.36x with
+1.77% PPL, KL 0.00504, and 96.88% top-1. K4/V4 reaches 2.72x but increases
PPL by 9.05%, so it is not the recommended point. Contiguous page storage more
than doubled the K8/V4 reference decoder from 8.92 to 21.05 tokens/s. Its
remaining 30.4 MB temporary peak comes from materializing decoded cold pages;
a packed-page-to-attention kernel is the next direct optimization.

The direct Triton path now performs packed key dot products, page-local
softmax statistics, packed value accumulation, and pagewise online-softmax
merging without materializing cold K/V. Parallel `(page, head)` programs raise
K8/V4 hot-32 throughput to 24.61 tokens/s, reduce temporary peak from 30.4 MB
to 18.4 MB, and preserve -0.69% PPL with 96.48% top-1 at 2.09x compression.
K6/V4 hot-32 reaches 22.72 tokens/s, 2.36x compression, and +1.74% PPL. A
fully compressed hot-0 K8/V4 cache reaches 2.42x compression and -0.05% PPL,
but slows to 17.94 tokens/s because every token participates in cold packed
attention. The fastest direct path is 7.15x faster than the initial packed
residual prototype and remains 2.08x behind its paired native run.

Long-context validation scores 1,024 held-out tokens after a 32-token prefix
and repeats every configuration three times. K8/V4 hot-32 reaches median 27.29
tokens/s (25.66--27.33), 2.32x compression, +0.043% PPL, and 97.07% top-1.
K6/V4 hot-32 reaches median 26.12 tokens/s (23.72--26.35), 2.69x compression,
+1.29% PPL, and 95.51% top-1. Fully compressed K8/V4 hot-0 reaches median
25.79 tokens/s (25.61--26.50), 2.42x compression, +0.50% PPL, and 96.97%
top-1. All quality metrics are deterministic across repeats; median compressed
throughput is within 1.95--1.98x of each paired native median.

Quality was then evaluated across five non-overlapping 1,024-token windows at
offsets 0, 1,056, 2,112, 3,168, and 4,224. K8/V4 hot-32 has median +0.115%
PPL change (range -0.275% to +0.232%), median KL 0.00207, and median 97.66%
top-1 agreement. K6/V4 hot-32 has median +0.813% PPL change (range +0.246% to
+1.293%), median KL 0.00642, and median 95.70% top-1. Median compressed
throughput across the five independent windows is 25.77 and 25.52 tokens/s.
This supersedes the original single-window quality claim.

The generation validator supports reusable `--token-cache` and
`--prefix-cache` artifacts plus an immediate `--phase-log`. A local
`--dataset-arrow` path bypasses a Hugging Face offline-builder hang discovered
during this work. With warm caches, the complete 256-step validation finishes
in about 38 seconds instead of timing out after six minutes.

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

Static residual repacking is available as a reproducible negative experiment
through `benchmark_residual_repacking.py` and `validate_autoregressive.py
--repacking`. On held-out Pythia-70M activations, a projection-aware basis cut
open-loop projected-QKV delta MSE substantially, but the improvement did not
survive closed-loop generation. At a 10.56x persistent-cache ratio over 64
tokens, identity, reader-aware, and PCA repacking increased perplexity by
523.80%, 89.81%, and 92.16%, respectively, with the corrected Lloyd–Max
codec. The decoder-visible delta
distribution shifts after every quantized layer, so a basis fitted to raw
adjacent deltas is not a stable importance map. Three rounds of closed-loop
reader-aware refitting also failed at +145.67%, rejecting a frozen per-layer
basis even when calibrated on decoder-visible predecessors.

Depth-block repacking is more promising but not yet viable. Sharing one
reader-aware basis across all six Pythia-70M layers reduced the two-bit PPL
regression from +89.81% to +44.47% at the same 10.56x cache ratio. Two- and
three-layer blocks measured +52.84% at 6.36x and +66.48% at 7.94x because
their extra 8-bit boundary anchors consume more storage. At three bits the
six-layer block reached 8.28x/+23.05%; four bits was non-monotonic at
6.81x/+27.65%. The block scope clearly helps, but the current importance map
and scalar codebook still do not preserve generation quality.

A three-band block allocation protects the highest-ranked 10% of shared-basis
directions at 8 bits, quantizes an ordinary band at 3 bits, and drops the
remainder while holding the average near 2 bits. The 10% fraction was selected
by closed-loop KL on a 32-token calibration trace, then evaluated on a separate
64-token window. It reduced holdout PPL damage from +55.63% to +41.87% and KL
from 0.70030 to 0.33260 at essentially unchanged compression (10.56x versus
10.47x). Top-1 agreement fell from 62.50% to 59.38%, so this is a directional
improvement rather than a usable operating point.

Message-axis repacking tests the residual stream explicitly as a communication
channel. Hooks capture the attention and MLP writes produced inside each layer,
then a transform operates across message depth. At two average bits, generic
message coordinates increased PPL by +225.36%, PCA by +197.09%, and a
prefix-sum-aware basis by +52.29%. The latter is a large functional improvement
but reaches only 4.00x cache compression because it stores two writes per
layer. Combining attention and MLP into the single update actually passed to
the next layer reduced storage but destroyed quality: identity, PCA, and
prefix-aware combined coding changed PPL by +552.97%, +320.11%, and +788.97%
at 5.33x. Internal writes contain cancellation structure that the current
scalar codec loses when they are summed first.

On the deeper 24-layer Pythia-410M model, adjacent update CKA and norm shifts
detected candidate phase boundaries at layers `0,5,9,20,24`: short early and
transition regions, a long middle region, and a short terminal region. These
boundaries resemble an identify/plan/produce/polish staging, but they are poor
codec boundaries. At identical four-anchor overhead and 10.61x compression,
detected phases increased PPL by +660.65% versus +138.82% for uniform
six-layer blocks (`0,6,12,18,24`). Capping the detected layout to six-layer
chains still measured +538.36% at 9.79x. Mechanistic discontinuities identify
changes in computation, not necessarily stable rate-distortion regions.

The final whole-stream controls close this branch. One reader-aware basis over
all 24 Pythia-410M layers recovered 14.11x compression, confirming that larger
objects amortize anchors and metadata, but increased PPL by +199.48%. One
prefix-aware transform over all 48 attention/MLP writes measured 6.40x and
+153.44% PPL. Neither improves on four uniform six-layer blocks in quality,
and both remain far behind direct packed projected-K/V compression. Residual
repacking is therefore retained as a mechanistic negative result rather than a
serving candidate under the current scalar codec.

Qwen2.5 `down_proj` geometry provides a more positive result. On the cached
0.5B model, we compared arbitrary residual coordinates, raw `down_proj` SVD,
gate-activation-weighted `down_proj`, and prompt-fitted message PCA on unseen
tokens. Activation weighting consistently improved the weight-only basis. At
the final sampled layer it beat message PCA at ranks 8, 16, 32, and 128; for
rank 16, next-reader MSE was 0.35694 versus 0.62653. Post-`down_proj` message
effective rank was initially 8–20 across sampled layers, although a longer
256-token calibration raises the all-layer median to 28.32 (range 1.04–37.11),
so the exact rank estimate is calibration-sensitive.

A direct intervention projected MLP writes at layers 5, 11, 17, and 23 into
the learned subspaces. At rank 64 out of residual width 896, applying all four
activation-weighted bases changed PPL by -0.57% on the short held-out passage,
with 0.26358 KL and 92.06% top-1 agreement. Prompt-fitted PCA measured +2.18%,
0.30495 KL, and 84.13%. The negative PPL delta is sampling variation, not an
improvement claim. This establishes a model-derived compact MLP-message
subspace, not yet a persistent-cache compression result.

An all-24-layer rank-64 scan adds the important control. The correctly paired
activation-weighted basis beat a basis with activation statistics shuffled
between channels on KL in 22/24 layers, supporting the claim that the *used*
dictionary matters rather than weighting alone. It beat message PCA in 11/24
layers. In six layers (4, 5, 6, 9, 11, 12), PCA reconstructed the MLP message
more accurately while the active `down_proj` basis preserved behavior better,
direct evidence that variance-optimal reconstruction and computation-optimal
representation can diverge. Independently detected four-phase boundaries were
0/4/12/16/24; the advantage is depth-dependent, but this single short run does
not establish phase alignment.

The same rank-64 all-layer protocol replicates on Pythia-410M, whose
`dense_4h_to_h` is the architectural equivalent of `down_proj`. Active
weighting again beat shuffled weighting in 22/24 layers and message PCA in
11/24. Eight layers (3, 5, 6, 8, 9, 10, 13, 18) showed the stricter inversion:
PCA had lower message MSE while the learned active basis had lower KL. The
effective-rank median was 19.20 (range 1.50–37.99). Matching win counts across
Qwen and Pythia are notable cross-family replication, but the 64-token
holdouts remain too small for a generality claim.

A first layer-dynamic allocator converts that observation into an allocation
experiment. On Pythia-410M, ranks 64/128/256/512/768/1024 were profiled per
layer on a separate passage, then an exact multiple-choice knapsack minimized
the sum of isolated intervention KL under fixed total-rank budgets. Each chosen
allocation was finally applied to all 24 MLP writes simultaneously on a third
passage. Dynamic allocation beat uniform rank at every tested budget. At 3.1%
coordinate savings, dynamic allocation reached +0.60% PPL, 0.01206 KL, and
96.83% top-1 agreement, versus +9.51%, 0.09946, and 87.30% for uniform rank at
the same total rank. At 6.3% savings, dynamic measured +3.77% PPL and 0.02831
KL versus +8.72% and 0.12729 for uniform. This is transient MLP-message rank
restriction—not weight, KV-cache, or end-to-end memory compression—and uses a
short 64-token validation passage.

We then tested whether the layerwise geometry forms a depth phase field and
whether next-token gradients supply the missing sensitivity variable. On
Pythia-410M, three 64-token calibration domains produced strongly
prompt-conditioned active-write subspaces: mean same-layer cross-domain
similarity was 0.568, ranging from 0.244 to 0.818. However, independently
detected phase boundaries were not subspace discontinuities at rank 64: mean
adjacent similarity was 0.210 at boundaries versus 0.197 elsewhere. On a
separate 32-token intervention passage, message PCA won 14/24 layers, active
`dense_4h_to_h` geometry won 8/24, and a joint write-times-Fisher basis won
2/24. Fisher-only geometry won none and had mean KL 0.369, versus 0.179 for
active geometry and 0.173 for PCA. Thus prompt-conditioned usage is a real
dynamic variable, while this first-order Fisher construction and the proposed
phase-boundary alignment are negative results.

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
