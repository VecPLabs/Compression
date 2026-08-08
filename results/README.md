# GIHKCC research results

This directory contains machine-readable benchmark artifacts. Treat JSON files
as the source of truth; each records the model revision, dataset fingerprint,
sample seed and offset, codec configuration, device, and software versions.

## Fixed configuration

- Model: `EleutherAI/pythia-410m`
- Revision: `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`
- Cache codec: GIHKCC v2 adjacent closed-loop residual prediction
- Quantizer: paper-reference TurboQuant, 2-bit deltas and one 8-bit layer-0 anchor
- Persistent compression ratio: 14.12x versus an FP16 K/V cache
- Runtime: PyTorch 2.11.0+cu128 on an NVIDIA RTX 4070 Ti SUPER

## Language modeling

Five non-overlapping 1,024-token WikiText-2 validation windows were evaluated
with real token-by-token compressed-cache inference.

| Token offset | FP16 PPL | Compressed PPL | Change | KL | Top-1 agreement |
|---:|---:|---:|---:|---:|---:|
| 0 | 11.1974 | 11.9475 | +6.70% | 0.06705 | 87.60% |
| 1,024 | 15.1080 | 16.3339 | +8.11% | 0.08445 | 85.64% |
| 2,048 | 11.5366 | 12.5572 | +8.85% | 0.07839 | 87.50% |
| 3,072 | 15.6336 | 16.8038 | +7.49% | 0.07963 | 84.47% |
| 4,096 | 12.9437 | 13.9918 | +8.10% | 0.07982 | 84.08% |
| Equal-token aggregate | 13.1615 | 14.1941 | **+7.85%** | 0.07787 | 85.86% |

The standard deviation of the five relative perplexity changes is 0.80
percentage points. The aggregate perplexities are computed by averaging token
negative log likelihoods (equivalently log perplexities), not arithmetic PPL.

## Downstream paired evaluations

All task samples use deterministic shuffle seed 1234. Accuracy intervals are
95% Wilson intervals. `Agreement` measures whether native and compressed cache
evaluation made the same correctness decision for LAMBADA or selected the same
answer choice for multiple-choice tasks.

| Task | n | FP16 cache | Compressed cache | Agreement |
|---|---:|---:|---:|---:|
| LAMBADA exact match | 1,000 | 493 (49.3%) | 492 (49.2%) | 94.5% |
| HellaSwag normalized | 100 | 41 (41.0%) | 41 (41.0%) | 90.0% |
| ARC-Easy normalized | 100 | 45 (45.0%) | 44 (44.0%) | 93.0% |
| ARC-Challenge normalized | 100 | 30 (30.0%) | 29 (29.0%) | 94.0% |

The net difference across these 1,300 paired examples is -3 correct decisions.
This is evidence of small degradation, not proof of equivalence. Larger task
samples and paired significance tests are required before making a formal
non-inferiority claim.

## Limitations

- The downstream harness is a transparent custom zero-shot implementation. Its
  prompt formatting and normalized continuation scoring must be cross-checked
  against a pinned `lm-evaluation-harness` release before comparing scores with
  external leaderboards.
- HellaSwag is stored as two non-overlapping 50-example shards because full
  autoregressive choice scoring is expensive. Aggregate counts sum both shards.
- The WikiText windows are non-overlapping but contiguous, not independent
  random draws from a population.
- 14.12x is persistent payload compression. The reference path reconstructs K/V
  into a standard cache and therefore does not yet establish peak-VRAM savings,
  throughput, or production latency.
- Publication-scale downstream results currently cover only Pythia-410M;
  Qwen has a shorter architecture-boundary quality sweep.

## Cross-architecture boundary: Qwen GQA

The lossless parity adapter was also validated on pinned
`Qwen/Qwen2.5-0.5B` revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. Qwen uses 14 query heads but
only 2 K/V heads, so GQA makes the native K/V cache narrower than the hidden
residual stream and removes GIHKCC's structural multiplier.

| Delta bits | Compression | PPL change | KL | Top-1 agreement |
|---:|---:|---:|---:|---:|
| 2 | 2.01x | +11.63% | 0.06243 | 86.72% |
| 3 | 1.42x | +2.49% | 0.01464 | 94.92% |
| 4 | 1.09x | -0.35% | 0.00378 | 97.66% |

These 256-token results show numerical portability but poor practical value.
Residual-native compression should not be presented as architecture-agnostic;
models with aggressive GQA likely require a KV-native or lower-dimensional
latent predictor.

## Reference runtime and VRAM

`benchmark_runtime.py` compares native generation with the current transparent
reference codec over 256 generated tokens. The reference implementation is a
quality validator: it reconstructs a conventional FP16 K/V cache before
attention and is not a production compressed-attention kernel.

| Model/configuration | Native tok/s | Reference tok/s | Relative throughput |
|---|---:|---:|---:|
| Pythia-410M, 2-bit | 56.69 | 16.18 | 0.285x |
| Qwen2.5-0.5B, 3-bit | 30.64 | 11.92 | 0.389x |

For Pythia, the theoretical persistent payload at 288 total tokens was about
2.01 MB versus 28.31 MB of FP16 K/V, but the reference GPU path still
materialized approximately the full K/V allocation. This establishes no
production VRAM or speed benefit. A fused attention path that consumes decoded
blocks without retaining the full FP16 cache is required for a systems claim.

## Packed blockwise-attention prototype

`blockwise_attention.py` is the first step toward that fused path. It stores
quantizer indices densely at their true 2/3/4-bit width, decodes one token block
and one residual chain at a time, projects K/V just in time, and uses an online
softmax accumulator. It never constructs the full historical K/V tensor during
the measured attention call.

On Pythia-410M with 1,024 tokens, the actual packed residual tensors occupied
7,127,040 bytes versus 7,127,808 accounted bytes and 100,663,296 bytes for an
FP16 K/V cache: an actual tensor-resident ratio of 14.12x. Deepest-layer results:

| Block | Temporary peak | Peak reduction | Blockwise time | Materialized time | Max output error |
|---:|---:|---:|---:|---:|---:|
| 32 | 11.82 MB | 12.78x | 444.4 ms | 30.3 ms | 8.54e-3 |
| 64 | 13.40 MB | 11.27x | 231.3 ms | 29.9 ms | 8.54e-3 |
| 128 | 22.85 MB | 6.61x | 128.1 ms | 36.1 ms | 1.46e-3 |
| 256 | 43.85 MB | 3.44x | 56.9 ms | 31.3 ms | 6.10e-5 |

The materialized reference peaked at 151.01 MB of temporary allocation. Block
256 is the current prototype Pareto point: 3.44x lower temporary peak and 1.82x
slower attention with near-FP16 numerical agreement. Block 64 minimizes memory
more aggressively but is 7.75x slower. These are single-layer attention-kernel
measurements, not end-to-end generation results; Python launch overhead and
repeated residual-chain decoding should be fused before serving claims.

## End-to-end packed autoregressive generation

`blockwise_generation.py` patches every GPT-NeoX attention layer during
one-token decoding. It supplies absolute RoPE positions, attends directly over
the packed residual history with online blockwise softmax, includes the current
token's K/V without caching it, and appends the completed token as a densely
packed adjacent-delta payload. No historical K/V tensor is created.

On the pinned Pythia-410M WikiText-2 trace with a 32-token prefix and 256 scored
tokens, 2-bit/block-256 generation measured:

- 12.2199 FP16-cache baseline perplexity versus 12.8906 packed (+5.49%).
- 0.07304 mean logit KL and 87.89% top-1 agreement.
- 2,004,480 bytes of packed history versus 28,311,552 bytes FP16 K/V: 14.12x.
- 39,963,648 bytes peak temporary allocation above the model/packed baseline.
- 3.44 tokens/s versus 48.90 native: 0.070x throughput.

The quality is close to the materialized compressed-cache validator on the same
regime, demonstrating correct integration. The throughput is deliberately not
competitive: this Python prototype repeatedly launches unpack, inverse-rotate,
projection, and online-softmax operations for every block and layer, and grows
packed tensors with concatenation. Kernel fusion and chunked append allocation
are the next engineering requirements.
