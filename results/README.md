# GIHKCC research results

This directory contains machine-readable benchmark artifacts. Treat JSON files
as the source of truth; each records the model revision, dataset fingerprint,
sample seed and offset, codec configuration, device, and software versions.

## Static residual repacking (negative result)

We tested whether an orthogonal change of basis could concentrate adjacent
residual deltas before a fixed two-bit-average mixed-precision code. Bases were
calibrated separately from evaluation tokens and reused during incremental
closed-loop generation. The stored basis is treated as model-side codec state,
as TurboQuant treats its fixed rotation, and is not counted as cache payload.

On an open-loop 256-token diagnostic, learned bases reduced raw residual MSE,
but that metric did not predict live-cache behavior. The paired 64-token
Pythia-70M generation run produced:

| Basis | Cache ratio | PPL change | Mean KL | Top-1 agreement |
|---|---:|---:|---:|---:|
| Identity allocation | 10.56x | +523.80% | 1.714002 | 39.06% |
| Reader-aware | 10.56x | +89.81% | 0.532261 | 54.69% |
| Delta PCA | 10.56x | +92.16% | 0.600651 | 59.38% |
| Reader-aware, closed-loop fit | 10.56x | +145.67% | 0.795298 | 48.44% |

Artifacts: `pythia70m_repacking_identity_n64_2bit.json`,
`pythia70m_repacking_reader_n64_2bit.json`, and
`pythia70m_repacking_pca_n64_2bit.json`, plus the closed-loop control
`pythia70m_repacking_reader_closedloop_n64_2bit.json`. These are short diagnostic runs, not
publication-quality estimates. Their purpose is to reject static raw-delta
bases before spending compute on larger models.

### Depth-block controls

One basis was then shared across consecutive layer blocks, with an 8-bit
anchor at every boundary. All rows use the corrected norm-scaled Lloyd–Max
codec and include boundary-anchor overhead.

| Basis/block | Bits | Cache ratio | PPL change | Mean KL | Top-1 |
|---|---:|---:|---:|---:|---:|
| Reader, 2 layers | 2 | 6.36x | +52.84% | 0.412203 | 60.94% |
| Reader, 3 layers | 2 | 7.94x | +66.48% | 0.343334 | 59.38% |
| Reader, 6 layers | 2 | 10.56x | +44.47% | 0.444301 | 67.19% |
| Reader, 6 layers | 3 | 8.28x | +23.05% | 0.388733 | 57.81% |
| Reader, 6 layers | 4 | 6.81x | +27.65% | 0.275566 | 70.31% |
| PCA, 6 layers | 2 | 10.56x | +147.43% | 1.017875 | 37.50% |

The six-layer reader basis halves the relative PPL damage of its per-layer
counterpart at identical storage. This supports a block-stationarity effect,
but none of the tested points is competitive with the established direct-K/V
cache frontier.

### KL-selected protected band

At the same two-bit-average budget, candidate protected fractions of 0%, 5%,
10%, 15%, and 20% were swept on a 32-token calibration trace. Protected
directions use 8 bits, the ordinary band uses 3 bits, and the remaining
directions are dropped. The 10% candidate had the lowest calibration KL
(0.231860) and was frozen before evaluation on the non-overlapping token-offset
96 window.

| Holdout allocation | Cache ratio | PPL change | Mean KL | Top-1 |
|---|---:|---:|---:|---:|
| Two-band block baseline | 10.56x | +55.63% | 0.700297 | 62.50% |
| Three-band, 10% protected | 10.47x | +41.87% | 0.332601 | 59.38% |

The protected band generalizes in PPL and KL, but not top-1 agreement, and the
absolute regression remains too large. See the `threeband_cal_*` artifacts and
`pythia70m_repacking_threeband_f010_holdout_n64_2bit.json`.

### Attention/MLP message-axis coding

Pythia's parallel residual layers export `x + attention(x) + MLP(x)`. We
captured the two internal writes and transformed along the 12-message depth
axis before rebuilding the live K/V cache.

| Message representation/basis | Cache ratio | PPL change | Mean KL | Top-1 |
|---|---:|---:|---:|---:|
| Separate writes, identity | 4.00x | +225.36% | 1.026909 | 46.88% |
| Separate writes, PCA | 4.00x | +197.09% | 0.883755 | 42.19% |
| Separate writes, prefix-aware | 4.00x | +52.29% | 0.399160 | 65.62% |
| Combined layer update, identity | 5.33x | +552.97% | 1.875011 | 23.44% |
| Combined layer update, PCA | 5.33x | +320.11% | 1.783722 | 37.50% |
| Combined layer update, prefix-aware | 5.33x | +788.97% | 1.848445 | 29.69% |

Prefix-sum weighting validates the communication-channel framing for separate
writes, but the representation doubles the objects being stored. Combining
the writes before scalar coding loses cancellation structure and is rejected.
The source artifacts use the `pythia70m_message_repacking_*` prefix.

### Detected functional phases on Pythia-410M

Linear CKA between adjacent layer updates plus update-norm shifts selected
four phases at `0–5–9–20–24` under a four-layer minimum. The strongest changes
occurred near layers 5 and 20–22, with another clear break at 9. Equal-anchor
closed-loop controls produced:

| Boundaries | Cache ratio | PPL change | Mean KL | Top-1 |
|---|---:|---:|---:|---:|
| Uniform `0,6,12,18,24` | 10.61x | +138.82% | 0.794054 | 56.25% |
| Detected `0,5,9,20,24` | 10.61x | +660.65% | 1.455376 | 35.94% |
| Detected plus six-layer cap `0,5,9,15,20,24` | 9.79x | +538.36% | 1.406635 | 40.62% |

The phase changes are measurable but do not define useful codec resets. The
detector artifact is `pythia410m_detected_phases_n128.json`; generation
artifacts use `pythia410m_repacking_*phase*` and `*capped6*`.

### Whole-stream closing controls

| Whole-depth representation | Cache ratio | PPL change | Mean KL | Top-1 |
|---|---:|---:|---:|---:|
| One reader basis over 24 residual layers | 14.11x | +199.48% | 1.292619 | 51.56% |
| One prefix-aware basis over 48 internal writes | 6.40x | +153.44% | 0.751582 | 54.69% |

The 24-layer block validates the expected size effect in storage accounting,
but not in quality. The 48-write representation has lower KL but doubles the
source objects and therefore loses compression. Artifacts are
`pythia410m_repacking_whole24_reader_n64_2bit.json` and
`pythia410m_message_repacking_whole48_prefix_n64_2bit.json`.

### Qwen2.5 activation-weighted `down_proj` geometry

On Qwen2.5-0.5B, gated intermediate coefficient variance was used to weight
the columns of each MLP `down_proj`, producing a residual-output covariance
basis. Bases were calibrated separately from the 64-token holdout. Sampled
post-`down_proj` messages had effective ranks from 8.31 to 19.97. A broader
256-token, all-layer calibration produced a median of 28.32 and a range of
1.04–37.11, showing that the exact estimate is sample-sensitive.

At the final sampled layer, activation-weighted geometry beat message PCA at
every evaluated rank (8, 16, 32, 128) on next-reader MSE. A direct intervention
at four layers produced:

| Rank/basis, all sampled layers | PPL change | Mean KL | Top-1 |
|---|---:|---:|---:|
| 16 activation-weighted | +12.85% | 0.559252 | 88.89% |
| 16 message PCA | +45.38% | 0.754446 | 79.37% |
| 32 activation-weighted | +12.73% | 0.542594 | 88.89% |
| 32 message PCA | +19.06% | 0.504988 | 84.13% |
| 64 activation-weighted | -0.57%* | 0.263576 | 92.06% |
| 64 message PCA | +2.18% | 0.304945 | 84.13% |

\*Short-passage variation, not evidence of improved modeling. These experiments
project transient MLP messages and do not establish cache compression. Raw
artifacts use `qwen2.5_0.5b_downproj_geometry_*` and
`qwen2.5_0.5b_downproj_intervention_*`.

The all-layer rank-64 scan is
`qwen2.5_0.5b_downproj_all_layers_r64_cal256_eval64.json`. Active `down_proj`
beat activation weights shuffled across channels in 22/24 layers and message
PCA in 11/24 by live-intervention KL. Layers 4, 5, 6, 9, 11, and 12 form a
particularly informative inversion: PCA had lower reconstruction MSE, but the
model-derived basis had lower downstream KL. The companion phase artifact,
`qwen2.5_0.5b_detected_phases_n256.json`, reports boundaries 0/4/12/16/24;
these are an exploratory overlay rather than evidence of causal phase
alignment.

The cross-family replication is
`pythia410m_downproj_all_layers_r64_cal256_eval64.json`. Pythia-410M active
`dense_4h_to_h` geometry also beat shuffled weighting in 22/24 layers and
message PCA in 11/24 by KL. PCA reconstructed more accurately while the active
basis preserved behavior better in layers 3, 5, 6, 8, 9, 10, 13, and 18. Its
effective-rank median was 19.20 (range 1.50–37.99). The independently detected
Pythia boundaries are 0/5/9/20/24; phase alignment remains exploratory.

### Layer-dynamic MLP message rank allocation

`pythia410m_dynamic_mlp_rank_corrected_cal256_profile64_eval64.json` contains
the corrected Pythia-410M frontier. Basis calibration, layer sensitivity
profiling, and simultaneous validation use separate text passages. At equal
total rank, dynamic versus uniform results include:

| Coordinate saving | Dynamic PPL / KL / top-1 | Uniform PPL / KL / top-1 |
|---:|---:|---:|
| 3.1% | +0.60% / 0.01206 / 96.83% | +9.51% / 0.09946 / 87.30% |
| 6.3% | +3.77% / 0.02831 / 95.24% | +8.72% / 0.12729 / 87.30% |
| 12.5% | +12.35% / 0.06332 / 92.06% | +26.67% / 0.23357 / 87.30% |
| 25.0% | +34.34% / 0.21693 / 90.48% | +58.74% / 0.43806 / 76.19% |

These percentages describe MLP-message coordinate rank, not model weights,
KV-cache size, or measured application memory.

### Prompt- and gradient-conditioned geometry field

`pythia410m_mlp_geometry_field_r64_cal3x64_eval32.json` compares rank-64
active output-projection, message-PCA, Fisher-only, and joint write-times-Fisher
bases. Three calibration domains are separate from the intervention passage.
Same-layer cross-domain active-basis similarity averaged 0.568 (range
0.244–0.818), demonstrating input-conditioned usage. Phase-boundary adjacent
similarity averaged 0.210 versus 0.197 elsewhere, so the independently detected
boundaries are not active-subspace discontinuities in this run. Message PCA won
14/24 held-out interventions, active geometry 8, joint geometry 2, and Fisher
zero. Mean KL was 0.17305, 0.17924, 0.21576, and 0.36913 respectively. The
32-token intervention holdout makes this diagnostic evidence, not a broad
generality claim.

### Dynamic residual width (negative result)

`pythia410m_dynamic_residual_width_reader_oracle_n64.json` tests direct
residual-stream restriction using nested next-reader weight bases and an
oracle sequence-by-layer allocator. At 6.3% coordinate reduction, the oracle
passage measured +13,438% PPL, 4.3735 KL, and 22.22% top-1; transfer measured
+1,617%, 2.8003, and 25.40%. Uniform restriction was worse, but neither is a
viable operating point. This result rules out the tested static post-training
basis; it does not test a transformer trained to use conditional width.

### Reversible residual folding (mixed/negative result)

`pythia70m_residual_folding_cal128_eval128.json` and
`pythia410m_residual_folding_cal128_eval128.json` test adjacent Haar, random,
and correlation-matched lifting on held-out residuals and decoder-visible
adjacent GIHKCC deltas. Equal-bit Haar gained 0.11–0.18 dB projected-K/V PSNR
on the six-layer model but not on the 24-layer model. The best deeper-model
point was correlation lifting at coarse-4/detail-2 versus direct 3-bit deltas:
21.60 versus 20.57 dB residual PSNR, 34.19 versus 34.09 dB projected-K/V PSNR,
and 4.93x versus 4.96x compression. At nominal 2 bits the residual improvement
did not preserve projected K/V. These are static reconstruction diagnostics,
not live autoregressive quality results.

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

### Triton decode and adjacent-chain fusion

The Windows environment uses optional `triton-windows==3.7.1.post27`. A custom
kernel fuses cross-byte bit extraction, codebook lookup, and per-vector norm
scaling. It reconstructs 2/4-bit payloads exactly relative to the reference and
3-bit payloads within 9.54e-7. Because every adjacent payload uses the same
orthogonal rotation, scaled anchor/delta codewords are summed before applying a
single inverse rotation instead of one rotation per dependency.

At the deepest layer over 1,024 tokens, the combined path produced exact
materialized/blockwise attention parity. Materialized attention took 16.24 ms
and peaked at 19.09 MB temporary allocation; block-256 took 18.46 ms and peaked
at 7.67 MB. Thus blockwise attention is only 1.14x slower at that layer while
using 2.49x less temporary memory.

End-to-end optimization progression on the same 32-prefix/256-token trace:

| Packed implementation | tok/s | Temporary peak | PPL change | Top-1 |
|---|---:|---:|---:|---:|
| Unfused PyTorch | 3.44 | 39.96 MB | +5.49% | 87.89% |
| Triton packed decode | 5.96 | 16.83 MB | +5.42% | 85.94% |
| Triton + one-rotation chain | **6.56** | **7.19 MB** | **+5.17%** | **89.45%** |
| Triton + fused chain kernel | **10.28** | 11.56 MB | +6.12% | 87.89% |
| Fused chain + capacity buffer | **10.32** | 9.61 MB | +6.12% | 87.89% |
| Shared-rotation streaming encode | **11.57** | 9.61 MB | **+4.00%** | 87.50% |
| FP16 fused SDPA (single block) | **13.05** | **8.65 MB** | +4.77% | 86.72% |
| Native FP16 K/V (paired latest run) | 51.64 | -- | baseline | 100% |

The one-rotation path is 1.91x faster and uses 5.56x less temporary memory than
the initial packed prototype. The newer fused-chain result supersedes its
throughput lead; remaining targets are fused inverse rotation/KV
projection/online softmax and capacity-managed packed append buffers.

The fused-chain kernel lays adjacent delta indices and norms out as contiguous
layer-major matrices. A runtime-depth Triton loop decodes and sums the complete
anchor/delta ancestry in one launch, followed by the shared inverse rotation.
The isolated 24-layer CUDA test matches the reference decoder. On the full
256-step trace this is 1.57x faster than the one-rotation chain and 2.99x faster
than the initial packed prototype, while remaining 5.18x slower than its paired
native run. The temporary peak regression from 7.19 MB to 11.56 MB comes from
matrix-wide `torch.cat` during streaming growth, not historical K/V
materialization.

Reserving the exact 288-token capacity keeps physical and logical packed
storage equal at 2,004,480 bytes and prevents backing-pointer changes during
append. Temporary allocation above the resident base falls 16.9%, from 11.56
MB to 9.61 MB. Absolute peak allocation remains essentially flat because the
final packed history is resident state; the negligible throughput change
(10.28 to 10.32 tokens/s) shows append copying was not the main runtime limit.

The streaming encoder preserves closed-loop adjacent prediction in rotated
space because `(state - reconstructed) R^T` equals `state R^T -
reconstructed_rotated`. It batches all 24 forward rotations into one matmul and
avoids inverse-rotating every reconstructed delta during encoding. The
32-token synchronized profile reduced append compression from 1.112 seconds to
0.687 seconds (38.2%). On the full trace throughput reaches 11.57 tokens/s,
3.36x the initial packed prototype and 4.46x behind its paired 51.64-token/s
native run. PPL change improves to +4.00% and mean KL to 0.06802; top-1
agreement moves slightly lower to 87.50%.

### Folded normalized K/V projection: rejected

The algebraically fused projection computes LayerNorm statistics in the
pre-rotation basis and folds `rotation Ã— LayerNorm scale Ã— K/V weight` into a
transformed matrix. Focused tests match the conventional path, but the full
run reaches only 9.75 tokens/s, 15.7% below the 11.57-token/s Pareto path, while
adding 101,154,816 bytes of transformed weights. PPL change also regresses from
+4.00% to +4.84%. This remains available behind `--fused-projection` as a
documented experiment, not a recommended serving configuration.

### Fused attention Pareto point

When all historical tokens fit one block, `--fused-attention` concatenates the
current K/V transiently and uses fused FP16 scaled-dot-product attention. At
13.05 tokens/s it is 12.8% faster than the online-FP32 path and 3.79x faster
than the initial packed implementation; temporary peak above the resident base
falls to 8.65 MB. Quality moves to +4.77% PPL and 86.72% top-1 agreement, so
the 11.57-token/s online path remains the quality-oriented point at +4.00% and
87.50%. The last 32 steps of the 256-token run exceed the block size and
exercise the online fallback. FP32 SDPA was tested separately but fell to 8.90
tokens/s at full length, so it was rejected.

### Projected K/V cache: first diagnostic

`quantized_kv_generation.py` removes historical reprojection entirely and
maintains post-RoPE K/V with an FP16 hot tail plus packed cold storage. The
lossless hot-only control measured +0.55% PPL and 100% top-1 over 16 tokens, so
cache ordering and single-token attention are sound. Packed-stream unit tests
verify int8 reconstruction and chronological decode directly.

Using the same per-head symmetric quantizer for both keys and values is not a
Pareto result. Hot-32/int8 cold produced +12.59% PPL, KL 0.02097, and 100%
top-1 over 16 tokens. Hot-32/4-bit cold reached 19.93 tokens/s on 32 tokens but
increased PPL by 323%. These diagnostics support separating the geometries:
pagewise per-channel quantization for keys, which are dot-product sensitive and
outlier-heavy, and per-token quantization for values.

### Pagewise projected K/V Pareto frontier

The corrected design uses 32-token pages, per-channel affine keys across each
page, per-token/per-head symmetric values, and an exact FP16 hot-32 tail.
Completed pages are stored contiguously so each layer performs one key unpack
and one value unpack rather than decoding every page independently.

| Cold format | Resident ratio | tok/s | PPL change | KL | Top-1 |
|---|---:|---:|---:|---:|---:|
| K8 / V4 | 2.09x | **21.05** | **-0.45%** | **0.00175** | 96.48% |
| K6 / V4 | **2.36x** | 16.53 | +1.77% | 0.00504 | **96.88%** |
| K4 / V4 | 2.72x | 16.75 | +9.05% | 0.06396 | 85.94% |

K8/V4 is the quality-and-speed point; K6/V4 is the balanced compression point.
K4 keys are too aggressive on this trace. Before contiguous storage, K8/V4
decoded at 8.92 tokens/s; the same payload and logits reach 21.05 tokens/s once
page launches are collapsed. Temporary allocation is still 30.4 MB because
the PyTorch reference materializes decoded cold pages before SDPA. Direct
packed-page attention should remove that allocation and close more of the
remaining 2.53x throughput gap to the paired native run.

### Direct packed-page attention

A Triton kernel now decodes neither full keys nor values. Each `(page, head)`
program extracts packed key codes, computes query scores and page-local
softmax statistics, extracts packed values, and produces a weighted
accumulator. Tiny page statistics are merged with exact pending/hot/current
tokens using online softmax. A CUDA test matches explicit decode plus attention.

| Configuration | Ratio | tok/s | PPL change | KL | Top-1 | Temp peak |
|---|---:|---:|---:|---:|---:|---:|
| K8/V4 hot-32 | 2.09x | **24.61** | **-0.69%** | **0.00167** | 96.48% | 18.43 MB |
| K6/V4 hot-32 | 2.36x | 22.72 | +1.74% | 0.00487 | **96.88%** | **17.04 MB** |
| K8/V4 hot-0 | **2.42x** | 17.94 | -0.05% | 0.00218 | 96.48% | 18.16 MB |

The first head-wide kernel reduced temporary memory but reached only 19.79
tokens/s. Splitting work into parallel page/head programs raised throughput by
24.4% to 24.61 tokens/s. Relative to materialized page decode, the final path
is 16.9% faster and uses 39% less temporary memory. It is 7.15x faster than the
initial 3.44-token/s packed-residual prototype and 2.08x behind the paired
51.29-token/s native run.

### 1,024-token repeated validation

Each configuration below scores 1,024 held-out WikiText tokens after the same
32-token prefix. Three independent processes provide paired native/compressed
timings; quality metrics are identical across repeats. The machine-readable
aggregate is `pythia410m_projectedkv_directpacked_n1024_repeats3_aggregate.json`.

| Configuration | Ratio | Compressed tok/s median (range) | Native median | PPL change | KL | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| K8/V4 hot-32 | 2.32x | **27.29** (25.66--27.33) | 53.48 | **+0.043%** | **0.00216** | **97.07%** |
| K6/V4 hot-32 | **2.69x** | 26.12 (23.72--26.35) | 51.60 | +1.29% | 0.00642 | 95.51% |
| K8/V4 hot-0 | 2.42x | 25.79 (25.61--26.50) | 50.19 | +0.50% | 0.00274 | 96.97% |

At this length the direct packed configurations sustain roughly half native
throughput. K8/V4 hot-32 remains the quality/speed point, K6/V4 hot-32 is the
compression point, and hot-0 demonstrates that the FP16 tail is optional when
near-baseline quality matters more than peak speed.

### Five-window quality validation

Five non-overlapping windows begin at token offsets 0, 1,056, 2,112, 3,168,
and 4,224. Each uses a 32-token prefix followed by 1,024 scored tokens. Window
zero uses the median-timing repeat from the repeat study; the other windows are
independent paired processes. The aggregate is stored in
`pythia410m_projectedkv_directpacked_n1024_windows5_aggregate.json`.

| Configuration | Ratio | PPL median (range) | KL median (range) | Top-1 median (range) | tok/s median (range) |
|---|---:|---:|---:|---:|---:|
| K8/V4 hot-32 | 2.32x | **+0.115%** (-0.275%--+0.232%) | **0.00207** (0.00172--0.00216) | **97.66%** (97.07%--97.85%) | **25.77** (23.17--27.29) |
| K6/V4 hot-32 | **2.69x** | +0.813% (+0.246%--+1.293%) | 0.00642 (0.00597--0.00706) | 95.70% (95.31%--95.90%) | 25.52 (23.30--26.12) |

The narrow K8 quality range supports a near-baseline claim on this model and
dataset. K6 remains below +1.3% PPL on every window and is the stronger
compression point. Runtime varies more than quality because these are
wall-clock desktop GPU processes rather than isolated server measurements.

The earlier six-minute timeouts were traced to Hugging Face's offline dataset
builder, not residual capture or compression. The validator now records phase
timings and supports tokenized-text and packed-prefix checkpoints. Loading the
local validation Arrow file directly took 2 ms; warm token/prefix checkpoints
took 3 ms and 10 ms respectively. The measured compressed generation phase was
24.90 seconds for 256 tokens.
