"""
KVTC Test Suite

Validates the transform coding pipeline: DCT accuracy, energy thresholding
behavior, sparse quantization, RLE encoding, and full pipeline compression
ratios on synthetic delta tensors that mimic real L1/L2 output.

Run:  python test_kvtc.py
"""

import sys
import time
import math
import torch

from kvtc import (
    KVTCConfig,
    dct_1d, idct_1d,
    dct_2d, idct_2d,
    energy_threshold,
    topk_threshold,
    sparse_quantize,
    sparse_dequantize,
    rle_encode,
    rle_decode,
    kvtc_compress_delta,
    kvtc_decompress_delta,
    kvtc_compress_all_deltas,
    kvtc_decompress_all_deltas,
)

# Also import from gihkcc to generate realistic deltas
from gihkcc import (
    GIHKCCConfig,
    l1_compress,
    l1_decompress,
)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# --- Helpers to generate realistic delta tensors ---

def generate_smooth_deltas(
    count: int = 28,
    num_heads: int = 32,
    seq_len: int = 512,
    head_dim: int = 128,
    noise_scale: float = 0.03,
    rank: int = 8,
) -> list:
    """
    Generate delta tensors that mimic L1 fold output.

    Real KV cache deltas between adjacent layers are NOT iid Gaussian.
    They're the output of learned linear projections (attention + MLP
    residual), which means they're approximately low-rank: most of the
    signal lives in a few principal components, with small full-rank noise.

    This generator creates low-rank + noise deltas, which concentrate
    energy in low-frequency DCT coefficients the same way real deltas do.
    """
    torch.manual_seed(42)
    deltas = []
    for i in range(count):
        # Low-rank component: rank-r outer product structure
        # Simulates the output of a learned projection (Wk @ residual)
        # where most variance is captured by the top singular vectors
        U = torch.randn(num_heads * seq_len, rank) * noise_scale * 3.0
        V = torch.randn(rank, head_dim) * 1.0
        low_rank = (U @ V).reshape(num_heads, seq_len, head_dim)

        # Small full-rank noise
        noise = torch.randn(num_heads, seq_len, head_dim) * noise_scale * 0.1

        delta = (low_rank + noise).to(torch.float16)
        deltas.append(delta)
    return deltas


# Inline F import for the smoothing kernel
import torch.nn.functional as F


# ── Tests ────────────────────────────────────────────────────────────────

def test_dct_roundtrip():
    """Verify DCT → IDCT is identity (within float precision)."""
    print_header("Test: DCT Round-Trip Fidelity")

    # 1D
    x1d = torch.randn(64)
    recon_1d = idct_1d(dct_1d(x1d))
    err_1d = (recon_1d - x1d).abs().max().item()
    print(f"  1D ({x1d.shape}): max error = {err_1d:.2e}")
    assert err_1d < 1e-4, f"1D DCT round-trip error too high: {err_1d}"

    # 2D
    x2d = torch.randn(32, 128)
    recon_2d = idct_2d(dct_2d(x2d))
    err_2d = (recon_2d - x2d).abs().max().item()
    print(f"  2D ({x2d.shape}): max error = {err_2d:.2e}")
    assert err_2d < 1e-4, f"2D DCT round-trip error too high: {err_2d}"

    # 3D (multi-head KV shape)
    x3d = torch.randn(8, 64, 128)
    recon_3d = idct_2d(dct_2d(x3d))
    err_3d = (recon_3d - x3d).abs().max().item()
    print(f"  3D ({x3d.shape}): max error = {err_3d:.2e}")
    assert err_3d < 1e-4, f"3D DCT round-trip error too high: {err_3d}"

    print("  ✓ PASSED")


def test_energy_concentration():
    """
    Verify that smooth signals concentrate energy in few DCT coefficients.
    A smooth delta should have >95% energy in <20% of coefficients.
    """
    print_header("Test: Energy Concentration in Smooth Deltas")

    # Generate a smooth delta (low-frequency dominant)
    torch.manual_seed(42)
    smooth = torch.randn(64, 128) * 0.05
    # Smooth it
    kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
    smooth_flat = smooth.reshape(-1, 1, 128)
    smooth_flat = F.conv1d(smooth_flat, kernel, padding=1)
    smooth = smooth_flat.reshape(64, 128)

    coeffs = dct_2d(smooth)

    # Measure energy distribution
    energies = (coeffs ** 2).flatten().sort(descending=True).values
    total_energy = energies.sum().item()
    cum_energy = energies.cumsum(0) / total_energy

    # How many coefficients for 95% energy?
    idx_95 = (cum_energy >= 0.95).nonzero(as_tuple=False)[0].item()
    pct_95 = (idx_95 + 1) / energies.numel() * 100

    # How many for 99%?
    idx_99 = (cum_energy >= 0.99).nonzero(as_tuple=False)[0].item()
    pct_99 = (idx_99 + 1) / energies.numel() * 100

    print(f"  Smooth delta [{smooth.shape}]:")
    print(f"    95% energy in top {idx_95+1}/{energies.numel()} coeffs ({pct_95:.1f}%)")
    print(f"    99% energy in top {idx_99+1}/{energies.numel()} coeffs ({pct_99:.1f}%)")

    # Compare: random (non-smooth) delta
    random = torch.randn(64, 128) * 0.05
    coeffs_r = dct_2d(random)
    energies_r = (coeffs_r ** 2).flatten().sort(descending=True).values
    cum_r = energies_r.cumsum(0) / energies_r.sum()
    idx_95_r = (cum_r >= 0.95).nonzero(as_tuple=False)[0].item()
    pct_95_r = (idx_95_r + 1) / energies_r.numel() * 100

    print(f"\n  Random delta [{random.shape}]:")
    print(f"    95% energy in top {idx_95_r+1}/{energies_r.numel()} coeffs ({pct_95_r:.1f}%)")

    print(f"\n  Concentration advantage: smooth uses {pct_95:.1f}% vs random {pct_95_r:.1f}% of coeffs")
    assert pct_95 < pct_95_r, "Smooth deltas should concentrate energy better than random"
    print("  ✓ PASSED")


def test_energy_threshold():
    """Test energy-based coefficient thresholding."""
    print_header("Test: Energy Thresholding")

    torch.manual_seed(42)
    coeffs = torch.randn(8, 64, 128)

    for retention in [0.90, 0.95, 0.99]:
        sparse, num_nz, thresh = energy_threshold(coeffs, retention)
        actual_sparsity = 1.0 - (num_nz / coeffs.numel())

        # Verify energy is actually retained
        original_energy = (coeffs ** 2).sum().item()
        retained_energy = (sparse ** 2).sum().item()
        actual_retention = retained_energy / original_energy

        print(f"  retention={retention:.2f}: "
              f"kept {num_nz}/{coeffs.numel()} ({actual_sparsity:.1%} sparse), "
              f"actual energy retained={actual_retention:.4f}")

        assert actual_retention >= retention - 0.01, \
            f"Energy retention too low: {actual_retention} < {retention}"

    print("  ✓ PASSED")


def test_sparse_quantize_roundtrip():
    """Test sparse quantization → dequantization accuracy."""
    print_header("Test: Sparse Quantization Round-Trip")

    torch.manual_seed(42)
    # Create a sparse tensor (90% zeros)
    dense = torch.randn(8, 64, 128) * 0.1
    mask = torch.rand_like(dense) > 0.9  # Keep 10%
    sparse = dense * mask.float()

    for bits in [8, 6, 4]:
        sqt = sparse_quantize(sparse, bits=bits)
        recon = sparse_dequantize(sqt)

        err = (recon - sparse).abs()
        nz_err = err[sparse != 0]

        print(f"  {bits}-bit: "
              f"sparsity={sqt.sparsity:.1%}, "
              f"mean_nz_err={nz_err.mean():.6f}, "
              f"max_nz_err={nz_err.max():.6f}, "
              f"ratio={sqt.ratio:.1f}x, "
              f"bytes={sqt.compressed_bytes}")

    print("  ✓ PASSED")


def test_rle_roundtrip():
    """Test RLE encode → decode preserves data."""
    print_header("Test: RLE Round-Trip")

    torch.manual_seed(42)
    dense = torch.randn(8, 64, 128) * 0.1
    mask = torch.rand_like(dense) > 0.95  # Very sparse: 5% non-zero
    sparse = dense * mask.float()

    sqt = sparse_quantize(sparse, bits=8)
    rle = rle_encode(sqt)
    sqt_recon = rle_decode(rle)

    # Verify indices and values match
    assert sqt.num_nonzero == sqt_recon.num_nonzero, \
        f"Non-zero count mismatch: {sqt.num_nonzero} vs {sqt_recon.num_nonzero}"

    idx_match = torch.equal(sqt.indices.sort().values, sqt_recon.indices.sort().values)
    assert idx_match, "Indices don't match after RLE round-trip"

    # Verify full reconstruction
    recon_original = sparse_dequantize(sqt)
    recon_rle = sparse_dequantize(sqt_recon)
    err = (recon_original - recon_rle).abs().max().item()

    print(f"  Non-zeros: {sqt.num_nonzero}")
    print(f"  RLE pairs: {rle.runs.numel()}")
    print(f"  Sparse bytes: {sqt.compressed_bytes}")
    print(f"  RLE bytes: {rle.compressed_bytes}")
    print(f"  Reconstruction error: {err:.2e}")
    print(f"  RLE ratio: {rle.ratio:.1f}x")

    assert err < 1e-6, f"RLE round-trip introduced error: {err}"
    print("  ✓ PASSED")


def test_full_kvtc_pipeline():
    """End-to-end KVTC pipeline on realistic delta tensors."""
    print_header("Test: Full KVTC Pipeline")

    # Generate realistic deltas
    deltas = generate_smooth_deltas(
        count=28, num_heads=8, seq_len=64, head_dim=64,
        noise_scale=0.03,
    )

    configs = {
        "Conservative (99% energy)": KVTCConfig(
            energy_retention=0.99,
            coeff_quant_bits=8,
            rle_enabled=True,
        ),
        "Balanced (95% energy)": KVTCConfig(
            energy_retention=0.95,
            coeff_quant_bits=8,
            rle_enabled=True,
        ),
        "Aggressive (90% energy)": KVTCConfig(
            energy_retention=0.90,
            coeff_quant_bits=6,
            rle_enabled=True,
        ),
        "Extreme (85% energy, 4-bit)": KVTCConfig(
            energy_retention=0.85,
            coeff_quant_bits=4,
            rle_enabled=True,
        ),
    }

    for name, cfg in configs.items():
        t0 = time.perf_counter()
        compressed, stats = kvtc_compress_all_deltas(deltas, cfg)
        t_comp = time.perf_counter() - t0

        t1 = time.perf_counter()
        reconstructed = kvtc_decompress_all_deltas(compressed)
        t_decomp = time.perf_counter() - t1

        # Measure reconstruction quality
        total_mse = 0.0
        total_mae = 0.0
        max_err = 0.0
        for orig, recon in zip(deltas, reconstructed):
            diff = (recon.float() - orig.float())
            total_mse += (diff ** 2).mean().item()
            total_mae += diff.abs().mean().item()
            max_err = max(max_err, diff.abs().max().item())

        n = len(deltas)
        psnr = -10 * math.log10(total_mse / n) if total_mse > 0 else float("inf")

        print(f"  {name}:")
        print(f"    Ratio: {stats['ratio']}x  "
              f"({stats['original_bytes']/1024:.0f} KB → {stats['compressed_bytes']/1024:.0f} KB)")
        print(f"    Sparsity: {stats['mean_sparsity']:.1%}")
        print(f"    PSNR: {psnr:.1f} dB  |  MAE: {total_mae/n:.6f}  |  Max: {max_err:.6f}")
        print(f"    Compress: {t_comp*1000:.1f} ms  |  Decompress: {t_decomp*1000:.1f} ms")
        print()

    print("  ✓ PASSED")


def test_kvtc_on_real_l1_deltas():
    """
    Generate a synthetic KV cache with low-rank inter-layer evolution,
    run L1 fold to get actual deltas, then apply KVTC.
    This is the real pipeline path.
    """
    print_header("Test: KVTC on L1 Fold Output (Full Pipeline)")

    # Generate KV stack with low-rank evolution (realistic)
    torch.manual_seed(42)
    num_layers, num_heads, seq_len, head_dim = 32, 8, 128, 64
    rank = 6
    shift_layers = [8, 16, 24]

    kv = []
    base = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16)
    kv.append(base.clone())
    current = base.clone()

    for layer in range(1, num_layers):
        if layer in shift_layers:
            # Major shift — new low-rank projection
            U = torch.randn(num_heads * seq_len, rank) * 0.3
            V = torch.randn(rank, head_dim)
            shift = (U @ V).reshape(num_heads, seq_len, head_dim).to(torch.float16)
            current = current + shift
        else:
            # Small low-rank evolution
            U = torch.randn(num_heads * seq_len, rank) * 0.02
            V = torch.randn(rank, head_dim)
            delta = (U @ V).reshape(num_heads, seq_len, head_dim).to(torch.float16)
            noise = torch.randn_like(current) * 0.002
            current = current + delta + noise
        kv.append(current.clone())

    l1_config = GIHKCCConfig(l1_snr_threshold=0.92, l1_max_keyframe_span=8)
    l1_cache = l1_compress(kv, l1_config)

    print(f"  L1 output: {len(l1_cache.keyframes)} keyframes, {len(l1_cache.deltas)} deltas")

    # Extract raw delta tensors
    raw_deltas = [d.delta for d in l1_cache.deltas]

    if not raw_deltas:
        print("  No deltas to compress (all keyframes). Skipping KVTC.")
        print("  ✓ PASSED")
        return

    # Apply KVTC
    kvtc_config = KVTCConfig(
        energy_retention=0.95,
        coeff_quant_bits=8,
        rle_enabled=True,
    )

    compressed, stats = kvtc_compress_all_deltas(raw_deltas, kvtc_config)
    reconstructed = kvtc_decompress_all_deltas(compressed)

    # Measure combined compression
    per_layer_bytes = raw_deltas[0].nelement() * 2  # float16
    kf_bytes = len(l1_cache.keyframes) * per_layer_bytes
    delta_bytes_original = len(raw_deltas) * per_layer_bytes
    delta_bytes_kvtc = stats["compressed_bytes"]
    total_original = num_layers * per_layer_bytes
    total_compressed = kf_bytes + delta_bytes_kvtc

    combined_ratio = total_original / total_compressed if total_compressed > 0 else 0

    # Reconstruction fidelity
    errors = []
    for orig, recon in zip(raw_deltas, reconstructed):
        errors.append((recon.float() - orig.float()).abs().mean().item())

    print(f"\n  Per-layer size: {per_layer_bytes / 1024:.1f} KB")
    print(f"  Original {num_layers} layers: {total_original / 1024:.0f} KB")
    print(f"  L1 keyframes ({len(l1_cache.keyframes)}): {kf_bytes / 1024:.0f} KB")
    print(f"  L1 deltas raw ({len(raw_deltas)}): {delta_bytes_original / 1024:.0f} KB")
    print(f"  L1 deltas KVTC: {delta_bytes_kvtc / 1024:.0f} KB")
    print(f"  Total compressed: {total_compressed / 1024:.0f} KB")
    print(f"\n  L1 structural ratio: {l1_cache.compression_ratio:.2f}x")
    print(f"  KVTC delta ratio: {stats['ratio']}x")
    print(f"  KVTC sparsity: {stats['mean_sparsity']:.1%}")
    print(f"  Combined L1+KVTC ratio: {combined_ratio:.1f}x")
    print(f"  Mean delta reconstruction error: {sum(errors)/len(errors):.6f}")

    assert stats['ratio'] > 1.0, f"KVTC should provide some compression: {stats['ratio']}"
    print("\n  ✓ PASSED")


def test_kvtc_scaling():
    """Test how KVTC compression scales with delta structure (rank)."""
    print_header("Test: KVTC Scaling — Low-Rank vs High-Rank Deltas")

    torch.manual_seed(42)
    cfg = KVTCConfig(energy_retention=0.95, coeff_quant_bits=8, rle_enabled=True)

    for label, rank in [("Rank 2 (very structured)", 2),
                        ("Rank 4", 4),
                        ("Rank 8", 8),
                        ("Rank 16", 16),
                        ("Rank 32 (less structured)", 32),
                        ("Rank 64 (approaching full)", 64)]:
        deltas = generate_smooth_deltas(count=10, num_heads=8, seq_len=64,
                                         head_dim=64, noise_scale=0.03, rank=rank)
        compressed, stats = kvtc_compress_all_deltas(deltas, cfg)

        # Reconstruction quality
        recon = kvtc_decompress_all_deltas(compressed)
        mae = sum((r.float() - o.float()).abs().mean().item()
                   for r, o in zip(recon, deltas)) / len(deltas)

        print(f"  {label:35s}  ratio={stats['ratio']:6.1f}x  "
              f"sparsity={stats['mean_sparsity']:.1%}  MAE={mae:.6f}")

    print("\n  ✓ PASSED")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  KVTC — Transform Coding Test Suite")
    print("  VecP Labs LLC | Patent Pending")
    print("="*60)

    tests = [
        test_dct_roundtrip,
        test_energy_concentration,
        test_energy_threshold,
        test_sparse_quantize_roundtrip,
        test_rle_roundtrip,
        test_full_kvtc_pipeline,
        test_kvtc_on_real_l1_deltas,
        test_kvtc_scaling,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
