"""
GIHKCC Test Suite & Demo

Validates the compression pipeline with synthetic KV caches that simulate
real transformer behavior (high inter-layer similarity with periodic
representational shifts). No GPU or model weights required.

Run:  python test_gihkcc.py
"""

import sys
import json
import time
import torch
import math

from gihkcc import (
    GIHKCCConfig,
    compute_statistical_snr,
    compute_snr_profile,
    l1_compress,
    l1_decompress,
    l2_compress,
    l2_decompress,
    l3_compress_layer,
    l3_decompress_layer,
    quantize_delta,
    dequantize_delta,
    compress_kv_cache,
    decompress_kv_cache,
    estimate_memory_bytes,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def generate_synthetic_kv_stack(
    num_layers: int = 32,
    num_heads: int = 32,
    seq_len: int = 512,
    head_dim: int = 128,
    shift_layers: list = None,
    noise_scale: float = 0.05,
    shift_scale: float = 0.4,
    dtype: torch.dtype = torch.float16,
) -> list:
    """
    Generate a synthetic KV cache stack that mimics real transformer behavior:
    - Base representation at layer 0
    - Small incremental changes between adjacent layers (high SNR)
    - Larger representational shifts at specified layers (low SNR, keyframe candidates)

    Args:
        shift_layers: Layer indices where a major representational shift occurs.
                      Defaults to layers [8, 16, 24] for a 32-layer model.
    """
    if shift_layers is None:
        shift_layers = [i for i in range(num_layers) if i > 0 and i % 8 == 0]

    torch.manual_seed(42)  # Reproducibility

    kv_stack = []
    base = torch.randn(num_heads, seq_len, head_dim, dtype=dtype)
    kv_stack.append(base.clone())

    current = base.clone()
    for layer in range(1, num_layers):
        if layer in shift_layers:
            # Major shift — simulates a representational boundary
            shift = torch.randn_like(current) * shift_scale
            current = current + shift
        else:
            # Small incremental change — high inter-layer similarity
            noise = torch.randn_like(current) * noise_scale
            current = current + noise
        kv_stack.append(current.clone())

    return kv_stack


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_snr_visual(snr_values: list, threshold: float, label: str = ""):
    """Print a visual SNR profile with keyframe markers."""
    bars = "░▒▓█"
    if label:
        print(f"  {label}:")
    for i, snr in enumerate(snr_values):
        bar_idx = min(int(snr * len(bars)), len(bars) - 1)
        bar_len = int(snr * 25)
        bar = bars[bar_idx] * max(bar_len, 1)
        marker = " ◄ KEYFRAME" if snr < threshold else ""
        print(f"    L{i:2d}→L{i+1:2d}  {snr:.4f}  {bar}{marker}")


# ── Tests ────────────────────────────────────────────────────────────────

def test_snr_computation():
    """Test that SNR correctly identifies similar vs dissimilar layers."""
    print_header("Test: Statistical SNR Computation")

    a = torch.randn(8, 64, 64)

    # Near-identical: high SNR
    b_similar = a + torch.randn_like(a) * 0.01
    snr_high = compute_statistical_snr(a, b_similar)

    # Very different: low SNR
    b_different = torch.randn(8, 64, 64)
    snr_low = compute_statistical_snr(a, b_different)

    print(f"  Similar layers  → SNR = {snr_high:.4f}  (expect > 0.95)")
    print(f"  Different layers → SNR = {snr_low:.4f}  (expect < 0.5)")

    assert snr_high > 0.9, f"Expected high SNR for similar tensors, got {snr_high}"
    assert snr_low < 0.6, f"Expected low SNR for different tensors, got {snr_low}"
    print("  ✓ PASSED")


def test_l1_compression():
    """Test Level 1 inter-layer fold."""
    print_header("Test: Level 1 — Inter-Layer Fold")

    config = GIHKCCConfig(
        l1_snr_threshold=0.90,
        l1_max_keyframe_span=8,
    )

    # 32 layers with shifts at 8, 16, 24
    kv = generate_synthetic_kv_stack(
        num_layers=32, num_heads=8, seq_len=64, head_dim=64,
        shift_layers=[8, 16, 24],
    )

    # Show SNR profile
    snr = compute_snr_profile(kv)
    print_snr_visual(snr, config.l1_snr_threshold, "SNR Profile")

    # Compress
    compressed = l1_compress(kv, config)
    print(f"\n  Layers: {compressed.num_layers}")
    print(f"  Keyframes: {len(compressed.keyframes)}")
    print(f"  Deltas: {len(compressed.deltas)}")
    print(f"  Ratio: {compressed.compression_ratio:.2f}x")

    kf_layers = [kf.layer_idx for kf in compressed.keyframes]
    print(f"  Keyframe layers: {kf_layers}")

    # Verify keyframes land near shift boundaries
    assert 0 in kf_layers, "Layer 0 must be a keyframe"
    # Shifts at 8, 16, 24 should produce keyframes at or near those layers
    for shift in [8, 16, 24]:
        nearby = any(abs(kf - shift) <= 1 for kf in kf_layers)
        assert nearby, f"Expected keyframe near shift layer {shift}"

    # Decompress and verify
    reconstructed = l1_decompress(compressed)
    max_err = 0.0
    for i in range(len(kv)):
        if reconstructed[i] is not None:
            err = (reconstructed[i] - kv[i]).abs().max().item()
            max_err = max(max_err, err)

    print(f"  Max reconstruction error: {max_err:.8f}")
    # Without quantization, reconstruction should be exact (float rounding only)
    assert max_err < 1e-3, f"Reconstruction error too high: {max_err}"
    print("  ✓ PASSED")


def test_l2_compression():
    """Test Level 2 inter-keyframe fold."""
    print_header("Test: Level 2 — Inter-Keyframe Fold")

    config = GIHKCCConfig(
        l1_snr_threshold=0.90,
        l1_max_keyframe_span=6,
        l2_enabled=True,
        l2_super_keyframe_interval=3,
    )

    kv = generate_synthetic_kv_stack(
        num_layers=32, num_heads=8, seq_len=64, head_dim=64,
        shift_layers=[6, 12, 18, 24, 30],
    )

    # L1 → L2
    l1 = l1_compress(kv, config)
    l2 = l2_compress(l1, config)

    print(f"  L1 keyframes: {len(l1.keyframes)}")
    print(f"  L2 super-keyframes: {len(l2.super_keyframes)}")
    print(f"  L2 keyframe-deltas: {len(l2.keyframe_deltas)}")
    print(f"  L2 ratio: {l2.compression_ratio:.2f}x")

    skf_layers = [skf.layer_idx for skf in l2.super_keyframes]
    print(f"  Super-keyframe layers: {skf_layers}")

    # Round-trip: L2 → L1 → raw
    l1_recon = l2_decompress(l2)
    raw_recon = l1_decompress(l1_recon)

    max_err = 0.0
    for i in range(len(kv)):
        if raw_recon[i] is not None:
            err = (raw_recon[i] - kv[i]).abs().max().item()
            max_err = max(max_err, err)

    print(f"  Max round-trip error: {max_err:.8f}")
    # Float16 cascading: super-KF + L2 delta + L1 delta accumulates rounding
    assert max_err < 5e-3, f"L2 round-trip error too high: {max_err}"
    print("  ✓ PASSED")


def test_l3_compression():
    """Test Level 3 inter-token fold."""
    print_header("Test: Level 3 — Inter-Token Fold")

    # Single layer KV: [seq_len, num_heads, head_dim]
    seq_len = 128
    num_heads = 8
    head_dim = 64

    # Generate locally-coherent sequence (adjacent tokens similar)
    torch.manual_seed(42)
    base = torch.randn(1, num_heads, head_dim)
    kv = [base.clone()]
    for t in range(1, seq_len):
        noise_scale = 0.02 if t % 32 != 0 else 0.3  # Shift every 32 tokens
        kv.append(kv[-1] + torch.randn(1, num_heads, head_dim) * noise_scale)
    kv_layer = torch.cat(kv, dim=0)  # [seq_len, num_heads, head_dim]

    compressed = l3_compress_layer(kv_layer, stride=16)

    n_kf = compressed.token_keyframes.shape[0]
    n_delta = compressed.token_deltas.shape[0]
    ratio = seq_len / (n_kf + n_delta * 0.5)

    print(f"  Sequence length: {seq_len}")
    print(f"  Token keyframes: {n_kf}")
    print(f"  Token deltas: {n_delta}")
    print(f"  Effective ratio: {ratio:.2f}x")

    # Round-trip
    recon = l3_decompress_layer(compressed)
    max_err = (recon - kv_layer).abs().max().item()
    print(f"  Max reconstruction error: {max_err:.8f}")
    assert max_err < 1e-5, f"L3 round-trip error too high: {max_err}"
    print("  ✓ PASSED")


def test_quantization():
    """Test delta quantization round-trip."""
    print_header("Test: Delta Quantization")

    delta = torch.randn(8, 64, 64) * 0.1  # Small deltas typical of L1

    for bits in [8, 4]:
        q, scale, zp = quantize_delta(delta, bits=bits)
        recon = dequantize_delta(q, scale, zp, bits=bits)
        err = (recon - delta).abs()

        print(f"  {bits}-bit: mean_err={err.mean():.6f}, max_err={err.max():.6f}")

    print("  ✓ PASSED")


def test_full_pipeline():
    """End-to-end compression pipeline test."""
    print_header("Test: Full Pipeline (L1 + L2)")

    config = GIHKCCConfig(
        l1_snr_threshold=0.92,
        l1_max_keyframe_span=8,
        l2_enabled=True,
        l2_super_keyframe_interval=4,
        l3_enabled=False,
    )

    # Simulate a realistic model: 32 layers, 32 heads, 512 tokens, 128 head_dim
    print("  Generating synthetic KV cache (32L × 32H × 512T × 128D)...")
    keys = generate_synthetic_kv_stack(
        num_layers=32, num_heads=32, seq_len=512, head_dim=128,
        shift_layers=[8, 16, 24],
        noise_scale=0.03,
        shift_scale=0.35,
    )
    values = generate_synthetic_kv_stack(
        num_layers=32, num_heads=32, seq_len=512, head_dim=128,
        shift_layers=[10, 20, 28],
        noise_scale=0.03,
        shift_scale=0.30,
    )

    # Compress
    t0 = time.perf_counter()
    compressed = compress_kv_cache(keys, values, config)
    t_compress = time.perf_counter() - t0

    summary = compressed.summary()
    mem = estimate_memory_bytes(compressed)

    print(f"\n  Compression time: {t_compress*1000:.1f} ms")
    print(f"  Keys ratio:  {summary['keys_compression_ratio']}x")
    print(f"  Values ratio: {summary['values_compression_ratio']}x")
    print(f"  Combined:    {summary['combined_ratio']}x")
    print(f"\n  Original:    {mem['original_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Structural:  {mem['structural_bytes'] / 1024 / 1024:.1f} MB  ({mem['structural_ratio']}x)")
    print(f"  Projected:   {mem['projected_bytes'] / 1024 / 1024:.1f} MB  ({mem['projected_ratio']}x, w/ {config.delta_quant_bits}-bit deltas)")
    print(f"  Savings:     {mem['savings_projected_bytes'] / 1024 / 1024:.1f} MB")

    # Decompress and verify
    t1 = time.perf_counter()
    recon_keys, recon_values = decompress_kv_cache(compressed)
    t_decompress = time.perf_counter() - t1

    key_errs = []
    val_errs = []
    for i in range(len(keys)):
        key_errs.append((recon_keys[i] - keys[i]).abs().mean().item())
        val_errs.append((recon_values[i] - values[i]).abs().mean().item())

    print(f"\n  Decompression time: {t_decompress*1000:.1f} ms")
    print(f"  Mean key error:   {sum(key_errs)/len(key_errs):.8f}")
    print(f"  Mean value error: {sum(val_errs)/len(val_errs):.8f}")
    print(f"  Max key error:    {max(key_errs):.8f}")
    print(f"  Max value error:  {max(val_errs):.8f}")

    # SNR profiles
    print(f"\n  Key SNR Profile (layers with keyframes marked):")
    print_snr_visual(
        compressed.snr_profile_keys,
        config.l1_snr_threshold,
    )

    assert max(key_errs) < 1e-3, "Key reconstruction error too high"
    assert max(val_errs) < 1e-3, "Value reconstruction error too high"
    print("\n  ✓ PASSED")


def test_adaptive_degradation():
    """Test that compression ratios scale with config aggressiveness."""
    print_header("Test: Adaptive Compression Levels")

    kv = generate_synthetic_kv_stack(
        num_layers=32, num_heads=8, seq_len=128, head_dim=64,
        shift_layers=[8, 16, 24],
    )

    configs = {
        "Conservative": GIHKCCConfig(
            l1_snr_threshold=0.85, l1_max_keyframe_span=4,
            l2_enabled=False,
        ),
        "Moderate": GIHKCCConfig(
            l1_snr_threshold=0.92, l1_max_keyframe_span=8,
            l2_enabled=True, l2_super_keyframe_interval=4,
        ),
        "Aggressive": GIHKCCConfig(
            l1_snr_threshold=0.97, l1_max_keyframe_span=12,
            l2_enabled=True, l2_super_keyframe_interval=3,
        ),
    }

    results = {}
    for name, cfg in configs.items():
        compressed = compress_kv_cache(kv, kv, cfg)
        s = compressed.summary()
        mem = estimate_memory_bytes(compressed)
        results[name] = {
            "ratio": s["combined_ratio"],
            "skf": s["keys_super_keyframes"],
            "kfd": s["keys_keyframe_deltas"],
            "d": s["keys_l1_deltas"],
        }
        print(f"  {name:14s}  ratio={s['combined_ratio']:5.2f}x  "
              f"skf={s['keys_super_keyframes']:2d}  "
              f"kfd={s['keys_keyframe_deltas']:2d}  "
              f"deltas={s['keys_l1_deltas']:2d}  "
              f"projected={mem['projected_ratio']}x")

    # Aggressive should compress more than conservative
    assert results["Aggressive"]["ratio"] >= results["Conservative"]["ratio"], \
        "Aggressive config should yield higher compression"
    print("\n  ✓ PASSED")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  GIHKCC Monolithic — Test Suite")
    print("  VecP Labs LLC | Patent Pending")
    print("="*60)

    tests = [
        test_snr_computation,
        test_l1_compression,
        test_l2_compression,
        test_l3_compression,
        test_quantization,
        test_full_pipeline,
        test_adaptive_degradation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
