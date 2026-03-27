#!/usr/bin/env python3
"""
GIHKCC Stack Tests — Tonight's Experiments

Two configurations:
  Config A: GIHKCC → TurboQuant (clean two-stage)
  Config B: GIHKCC → Per-Layer PCA → TurboQuant (full orthogonal stack)

Both tested against:
  1. Synthetic low-rank KV caches (runs anywhere, immediate validation)
  2. Real model KV caches (requires GPU + transformers, run on 4070 Ti)

Error is tracked at EVERY stage boundary so you can see exactly where
quality degrades and make informed tradeoffs.

Usage:
  python test_tonight.py                    # Synthetic only
  python test_tonight.py --model qwen       # + Qwen3-4B (Nicodemus base)
  python test_tonight.py --model llama      # + Llama 3.2 1B (quick test)
  python test_tonight.py --model <path>     # + any HF model

VecP Labs LLC | vecplabs.com | Patent Pending
"""

import sys
import time
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch

from gihkcc import (
    GIHKCCConfig,
    compress_kv_cache,
    decompress_kv_cache,
    compute_snr_profile,
    l1_compress,
    l1_decompress,
    l2_compress,
    l2_decompress,
)
from turboquant import (
    TurboQuantConfig,
    turboquant_compress,
    turboquant_decompress,
    turboquant_compress_list,
    turboquant_decompress_list,
)
from pca_layer import (
    PCAConfig,
    PCACompressedLayer,
    pca_compress_layer,
    pca_decompress_layer,
    pca_compress_layers,
    pca_decompress_layers,
)


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def measure_error(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """Compute reconstruction error metrics."""
    diff = (original.float() - reconstructed.float())
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    # Relative error
    orig_norm = original.float().norm().item()
    rel_err = diff.norm().item() / orig_norm if orig_norm > 0 else 0
    # PSNR (using original's range)
    signal_power = (original.float() ** 2).mean().item()
    psnr = 10 * math.log10(signal_power / mse) if mse > 0 else float("inf")

    return {
        "mse": mse,
        "mae": mae,
        "max_err": max_err,
        "rel_err": rel_err,
        "psnr_db": psnr,
    }


def measure_list_error(
    originals: List[torch.Tensor],
    reconstructeds: List[torch.Tensor],
) -> Dict[str, float]:
    """Average error metrics across a list of tensors."""
    all_metrics = [measure_error(o, r) for o, r in zip(originals, reconstructeds)]
    n = len(all_metrics)
    return {
        "mean_mse": sum(m["mse"] for m in all_metrics) / n,
        "mean_mae": sum(m["mae"] for m in all_metrics) / n,
        "max_err": max(m["max_err"] for m in all_metrics),
        "mean_rel_err": sum(m["rel_err"] for m in all_metrics) / n,
        "mean_psnr_db": sum(m["psnr_db"] for m in all_metrics) / n,
    }


def fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    else:
        return f"{b/1024**2:.1f} MB"


def fmt_ratio(r: float) -> str:
    return f"{r:.1f}x"


def print_stage(name: str, original_bytes: int, compressed_bytes: int,
                error: Dict[str, float], time_ms: float):
    """Print a stage summary line."""
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
    print(f"    {name:30s}  {fmt_bytes(original_bytes):>10s} → {fmt_bytes(compressed_bytes):>10s}  "
          f"({fmt_ratio(ratio):>6s})  "
          f"PSNR={error['mean_psnr_db']:6.1f}dB  "
          f"MAE={error['mean_mae']:.6f}  "
          f"MaxErr={error['max_err']:.6f}  "
          f"{time_ms:.0f}ms")


def print_header(title: str):
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print(f"{'═'*80}")


def print_subheader(title: str):
    print(f"\n  ── {title} {'─'*(60-len(title))}")


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic KV Cache Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_realistic_kv_cache(
    num_layers: int = 32,
    num_heads: int = 32,
    seq_len: int = 512,
    head_dim: int = 128,
    shift_layers: Optional[List[int]] = None,
    rank: int = 8,
    evolution_scale: float = 0.03,
    shift_scale: float = 0.3,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate synthetic KV cache with realistic properties:
    - Low-rank inter-layer evolution (simulates learned projections)
    - Representational shifts at boundary layers
    - Correlated features within each layer (for PCA to exploit)
    """
    torch.manual_seed(seed)

    if shift_layers is None:
        shift_layers = [i for i in range(num_layers) if i > 0 and i % 8 == 0]

    def _make_stack():
        base = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16)
        stack = [base.clone()]
        current = base.clone()

        for layer in range(1, num_layers):
            if layer in shift_layers:
                U = torch.randn(num_heads * seq_len, rank) * shift_scale
                V = torch.randn(rank, head_dim)
                shift = (U @ V).reshape(num_heads, seq_len, head_dim).to(torch.float16)
                current = current + shift
            else:
                U = torch.randn(num_heads * seq_len, rank) * evolution_scale
                V = torch.randn(rank, head_dim)
                delta = (U @ V).reshape(num_heads, seq_len, head_dim).to(torch.float16)
                noise = torch.randn_like(current) * (evolution_scale * 0.1)
                current = current + delta + noise
            stack.append(current.clone())

        return stack

    keys = _make_stack()
    torch.manual_seed(seed + 1000)
    values = _make_stack()

    return keys, values


# ═══════════════════════════════════════════════════════════════════════════
# Config A: GIHKCC → TurboQuant
# ═══════════════════════════════════════════════════════════════════════════

def test_config_a(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    label: str = "Synthetic",
) -> Dict[str, Any]:
    """
    Config A: GIHKCC (L1+L2 structural fold) → TurboQuant (3-bit quantization)

    Clean two-stage pipeline. GIHKCC removes structural redundancy,
    TurboQuant compresses the remaining precision.
    """
    print_header(f"CONFIG A: GIHKCC → TurboQuant [{label}]")

    num_layers = len(keys)
    per_layer_bytes = keys[0].nelement() * keys[0].element_size()
    total_original = per_layer_bytes * num_layers * 2  # keys + values

    print(f"\n    Input: {num_layers} layers, "
          f"{keys[0].shape[0]} heads, "
          f"{keys[0].shape[1]} tokens, "
          f"{keys[0].shape[2]} head_dim")
    print(f"    Original size: {fmt_bytes(total_original)}")

    gihkcc_config = GIHKCCConfig(
        l1_snr_threshold=0.92,
        l1_max_keyframe_span=8,
        l2_enabled=True,
        l2_super_keyframe_interval=4,
        l3_enabled=True,
        l3_token_keyframe_stride=8,
    )

    # ── Stage 0: L3 Inter-Token Fold ─────────────────────────────────

    l3_active = gihkcc_config.l3_enabled and keys[0].shape[1] > gihkcc_config.l3_token_keyframe_stride * 2
    l3_ratio_a = 1.0

    if l3_active:
        from gihkcc import l3_compress_layer, l3_decompress_layer
        stride = gihkcc_config.l3_token_keyframe_stride
        print_subheader(f"Stage 0: L3 Inter-Token Fold (stride={stride})")

        t_l3 = time.perf_counter()
        l3_k = [l3_compress_layer(k, stride=stride) for k in keys]
        l3_v = [l3_compress_layer(v, stride=stride) for v in values]
        t_l3 = (time.perf_counter() - t_l3) * 1000

        seq_len = keys[0].shape[1]
        kf_count = l3_k[0].token_keyframes.shape[0]
        delta_count = l3_k[0].token_deltas.shape[0]

        # Measure delta vs keyframe magnitudes
        kf_mag = sum(c.token_keyframes.float().abs().mean().item() for c in l3_k) / len(l3_k)
        delta_mag = sum(c.token_deltas.float().abs().mean().item() for c in l3_k) / len(l3_k) if delta_count > 0 else 0
        delta_reduction = kf_mag / delta_mag if delta_mag > 0 else float('inf')

        # Byte accounting: deltas are smaller magnitude, quantize tighter
        # Conservative: deltas cost 1/delta_reduction as much per element
        delta_cost = min(1.0, 1.0 / max(delta_reduction, 1.0) * 2)
        l3_ratio_a = seq_len / (kf_count + delta_count * delta_cost)

        print(f"    Tokens: {seq_len} → {kf_count} keyframes + {delta_count} deltas")
        print(f"    Keyframe mag: {kf_mag:.4f}  |  Delta mag: {delta_mag:.4f}  |  Reduction: {delta_reduction:.1f}x")
        print(f"    L3 est. ratio: {l3_ratio_a:.2f}x")
        print(f"    Time: {t_l3:.0f}ms")

        # Round-trip and transpose back to [heads, seq, dim]
        keys = [l3_decompress_layer(c).transpose(0, 1) for c in l3_k]
        values = [l3_decompress_layer(c).transpose(0, 1) for c in l3_v]

    # ── Stage 1: GIHKCC ──────────────────────────────────────────────────

    print_subheader("Stage 1: GIHKCC Structural Fold (L1+L2)")

    t0 = time.perf_counter()
    compressed = compress_kv_cache(keys, values, gihkcc_config)
    t_gihkcc = (time.perf_counter() - t0) * 1000

    # Decompress to measure error
    recon_keys, recon_values = decompress_kv_cache(compressed)
    gihkcc_error = {
        "keys": measure_list_error(keys, recon_keys),
        "values": measure_list_error(values, recon_values),
    }

    # Collect all tensors that need further compression:
    # - Super-keyframes (full size)
    # - Keyframe deltas (from L2)
    # - Layer deltas (from L1)
    all_tensors_to_compress = []
    tensor_labels = []

    for skf in compressed.keys_l2.super_keyframes:
        all_tensors_to_compress.append(skf.data)
        tensor_labels.append(f"key_skf_L{skf.layer_idx}")
    for kfd in compressed.keys_l2.keyframe_deltas:
        all_tensors_to_compress.append(kfd.delta)
        tensor_labels.append(f"key_kfd_L{kfd.layer_idx}")
    for d in compressed.keys_l2.l1_deltas:
        all_tensors_to_compress.append(d.delta)
        tensor_labels.append(f"key_delta_L{d.layer_idx}")

    for skf in compressed.values_l2.super_keyframes:
        all_tensors_to_compress.append(skf.data)
        tensor_labels.append(f"val_skf_L{skf.layer_idx}")
    for kfd in compressed.values_l2.keyframe_deltas:
        all_tensors_to_compress.append(kfd.delta)
        tensor_labels.append(f"val_kfd_L{kfd.layer_idx}")
    for d in compressed.values_l2.l1_deltas:
        all_tensors_to_compress.append(d.delta)
        tensor_labels.append(f"val_delta_L{d.layer_idx}")

    gihkcc_bytes = sum(t.nelement() * t.element_size() for t in all_tensors_to_compress)

    n_skf = len(compressed.keys_l2.super_keyframes) + len(compressed.values_l2.super_keyframes)
    n_kfd = len(compressed.keys_l2.keyframe_deltas) + len(compressed.values_l2.keyframe_deltas)
    n_delta = len(compressed.keys_l2.l1_deltas) + len(compressed.values_l2.l1_deltas)
    gihkcc_ratio = total_original / gihkcc_bytes if gihkcc_bytes > 0 else 1

    # Combine key/value error for display
    combined_err = {
        "mean_psnr_db": (gihkcc_error["keys"]["mean_psnr_db"] + gihkcc_error["values"]["mean_psnr_db"]) / 2,
        "mean_mae": (gihkcc_error["keys"]["mean_mae"] + gihkcc_error["values"]["mean_mae"]) / 2,
        "max_err": max(gihkcc_error["keys"]["max_err"], gihkcc_error["values"]["max_err"]),
    }

    print(f"    Super-keyframes: {n_skf}  |  KF-deltas: {n_kfd}  |  L1-deltas: {n_delta}")
    print(f"    Structural: {fmt_bytes(total_original)} → {fmt_bytes(gihkcc_bytes)}  ({fmt_ratio(gihkcc_ratio)})")
    print(f"    Error: PSNR={combined_err['mean_psnr_db']:.1f}dB  MAE={combined_err['mean_mae']:.6f}  Max={combined_err['max_err']:.6f}")
    print(f"    Time: {t_gihkcc:.0f}ms")

    # ── Stage 2: TurboQuant ──────────────────────────────────────────────

    print_subheader("Stage 2: TurboQuant (3-bit quantization)")

    tq_config = TurboQuantConfig(
        target_bits=3,
        qjl_enabled=True,
        rotation_seed=42,
    )

    t1 = time.perf_counter()
    tq_compressed, tq_stats = turboquant_compress_list(all_tensors_to_compress, tq_config)
    t_tq = (time.perf_counter() - t1) * 1000

    # Decompress to measure error
    t2 = time.perf_counter()
    tq_recon = turboquant_decompress_list(tq_compressed)
    t_tq_decomp = (time.perf_counter() - t2) * 1000

    # Error from TurboQuant stage alone (input vs output of this stage)
    tq_stage_error = measure_list_error(all_tensors_to_compress, tq_recon)

    tq_bytes = tq_stats["compressed_bytes"]
    tq_ratio = gihkcc_bytes / tq_bytes if tq_bytes > 0 else 1

    print(f"    Tensors: {tq_stats['num_tensors']}  |  Bits: {tq_config.target_bits}  |  QJL: {tq_config.qjl_enabled}")
    print(f"    Stage:  {fmt_bytes(gihkcc_bytes)} → {fmt_bytes(tq_bytes)}  ({fmt_ratio(tq_ratio)})")
    print(f"    Error:  PSNR={tq_stage_error['mean_psnr_db']:.1f}dB  MAE={tq_stage_error['mean_mae']:.6f}  Max={tq_stage_error['max_err']:.6f}")
    print(f"    Time: {t_tq:.0f}ms compress, {t_tq_decomp:.0f}ms decompress")

    # ── End-to-End ───────────────────────────────────────────────────────

    print_subheader("End-to-End: Original → GIHKCC → TurboQuant → Reconstruct")

    # Full reconstruction: TQ decompress → re-inject into GIHKCC structure → GIHKCC decompress
    # For error measurement, compare original KV against full round-trip
    # The simplest approach: measure error of TQ output vs original tensors,
    # which captures GIHKCC loss + TQ loss combined

    # We need to reconstruct the full KV cache from the TQ-decompressed tensors
    # by re-injecting them into the GIHKCC structure
    idx = 0
    # Re-inject into compressed structure
    for skf in compressed.keys_l2.super_keyframes:
        skf.data = tq_recon[idx].to(skf.data.dtype)
        idx += 1
    for kfd in compressed.keys_l2.keyframe_deltas:
        kfd.delta = tq_recon[idx].to(kfd.delta.dtype)
        idx += 1
    for d in compressed.keys_l2.l1_deltas:
        d.delta = tq_recon[idx].to(d.delta.dtype)
        idx += 1
    for skf in compressed.values_l2.super_keyframes:
        skf.data = tq_recon[idx].to(skf.data.dtype)
        idx += 1
    for kfd in compressed.values_l2.keyframe_deltas:
        kfd.delta = tq_recon[idx].to(kfd.delta.dtype)
        idx += 1
    for d in compressed.values_l2.l1_deltas:
        d.delta = tq_recon[idx].to(d.delta.dtype)
        idx += 1

    # Now decompress the modified GIHKCC cache
    e2e_keys, e2e_values = decompress_kv_cache(compressed)

    e2e_error_keys = measure_list_error(keys, e2e_keys)
    e2e_error_values = measure_list_error(values, e2e_values)
    e2e_combined = {
        "mean_psnr_db": (e2e_error_keys["mean_psnr_db"] + e2e_error_values["mean_psnr_db"]) / 2,
        "mean_mae": (e2e_error_keys["mean_mae"] + e2e_error_values["mean_mae"]) / 2,
        "max_err": max(e2e_error_keys["max_err"], e2e_error_values["max_err"]),
    }

    combined_ratio = total_original / tq_bytes if tq_bytes > 0 else 1
    # Factor in L3 token-level compression
    combined_with_l3 = combined_ratio * l3_ratio_a if l3_active else combined_ratio

    l3_label = f" × L3 {fmt_ratio(l3_ratio_a)}" if l3_active else ""
    print(f"\n    ┌───────────────────────────────────────────────────────────┐")
    print(f"    │  COMBINED: {fmt_bytes(total_original)} → {fmt_bytes(tq_bytes)}  =  {fmt_ratio(combined_ratio)} (storage)  │")
    if l3_active:
        print(f"    │  With L3 token fold factor: {fmt_ratio(combined_with_l3)} (effective)       │")
    print(f"    │  L1/L2 {fmt_ratio(gihkcc_ratio)} × TQ {fmt_ratio(tq_ratio)}{l3_label}              │")
    print(f"    │  PSNR: {e2e_combined['mean_psnr_db']:.1f} dB                                      │")
    print(f"    │  MAE:  {e2e_combined['mean_mae']:.6f}                                  │")
    print(f"    │  Max:  {e2e_combined['max_err']:.6f}                                  │")
    print(f"    └───────────────────────────────────────────────────────────┘")

    return {
        "config": "A",
        "total_original": total_original,
        "gihkcc_bytes": gihkcc_bytes,
        "final_bytes": tq_bytes,
        "gihkcc_ratio": gihkcc_ratio,
        "tq_ratio": tq_ratio,
        "l3_ratio": l3_ratio_a if l3_active else 1.0,
        "combined_ratio": combined_ratio,
        "combined_with_l3": combined_with_l3,
        "e2e_error": e2e_combined,
        "gihkcc_time_ms": t_gihkcc,
        "tq_time_ms": t_tq,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Config B: GIHKCC → Per-Layer PCA → TurboQuant
# ═══════════════════════════════════════════════════════════════════════════

def test_config_b(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    label: str = "Synthetic",
) -> Dict[str, Any]:
    """
    Config B: GIHKCC → Per-Layer PCA → TurboQuant

    Three-stage pipeline. GIHKCC handles depth axis, PCA handles feature
    axis, TurboQuant handles precision. All three target different
    redundancy sources.
    """
    print_header(f"CONFIG B: GIHKCC → PCA → TurboQuant [{label}]")

    num_layers = len(keys)
    per_layer_bytes = keys[0].nelement() * keys[0].element_size()
    total_original = per_layer_bytes * num_layers * 2

    print(f"\n    Input: {num_layers} layers, {keys[0].shape}")
    print(f"    Original size: {fmt_bytes(total_original)}")

    gihkcc_config = GIHKCCConfig(
        l1_snr_threshold=0.92,
        l1_max_keyframe_span=8,
        l2_enabled=True,
        l2_super_keyframe_interval=4,
        l3_enabled=True,
        l3_token_keyframe_stride=8,
    )

    # ── Stage 0: L3 Inter-Token Fold ─────────────────────────────────

    l3_active = gihkcc_config.l3_enabled and keys[0].shape[1] > gihkcc_config.l3_token_keyframe_stride * 2
    l3_ratio_b = 1.0

    if l3_active:
        from gihkcc import l3_compress_layer, l3_decompress_layer
        stride = gihkcc_config.l3_token_keyframe_stride
        print_subheader(f"Stage 0: L3 Inter-Token Fold (stride={stride})")

        t_l3 = time.perf_counter()
        l3_k = [l3_compress_layer(k, stride=stride) for k in keys]
        l3_v = [l3_compress_layer(v, stride=stride) for v in values]
        t_l3 = (time.perf_counter() - t_l3) * 1000

        seq_len = keys[0].shape[1]
        kf_count = l3_k[0].token_keyframes.shape[0]
        delta_count = l3_k[0].token_deltas.shape[0]

        kf_mag = sum(c.token_keyframes.float().abs().mean().item() for c in l3_k) / len(l3_k)
        delta_mag = sum(c.token_deltas.float().abs().mean().item() for c in l3_k) / len(l3_k) if delta_count > 0 else 0
        delta_reduction = kf_mag / delta_mag if delta_mag > 0 else float('inf')

        delta_cost = min(1.0, 1.0 / max(delta_reduction, 1.0) * 2)
        l3_ratio_b = seq_len / (kf_count + delta_count * delta_cost)

        print(f"    Tokens: {seq_len} → {kf_count} keyframes + {delta_count} deltas")
        print(f"    Keyframe mag: {kf_mag:.4f}  |  Delta mag: {delta_mag:.4f}  |  Reduction: {delta_reduction:.1f}x")
        print(f"    L3 est. ratio: {l3_ratio_b:.2f}x")
        print(f"    Time: {t_l3:.0f}ms")

        keys = [l3_decompress_layer(c).transpose(0, 1) for c in l3_k]
        values = [l3_decompress_layer(c).transpose(0, 1) for c in l3_v]

    # ── Stage 1: GIHKCC ──────────────────────────────────────────────────

    print_subheader("Stage 1: GIHKCC Structural Fold (L1+L2)")

    t0 = time.perf_counter()
    compressed = compress_kv_cache(keys, values, gihkcc_config)
    t_gihkcc = (time.perf_counter() - t0) * 1000

    # Collect tensors
    all_tensors = []
    for skf in compressed.keys_l2.super_keyframes:
        all_tensors.append(("key_skf", skf.data))
    for kfd in compressed.keys_l2.keyframe_deltas:
        all_tensors.append(("key_kfd", kfd.delta))
    for d in compressed.keys_l2.l1_deltas:
        all_tensors.append(("key_d", d.delta))
    for skf in compressed.values_l2.super_keyframes:
        all_tensors.append(("val_skf", skf.data))
    for kfd in compressed.values_l2.keyframe_deltas:
        all_tensors.append(("val_kfd", kfd.delta))
    for d in compressed.values_l2.l1_deltas:
        all_tensors.append(("val_d", d.delta))

    gihkcc_bytes = sum(t.nelement() * t.element_size() for _, t in all_tensors)
    gihkcc_ratio = total_original / gihkcc_bytes if gihkcc_bytes > 0 else 1

    print(f"    Tensors: {len(all_tensors)}  |  {fmt_bytes(total_original)} → {fmt_bytes(gihkcc_bytes)}  ({fmt_ratio(gihkcc_ratio)})")
    print(f"    Time: {t_gihkcc:.0f}ms")

    # ── Stage 2: Per-Layer PCA ───────────────────────────────────────────

    print_subheader("Stage 2: Per-Layer PCA (feature decorrelation)")

    pca_config = PCAConfig(
        variance_threshold=0.95,
        min_components=8,
        per_head=True,
    )

    t1 = time.perf_counter()
    pca_compressed = []
    for label_t, tensor in all_tensors:
        pc = pca_compress_layer(tensor, pca_config)
        pca_compressed.append((label_t, pc))
    t_pca = (time.perf_counter() - t1) * 1000

    # PCA compressed size = coefficients (reduced dim) at original dtype
    pca_bytes = sum(pc.compressed_bytes for _, pc in pca_compressed)
    pca_ratio = gihkcc_bytes / pca_bytes if pca_bytes > 0 else 1

    # PCA reconstruction error (stage-level)
    pca_recon = [pca_decompress_layer(pc) for _, pc in pca_compressed]
    pca_originals = [t for _, t in all_tensors]
    pca_error = measure_list_error(pca_originals, pca_recon)

    mean_dim_red = sum(pc.dimensionality_reduction for _, pc in pca_compressed) / len(pca_compressed)

    print(f"    Dim reduction: {mean_dim_red:.1f}x mean  |  Variance retained: {pca_config.variance_threshold:.0%}")
    print(f"    Stage: {fmt_bytes(gihkcc_bytes)} → {fmt_bytes(pca_bytes)}  ({fmt_ratio(pca_ratio)})")
    print(f"    Error: PSNR={pca_error['mean_psnr_db']:.1f}dB  MAE={pca_error['mean_mae']:.6f}")
    print(f"    Time: {t_pca:.0f}ms")

    # ── Stage 3: TurboQuant ──────────────────────────────────────────────

    print_subheader("Stage 3: TurboQuant (3-bit quantization)")

    tq_config = TurboQuantConfig(target_bits=3, qjl_enabled=True)

    # TurboQuant on PCA coefficients
    pca_coefficients = [pc.coefficients for _, pc in pca_compressed]

    t2 = time.perf_counter()
    tq_compressed, tq_stats = turboquant_compress_list(pca_coefficients, tq_config)
    t_tq = (time.perf_counter() - t2) * 1000

    tq_recon = turboquant_decompress_list(tq_compressed)
    tq_stage_error = measure_list_error(pca_coefficients, tq_recon)

    tq_bytes = tq_stats["compressed_bytes"]
    tq_ratio = pca_bytes / tq_bytes if tq_bytes > 0 else 1

    print(f"    Stage: {fmt_bytes(pca_bytes)} → {fmt_bytes(tq_bytes)}  ({fmt_ratio(tq_ratio)})")
    print(f"    Error: PSNR={tq_stage_error['mean_psnr_db']:.1f}dB  MAE={tq_stage_error['mean_mae']:.6f}")
    print(f"    Time: {t_tq:.0f}ms")

    # ── End-to-End ───────────────────────────────────────────────────────

    print_subheader("End-to-End: Original → GIHKCC → PCA → TQ → Reconstruct")

    # Full round-trip: TQ decompress → PCA reconstruct → re-inject into GIHKCC → GIHKCC decompress
    # Step 1: TQ → PCA coefficients (already done: tq_recon)
    # Step 2: PCA coefficients → layer tensors
    e2e_tensors = []
    for i, ((label_t, pc), tq_coeffs) in enumerate(zip(pca_compressed, tq_recon)):
        # Replace coefficients with TQ-reconstructed ones
        pc_copy = PCACompressedLayer(
            coefficients=tq_coeffs.to(pc.coefficients.dtype),
            basis=pc.basis,
            all_bases=pc.all_bases,
            original_shape=pc.original_shape,
            original_dtype=pc.original_dtype,
            per_head=pc.per_head,
        )
        recon = pca_decompress_layer(pc_copy)
        e2e_tensors.append(recon)

    # Step 3: Re-inject into GIHKCC structure
    idx = 0
    for skf in compressed.keys_l2.super_keyframes:
        skf.data = e2e_tensors[idx].to(skf.data.dtype)
        idx += 1
    for kfd in compressed.keys_l2.keyframe_deltas:
        kfd.delta = e2e_tensors[idx].to(kfd.delta.dtype)
        idx += 1
    for d in compressed.keys_l2.l1_deltas:
        d.delta = e2e_tensors[idx].to(d.delta.dtype)
        idx += 1
    for skf in compressed.values_l2.super_keyframes:
        skf.data = e2e_tensors[idx].to(skf.data.dtype)
        idx += 1
    for kfd in compressed.values_l2.keyframe_deltas:
        kfd.delta = e2e_tensors[idx].to(kfd.delta.dtype)
        idx += 1
    for d in compressed.values_l2.l1_deltas:
        d.delta = e2e_tensors[idx].to(d.delta.dtype)
        idx += 1

    # Step 4: GIHKCC decompress
    e2e_keys, e2e_values = decompress_kv_cache(compressed)

    e2e_error_keys = measure_list_error(keys, e2e_keys)
    e2e_error_values = measure_list_error(values, e2e_values)
    e2e_combined = {
        "mean_psnr_db": (e2e_error_keys["mean_psnr_db"] + e2e_error_values["mean_psnr_db"]) / 2,
        "mean_mae": (e2e_error_keys["mean_mae"] + e2e_error_values["mean_mae"]) / 2,
        "max_err": max(e2e_error_keys["max_err"], e2e_error_values["max_err"]),
    }

    combined_ratio = total_original / tq_bytes if tq_bytes > 0 else 1
    combined_with_l3 = combined_ratio * l3_ratio_b if l3_active else combined_ratio

    l3_label = f" × L3 {fmt_ratio(l3_ratio_b)}" if l3_active else ""
    print(f"\n    ┌───────────────────────────────────────────────────────────┐")
    print(f"    │  COMBINED: {fmt_bytes(total_original)} → {fmt_bytes(tq_bytes)}  =  {fmt_ratio(combined_ratio)} (storage)  │")
    if l3_active:
        print(f"    │  With L3 token fold factor: {fmt_ratio(combined_with_l3)} (effective)       │")
    print(f"    │  L1/L2 {fmt_ratio(gihkcc_ratio)} × PCA {fmt_ratio(pca_ratio)} × TQ {fmt_ratio(tq_ratio)}{l3_label}  │")
    print(f"    │  PSNR: {e2e_combined['mean_psnr_db']:.1f} dB                                      │")
    print(f"    │  MAE:  {e2e_combined['mean_mae']:.6f}                                  │")
    print(f"    │  Max:  {e2e_combined['max_err']:.6f}                                  │")
    print(f"    └───────────────────────────────────────────────────────────┘")

    return {
        "config": "B",
        "total_original": total_original,
        "gihkcc_bytes": gihkcc_bytes,
        "pca_bytes": pca_bytes,
        "final_bytes": tq_bytes,
        "gihkcc_ratio": gihkcc_ratio,
        "pca_ratio": pca_ratio,
        "tq_ratio": tq_ratio,
        "l3_ratio": l3_ratio_b if l3_active else 1.0,
        "combined_ratio": combined_ratio,
        "combined_with_l3": combined_with_l3,
        "e2e_error": e2e_combined,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Real Model Testing
# ═══════════════════════════════════════════════════════════════════════════

def extract_real_kv_cache(model_name: str, prompt: str = None):
    """
    Extract KV cache from a real HuggingFace model.
    Returns (keys, values) as lists of tensors per layer.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("    ✗ transformers not installed. Run: pip install transformers")
        return None, None

    model_map = {
        "qwen": "Qwen/Qwen2.5-1.5B",
        "llama": "meta-llama/Llama-3.2-1B",
    }
    actual_name = model_map.get(model_name, model_name)

    if prompt is None:
        base = (
            "The transformer architecture revolutionized natural language processing "
            "by introducing self-attention mechanisms that allow models to weigh the "
            "importance of different parts of the input sequence. Key-value caches "
            "store previously computed attention states, enabling efficient "
            "autoregressive generation without redundant computation. However, as "
            "context windows grow to hundreds of thousands of tokens, the memory "
            "required for these caches becomes a significant bottleneck, motivating "
            "research into compression techniques that can reduce this footprint "
            "while preserving model output quality. "
        )
        # Tile to ~3.5K tokens
        prompt = base * 40

    print(f"    Loading model: {actual_name}")
    tokenizer = AutoTokenizer.from_pretrained(actual_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        actual_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"    Tokenizing ({len(prompt)} chars)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    print(f"    Sequence length: {seq_len} tokens")

    print(f"    Forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    past_kv = outputs.past_key_values
    keys, values = [], []

    # DynamicCache (transformers >= 4.36) — has key_cache/value_cache attrs
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i in range(len(past_kv.key_cache)):
            keys.append(past_kv.key_cache[i][0].cpu())
            values.append(past_kv.value_cache[i][0].cpu())
    # DynamicCache that acts as a list of tuples — try indexing
    elif hasattr(past_kv, "__getitem__"):
        # Probe the first element to figure out the format
        first = past_kv[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            # Tuple of (key, value, ...) per layer — possibly extra elements
            for layer_kv in past_kv:
                keys.append(layer_kv[0][0].cpu())  # key: [batch, heads, seq, dim] → remove batch
                values.append(layer_kv[1][0].cpu())
        elif hasattr(first, "shape"):
            # Some formats return alternating k,v tensors
            print(f"    WARNING: Unknown KV format, first element shape: {first.shape}")
            raise ValueError(f"Unrecognized KV cache format: {type(first)}")
        else:
            raise ValueError(f"Unrecognized KV cache format: {type(first)}")
    else:
        # Last resort: try to iterate and take first two from each
        print(f"    KV cache type: {type(past_kv)}")
        print(f"    Attempting generic extraction...")
        for layer_kv in past_kv:
            layer_list = list(layer_kv)
            keys.append(layer_list[0][0].cpu())
            values.append(layer_list[1][0].cpu())

    print(f"    Extracted: {len(keys)} layers, shape {keys[0].shape}")
    return keys, values


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GIHKCC Stack Tests")
    parser.add_argument("--model", type=str, default=None,
                        help="HF model: 'qwen', 'llama', or a model path")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    print("\n" + "█"*80)
    print("█  GIHKCC COMPRESSION STACK — TONIGHT'S EXPERIMENTS")
    print("█  VecP Labs LLC | Patent Pending")
    print("█"*80)

    # ── Synthetic Tests ──────────────────────────────────────────────────

    print(f"\n  Generating synthetic KV cache: {args.layers}L × {args.heads}H × {args.seq_len}T × {args.head_dim}D")
    keys, values = generate_realistic_kv_cache(
        num_layers=args.layers,
        num_heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        rank=6,
    )

    result_a = test_config_a(keys, values, label="Synthetic")

    # Regenerate because Config A mutates the compressed structure
    keys, values = generate_realistic_kv_cache(
        num_layers=args.layers,
        num_heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        rank=6,
    )

    result_b = test_config_b(keys, values, label="Synthetic")

    # ── Real Model Tests ─────────────────────────────────────────────────

    result_a_real = None
    result_b_real = None

    if args.model:
        print_header(f"REAL MODEL: {args.model}")

        real_keys, real_values = extract_real_kv_cache(args.model)

        if real_keys is not None:
            # Clone for Config B before Config A mutates them
            real_keys_b = [k.clone() for k in real_keys]
            real_values_b = [v.clone() for v in real_values]

            result_a_real = test_config_a(real_keys, real_values, label=args.model)
            result_b_real = test_config_b(real_keys_b, real_values_b, label=args.model)

    # ── Summary ──────────────────────────────────────────────────────────

    print_header("SUMMARY")

    all_results = [("Synth A", result_a), ("Synth B", result_b)]
    if result_a_real is not None:
        all_results.append(("Real A", result_a_real))
    if result_b_real is not None:
        all_results.append(("Real B", result_b_real))

    print(f"\n    {'Config':<12s} {'Original':>12s} {'Final':>12s} {'Ratio':>8s} {'w/ L3':>8s} {'PSNR':>8s} {'MAE':>10s}")
    print(f"    {'─'*72}")

    for label, r in all_results:
        l3r = r.get('combined_with_l3', r['combined_ratio'])
        print(f"    {label:<12s} "
              f"{fmt_bytes(r['total_original']):>12s} "
              f"{fmt_bytes(r['final_bytes']):>12s} "
              f"{fmt_ratio(r['combined_ratio']):>8s} "
              f"{fmt_ratio(l3r):>8s} "
              f"{r['e2e_error']['mean_psnr_db']:>7.1f}dB "
              f"{r['e2e_error']['mean_mae']:>10.6f}")

    print(f"\n    Config A: L3 → GIHKCC L1/L2 → TurboQuant")
    print(f"    Config B: L3 → GIHKCC L1/L2 → PCA → TurboQuant")

    print(f"\n{'█'*80}")
    print(f"█  DONE. Compare PSNR and ratio to decide which stack ships.")
    print(f"█  >30dB PSNR = visually lossless for most generation tasks.")
    print(f"█  >20dB PSNR = acceptable for non-critical workloads.")
    print(f"{'█'*80}\n")


if __name__ == "__main__":
    main()
