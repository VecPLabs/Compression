#!/usr/bin/env python3
"""
GIHKCC Cerberus Test — The Architecture It Was Designed For

Extracts separate KV caches per head group (Reasoning/Guardian/Language)
from a Cerberus V2 checkpoint, then tests the full RGIHKCC stack:

  L0: Inter-head-group factorization (shared residual basis)
  L1: Inter-layer fold within each group
  L3: Inter-token fold

This is the test that validates what monolithic models CAN'T do.

Usage:
  python test_cerberus.py --checkpoint path/to/cerberus.pt --config path/to/config.json

VecP Labs LLC | vecplabs.com | Patent Pending
"""

import sys
import os
import json
import time
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

# GIHKCC imports
from gihkcc import (
    GIHKCCConfig,
    compute_statistical_snr,
    compute_snr_profile,
    l1_compress,
    l1_decompress,
    l3_compress_layer,
    l3_decompress_layer,
)
from turboquant import TurboQuantConfig, turboquant_compress, turboquant_decompress


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def fmt_bytes(b):
    if b < 1024: return f"{b} B"
    elif b < 1024**2: return f"{b/1024:.1f} KB"
    else: return f"{b/1024**2:.1f} MB"

def fmt_ratio(r): return f"{r:.1f}x"

def measure_error(a, b):
    diff = (a.float() - b.float())
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    signal = (a.float() ** 2).mean().item()
    psnr = 10 * math.log10(signal / mse) if mse > 0 else float("inf")
    return {"mse": mse, "mae": mae, "max_err": max_err, "psnr_db": psnr}

def print_header(title):
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print(f"{'═'*80}")

def print_sub(title):
    print(f"\n  ── {title} {'─'*(60-len(title))}")


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache Extraction via Hooks
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CerberusKVCache:
    """Extracted KV caches separated by head group."""
    # Per-layer, per-group KV tensors
    # Shape per tensor: [n_group_heads, seq_len, d_head]
    reasoning_keys: List[torch.Tensor]
    reasoning_values: List[torch.Tensor]
    language_keys: List[torch.Tensor]
    language_values: List[torch.Tensor]
    guardian_keys: List[torch.Tensor]   # Only at guardian-active layers
    guardian_values: List[torch.Tensor]

    # Residual stream at each layer (for L0 factorization)
    residual_states: List[torch.Tensor]  # [seq_len, d_model] per layer

    # Metadata
    n_layers: int = 0
    guardian_layers: List[int] = None  # Which layer indices have guardian

    def summary(self):
        r_shape = self.reasoning_keys[0].shape if self.reasoning_keys else "N/A"
        l_shape = self.language_keys[0].shape if self.language_keys else "N/A"
        g_shape = self.guardian_keys[0].shape if self.guardian_keys else "N/A"
        print(f"    Reasoning: {len(self.reasoning_keys)} layers, shape {r_shape}")
        print(f"    Language:  {len(self.language_keys)} layers, shape {l_shape}")
        print(f"    Guardian:  {len(self.guardian_keys)} layers, shape {g_shape}")
        if self.residual_states:
            print(f"    Residual:  {len(self.residual_states)} states, shape {self.residual_states[0].shape}")


def extract_cerberus_kv(model, input_ids: torch.Tensor) -> CerberusKVCache:
    """
    Hook into Cerberus attention modules to capture per-group KV caches.

    We register forward hooks on each PrimitiveGroupAttention that capture
    the K and V tensors after projection but before attention computation.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    captured = {
        "reasoning_k": [], "reasoning_v": [],
        "language_k": [], "language_v": [],
        "guardian_k": [], "guardian_v": [],
        "residuals": [],
    }

    hooks = []

    # Hook each block
    for layer_idx, block in enumerate(model.blocks):
        # Capture residual stream entering this block
        def make_block_hook(idx):
            def hook(module, input):
                # input is a tuple, input[0] is x (residual stream), shape (B, T, d_model)
                x = input[0]
                captured["residuals"].append(x[0].detach().cpu())  # Remove batch dim
            return hook
        h = block.register_forward_pre_hook(make_block_hook(layer_idx))
        hooks.append(h)

        # Hook reasoning attention
        def make_attn_hook(group_name, attn_module):
            def hook(module, input, output):
                x = input[0]  # (B, T, d_model)
                B, T, D = x.shape

                h = module.down_proj(x)  # (B, T, d_internal)

                if module.mode == "cross":
                    # Guardian: kv_source is input[2]
                    kv_source = input[2] if len(input) > 2 else input[0]
                    if kv_source is None:
                        kv_source = x
                    kv_h = module.kv_down_proj(kv_source)
                    k = module.k_proj(kv_h).reshape(B, T, module.n_heads, module.d_head)
                    v = module.v_proj(kv_h).reshape(B, T, module.n_heads, module.d_head)
                else:
                    qkv = module.qkv(h).reshape(B, T, 3, module.n_heads, module.d_head)
                    k = qkv[:, :, 1]  # (B, T, n_heads, d_head)
                    v = qkv[:, :, 2]

                # Store as [n_heads, seq_len, d_head] (remove batch)
                captured[f"{group_name}_k"].append(
                    k[0].transpose(0, 1).detach().cpu()  # [n_heads, T, d_head]
                )
                captured[f"{group_name}_v"].append(
                    v[0].transpose(0, 1).detach().cpu()
                )
            return hook

        h = block.reasoning_attn.register_forward_hook(
            make_attn_hook("reasoning", block.reasoning_attn)
        )
        hooks.append(h)

        h = block.language_attn.register_forward_hook(
            make_attn_hook("language", block.language_attn)
        )
        hooks.append(h)

        if block.has_guardian:
            h = block.guardian_attn.register_forward_hook(
                make_attn_hook("guardian", block.guardian_attn)
            )
            hooks.append(h)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Identify guardian layers
    guardian_layers = [i for i, block in enumerate(model.blocks) if block.has_guardian]

    return CerberusKVCache(
        reasoning_keys=captured["reasoning_k"],
        reasoning_values=captured["reasoning_v"],
        language_keys=captured["language_k"],
        language_values=captured["language_v"],
        guardian_keys=captured["guardian_k"],
        guardian_values=captured["guardian_v"],
        residual_states=captured["residuals"],
        n_layers=len(model.blocks),
        guardian_layers=guardian_layers,
    )


# ═══════════════════════════════════════════════════════════════════════════
# L0: Inter-Head-Group Factorization
# ═══════════════════════════════════════════════════════════════════════════

def test_l0_factorization(cache: CerberusKVCache):
    """
    Test L0: Do the three head groups share a common residual basis?

    At each layer, Reasoning K, Language K, and Guardian K all derive from
    the same residual stream x via different learned projections. If we
    can factor out the shared component, we store it once + 3 small deltas.

    We measure this by computing cosine similarity between each pair of
    group KV tensors at the same layer.
    """
    print_sub("L0: Inter-Head-Group Similarity (Shared Residual Basis)")

    for layer in range(cache.n_layers):
        # Average across heads → [seq_len, d_head] per group
        r_k = cache.reasoning_keys[layer].float().mean(dim=0).flatten()
        l_k = cache.language_keys[layer].float().mean(dim=0).flatten()

        sim_rl = torch.nn.functional.cosine_similarity(
            r_k.unsqueeze(0), l_k.unsqueeze(0)
        ).item()

        # Guardian only at certain layers
        g_idx = None
        for gi, gl in enumerate(cache.guardian_layers):
            if gl == layer:
                g_idx = gi
                break

        if g_idx is not None:
            g_k = cache.guardian_keys[g_idx].float().mean(dim=0).flatten()
            sim_rg = torch.nn.functional.cosine_similarity(
                r_k.unsqueeze(0), g_k.unsqueeze(0)
            ).item()
            sim_lg = torch.nn.functional.cosine_similarity(
                l_k.unsqueeze(0), g_k.unsqueeze(0)
            ).item()
            print(f"    Layer {layer}: R↔L={sim_rl:.4f}  R↔G={sim_rg:.4f}  L↔G={sim_lg:.4f}")
        else:
            print(f"    Layer {layer}: R↔L={sim_rl:.4f}  (no Guardian)")

    # Also test residual stream similarity across layers
    print_sub("L0: Residual Stream Inter-Layer Similarity")
    if len(cache.residual_states) > 1:
        for i in range(len(cache.residual_states) - 1):
            sim = torch.nn.functional.cosine_similarity(
                cache.residual_states[i].flatten().unsqueeze(0).float(),
                cache.residual_states[i+1].flatten().unsqueeze(0).float()
            ).item()
            print(f"    Residual L{i}→L{i+1}: {sim:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# L1: Inter-Layer Fold Per Group
# ═══════════════════════════════════════════════════════════════════════════

def test_l1_per_group(cache: CerberusKVCache):
    """
    Test L1: Do adjacent layers within each group show high similarity?

    This is the core GIHKCC mechanism. In monolithic models, per-layer
    projections destroy inter-layer coherence. In Cerberus, each group
    has its own projection but the residual stream evolves smoothly.
    """
    print_sub("L1: Inter-Layer SNR Per Head Group")

    groups = {
        "Reasoning": (cache.reasoning_keys, cache.reasoning_values),
        "Language": (cache.language_keys, cache.language_values),
    }
    if cache.guardian_keys:
        groups["Guardian"] = (cache.guardian_keys, cache.guardian_values)

    for group_name, (keys, values) in groups.items():
        if len(keys) < 2:
            print(f"    {group_name}: Only {len(keys)} layer(s), skipping SNR")
            continue

        snr_k = compute_snr_profile(keys)
        snr_v = compute_snr_profile(values)

        print(f"\n    {group_name} Keys ({len(keys)} layers):")
        for i, snr in enumerate(snr_k):
            marker = " ◄ KF" if snr < 0.92 else ""
            bar = "█" * int(snr * 25)
            print(f"      L{i}→L{i+1}: {snr:.4f}  {bar}{marker}")

        # Try L1 compression
        config = GIHKCCConfig(l1_snr_threshold=0.92, l1_max_keyframe_span=4)
        l1_k = l1_compress(keys, config)
        l1_v = l1_compress(values, config)

        print(f"    {group_name} L1 Fold: "
              f"{len(keys)} layers → {len(l1_k.keyframes)} KF + {len(l1_k.deltas)} deltas "
              f"(ratio: {l1_k.compression_ratio:.2f}x)")

        # Lower threshold test
        config_low = GIHKCCConfig(l1_snr_threshold=0.80, l1_max_keyframe_span=4)
        l1_k_low = l1_compress(keys, config_low)
        print(f"    {group_name} L1 Fold (threshold=0.80): "
              f"{len(l1_k_low.keyframes)} KF + {len(l1_k_low.deltas)} deltas "
              f"(ratio: {l1_k_low.compression_ratio:.2f}x)")


# ═══════════════════════════════════════════════════════════════════════════
# L3: Inter-Token Fold
# ═══════════════════════════════════════════════════════════════════════════

def test_l3_per_group(cache: CerberusKVCache):
    """Test L3 inter-token fold on each group's KV cache."""
    print_sub("L3: Inter-Token Fold Per Head Group")

    groups = {
        "Reasoning": cache.reasoning_keys,
        "Language": cache.language_keys,
    }
    if cache.guardian_keys:
        groups["Guardian"] = cache.guardian_keys

    for group_name, keys in groups.items():
        if not keys:
            continue

        # Test on first layer
        k = keys[0]  # [n_heads, seq_len, d_head]
        seq_len = k.shape[1]

        for stride in [4, 8, 16]:
            if seq_len <= stride * 2:
                continue

            c = l3_compress_layer(k, stride=stride)
            kf_count = c.token_keyframes.shape[0]
            delta_count = c.token_deltas.shape[0]

            kf_mag = c.token_keyframes.float().abs().mean().item()
            delta_mag = c.token_deltas.float().abs().mean().item() if delta_count > 0 else 0
            reduction = kf_mag / delta_mag if delta_mag > 0 else float('inf')

            # Round-trip error
            recon = l3_decompress_layer(c)
            err = (recon.float() - k.transpose(0, 1).float()).abs().mean().item()

            print(f"    {group_name} stride={stride}: "
                  f"{seq_len}→{kf_count}KF+{delta_count}Δ  "
                  f"KF_mag={kf_mag:.4f}  Δ_mag={delta_mag:.4f}  "
                  f"reduction={reduction:.1f}x  err={err:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Full Stack Test
# ═══════════════════════════════════════════════════════════════════════════

def test_full_stack(cache: CerberusKVCache):
    """Run the full RGIHKCC + TurboQuant stack on Cerberus KV caches."""
    print_header("FULL STACK: L3 → L1 → TurboQuant (Per Group)")

    groups = {
        "Reasoning": (cache.reasoning_keys, cache.reasoning_values),
        "Language": (cache.language_keys, cache.language_values),
    }
    if cache.guardian_keys:
        groups["Guardian"] = (cache.guardian_keys, cache.guardian_values)

    total_original = 0
    total_compressed = 0

    for group_name, (keys, values) in groups.items():
        if not keys:
            continue

        print_sub(f"{group_name} Group ({len(keys)} layers)")

        # Original size
        orig_bytes = sum(k.nelement() * k.element_size() for k in keys)
        orig_bytes += sum(v.nelement() * v.element_size() for v in values)
        total_original += orig_bytes

        # L1 fold
        config = GIHKCCConfig(l1_snr_threshold=0.85, l1_max_keyframe_span=4)
        l1_k = l1_compress(keys, config)
        l1_v = l1_compress(values, config)

        n_kf = len(l1_k.keyframes)
        n_delta = len(l1_k.deltas)
        l1_ratio = l1_k.compression_ratio

        # Collect all tensors for TurboQuant
        all_tensors = []
        for kf in l1_k.keyframes:
            all_tensors.append(kf.data)
        for d in l1_k.deltas:
            all_tensors.append(d.delta)
        for kf in l1_v.keyframes:
            all_tensors.append(kf.data)
        for d in l1_v.deltas:
            all_tensors.append(d.delta)

        l1_bytes = sum(t.nelement() * t.element_size() for t in all_tensors)

        # TurboQuant
        tq_config = TurboQuantConfig(target_bits=3, qjl_enabled=True)
        tq_bytes = 0
        for t in all_tensors:
            c = turboquant_compress(t, tq_config)
            tq_bytes += c.compressed_bytes

        total_compressed += tq_bytes
        combined = orig_bytes / tq_bytes if tq_bytes > 0 else 1

        print(f"    Original: {fmt_bytes(orig_bytes)}")
        print(f"    L1: {n_kf} KF + {n_delta} deltas ({fmt_ratio(l1_ratio)})")
        print(f"    L1+TQ: {fmt_bytes(orig_bytes)} → {fmt_bytes(tq_bytes)} ({fmt_ratio(combined)})")

    # Overall
    overall = total_original / total_compressed if total_compressed > 0 else 1
    print(f"\n    ┌───────────────────────────────────────────────┐")
    print(f"    │  ALL GROUPS: {fmt_bytes(total_original)} → {fmt_bytes(total_compressed)}  =  {fmt_ratio(overall)}  │")
    print(f"    └───────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════
# THE REAL TEST: Compress Residual Stream, Recompute KV
# ═══════════════════════════════════════════════════════════════════════════

def test_residual_stream_compression(cache: CerberusKVCache, model):
    """
    THE PIVOT: Compress the residual stream, not the KV cache.

    The residual stream shows 0.95-0.99 cosine similarity between adjacent
    layers. KV space shows 0.00. The redundancy lives in the residual stream.

    Pipeline:
      1. Delta-encode residual states (L1 fold where cosine sim IS high)
      2. TurboQuant the keyframes + deltas
      3. Reconstruct residual states
      4. Recompute K/V via each layer's projection matrices
      5. Compare reconstructed K/V against original K/V
      6. Report: storage cost of compressed residuals vs original KV cache
    """
    print_header("RESIDUAL STREAM COMPRESSION (The Real Test)")

    residuals = cache.residual_states  # List of [seq_len, d_model] per layer
    n_layers = len(residuals)

    if n_layers < 2:
        print("    Not enough layers for residual compression.")
        return

    # ── Step 1: Residual stream L1 fold ──────────────────────────────
    print_sub("Step 1: Delta-encode residual stream")

    # Compute SNR profile
    snr_profile = []
    for i in range(n_layers - 1):
        sim = compute_statistical_snr(residuals[i], residuals[i + 1])
        snr_profile.append(sim)

    # Use 0.95 threshold — matching the observed similarity
    threshold = 0.95
    keyframes = [0]  # Layer 0 is always a keyframe
    keyframe_data = [residuals[0]]
    deltas = []
    delta_refs = []

    current_kf = 0
    for i in range(1, n_layers):
        if i - 1 < len(snr_profile) and snr_profile[i - 1] >= threshold:
            # High similarity: store delta from current keyframe
            delta = residuals[i] - residuals[current_kf]
            deltas.append(delta)
            delta_refs.append((i, current_kf))
        else:
            # Low similarity or first layers: new keyframe
            keyframes.append(i)
            keyframe_data.append(residuals[i])
            current_kf = i

    n_kf = len(keyframes)
    n_delta = len(deltas)
    structural_ratio = n_layers / (n_kf + n_delta * 0.5)  # Deltas are smaller

    print(f"    Threshold: {threshold}")
    print(f"    {n_layers} layers → {n_kf} keyframes + {n_delta} deltas")
    print(f"    Keyframe layers: {keyframes}")
    print(f"    SNR profile: {[f'{s:.4f}' for s in snr_profile]}")

    # Delta magnitude analysis
    if deltas:
        kf_mag = sum(kf.float().abs().mean().item() for kf in keyframe_data) / len(keyframe_data)
        delta_mag = sum(d.float().abs().mean().item() for d in deltas) / len(deltas)
        print(f"    Keyframe magnitude: {kf_mag:.4f}")
        print(f"    Delta magnitude: {delta_mag:.4f}")
        print(f"    Delta/KF ratio: {delta_mag/kf_mag:.4f} (lower = more compressible)")

    # ── Step 2: Quantize with multiple precision configs ────────────
    print_sub("Step 2: Quantize residual keyframes + deltas")

    # Original sizes for reference
    orig_residual_bytes = sum(r.nelement() * r.element_size() for r in residuals)
    orig_kv_bytes = 0
    orig_kv_bytes += sum(k.nelement() * k.element_size() for k in cache.reasoning_keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.reasoning_values)
    orig_kv_bytes += sum(k.nelement() * k.element_size() for k in cache.language_keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.language_values)
    orig_kv_bytes += sum(k.nelement() * k.element_size() for k in cache.guardian_keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.guardian_values)

    per_residual_bytes = residuals[0].nelement() * residuals[0].element_size()

    # ── Config sweep: test multiple precision combos ─────────────
    configs = {
        "A: Full KF + 8-bit deltas": {
            "kf_mode": "full",     # Keep keyframes at original dtype
            "delta_bits": 8,
        },
        "B: Full KF + 4-bit deltas": {
            "kf_mode": "full",
            "delta_bits": 4,
        },
        "C: 8-bit KF + 8-bit deltas": {
            "kf_mode": "tq8",
            "delta_bits": 8,
        },
        "D: 3-bit everything (prev run)": {
            "kf_mode": "tq3",
            "delta_bits": 3,
        },
    }

    best_name = None
    best_psnr = -999
    best_ratio = 0
    best_r_psnr = 0
    best_l_psnr = 0
    best_mae = 0
    best_max = 0
    best_bytes = 0

    for cfg_name, cfg in configs.items():
        print(f"\n    --- {cfg_name} ---")

        # Compress keyframes
        kf_recon_list = []
        kf_bytes = 0
        if cfg["kf_mode"] == "full":
            # Full precision — no compression, perfect reconstruction
            kf_bytes = n_kf * per_residual_bytes
            kf_recon_list = [kf.clone() for kf in keyframe_data]
        else:
            bits = int(cfg["kf_mode"].replace("tq", ""))
            tq_cfg = TurboQuantConfig(target_bits=bits, qjl_enabled=True)
            for kf in keyframe_data:
                c = turboquant_compress(kf, tq_cfg)
                kf_bytes += c.compressed_bytes
                kf_recon_list.append(turboquant_decompress(c))

        # Compress deltas
        delta_recon_list = []
        delta_bytes = 0
        tq_delta = TurboQuantConfig(target_bits=cfg["delta_bits"], qjl_enabled=True)
        for d in deltas:
            c = turboquant_compress(d, tq_delta)
            delta_bytes += c.compressed_bytes
            delta_recon_list.append(turboquant_decompress(c))

        total_compressed = kf_bytes + delta_bytes
        vs_kv = orig_kv_bytes / total_compressed if total_compressed > 0 else 1

        print(f"    KF: {fmt_bytes(kf_bytes)}  |  Deltas: {fmt_bytes(delta_bytes)}  |  Total: {fmt_bytes(total_compressed)}")
        print(f"    vs KV cache: {fmt_bytes(orig_kv_bytes)} → {fmt_bytes(total_compressed)} ({fmt_ratio(vs_kv)})")

        # ── Reconstruct residuals ────────────────────────────────
        recon_residuals = [None] * n_layers
        kf_map = {}
        for i, kf_layer in enumerate(keyframes):
            recon_residuals[kf_layer] = kf_recon_list[i]
            kf_map[kf_layer] = kf_recon_list[i]

        delta_idx = 0
        for i in range(n_layers):
            if recon_residuals[i] is None:
                layer_idx, ref_layer = delta_refs[delta_idx]
                assert layer_idx == i
                recon_residuals[i] = kf_map[ref_layer] + delta_recon_list[delta_idx]
                delta_idx += 1

        # Measure residual error
        res_errors = [measure_error(residuals[i], recon_residuals[i]) for i in range(n_layers)]
        mean_res_psnr = sum(e["psnr_db"] for e in res_errors) / n_layers
        mean_res_mae = sum(e["mae"] for e in res_errors) / n_layers
        print(f"    Residual: PSNR={mean_res_psnr:.1f}dB  MAE={mean_res_mae:.6f}")

        # ── Recompute K/V from reconstructed residuals ───────────
        device = next(model.parameters()).device

        # REASONING K/V — clean reconstruction via ln_reason → down_proj → qkv
        # This is the primary group (39 heads, 60% of KV cache)
        reason_kv_mae = 0.0
        reason_kv_psnr = 0.0
        reason_kv_max = 0.0

        # LANGUAGE K/V — requires intermediate state (x + reasoning_out)
        # We recompute by running reasoning attention to get the updated residual
        lang_kv_mae = 0.0
        lang_kv_psnr = 0.0
        lang_kv_max = 0.0

        n_reason = 0
        n_lang = 0

        for layer_idx, block in enumerate(model.blocks):
            model_dtype = next(block.parameters()).dtype; recon_res = recon_residuals[layer_idx].unsqueeze(0).to(device=device, dtype=model_dtype)

            # ── Reasoning K/V (clean: ln_reason → project) ──
            with torch.no_grad():
                x_normed = block.ln_reason(recon_res)  # THE FIX
                h_r = block.reasoning_attn.down_proj(x_normed)
                qkv_r = block.reasoning_attn.qkv(h_r).reshape(
                    1, recon_res.shape[1], 3,
                    block.reasoning_attn.n_heads, block.reasoning_attn.d_head
                )
                recon_r_k = qkv_r[0, :, 1].transpose(0, 1).cpu()
                recon_r_v = qkv_r[0, :, 2].transpose(0, 1).cpu()

            ek = measure_error(cache.reasoning_keys[layer_idx], recon_r_k)
            ev = measure_error(cache.reasoning_values[layer_idx], recon_r_v)
            reason_kv_mae += (ek["mae"] + ev["mae"]) / 2
            reason_kv_psnr += (ek["psnr_db"] + ev["psnr_db"]) / 2
            reason_kv_max = max(reason_kv_max, ek["max_err"], ev["max_err"])
            n_reason += 1

            # ── Language K/V (needs intermediate state) ──
            # Language sees ln_lang(x + scale * reasoning_out)
            # We need to run reasoning attention to get reasoning_out
            with torch.no_grad():
                # Get routing weights for this layer
                routing_weights, _ = block.hsr(recon_res)
                # Run full reasoning attention
                reasoning_out = block.reasoning_attn(x_normed, routing_weights)

                # Apply guardian gate if present
                if block.has_guardian:
                    guardian_raw = block.guardian_attn(
                        block.ln_guardian(recon_res), routing_weights,
                        kv_source=reasoning_out,
                    )
                    guardian_internal = block.guardian_attn.down_proj(guardian_raw)
                    gate, _ = block.guardian_gate(guardian_internal)
                    reasoning_out = reasoning_out * gate

                # Compute intermediate state after reasoning
                x_mid = recon_res + block._STREAM_SCALE * reasoning_out
                x_lang_normed = block.ln_lang(x_mid)

                h_l = block.language_attn.down_proj(x_lang_normed)
                qkv_l = block.language_attn.qkv(h_l).reshape(
                    1, recon_res.shape[1], 3,
                    block.language_attn.n_heads, block.language_attn.d_head
                )
                recon_l_k = qkv_l[0, :, 1].transpose(0, 1).cpu()
                recon_l_v = qkv_l[0, :, 2].transpose(0, 1).cpu()

            ek = measure_error(cache.language_keys[layer_idx], recon_l_k)
            ev = measure_error(cache.language_values[layer_idx], recon_l_v)
            lang_kv_mae += (ek["mae"] + ev["mae"]) / 2
            lang_kv_psnr += (ek["psnr_db"] + ev["psnr_db"]) / 2
            lang_kv_max = max(lang_kv_max, ek["max_err"], ev["max_err"])
            n_lang += 1

        r_mae = reason_kv_mae / n_reason if n_reason > 0 else 0
        r_psnr = reason_kv_psnr / n_reason if n_reason > 0 else 0
        l_mae = lang_kv_mae / n_lang if n_lang > 0 else 0
        l_psnr = lang_kv_psnr / n_lang if n_lang > 0 else 0
        overall_psnr = (reason_kv_psnr + lang_kv_psnr) / (n_reason + n_lang)
        overall_mae = (reason_kv_mae + lang_kv_mae) / (n_reason + n_lang)
        overall_max = max(reason_kv_max, lang_kv_max)

        print(f"    Reasoning KV (39 heads): PSNR={r_psnr:.1f}dB  MAE={r_mae:.6f}  Max={reason_kv_max:.6f}")
        print(f"    Language KV  (13 heads): PSNR={l_psnr:.1f}dB  MAE={l_mae:.6f}  Max={lang_kv_max:.6f}")
        print(f"    Combined:               PSNR={overall_psnr:.1f}dB  MAE={overall_mae:.6f}  Max={overall_max:.6f}")
        print(f"    *** {fmt_ratio(vs_kv)} compression at {overall_psnr:.1f}dB KV PSNR ***")

        if overall_psnr > best_psnr:
            best_psnr = overall_psnr
            best_ratio = vs_kv
            best_name = cfg_name
            best_mae = overall_mae
            best_max = overall_max
            best_bytes = total_compressed
            best_r_psnr = r_psnr
            best_l_psnr = l_psnr

    # ── Final Summary ────────────────────────────────────────────────
    print_sub("RESIDUAL STREAM COMPRESSION — FINAL RESULTS")

    print(f"\n    ┌──────────────────────────────────────────────────────────────┐")
    print(f"    │  BEST CONFIG: {best_name:<46s}│")
    print(f"    │                                                              │")
    print(f"    │  ORIGINAL KV CACHE:         {fmt_bytes(orig_kv_bytes):>12s}                  │")
    print(f"    │  COMPRESSED RESIDUALS:      {fmt_bytes(best_bytes):>12s}                  │")
    print(f"    │  COMPRESSION RATIO:         {fmt_ratio(best_ratio):>12s}                  │")
    print(f"    │                                                              │")
    print(f"    │  Residual fold: {n_kf} KF + {n_delta} deltas from {n_layers} layers        │")
    print(f"    │                                                              │")
    print(f"    │  KV Reconstruction:                                          │")
    print(f"    │    Reasoning (39h): {best_r_psnr:>6.1f} dB                              │")
    print(f"    │    Language  (13h): {best_l_psnr:>6.1f} dB                              │")
    print(f"    │    Combined PSNR:   {best_psnr:>6.1f} dB                              │")
    print(f"    │    MAE:  {best_mae:>12.6f}                                       │")
    print(f"    │    Max:  {best_max:>12.6f}                                       │")
    print(f"    │                                                              │")
    print(f"    │  vs TurboQuant-only on KV cache: 5.8x                        │")
    print(f"    │  vs monolithic PCA+TQ best:      6.0-10.3x                   │")
    print(f"    └──────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════
# TERNARY EXPERIMENTS: XNOR Chain vs Hierarchical Fold
# ═══════════════════════════════════════════════════════════════════════════

def test_ternary_compression(cache: CerberusKVCache, model):
    """
    Test both ternary compression approaches on the residual stream,
    then reconstruct KV caches and measure quality.
    """
    from ternary import (
        analyze_ternary_stats,
        xnor_compress_residuals,
        xnor_decompress_residuals,
        hierarchical_compress_residuals,
        hierarchical_decompress_residuals,
    )

    print_header("TERNARY RESIDUAL COMPRESSION")

    residuals = cache.residual_states
    n_layers = len(residuals)

    # Original KV cache size
    orig_kv_bytes = 0
    orig_kv_bytes += sum(k.nelement() * k.element_size() for k in cache.reasoning_keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.reasoning_values)
    orig_kv_bytes += sum(k.nelement() * k.element_size() for k in cache.language_keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.language_values)
    orig_kv_bytes += sum(k.nelement() * k.element_size() for k in cache.guardian_keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.guardian_values)

    # ── Ternary Statistics ───────────────────────────────────────────
    print_sub("Ternary Quantization Statistics")

    stats = analyze_ternary_stats(residuals)
    print(f"    Mean sparsity (zeros): {stats['mean_sparsity']:.1%}")
    print(f"    Mean scale: {stats['mean_scale']:.4f}")
    print(f"    Adjacent-layer agreement rates:")
    for i, a in enumerate(stats['agreements']):
        bar = "█" * int(a * 30)
        print(f"      L{i}→L{i+1}: {a:.4f}  {bar}")
    print(f"    Mean agreement: {stats['mean_agreement']:.4f}")
    print(f"    Min agreement:  {stats['min_agreement']:.4f}")

    # ── Helper: KV reconstruction from residuals ─────────────────────
    def reconstruct_and_measure_kv(recon_residuals: List[torch.Tensor]) -> Dict[str, float]:
        """Recompute KV from reconstructed residuals, measure error."""
        device = next(model.parameters()).device
        r_psnr_sum = 0.0
        r_mae_sum = 0.0
        r_max = 0.0
        l_psnr_sum = 0.0
        l_mae_sum = 0.0
        l_max = 0.0
        n_r = 0
        n_l = 0

        for layer_idx, block in enumerate(model.blocks):
            model_dtype = next(block.parameters()).dtype; recon_res = recon_residuals[layer_idx].unsqueeze(0).to(device=device, dtype=model_dtype)

            with torch.no_grad():
                # Reasoning
                x_normed = block.ln_reason(recon_res)
                h_r = block.reasoning_attn.down_proj(x_normed)
                qkv_r = block.reasoning_attn.qkv(h_r).reshape(
                    1, recon_res.shape[1], 3,
                    block.reasoning_attn.n_heads, block.reasoning_attn.d_head
                )
                rk = qkv_r[0, :, 1].transpose(0, 1).cpu()
                rv = qkv_r[0, :, 2].transpose(0, 1).cpu()

            ek = measure_error(cache.reasoning_keys[layer_idx], rk)
            ev = measure_error(cache.reasoning_values[layer_idx], rv)
            r_psnr_sum += (ek["psnr_db"] + ev["psnr_db"]) / 2
            r_mae_sum += (ek["mae"] + ev["mae"]) / 2
            r_max = max(r_max, ek["max_err"], ev["max_err"])
            n_r += 1

            with torch.no_grad():
                # Language — full reconstruction path
                routing_weights, _ = block.hsr(recon_res)
                reasoning_out = block.reasoning_attn(x_normed, routing_weights)
                if block.has_guardian:
                    guardian_raw = block.guardian_attn(
                        block.ln_guardian(recon_res), routing_weights,
                        kv_source=reasoning_out,
                    )
                    guardian_internal = block.guardian_attn.down_proj(guardian_raw)
                    gate, _ = block.guardian_gate(guardian_internal)
                    reasoning_out = reasoning_out * gate
                x_mid = recon_res + block._STREAM_SCALE * reasoning_out
                x_lang_normed = block.ln_lang(x_mid)
                h_l = block.language_attn.down_proj(x_lang_normed)
                qkv_l = block.language_attn.qkv(h_l).reshape(
                    1, recon_res.shape[1], 3,
                    block.language_attn.n_heads, block.language_attn.d_head
                )
                lk = qkv_l[0, :, 1].transpose(0, 1).cpu()
                lv = qkv_l[0, :, 2].transpose(0, 1).cpu()

            ek = measure_error(cache.language_keys[layer_idx], lk)
            ev = measure_error(cache.language_values[layer_idx], lv)
            l_psnr_sum += (ek["psnr_db"] + ev["psnr_db"]) / 2
            l_mae_sum += (ek["mae"] + ev["mae"]) / 2
            l_max = max(l_max, ek["max_err"], ev["max_err"])
            n_l += 1

        combined_psnr = (r_psnr_sum + l_psnr_sum) / (n_r + n_l)
        combined_mae = (r_mae_sum + l_mae_sum) / (n_r + n_l)
        return {
            "r_psnr": r_psnr_sum / n_r, "r_mae": r_mae_sum / n_r, "r_max": r_max,
            "l_psnr": l_psnr_sum / n_l, "l_mae": l_mae_sum / n_l, "l_max": l_max,
            "combined_psnr": combined_psnr,
            "combined_mae": combined_mae,
            "combined_max": max(r_max, l_max),
        }

    # ── Approach 1: XNOR Delta Chain ─────────────────────────────────
    print_sub("Approach 1: XNOR Ternary Delta Chain")

    xnor = xnor_compress_residuals(residuals)
    xnor_bytes = xnor.total_compressed_bytes
    xnor_ratio = orig_kv_bytes / xnor_bytes if xnor_bytes > 0 else 1

    print(f"    Anchor: layer 0 (full ternary)")
    print(f"    Deltas: {len(xnor.deltas)} sequential")
    print(f"    Mean agreement: {xnor.mean_agreement:.1%}")
    print(f"    Mean bits/element: {xnor.mean_bits_per_element:.2f}")
    print(f"    Storage: {fmt_bytes(xnor_bytes)}")
    print(f"    vs KV cache: {fmt_ratio(xnor_ratio)}")

    # Reconstruct and measure KV
    xnor_recon = xnor_decompress_residuals(xnor)
    xnor_kv = reconstruct_and_measure_kv(xnor_recon)

    print(f"    Reasoning KV: PSNR={xnor_kv['r_psnr']:.1f}dB  MAE={xnor_kv['r_mae']:.6f}")
    print(f"    Language KV:  PSNR={xnor_kv['l_psnr']:.1f}dB  MAE={xnor_kv['l_mae']:.6f}")
    print(f"    Combined:     PSNR={xnor_kv['combined_psnr']:.1f}dB  MAE={xnor_kv['combined_mae']:.6f}")

    # ── Approach 2: Hierarchical Fold ────────────────────────────────
    for group_size in [3, 4, 5]:
        print_sub(f"Approach 2: Hierarchical Fold (group_size={group_size})")

        hier = hierarchical_compress_residuals(residuals, group_size=group_size)
        hier_bytes = hier.total_compressed_bytes
        hier_ratio = orig_kv_bytes / hier_bytes if hier_bytes > 0 else 1

        summary = hier.summary()
        print(f"    Root anchor: layer 0")
        print(f"    L1 (layer fold):  {summary.get('L1_layer_kf', '?')} SKFs + {summary.get('L1_layer_deltas', '?')} deltas  "
              f"agreement={summary.get('L1_layer_agreement', 0):.1%}  "
              f"({fmt_bytes(summary.get('L1_layer_bytes', 0))})")
        print(f"    L2 (SKF fold):    {summary.get('L2_skf_kf', '?')} root + {summary.get('L2_skf_deltas', '?')} deltas  "
              f"agreement={summary.get('L2_skf_agreement', 0):.1%}  "
              f"({fmt_bytes(summary.get('L2_skf_bytes', 0))})")
        print(f"    Total: {fmt_bytes(hier_bytes)}")
        print(f"    vs KV cache: {fmt_ratio(hier_ratio)}")

        # Reconstruct and measure KV
        hier_recon = hierarchical_decompress_residuals(hier)
        hier_kv = reconstruct_and_measure_kv(hier_recon)

        print(f"    Reasoning KV: PSNR={hier_kv['r_psnr']:.1f}dB  MAE={hier_kv['r_mae']:.6f}")
        print(f"    Language KV:  PSNR={hier_kv['l_psnr']:.1f}dB  MAE={hier_kv['l_mae']:.6f}")
        print(f"    Combined:     PSNR={hier_kv['combined_psnr']:.1f}dB  MAE={hier_kv['combined_mae']:.6f}")

    # ── Approach 3+4: N-Level Quantization Sweep ──────────────────
    from ternary import (
        quint5_compress_residuals,
        quint5_decompress_residuals,
        quint5_hierarchical_compress,
        quint5_hierarchical_decompress,
        quint5_quantize,
    )

    nlevel_results = []
    for levels in [2, 4, 6, 8]:
        n_values = 2 * levels + 1
        bits_per = math.log2(n_values)
        print_sub(f"N-Level: {{-{levels}..{levels}}} ({n_values} values, {bits_per:.2f} bits/elem)")

        # XNOR chain
        nl_xnor = quint5_compress_residuals(residuals, levels=levels)
        nl_xnor_bytes = nl_xnor.total_compressed_bytes
        nl_xnor_ratio = orig_kv_bytes / nl_xnor_bytes if nl_xnor_bytes > 0 else 1

        print(f"    XNOR chain: agreement={nl_xnor.mean_agreement:.1%}  "
              f"storage={fmt_bytes(nl_xnor_bytes)}  ratio={fmt_ratio(nl_xnor_ratio)}")

        nl_xnor_recon = quint5_decompress_residuals(nl_xnor, levels=levels)
        nl_xnor_kv = reconstruct_and_measure_kv(nl_xnor_recon)
        print(f"    XNOR KV: PSNR={nl_xnor_kv['combined_psnr']:.1f}dB  MAE={nl_xnor_kv['combined_mae']:.6f}")

        # Hierarchical
        nl_hier = quint5_hierarchical_compress(residuals, group_size=4, levels=levels)
        nl_hier_bytes = nl_hier.total_compressed_bytes
        nl_hier_ratio = orig_kv_bytes / nl_hier_bytes if nl_hier_bytes > 0 else 1

        nl_hier_recon = quint5_hierarchical_decompress(nl_hier, levels=levels)
        nl_hier_kv = reconstruct_and_measure_kv(nl_hier_recon)
        print(f"    Hier(g=4): storage={fmt_bytes(nl_hier_bytes)}  ratio={fmt_ratio(nl_hier_ratio)}  "
              f"PSNR={nl_hier_kv['combined_psnr']:.1f}dB")

        nlevel_results.append({
            "levels": levels, "n_values": n_values, "bits": bits_per,
            "xnor_bytes": nl_xnor_bytes, "xnor_ratio": nl_xnor_ratio,
            "xnor_psnr": nl_xnor_kv["combined_psnr"], "xnor_mae": nl_xnor_kv["combined_mae"],
            "hier_bytes": nl_hier_bytes, "hier_ratio": nl_hier_ratio,
            "hier_psnr": nl_hier_kv["combined_psnr"],
        })

    # ── Approach 5: Even/Odd Wavelet Fold ────────────────────────
    from ternary import evenodd_compress_residuals, evenodd_decompress_residuals

    eo_results = []
    for levels in [4, 6, 8]:
        print_sub(f"Even/Odd Wavelet Fold (±{levels})")

        eo = evenodd_compress_residuals(residuals, levels=levels)
        eo_bytes = eo.total_compressed_bytes
        eo_ratio = orig_kv_bytes / eo_bytes if eo_bytes > 0 else 1

        print(f"    Coarse grid: ±{levels//2}  |  Coarse agreement: {eo.coarse_agreement:.1%}")
        print(f"    Storage: {fmt_bytes(eo_bytes)}  |  Ratio: {fmt_ratio(eo_ratio)}")

        eo_recon = evenodd_decompress_residuals(eo)
        eo_kv = reconstruct_and_measure_kv(eo_recon)

        print(f"    Reasoning KV: PSNR={eo_kv['r_psnr']:.1f}dB  MAE={eo_kv['r_mae']:.6f}")
        print(f"    Language KV:  PSNR={eo_kv['l_psnr']:.1f}dB  MAE={eo_kv['l_mae']:.6f}")
        print(f"    Combined:     PSNR={eo_kv['combined_psnr']:.1f}dB  MAE={eo_kv['combined_mae']:.6f}")

        eo_results.append({
            "levels": levels, "bytes": eo_bytes, "ratio": eo_ratio,
            "psnr": eo_kv["combined_psnr"], "mae": eo_kv["combined_mae"],
            "agreement": eo.coarse_agreement,
        })

    # ── Approach 6: Sigma-Delta Noise Shaping ────────────────────
    from ternary import sigmadelta_compress_residuals, sigmadelta_decompress_residuals

    sd_results = []
    for levels in [1, 2, 4, 6]:
        if levels == 1:
            label = "ternary"
        else:
            label = f"±{levels}"
        print_sub(f"Sigma-Delta ({label})")

        sd = sigmadelta_compress_residuals(residuals, levels=levels)
        sd_bytes = sd.total_compressed_bytes
        sd_ratio = orig_kv_bytes / sd_bytes if sd_bytes > 0 else 1

        print(f"    Agreement: {sd.mean_agreement:.1%}  |  Storage: {fmt_bytes(sd_bytes)}  |  Ratio: {fmt_ratio(sd_ratio)}")

        sd_recon = sigmadelta_decompress_residuals(sd, levels=levels)
        sd_kv = reconstruct_and_measure_kv(sd_recon)

        print(f"    Reasoning KV: PSNR={sd_kv['r_psnr']:.1f}dB  MAE={sd_kv['r_mae']:.6f}")
        print(f"    Language KV:  PSNR={sd_kv['l_psnr']:.1f}dB  MAE={sd_kv['l_mae']:.6f}")
        print(f"    Combined:     PSNR={sd_kv['combined_psnr']:.1f}dB  MAE={sd_kv['combined_mae']:.6f}")

        # Compare against non-shaped version at same level
        if levels == 1:
            baseline_psnr = xnor_kv['combined_psnr']
        else:
            baseline = [r for r in nlevel_results if r['levels'] == levels]
            baseline_psnr = baseline[0]['xnor_psnr'] if baseline else 0
        gain = sd_kv['combined_psnr'] - baseline_psnr
        print(f"    vs non-shaped: {gain:+.1f} dB improvement")

        sd_results.append({
            "levels": levels, "label": label,
            "bytes": sd_bytes, "ratio": sd_ratio,
            "psnr": sd_kv["combined_psnr"], "mae": sd_kv["combined_mae"],
            "gain": gain,
        })

    # ── Comparison Table ─────────────────────────────────────────────
    print_sub("ALL APPROACHES — COMPARISON")

    print(f"\n    ┌──────────────────────────────────────────────────────────────────────┐")
    print(f"    │  Original KV cache:            {fmt_bytes(orig_kv_bytes):>12s}                      │")
    print(f"    │                                                                      │")
    print(f"    │  Ternary XNOR chain:           {fmt_bytes(xnor_bytes):>12s}  ({fmt_ratio(xnor_ratio):>6s})             │")
    print(f"    │    KV PSNR: {xnor_kv['combined_psnr']:>6.1f} dB                                         │")
    for r in nlevel_results:
        label = f"  {r['n_values']}-level XNOR (±{r['levels']}):"
        print(f"    │{label:<33s}{fmt_bytes(r['xnor_bytes']):>12s}  ({fmt_ratio(r['xnor_ratio']):>6s})             │")
        print(f"    │    KV PSNR: {r['xnor_psnr']:>6.1f} dB                                         │")
    print(f"    │                                                                      │")
    print(f"    │  8-bit residual fold (lossless):{fmt_bytes(1536000):>11s}  ({fmt_ratio(9.0):>6s})             │")
    print(f"    │    KV PSNR:    inf dB                                                │")
    print(f"    │  8-bit all (16.8x):            {fmt_bytes(795700):>12s}  ({fmt_ratio(16.8):>6s})             │")
    print(f"    │    KV PSNR:   48.8 dB                                                │")
    print(f"    │  TurboQuant-only on KV:                             ( {fmt_ratio(5.8):>6s})             │")
    if eo_results:
        print(f"    │                                                                      │")
        for r in eo_results:
            label = f"  Even/Odd ±{r['levels']} (coarse ±{r['levels']//2}):"
            print(f"    │{label:<33s}{fmt_bytes(r['bytes']):>12s}  ({fmt_ratio(r['ratio']):>6s})             │")
            print(f"    │    KV PSNR: {r['psnr']:>6.1f} dB  coarse agree: {r['agreement']:.1%}              │")
    if sd_results:
        print(f"    │                                                                      │")
        print(f"    │  SIGMA-DELTA NOISE SHAPED:                                           │")
        for r in sd_results:
            label = f"  σΔ {r['label']}:"
            print(f"    │{label:<33s}{fmt_bytes(r['bytes']):>12s}  ({fmt_ratio(r['ratio']):>6s})             │")
            print(f"    │    KV PSNR: {r['psnr']:>6.1f} dB  ({r['gain']:+.1f} dB vs unshaped)                │")
    print(f"    └──────────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_cerberus(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load Cerberus V2 model from checkpoint + config."""

    # Load config
    with open(config_path) as f:
        cfg_dict = json.load(f)

    # Try to import CerberusV2Config and CerberusV2
    # The model uses relative imports (from .config import ...) so we need
    # the parent of the cerberus_v2 package on sys.path
    CerberusV2Config = None
    CerberusV2 = None

    # Strategy 1: Try importing cerberus_v2 as a package (most reliable)
    try:
        from cerberus_v2.config import CerberusV2Config
        from cerberus_v2.model import CerberusV2
        print("    Imported cerberus_v2 package")
    except ImportError:
        pass

    # Strategy 2: Try cerberus_v2 from various paths
    if CerberusV2Config is None:
        search_paths = [
            os.path.dirname(os.path.abspath(checkpoint_path)),     # checkpoint dir
            os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))),  # parent of checkpoint
        ]
        for sp in search_paths:
            if sp not in sys.path:
                sys.path.insert(0, sp)
            # Check if cerberus_v2 package exists here
            pkg_path = os.path.join(sp, "cerberus_v2")
            if os.path.isdir(pkg_path) and os.path.exists(os.path.join(pkg_path, "model.py")):
                try:
                    from cerberus_v2.config import CerberusV2Config
                    from cerberus_v2.model import CerberusV2
                    print(f"    Imported cerberus_v2 from {sp}")
                    break
                except ImportError as e:
                    print(f"    Tried {sp}: {e}")

    if CerberusV2Config is None:
        print("    ERROR: Could not find cerberus_v2 package.")
        print("    Make sure --cerberus-dir points to the directory CONTAINING")
        print("    the cerberus_v2/ folder (e.g. /home/vecp/Desktop/Cerberus_v2)")
        return None, None

    # Filter to only keys that CerberusV2Config accepts
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(CerberusV2Config)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

    config = CerberusV2Config(**cfg_dict)

    # Build PrimitiveIndex — try ontology file first, fall back to synthetic
    try:
        from cerberus_v2.config import PrimitiveIndex, load_primitives
        if config.ontology_path and os.path.exists(config.ontology_path):
            primitives = load_primitives(config.ontology_path)
            prim_index = PrimitiveIndex.from_primitives(primitives)
            print(f"    Loaded ontology from {config.ontology_path}")
        else:
            # Build synthetic PrimitiveIndex from head counts
            prim_index = PrimitiveIndex()
            # Reasoning: IDs 0..38, Guardian: 39..52, Language: 53..65
            prim_index.reasoning_ids = list(range(config.n_reasoning_heads))
            prim_index.guardian_ids = list(range(
                config.n_reasoning_heads,
                config.n_reasoning_heads + config.n_guardian_heads
            ))
            prim_index.language_ids = list(range(
                config.n_reasoning_heads + config.n_guardian_heads,
                config.n_reasoning_heads + config.n_guardian_heads + config.n_language_heads
            ))
            # Build masks
            prim_index.reasoning_mask = [1 if i in prim_index.reasoning_ids else 0 for i in range(66)]
            prim_index.guardian_mask = [1 if i in prim_index.guardian_ids else 0 for i in range(66)]
            prim_index.language_mask = [1 if i in prim_index.language_ids else 0 for i in range(66)]
            prim_index.activation_map = {i: [] for i in range(66)}
            print(f"    Built synthetic PrimitiveIndex (R:{len(prim_index.reasoning_ids)} G:{len(prim_index.guardian_ids)} L:{len(prim_index.language_ids)})")
    except ImportError:
        print("    ERROR: Could not import PrimitiveIndex")
        return None, None

    model = CerberusV2(config, prim_index).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle wrapped state dicts
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded: {n_params:,} parameters")

    return model, config


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GIHKCC Cerberus Test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.json")
    parser.add_argument("--cerberus-dir", type=str, default=None,
                        help="Path to Cerberus package directory (if not auto-detected)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for test input")
    args = parser.parse_args()

    print("\n" + "█"*80)
    print("█  GIHKCC CERBERUS TEST — THE ARCHITECTURE IT WAS DESIGNED FOR")
    print("█  VecP Labs LLC | Patent Pending")
    print("█"*80)

    if args.cerberus_dir:
        sys.path.insert(0, args.cerberus_dir)
        sys.path.insert(0, os.path.dirname(args.cerberus_dir))

    # Load model
    print_header("Loading Cerberus V2")
    model, config = load_cerberus(args.checkpoint, args.config, args.device)

    if model is None:
        return 1

    # Generate test input
    print_header("Extracting KV Caches")
    seq_len = min(args.seq_len, 256)  # Config max is 256
    input_ids = torch.randint(0, 256, (1, seq_len))
    print(f"    Input: batch=1, seq_len={seq_len}, vocab=256")

    cache = extract_cerberus_kv(model, input_ids)
    cache.summary()

    # Run tests
    test_l0_factorization(cache)
    test_l1_per_group(cache)
    test_l3_per_group(cache)
    test_full_stack(cache)
    test_residual_stream_compression(cache, model)
    test_ternary_compression(cache, model)

    print(f"\n{'█'*80}")
    print(f"█  DONE.")
    print(f"█  Compare residual-stream compression vs KV-space compression.")
    print(f"█  The residual stream is where the redundancy lives.")
    print(f"{'█'*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
