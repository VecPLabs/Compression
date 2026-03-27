#!/usr/bin/env python3
"""
Residual Stream Compression — Monolithic Model Test

The same experiment that produced ∞ dB at 9x and 48.8 dB at 16.8x on
Cerberus, now applied to standard HuggingFace models.

Key question: Does the residual stream show the same 0.95-0.99 cosine
similarity between adjacent layers in monolithic architectures?

If yes → residual fold + recompute KV works here too.
If no → GIHKCC residual compression is Cerberus-only.

Usage:
  python test_residual_monolithic.py --model Qwen/Qwen2.5-1.5B
  python test_residual_monolithic.py --model meta-llama/Llama-3.2-1B
  python test_residual_monolithic.py --model Qwen/Qwen3.5-2B  # tomorrow's test

VecP Labs LLC | vecplabs.com | Patent Pending
"""

import sys
import os
import time
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from gihkcc import compute_statistical_snr
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
# Model Loading + Residual/KV Extraction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MonolithicCache:
    """Extracted residual states and KV caches from a monolithic model."""
    residual_states: List[torch.Tensor]  # [seq_len, d_model] per layer
    keys: List[torch.Tensor]             # [num_kv_heads, seq_len, head_dim] per layer
    values: List[torch.Tensor]
    n_layers: int = 0
    d_model: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    model_name: str = ""


def load_and_extract(
    model_name: str,
    prompt: str = None,
    device: str = "auto",
) -> Tuple[Any, Any, MonolithicCache]:
    """
    Load a HuggingFace model and extract both residual states and KV caches.

    Returns (model, tokenizer, cache).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"    Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    if prompt is None:
        base = (
            "The transformer architecture revolutionized natural language processing "
            "by introducing self-attention mechanisms that allow models to weigh the "
            "importance of different parts of the input sequence. Key-value caches "
            "store previously computed attention states, enabling efficient "
            "autoregressive generation without redundant computation. However, as "
            "context windows grow to hundreds of thousands of tokens, the memory "
            "required for these caches becomes a significant bottleneck. "
        )
        prompt = base * 20  # ~2K tokens

    print(f"    Tokenizing ({len(prompt)} chars)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    print(f"    Sequence length: {seq_len} tokens")

    # ── Capture residual states via hooks ─────────────────────────
    residual_states = []

    # Find decoder layers — works for most HF architectures
    decoder_layers = None
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            decoder_layers = obj
            print(f"    Found decoder layers at: {attr} ({len(decoder_layers)} layers)")
            break
        except AttributeError:
            continue

    if decoder_layers is None:
        print("    ERROR: Could not find decoder layers")
        return model, tokenizer, None

    hooks = []
    for layer_idx, layer in enumerate(decoder_layers):
        def make_hook(idx):
            def hook(module, input):
                # Pre-hook: input[0] is the hidden states entering this layer
                x = input[0]
                if isinstance(x, tuple):
                    x = x[0]
                residual_states.append(x[0].detach().cpu())  # Remove batch dim
            return hook
        h = layer.register_forward_pre_hook(make_hook(layer_idx))
        hooks.append(h)

    # ── Forward pass ─────────────────────────────────────────────
    print(f"    Forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # ── Extract KV caches ────────────────────────────────────────
    past_kv = outputs.past_key_values
    keys, values = [], []

    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i in range(len(past_kv.key_cache)):
            keys.append(past_kv.key_cache[i][0].cpu())
            values.append(past_kv.value_cache[i][0].cpu())
    elif hasattr(past_kv, "__getitem__"):
        first = past_kv[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            for layer_kv in past_kv:
                keys.append(layer_kv[0][0].cpu())
                values.append(layer_kv[1][0].cpu())
        else:
            print(f"    KV cache type: {type(past_kv)}")
            for layer_kv in past_kv:
                layer_list = list(layer_kv)
                keys.append(layer_list[0][0].cpu())
                values.append(layer_list[1][0].cpu())

    n_layers = len(keys)
    d_model = residual_states[0].shape[-1] if residual_states else 0
    num_kv_heads = keys[0].shape[0] if keys else 0
    head_dim = keys[0].shape[-1] if keys else 0

    cache = MonolithicCache(
        residual_states=residual_states,
        keys=keys,
        values=values,
        n_layers=n_layers,
        d_model=d_model,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_name=model_name,
    )

    print(f"    Extracted: {n_layers} layers")
    print(f"    Residual: {len(residual_states)} states, d_model={d_model}")
    print(f"    KV: {num_kv_heads} heads, head_dim={head_dim}")
    print(f"    KV shape: {keys[0].shape}")
    print(f"    Residual shape: {residual_states[0].shape}")

    return model, tokenizer, cache


# ═══════════════════════════════════════════════════════════════════════════
# KV Reconstruction from Residuals (Monolithic)
# ═══════════════════════════════════════════════════════════════════════════

def find_attention_modules(model, decoder_layers):
    """
    Find the attention projection paths in each decoder layer.
    Returns a list of dicts with 'norm' and 'attn' modules per layer.
    """
    layer_info = []
    for layer in decoder_layers:
        info = {"layer": layer, "norm": None, "attn": None}

        # Find input layernorm / RMSNorm
        for attr in ["input_layernorm", "ln_1", "norm1", "self_attn_layer_norm"]:
            if hasattr(layer, attr):
                info["norm"] = getattr(layer, attr)
                break

        # Find self-attention module
        for attr in ["self_attn", "attn", "attention"]:
            if hasattr(layer, attr):
                info["attn"] = getattr(layer, attr)
                break

        layer_info.append(info)

    return layer_info


def reconstruct_kv_from_residuals(
    recon_residuals: List[torch.Tensor],
    model,
    decoder_layers,
    original_keys: List[torch.Tensor],
    original_values: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Reconstruct KV caches from compressed residual states by running
    the residual through each layer's norm + QKV projection.

    For monolithic models: residual → LayerNorm/RMSNorm → QKV projection → K, V
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    layer_info = find_attention_modules(model, decoder_layers)

    total_k_psnr = 0.0
    total_v_psnr = 0.0
    total_k_mae = 0.0
    total_v_mae = 0.0
    total_max = 0.0
    n = 0

    for layer_idx, info in enumerate(layer_info):
        if layer_idx >= len(recon_residuals):
            break
        if info["norm"] is None or info["attn"] is None:
            continue

        recon_res = recon_residuals[layer_idx].unsqueeze(0).to(device=device, dtype=model_dtype)
        attn = info["attn"]

        with torch.no_grad():
            # Apply layer norm
            normed = info["norm"](recon_res)

            # Find QKV projection — architecture-dependent
            recon_k = None
            recon_v = None

            # Strategy 1: Separate q/k/v projections (Llama, Qwen, Mistral)
            if hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                k_out = attn.k_proj(normed)
                v_out = attn.v_proj(normed)
                # Reshape to [batch, seq, num_kv_heads, head_dim] → [num_kv_heads, seq, head_dim]
                num_kv_heads = original_keys[layer_idx].shape[0]
                head_dim = original_keys[layer_idx].shape[-1]
                seq_len = normed.shape[1]
                recon_k = k_out.reshape(1, seq_len, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()
                recon_v = v_out.reshape(1, seq_len, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()

            # Strategy 2: Combined QKV projection (GPT-2 style)
            elif hasattr(attn, "c_attn"):
                qkv = attn.c_attn(normed)
                d = qkv.shape[-1] // 3
                k_out = qkv[..., d:2*d]
                v_out = qkv[..., 2*d:]
                num_kv_heads = original_keys[layer_idx].shape[0]
                head_dim = original_keys[layer_idx].shape[-1]
                seq_len = normed.shape[1]
                recon_k = k_out.reshape(1, seq_len, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()
                recon_v = v_out.reshape(1, seq_len, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()

            # Strategy 3: qkv combined linear
            elif hasattr(attn, "qkv_proj"):
                qkv = attn.qkv_proj(normed)
                num_heads = original_keys[layer_idx].shape[0]
                head_dim = original_keys[layer_idx].shape[-1]
                # This varies by model — skip if we can't parse it
                recon_k = None

            if recon_k is None:
                # Skip layers we can't reconstruct
                continue

        # Measure error
        ek = measure_error(original_keys[layer_idx], recon_k)
        ev = measure_error(original_values[layer_idx], recon_v)
        total_k_psnr += ek["psnr_db"]
        total_v_psnr += ev["psnr_db"]
        total_k_mae += ek["mae"]
        total_v_mae += ev["mae"]
        total_max = max(total_max, ek["max_err"], ev["max_err"])
        n += 1

    if n == 0:
        return {"k_psnr": 0, "v_psnr": 0, "combined_psnr": 0,
                "combined_mae": 0, "combined_max": 0, "n_layers": 0}

    return {
        "k_psnr": total_k_psnr / n,
        "v_psnr": total_v_psnr / n,
        "combined_psnr": (total_k_psnr + total_v_psnr) / (2 * n),
        "combined_mae": (total_k_mae + total_v_mae) / (2 * n),
        "combined_max": total_max,
        "n_layers": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Test Suite
# ═══════════════════════════════════════════════════════════════════════════

def run_tests(model, tokenizer, cache: MonolithicCache, decoder_layers):
    """Run the full residual stream compression test suite."""

    residuals = cache.residual_states
    n_layers = len(residuals)

    # ── Original sizes ───────────────────────────────────────────
    orig_kv_bytes = sum(k.nelement() * k.element_size() for k in cache.keys)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.values)
    orig_res_bytes = sum(r.nelement() * r.element_size() for r in residuals)
    per_res_bytes = residuals[0].nelement() * residuals[0].element_size()

    print(f"    Original KV cache: {fmt_bytes(orig_kv_bytes)}")
    print(f"    Original residuals: {fmt_bytes(orig_res_bytes)} ({n_layers} × {fmt_bytes(per_res_bytes)})")
    print(f"    KV/Residual size ratio: {orig_kv_bytes / orig_res_bytes:.2f}x")

    # ── Residual stream similarity ───────────────────────────────
    print_sub("Residual Stream Inter-Layer Similarity")

    snr_profile = []
    for i in range(n_layers - 1):
        sim = compute_statistical_snr(residuals[i], residuals[i + 1])
        snr_profile.append(sim)
        marker = " ◄ HIGH" if sim > 0.95 else ""
        bar = "█" * int(sim * 25) if sim > 0 else ""
        print(f"    L{i}→L{i+1}: {sim:.4f}  {bar}{marker}")

    high_sim = sum(1 for s in snr_profile if s > 0.95)
    print(f"\n    Layers with >0.95 similarity: {high_sim}/{len(snr_profile)}")
    print(f"    Mean similarity (L2+): {sum(snr_profile[2:])/max(len(snr_profile)-2,1):.4f}")

    # ── Delta-encode residual stream ─────────────────────────────
    print_sub("Residual Stream Delta Encoding")

    threshold = 0.95
    keyframes = [0]
    keyframe_data = [residuals[0]]
    deltas = []
    delta_refs = []
    current_kf = 0

    for i in range(1, n_layers):
        if i - 1 < len(snr_profile) and snr_profile[i - 1] >= threshold:
            delta = residuals[i] - residuals[current_kf]
            deltas.append(delta)
            delta_refs.append((i, current_kf))
        else:
            keyframes.append(i)
            keyframe_data.append(residuals[i])
            current_kf = i

    n_kf = len(keyframes)
    n_delta = len(deltas)

    print(f"    Threshold: {threshold}")
    print(f"    {n_layers} layers → {n_kf} keyframes + {n_delta} deltas")
    print(f"    Keyframe layers: {keyframes}")

    if deltas:
        kf_mag = sum(kf.float().abs().mean().item() for kf in keyframe_data) / len(keyframe_data)
        delta_mag = sum(d.float().abs().mean().item() for d in deltas) / len(deltas)
        print(f"    Keyframe magnitude: {kf_mag:.4f}")
        print(f"    Delta magnitude: {delta_mag:.4f}")
        print(f"    Delta/KF ratio: {delta_mag/kf_mag:.4f}")

    # ── Helper: reconstruct residuals from compressed form ───────
    def compress_and_reconstruct(kf_mode, delta_bits):
        """Compress keyframes+deltas, reconstruct, return (bytes, residuals)."""
        kf_recon = []
        kf_bytes = 0

        if kf_mode == "full":
            kf_bytes = n_kf * per_res_bytes
            kf_recon = [kf.clone() for kf in keyframe_data]
        else:
            bits = int(kf_mode.replace("tq", ""))
            cfg = TurboQuantConfig(target_bits=bits, qjl_enabled=True)
            for kf in keyframe_data:
                c = turboquant_compress(kf, cfg)
                kf_bytes += c.compressed_bytes
                kf_recon.append(turboquant_decompress(c))

        delta_recon = []
        delta_bytes = 0
        cfg = TurboQuantConfig(target_bits=delta_bits, qjl_enabled=True)
        for d in deltas:
            c = turboquant_compress(d, cfg)
            delta_bytes += c.compressed_bytes
            delta_recon.append(turboquant_decompress(c))

        # Rebuild residual stack
        recon = [None] * n_layers
        kf_map = {}
        for i, kf_layer in enumerate(keyframes):
            recon[kf_layer] = kf_recon[i]
            kf_map[kf_layer] = kf_recon[i]

        d_idx = 0
        for i in range(n_layers):
            if recon[i] is None:
                li, ref = delta_refs[d_idx]
                assert li == i
                recon[i] = kf_map[ref] + delta_recon[d_idx]
                d_idx += 1

        return kf_bytes + delta_bytes, recon

    # ── Compression configs sweep ────────────────────────────────
    print_sub("Residual Compression + KV Reconstruction")

    configs = [
        ("A: Full KF + 8-bit Δ", "full", 8),
        ("B: Full KF + 4-bit Δ", "full", 4),
        ("C: 8-bit KF + 8-bit Δ", "tq8", 8),
        ("D: 3-bit everything", "tq3", 3),
    ]

    results = []
    for name, kf_mode, delta_bits in configs:
        print(f"\n    --- {name} ---")

        total_bytes, recon_residuals = compress_and_reconstruct(kf_mode, delta_bits)
        ratio = orig_kv_bytes / total_bytes if total_bytes > 0 else 1

        print(f"    Storage: {fmt_bytes(total_bytes)}  |  vs KV: {fmt_ratio(ratio)}")

        # Residual reconstruction error
        res_errors = [measure_error(residuals[i], recon_residuals[i]) for i in range(n_layers)]
        res_psnr = sum(e["psnr_db"] for e in res_errors) / n_layers
        res_mae = sum(e["mae"] for e in res_errors) / n_layers
        print(f"    Residual: PSNR={res_psnr:.1f}dB  MAE={res_mae:.6f}")

        # KV reconstruction
        kv_result = reconstruct_kv_from_residuals(
            recon_residuals, model, decoder_layers,
            cache.keys, cache.values,
        )

        if kv_result["n_layers"] > 0:
            print(f"    KV recon ({kv_result['n_layers']} layers): "
                  f"PSNR={kv_result['combined_psnr']:.1f}dB  "
                  f"MAE={kv_result['combined_mae']:.6f}  "
                  f"Max={kv_result['combined_max']:.6f}")
        else:
            print(f"    KV recon: SKIPPED (could not find projection matrices)")

        results.append({
            "name": name, "bytes": total_bytes, "ratio": ratio,
            "res_psnr": res_psnr, "kv_psnr": kv_result["combined_psnr"],
            "kv_mae": kv_result["combined_mae"], "n_kv_layers": kv_result["n_layers"],
        })

    # ── N-Level XNOR sweep ───────────────────────────────────────
    print_sub("N-Level XNOR Chain on Residuals")

    from ternary import (
        quint5_compress_residuals,
        quint5_decompress_residuals,
        analyze_ternary_stats,
    )

    stats = analyze_ternary_stats(residuals)
    print(f"    Mean ternary agreement: {stats['mean_agreement']:.1%}")

    nlevel_results = []
    for levels in [1, 2, 4, 6]:
        if levels == 1:
            from ternary import xnor_compress_residuals, xnor_decompress_residuals
            comp = xnor_compress_residuals(residuals)
            recon = xnor_decompress_residuals(comp)
        else:
            comp = quint5_compress_residuals(residuals, levels=levels)
            recon = quint5_decompress_residuals(comp, levels=levels)

        comp_bytes = comp.total_compressed_bytes
        comp_ratio = orig_kv_bytes / comp_bytes if comp_bytes > 0 else 1

        kv_result = reconstruct_kv_from_residuals(
            recon, model, decoder_layers, cache.keys, cache.values,
        )

        label = "ternary" if levels == 1 else f"±{levels}"
        print(f"    {label:>8s}: {fmt_bytes(comp_bytes):>10s}  ({fmt_ratio(comp_ratio):>6s})  "
              f"KV PSNR={kv_result['combined_psnr']:.1f}dB  "
              f"agree={comp.mean_agreement:.1%}")

        nlevel_results.append({
            "levels": levels, "label": label,
            "bytes": comp_bytes, "ratio": comp_ratio,
            "kv_psnr": kv_result["combined_psnr"],
            "agreement": comp.mean_agreement,
        })

    # ── Comparison Table ─────────────────────────────────────────
    print_sub(f"FULL COMPARISON — {cache.model_name}")

    print(f"\n    ┌──────────────────────────────────────────────────────────────────────┐")
    print(f"    │  Model: {cache.model_name:<60s}│")
    print(f"    │  Layers: {n_layers}  |  d_model: {cache.d_model}  |  KV heads: {cache.num_kv_heads}  |  head_dim: {cache.head_dim:<6s}│")
    print(f"    │  Original KV cache: {fmt_bytes(orig_kv_bytes):>12s}                                    │")
    print(f"    │                                                                      │")
    print(f"    │  RESIDUAL STREAM FOLD:                                                │")
    for r in results:
        kv_str = f"{r['kv_psnr']:.1f}" if r['n_kv_layers'] > 0 else "N/A"
        print(f"    │    {r['name']:<25s} {fmt_bytes(r['bytes']):>10s}  ({fmt_ratio(r['ratio']):>6s})  KV={kv_str:>6s}dB │")
    print(f"    │                                                                      │")
    print(f"    │  N-LEVEL XNOR CHAIN:                                                 │")
    for r in nlevel_results:
        print(f"    │    {r['label']:>8s}:                  {fmt_bytes(r['bytes']):>10s}  ({fmt_ratio(r['ratio']):>6s})  KV={r['kv_psnr']:>6.1f}dB │")
    print(f"    │                                                                      │")
    print(f"    │  CERBERUS REFERENCE (56M, 15 layers):                                │")
    print(f"    │    Full KF + 8-bit Δ          1.5 MB  (  9.0x)  KV=  inf dB │")
    print(f"    │    8-bit all                 795.7 KB  ( 16.8x)  KV= 48.8dB │")
    print(f"    │    9-level XNOR              136.1 KB  ( 98.3x)  KV= 16.6dB │")
    print(f"    └──────────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Residual Stream Compression — Monolithic Models")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prompt-repeat", type=int, default=20,
                        help="Number of times to repeat base prompt (~100 tokens each)")
    args = parser.parse_args()

    print("\n" + "█"*80)
    print("█  RESIDUAL STREAM COMPRESSION — MONOLITHIC MODEL TEST")
    print("█  VecP Labs LLC | Patent Pending")
    print("█"*80)

    # Load and extract
    print_header(f"Loading {args.model}")

    model, tokenizer, cache = load_and_extract(
        args.model, device=args.device,
    )

    if cache is None:
        print("    Failed to extract. Exiting.")
        return 1

    # Find decoder layers for KV reconstruction
    decoder_layers = None
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            decoder_layers = obj
            break
        except AttributeError:
            continue

    if decoder_layers is None:
        print("    Could not find decoder layers for KV reconstruction.")
        return 1

    # Run tests
    print_header("Residual Stream Compression Tests")
    run_tests(model, tokenizer, cache, decoder_layers)

    print(f"\n{'█'*80}")
    print(f"█  DONE.")
    print(f"█  Compare these numbers against Cerberus results.")
    print(f"█  Key question: does monolithic residual fold match Cerberus?")
    print(f"{'█'*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
