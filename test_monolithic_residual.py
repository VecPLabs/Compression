#!/usr/bin/env python3
"""
GIHKCC Monolithic Residual Stream Test — Cerberus vs Everyone

Captures residual states from any HuggingFace model and runs the same
compression experiments as test_cerberus.py. Direct comparison:
  - Does the residual stream show 0.95+ cosine similarity? (Cerberus: YES)
  - Can we delta-encode residuals and recompute KV? (Cerberus: YES, ∞ PSNR at 9x)
  - Where's the Pareto frontier? (Cerberus: 16.8x at 48.8 dB)

Usage:
  python test_monolithic_residual.py --model Qwen/Qwen3.5-2B
  python test_monolithic_residual.py --model meta-llama/Llama-3.2-1B
  python test_monolithic_residual.py --model <any-hf-model>

VecP Labs LLC | vecplabs.com | Patent Pending
"""

import sys
import os
import time
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from gihkcc import (
    compute_statistical_snr,
    compute_snr_profile,
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
# Residual + KV Extraction from HuggingFace Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MonolithicCache:
    """Extracted residual states and KV caches from a HF model."""
    residual_states: List[torch.Tensor]  # [seq_len, d_model] per layer
    keys: List[torch.Tensor]             # [num_kv_heads, seq_len, head_dim] per layer
    values: List[torch.Tensor]           # same shape
    n_layers: int = 0
    model_name: str = ""
    d_model: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    seq_len: int = 0


def extract_hf_residuals_and_kv(
    model_name: str,
    prompt: str = None,
    device: str = "auto",
) -> Tuple[MonolithicCache, Any]:
    """
    Extract residual states AND KV caches from a HuggingFace model.

    Hooks into each decoder layer to capture:
      1. The residual stream (input to each layer)
      2. The KV cache (output of the forward pass)

    Returns (cache, model) so the model can be reused for reconstruction.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"    Loading model: {model_name}")
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
            "required for these caches becomes a significant bottleneck, motivating "
            "research into compression techniques that can reduce this footprint "
            "while preserving model output quality. "
        )
        prompt = base * 20  # ~2K tokens

    print(f"    Tokenizing ({len(prompt)} chars)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    print(f"    Sequence length: {seq_len} tokens")

    # ── Hook into decoder layers to capture residual states ──────
    captured_residuals = []
    hooks = []

    # Find the decoder layers — different model architectures name them differently
    decoder_layers = None
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
        try:
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part)
            decoder_layers = obj
            print(f"    Found decoder layers at: {attr} ({len(decoder_layers)} layers)")
            break
        except AttributeError:
            continue

    if decoder_layers is None:
        print("    ERROR: Could not find decoder layers. Model structure:")
        for name, _ in model.named_children():
            print(f"      {name}")
        return None, model

    for layer_idx, layer in enumerate(decoder_layers):
        def make_hook(idx):
            def hook(module, input):
                # input is a tuple; input[0] is the hidden states
                x = input[0]
                if isinstance(x, tuple):
                    x = x[0]
                # x shape: [batch, seq_len, d_model]
                captured_residuals.append(x[0].detach().cpu())  # Remove batch dim
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

    # ── Extract KV cache ─────────────────────────────────────────
    past_kv = outputs.past_key_values
    keys, values = [], []

    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i in range(len(past_kv.key_cache)):
            if past_kv.key_cache[i] is not None:
                keys.append(past_kv.key_cache[i][0].cpu())
                values.append(past_kv.value_cache[i][0].cpu())
            else:
                # Some layers don't produce KV (linear attention, etc.)
                # Insert a placeholder so layer indices stay aligned
                keys.append(None)
                values.append(None)
        # Report how many actual KV layers we got
        actual_kv = sum(1 for k in keys if k is not None)
        print(f"    KV cache: {actual_kv}/{len(keys)} layers have KV entries")
    elif hasattr(past_kv, "__getitem__"):
        first = past_kv[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            for layer_kv in past_kv:
                keys.append(layer_kv[0][0].cpu())
                values.append(layer_kv[1][0].cpu())
        else:
            print(f"    WARNING: Unknown KV format: {type(first)}")
    else:
        for layer_kv in past_kv:
            layer_list = list(layer_kv)
            keys.append(layer_list[0][0].cpu())
            values.append(layer_list[1][0].cpu())

    n_layers = len(captured_residuals)
    print(f"    Extracted: {n_layers} residual states, shape {captured_residuals[0].shape}")

    # Find first non-None KV entry for shape info
    first_valid_k = next((k for k in keys if k is not None), None)
    if first_valid_k is not None:
        print(f"    KV cache: {sum(1 for k in keys if k is not None)}/{len(keys)} layers, shape {first_valid_k.shape}")
        num_kv_heads = first_valid_k.shape[0]
        head_dim = first_valid_k.shape[-1]
    else:
        print(f"    WARNING: No valid KV cache entries found!")
        num_kv_heads = 0
        head_dim = 0

    d_model = captured_residuals[0].shape[-1]

    cache = MonolithicCache(
        residual_states=captured_residuals,
        keys=keys,
        values=values,
        n_layers=n_layers,
        model_name=model_name,
        d_model=d_model,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
    )

    return cache, model


# ═══════════════════════════════════════════════════════════════════════════
# KV Reconstruction from Residuals (Monolithic)
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_kv_monolithic(
    model,
    recon_residuals: List[torch.Tensor],
    original_cache: MonolithicCache,
) -> Dict[str, float]:
    """
    Recompute KV from reconstructed residuals through the model's
    actual attention layers.

    For monolithic models, each layer has its own attention with:
      input_layernorm → self_attn (q_proj, k_proj, v_proj)

    We run the reconstructed residual through the norm + KV projection
    to get reconstructed keys and values, then compare against originals.
    """
    device = next(model.parameters()).device

    # Find decoder layers
    decoder_layers = None
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
        try:
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part)
            decoder_layers = obj
            break
        except AttributeError:
            continue

    if decoder_layers is None:
        return {"combined_psnr": 0, "combined_mae": 0, "combined_max": 0}

    total_psnr = 0.0
    total_mae = 0.0
    total_max = 0.0
    n_cmp = 0

    for layer_idx, layer in enumerate(decoder_layers):
        if layer_idx >= len(recon_residuals) or layer_idx >= len(original_cache.keys):
            break

        # Skip layers that don't have KV cache (linear attention, etc.)
        if original_cache.keys[layer_idx] is None:
            continue

        model_dtype = next(layer.parameters()).dtype
        recon_res = recon_residuals[layer_idx].unsqueeze(0).to(device=device, dtype=model_dtype)

        with torch.no_grad():
            # Find the layer norm — try common attribute names
            ln = None
            for attr in ["input_layernorm", "ln_1", "layer_norm", "norm1"]:
                if hasattr(layer, attr):
                    ln = getattr(layer, attr)
                    break

            if ln is not None:
                normed = ln(recon_res)
            else:
                normed = recon_res

            # Find the self-attention module
            attn = None
            for attr in ["self_attn", "attn", "attention", "self_attention"]:
                if hasattr(layer, attr):
                    attn = getattr(layer, attr)
                    break

            if attn is None:
                continue

            # Extract K and V projections
            # Most models have k_proj and v_proj as linear layers
            recon_k = None
            recon_v = None

            if hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                # Standard: separate k_proj, v_proj
                k_raw = attn.k_proj(normed)  # [1, seq, num_kv_heads * head_dim]
                v_raw = attn.v_proj(normed)
                num_kv_heads = original_cache.num_kv_heads
                head_dim = original_cache.head_dim
                recon_k = k_raw.reshape(1, -1, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()
                recon_v = v_raw.reshape(1, -1, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()

            elif hasattr(attn, "qkv_proj"):
                # Combined QKV projection
                qkv = attn.qkv_proj(normed)
                # Split — this varies by model, try common patterns
                d = qkv.shape[-1] // 3
                k_raw = qkv[..., d:2*d]
                v_raw = qkv[..., 2*d:]
                num_kv_heads = original_cache.num_kv_heads
                head_dim = original_cache.head_dim
                recon_k = k_raw.reshape(1, -1, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()
                recon_v = v_raw.reshape(1, -1, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()

            elif hasattr(attn, "c_attn"):
                # GPT-2 style combined
                qkv = attn.c_attn(normed)
                d = qkv.shape[-1] // 3
                k_raw = qkv[..., d:2*d]
                v_raw = qkv[..., 2*d:]
                num_kv_heads = original_cache.num_kv_heads
                head_dim = original_cache.head_dim
                recon_k = k_raw.reshape(1, -1, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()
                recon_v = v_raw.reshape(1, -1, num_kv_heads, head_dim)[0].transpose(0, 1).cpu()

            if recon_k is not None:
                ek = measure_error(original_cache.keys[layer_idx], recon_k)
                ev = measure_error(original_cache.values[layer_idx], recon_v)
                total_psnr += (ek["psnr_db"] + ev["psnr_db"]) / 2
                total_mae += (ek["mae"] + ev["mae"]) / 2
                total_max = max(total_max, ek["max_err"], ev["max_err"])
                n_cmp += 1

    if n_cmp == 0:
        print("    WARNING: Could not reconstruct any KV pairs")
        return {"combined_psnr": 0, "combined_mae": 0, "combined_max": 0}

    return {
        "combined_psnr": total_psnr / n_cmp,
        "combined_mae": total_mae / n_cmp,
        "combined_max": total_max,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Test
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GIHKCC Monolithic Residual Stream Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("\n" + "█"*80)
    print("█  GIHKCC MONOLITHIC RESIDUAL STREAM TEST")
    print("█  Cerberus vs Everyone — Same Compression, Different Architecture")
    print("█  VecP Labs LLC | Patent Pending")
    print("█"*80)

    # ── Extract ──────────────────────────────────────────────────
    print_header(f"Extracting: {args.model}")
    cache, model = extract_hf_residuals_and_kv(args.model, device=args.device)

    if cache is None:
        print("    Failed to extract. Exiting.")
        return 1

    residuals = cache.residual_states
    n_layers = cache.n_layers

    orig_kv_bytes = sum(k.nelement() * k.element_size() for k in cache.keys if k is not None)
    orig_kv_bytes += sum(v.nelement() * v.element_size() for v in cache.values if v is not None)
    orig_res_bytes = sum(r.nelement() * r.element_size() for r in residuals)

    print(f"\n    Model: {cache.model_name}")
    print(f"    Layers: {n_layers}")
    print(f"    d_model: {cache.d_model}")
    print(f"    KV heads: {cache.num_kv_heads} × {cache.head_dim} head_dim")
    print(f"    Seq len: {cache.seq_len}")
    print(f"    KV cache size: {fmt_bytes(orig_kv_bytes)}")
    print(f"    Residual size: {fmt_bytes(orig_res_bytes)}")
    if orig_kv_bytes > 0:
        print(f"    Residual/KV ratio: {orig_res_bytes/orig_kv_bytes:.2f}x")
    else:
        print(f"    WARNING: No KV cache data — linear attention only?")

    # ── Residual Stream Cosine Similarity ────────────────────────
    print_header("Residual Stream Inter-Layer Similarity")

    snr_profile = []
    for i in range(n_layers - 1):
        sim = compute_statistical_snr(residuals[i], residuals[i + 1])
        snr_profile.append(sim)
        marker = " ◄ KEYFRAME" if sim < 0.95 else ""
        bar = "█" * int(sim * 25)
        print(f"    L{i}→L{i+1}: {sim:.4f}  {bar}{marker}")

    mean_sim = sum(snr_profile) / len(snr_profile) if snr_profile else 0
    high_sim = sum(1 for s in snr_profile if s >= 0.95)
    print(f"\n    Mean similarity: {mean_sim:.4f}")
    print(f"    Layers with ≥0.95 similarity: {high_sim}/{len(snr_profile)}")

    # ── KV Reconstruction Helper ─────────────────────────────────
    def measure_kv(recon_residuals):
        return reconstruct_kv_monolithic(model, recon_residuals, cache)

    # ── Residual Stream Compression Sweep ────────────────────────
    print_header("Residual Stream Compression")

    # Delta-encode residuals
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
        print(f"    Keyframe mag: {kf_mag:.4f}  |  Delta mag: {delta_mag:.4f}  |  Ratio: {delta_mag/kf_mag:.4f}")

    per_residual_bytes = residuals[0].nelement() * residuals[0].element_size()

    # ── Compression Configs ──────────────────────────────────────
    print_sub("Mixed-Precision Sweep")

    configs = {
        "Full KF + 8-bit deltas": {"kf_mode": "full", "delta_bits": 8},
        "Full KF + 4-bit deltas": {"kf_mode": "full", "delta_bits": 4},
        "8-bit KF + 8-bit deltas": {"kf_mode": "tq8", "delta_bits": 8},
        "3-bit everything": {"kf_mode": "tq3", "delta_bits": 3},
    }

    results = []

    for cfg_name, cfg in configs.items():
        print(f"\n    --- {cfg_name} ---")

        # Compress keyframes
        kf_recon_list = []
        kf_bytes = 0
        if cfg["kf_mode"] == "full":
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

        # Reconstruct residuals
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

        # Measure KV reconstruction
        kv_result = measure_kv(recon_residuals)

        print(f"    Storage: {fmt_bytes(total_compressed)} ({fmt_ratio(vs_kv)} vs KV cache)")
        print(f"    Residual PSNR: {mean_res_psnr:.1f} dB")
        print(f"    KV PSNR: {kv_result['combined_psnr']:.1f} dB  MAE: {kv_result['combined_mae']:.6f}")

        results.append({
            "name": cfg_name,
            "bytes": total_compressed,
            "ratio": vs_kv,
            "kv_psnr": kv_result["combined_psnr"],
            "kv_mae": kv_result["combined_mae"],
            "kv_max": kv_result["combined_max"],
        })

    # ── N-Level Quantized Chains ─────────────────────────────────
    from ternary import (
        quint5_compress_residuals,
        quint5_decompress_residuals,
        analyze_ternary_stats,
        xnor_compress_residuals,
        xnor_decompress_residuals,
    )

    print_sub("Ternary / N-Level Statistics")
    stats = analyze_ternary_stats(residuals)
    print(f"    Mean ternary agreement: {stats['mean_agreement']:.1%}")
    print(f"    Min agreement: {stats['min_agreement']:.4f}")

    nlevel_results = []
    for levels in [1, 2, 4, 6]:
        if levels == 1:
            label = "ternary"
            comp = xnor_compress_residuals(residuals)
            recon = xnor_decompress_residuals(comp)
        else:
            label = f"±{levels}"
            comp = quint5_compress_residuals(residuals, levels=levels)
            recon = quint5_decompress_residuals(comp, levels=levels)

        comp_bytes = comp.total_compressed_bytes
        comp_ratio = orig_kv_bytes / comp_bytes if comp_bytes > 0 else 1

        kv_result = measure_kv(recon)

        print(f"    {label:>10s}: agreement={comp.mean_agreement:.1%}  "
              f"storage={fmt_bytes(comp_bytes)}  ratio={fmt_ratio(comp_ratio)}  "
              f"KV PSNR={kv_result['combined_psnr']:.1f}dB")

        nlevel_results.append({
            "label": label, "levels": levels,
            "bytes": comp_bytes, "ratio": comp_ratio,
            "psnr": kv_result["combined_psnr"],
            "mae": kv_result["combined_mae"],
        })

    # ── Comparison Table ─────────────────────────────────────────
    print_header(f"RESULTS: {cache.model_name}")

    print(f"\n    Model: {cache.model_name}")
    print(f"    {n_layers} layers, d_model={cache.d_model}, {cache.num_kv_heads} KV heads × {cache.head_dim}")
    print(f"    Seq len: {cache.seq_len} tokens")
    print(f"    Residual/KV size ratio: {orig_res_bytes/max(orig_kv_bytes,1):.2f}x")
    print(f"    Mean inter-layer similarity: {mean_sim:.4f}")

    print(f"\n    ┌──────────────────────────────────────────────────────────────────┐")
    print(f"    │  Original KV cache:         {fmt_bytes(orig_kv_bytes):>12s}                    │")
    print(f"    │                                                                  │")
    print(f"    │  RESIDUAL STREAM COMPRESSION:                                    │")
    for r in results:
        print(f"    │    {r['name']:<28s} {fmt_bytes(r['bytes']):>10s}  {fmt_ratio(r['ratio']):>6s}  {r['kv_psnr']:>6.1f}dB │")
    print(f"    │                                                                  │")
    print(f"    │  N-LEVEL QUANTIZED CHAINS:                                       │")
    for r in nlevel_results:
        print(f"    │    {r['label']:<28s} {fmt_bytes(r['bytes']):>10s}  {fmt_ratio(r['ratio']):>6s}  {r['psnr']:>6.1f}dB │")
    print(f"    │                                                                  │")
    print(f"    │  CERBERUS COMPARISON (56M, 15 layers):                            │")
    print(f"    │    Full KF + 8-bit deltas      1.5 MB    9.0x     inf dB │")
    print(f"    │    8-bit all                  795.7 KB   16.8x   48.8 dB │")
    print(f"    │    9-level XNOR               136.1 KB   98.3x   16.6 dB │")
    print(f"    └──────────────────────────────────────────────────────────────────┘")

    # ── Key Architectural Difference ─────────────────────────────
    print_sub("ARCHITECTURAL ANALYSIS")

    # Can the monolithic model reconstruct KV from residuals?
    # The key question: does the residual-to-KV reconstruction work?
    best = max(results, key=lambda r: r["kv_psnr"])

    if best["kv_psnr"] > 30:
        verdict = "YES — residual stream compression works for this model"
    elif best["kv_psnr"] > 15:
        verdict = "PARTIAL — usable for edge/cold storage, not production"
    else:
        verdict = "NO — KV projection amplifies residual error too much"

    print(f"\n    Can {cache.model_name} use residual stream compression?")
    print(f"    Best config: {best['name']}")
    print(f"    KV PSNR: {best['kv_psnr']:.1f} dB at {fmt_ratio(best['ratio'])}")
    print(f"    Verdict: {verdict}")

    # Cerberus advantage
    print(f"\n    Cerberus structural advantage:")
    print(f"      Cerberus stores 1 residual stream → recomputes 3 head groups (66 heads)")
    print(f"      {cache.model_name} stores 1 residual → recomputes {cache.num_kv_heads} KV heads")
    if cache.num_kv_heads > 0 and cache.d_model > 0:
        kv_total_dim = cache.num_kv_heads * cache.head_dim * 2
        kv_overhead = kv_total_dim / cache.d_model
        if kv_total_dim > cache.d_model:
            print(f"      KV dim ({cache.num_kv_heads}×{cache.head_dim}×2={kv_total_dim}) > d_model ({cache.d_model})")
            print(f"      → {kv_overhead:.1f}x expansion from residual to KV = favorable for compression")
        else:
            print(f"      KV dim ({cache.num_kv_heads}×{cache.head_dim}×2={kv_total_dim}) ≤ d_model ({cache.d_model})")
            print(f"      → {kv_overhead:.1f}x compression from residual to KV (GQA)")
            print(f"      → Residual is LARGER than KV — compression ratio is inherently limited")
            print(f"      → Cerberus's 66-head expansion avoids this limitation")
    else:
        print(f"      No standard KV cache (linear attention?) — different compression axis")

    print(f"\n{'█'*80}")
    print(f"█  DONE. Compare these numbers against Cerberus test_cerberus.py output.")
    print(f"{'█'*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
