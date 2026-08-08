"""Real-model residual/KV hybrid benchmark for GPT-NeoX/Pythia models."""

from __future__ import annotations

import argparse
import math
from typing import List

import torch

from benchmark_compression import measure, measure_inner_products, tensor_bytes
from gihkcc import compute_statistical_snr
from gihkcc_v2 import (
    GIHKCCV2Config,
    compress_predictive_stack,
    decompress_predictive_stack,
)
from turboquant_paper import (
    paper_turboquant_compress_list,
    paper_turboquant_decompress_list,
)


def project_neox_kv(model, residuals: List[torch.Tensor]):
    keys, values = [], []
    for layer, residual in zip(model.gpt_neox.layers, residuals):
        dtype = next(layer.parameters()).dtype
        device = next(layer.parameters()).device
        hidden = layer.input_layernorm(
            residual.unsqueeze(0).to(device=device, dtype=dtype)
        )
        qkv = layer.attention.query_key_value(hidden)
        heads = model.config.num_attention_heads
        head_dim = layer.attention.head_size
        qkv = qkv.reshape(1, residual.shape[0], heads, 3, head_dim)
        keys.append(qkv[0, :, :, 1].permute(1, 0, 2).detach().cpu().half())
        values.append(qkv[0, :, :, 2].permute(1, 0, 2).detach().cpu().half())
    return keys, values


def report(name, original, reconstructed, payload_bytes):
    psnr, mae, _ = measure(original, reconstructed)
    ip_nrmse, ip_bias = measure_inner_products(original, reconstructed)
    ratio = tensor_bytes(original) / payload_bytes
    print(
        f"{name:34} {ratio:7.2f}x {psnr:8.2f}dB {mae:10.6f} "
        f"{ip_nrmse:9.4f} {ip_bias:9.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    text = ("Residual streams can act as shared predictors for projected key and value caches. " * 80)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.tokens)

    residuals = []
    hooks = []
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_pre_hook(
            lambda module, arguments: residuals.append(arguments[0][0].detach().cpu())
        ))
    with torch.no_grad():
        model(**inputs, use_cache=True)
    for hook in hooks:
        hook.remove()

    # Benchmark the deployment storage format, independent of CPU inference
    # dtype. FP16 is the baseline used by the synthetic and paper comparisons.
    residuals = [residual.half() for residual in residuals]

    keys, values = project_neox_kv(model, residuals)
    original_kv = keys + values
    kv_bytes = tensor_bytes(original_kv)
    residual_bytes = tensor_bytes(residuals)
    print(f"\nModel: {args.model}; layers={len(residuals)}; tokens={residuals[0].shape[0]}")
    print(f"Raw KV={kv_bytes/1024**2:.2f} MiB; residual={residual_bytes/1024**2:.2f} MiB; structural={kv_bytes/residual_bytes:.2f}x")
    similarities = [
        compute_statistical_snr(residuals[index - 1], residuals[index])
        for index in range(1, len(residuals))
    ]
    print(f"Adjacent residual cosine: {[round(value, 4) for value in similarities]}")
    print(f"{'Method':34} {'Ratio':>8} {'PSNR':>10} {'MAE':>10} {'IP nRMSE':>9} {'IP bias':>9}")
    print("-" * 88)

    for bits in (3, 4):
        encoded, size = paper_turboquant_compress_list(original_kv, bits)
        restored = paper_turboquant_decompress_list(encoded)
        report(f"KV-only Paper TQ {bits}b", original_kv, restored, size)

    # Force a single predictive group for this six-layer diagnostic. Longer
    # models should select boundaries with a measured error budget.
    config = GIHKCCV2Config(similarity_threshold=0.0, max_keyframe_span=8)
    residual_candidates = {}
    for residual_bits in (1, 2, 3):
        residual_stack = compress_predictive_stack(
            residuals, 8, residual_bits, config
        )
        restored_residuals = decompress_predictive_stack(residual_stack)
        predicted_keys, predicted_values = project_neox_kv(model, restored_residuals)
        predicted_kv = predicted_keys + predicted_values
        residual_candidates[residual_bits] = (residual_stack, predicted_kv)
        report(
            f"Residual GIHKCC v2 {residual_bits}b", original_kv, predicted_kv,
            residual_stack.compressed_bytes,
        )

    # Corrections are most useful on the aggressive 1-bit residual payload.
    residual_stack, predicted_kv = residual_candidates[1]
    corrections = [
        truth - prediction for truth, prediction in zip(original_kv, predicted_kv)
    ]
    for bits in (1, 2):
        encoded, correction_bytes = paper_turboquant_compress_list(corrections, bits)
        restored_corrections = paper_turboquant_decompress_list(encoded)
        restored_kv = [
            prediction + correction
            for prediction, correction in zip(predicted_kv, restored_corrections)
        ]
        report(
            f"Residual 1b + KV correction {bits}b", original_kv, restored_kv,
            residual_stack.compressed_bytes + correction_bytes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
