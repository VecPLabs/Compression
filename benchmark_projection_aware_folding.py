"""Allocate folded GIHKCC bits by held-out projected-K/V distortion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark_compression import measure, tensor_bytes
from benchmark_residual_folding import quantize
from benchmark_real_residual import project_neox_kv
from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack, decompress_predictive_stack
from residual_folding import FoldSpec, fit_fold, fold, fold_spec_to_dict, unfold


def project_layer_kv(model, layer_index, residual):
    layer = model.gpt_neox.layers[layer_index]
    dtype = next(layer.parameters()).dtype
    hidden = layer.input_layernorm(residual.unsqueeze(0).to(dtype=dtype))
    qkv = layer.attention.query_key_value(hidden)
    heads, head_dim = model.config.num_attention_heads, layer.attention.head_size
    qkv = qkv.reshape(1, residual.shape[0], heads, 3, head_dim)
    key = qkv[0, :, :, 1].permute(1, 0, 2).detach().cpu().float()
    value = qkv[0, :, :, 2].permute(1, 0, 2).detach().cpu().float()
    return key, value


def projection_sse(model, layer_index, truth, restored):
    truth_kv = project_layer_kv(model, layer_index, truth)
    restored_kv = project_layer_kv(model, layer_index, restored)
    return sum((left - right).square().sum().item() for left, right in zip(truth_kv, restored_kv))


def optimize_allocation(model, states, specs, target_bits):
    candidates = (
        [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)] if target_bits == 2 else
        [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2)]
    )
    profiles = []
    for layer_index in range(1, len(states)):
        error = states[layer_index] - states[layer_index - 1]
        coarse, detail = fold(error, specs[layer_index - 1])
        layer_records = []
        for coarse_bits, detail_bits in candidates:
            decoded_coarse, _ = quantize(coarse, coarse_bits)
            decoded_detail, _ = quantize(detail, detail_bits)
            decoded_error = unfold(decoded_coarse, decoded_detail, specs[layer_index - 1])
            restored = states[layer_index - 1].float() + decoded_error
            layer_records.append((
                coarse_bits + detail_bits, (coarse_bits, detail_bits),
                projection_sse(model, layer_index, states[layer_index], restored),
            ))
        profiles.append(layer_records)
        print(f"profiled layer={layer_index:02d}", flush=True)
    budget = 2 * target_bits * (len(states) - 1)
    states_by_cost = {0: (0.0, [])}
    for layer_records in profiles:
        expanded = {}
        for prior_cost, (prior_error, allocation) in states_by_cost.items():
            for cost, choice, error in layer_records:
                total = prior_cost + cost
                if total > budget:
                    continue
                candidate = (prior_error + error, allocation + [choice])
                if total not in expanded or candidate[0] < expanded[total][0]:
                    expanded[total] = candidate
        states_by_cost = expanded
    if budget not in states_by_cost:
        raise RuntimeError(f"no allocation exactly matches nominal {target_bits}-bit budget")
    score, allocation = states_by_cost[budget]
    return allocation, score


def optimize_direct_allocation(model, states, target_bits):
    candidates = [1, 2, 3] if target_bits == 2 else [2, 3, 4, 5]
    profiles = []
    for layer_index in range(1, len(states)):
        error = states[layer_index] - states[layer_index - 1]
        layer_records = []
        for bits in candidates:
            decoded_error, _ = quantize(error, bits)
            restored = states[layer_index - 1].float() + decoded_error.float()
            layer_records.append((
                bits, bits,
                projection_sse(model, layer_index, states[layer_index], restored),
            ))
        profiles.append(layer_records)
    budget = target_bits * (len(states) - 1)
    states_by_cost = {0: (0.0, [])}
    for layer_records in profiles:
        expanded = {}
        for prior_cost, (prior_error, allocation) in states_by_cost.items():
            for cost, choice, error in layer_records:
                total = prior_cost + cost
                if total > budget:
                    continue
                candidate = (prior_error + error, allocation + [choice])
                if total not in expanded or candidate[0] < expanded[total][0]:
                    expanded[total] = candidate
        states_by_cost = expanded
    score, allocation = states_by_cost[budget]
    return allocation, score


def decode_folded(states, specs, allocation):
    restored, payload_bytes = [], 0
    anchor, size = quantize(states[0], 8)
    restored.append(anchor.half())
    payload_bytes += size + 8
    for layer_index, (coarse_bits, detail_bits) in enumerate(allocation, 1):
        error = states[layer_index] - restored[-1]
        coarse, detail = fold(error, specs[layer_index - 1])
        decoded_coarse, coarse_size = quantize(coarse, coarse_bits)
        decoded_detail, detail_size = quantize(detail, detail_bits)
        decoded_error = unfold(decoded_coarse, decoded_detail, specs[layer_index - 1])
        restored.append((restored[-1].float() + decoded_error).half())
        payload_bytes += coarse_size + detail_size + 8
    return restored, payload_bytes


def evaluate(model, truth, restored, payload_bytes):
    truth_kv = sum(project_neox_kv(model, truth), [])
    restored_kv = sum(project_neox_kv(model, restored), [])
    residual_psnr, residual_mae, _ = measure(truth, restored)
    kv_psnr, kv_mae, _ = measure(truth_kv, restored_kv)
    return {
        "compression_ratio": tensor_bytes(truth) / payload_bytes,
        "payload_bytes": payload_bytes, "residual_psnr": residual_psnr,
        "residual_mae": residual_mae, "projected_kv_psnr": kv_psnr,
        "projected_kv_mae": kv_mae,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--fit-tokens", type=int, default=64)
    parser.add_argument("--profile-tokens", type=int, default=64)
    parser.add_argument("--evaluation-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    text = (
        "Efficient representations preserve consequences while assigning fewer "
        "bits to predictable variation and more bits to sensitive computation. " * 220
    )
    total = args.fit_tokens + args.profile_tokens + args.evaluation_tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=total)
    captured, hooks = [], []
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, call: captured.append(call[0][0].detach().cpu())
        ))
    with torch.no_grad():
        model(**inputs, use_cache=False)
    for hook in hooks:
        hook.remove()
    first, second = args.fit_tokens, args.fit_tokens + args.profile_tokens
    fit = [value[:first].half() for value in captured]
    profile = [value[first:second].half() for value in captured]
    evaluation = [value[second:].half() for value in captured]
    fit_deltas = [fit[index] - fit[index - 1] for index in range(1, len(fit))]
    specs: list[FoldSpec] = [
        fit_fold(delta, "correlation_lifting", args.seed + index)
        for index, delta in enumerate(fit_deltas, 1)
    ]
    config = GIHKCCV2Config(
        similarity_threshold=0.0, max_keyframe_span=len(evaluation) + 1,
        prediction_mode="adjacent",
    )
    records = []
    for target_bits in (2, 3):
        allocation, profile_score = optimize_allocation(
            model, profile, specs, target_bits
        )
        restored, payload_bytes = decode_folded(evaluation, specs, allocation)
        dynamic = {
            "strategy": "projection_aware_folded", "target_bits": target_bits,
            "allocation": allocation,
            "profile_projected_sse": profile_score,
        }
        dynamic.update(evaluate(model, evaluation, restored, payload_bytes))
        records.append(dynamic)
        uniform_allocation = [(target_bits, target_bits)] * (len(evaluation) - 1)
        restored, payload_bytes = decode_folded(evaluation, specs, uniform_allocation)
        uniform = {
            "strategy": "uniform_folded", "target_bits": target_bits,
            "allocation": uniform_allocation,
        }
        uniform.update(evaluate(model, evaluation, restored, payload_bytes))
        records.append(uniform)
        direct_allocation, direct_profile_score = optimize_direct_allocation(
            model, profile, target_bits
        )
        layer_bits = [8] + direct_allocation
        allocated_direct_stack = compress_predictive_stack(
            evaluation, 8, layer_bits, config
        )
        allocated_direct = {
            "strategy": "projection_aware_direct", "target_bits": target_bits,
            "allocation": direct_allocation,
            "profile_projected_sse": direct_profile_score,
        }
        allocated_direct.update(evaluate(
            model, evaluation, decompress_predictive_stack(allocated_direct_stack),
            allocated_direct_stack.compressed_bytes,
        ))
        records.append(allocated_direct)
        direct_stack = compress_predictive_stack(evaluation, 8, target_bits, config)
        direct = {
            "strategy": "uniform_direct", "target_bits": target_bits,
            "allocation": None,
        }
        direct.update(evaluate(
            model, evaluation, decompress_predictive_stack(direct_stack),
            direct_stack.compressed_bytes,
        ))
        records.append(direct)
        print(f"target={target_bits} dynamic_kv={dynamic['projected_kv_psnr']:.2f}dB "
              f"direct_kv={direct['projected_kv_psnr']:.2f}dB", flush=True)
    report = {
        "model": args.model, "fit_tokens": len(fit[0]),
        "profile_tokens": len(profile[0]), "evaluation_tokens": len(evaluation[0]),
        "fold_specs": [fold_spec_to_dict(spec) for spec in specs],
        "records": records, "complete": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
