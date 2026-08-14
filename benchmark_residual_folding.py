"""Compare reversible residual folding under equal nominal bit budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark_compression import measure, tensor_bytes
from benchmark_real_residual import project_neox_kv
from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack, decompress_predictive_stack
from residual_folding import fit_fold, fold, unfold
from turboquant_paper import paper_turboquant_compress, paper_turboquant_decompress


def quantize(values, bits):
    payload = paper_turboquant_compress(values, bits)
    return paper_turboquant_decompress(payload), payload.compressed_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--calibration-tokens", type=int, default=128)
    parser.add_argument("--evaluation-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    text = (
        "Compression should preserve information used by later computation while "
        "reorganizing predictable structure into inexpensive detail streams. " * 160
    )
    total = args.calibration_tokens + args.evaluation_tokens
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
    calibration = [value[:args.calibration_tokens].half() for value in captured]
    evaluation = [value[args.calibration_tokens:].half() for value in captured]
    original_kv = sum(project_neox_kv(model, evaluation), [])
    original_bytes = tensor_bytes(evaluation)
    modes = ["direct", "adjacent_haar", "random_lifting", "correlation_lifting"]
    budgets = {
        3: [(3, 3), (4, 2), (5, 1)],
        4: [(4, 4), (5, 3), (6, 2)],
    }
    records = []
    for nominal_bits, allocations in budgets.items():
        for mode in modes:
            mode_allocations = [(nominal_bits, nominal_bits)] if mode == "direct" else allocations
            for coarse_bits, detail_bits in mode_allocations:
                restored, payload_bytes = [], 0
                detail_energy = []
                for index, values in enumerate(evaluation):
                    if mode == "direct":
                        decoded, size = quantize(values, nominal_bits)
                    else:
                        spec = fit_fold(calibration[index], mode, args.seed + index)
                        coarse, detail = fold(values, spec)
                        decoded_coarse, coarse_size = quantize(coarse, coarse_bits)
                        decoded_detail, detail_size = quantize(detail, detail_bits)
                        decoded = unfold(decoded_coarse, decoded_detail, spec).half()
                        size = coarse_size + detail_size
                        detail_energy.append(
                            (detail.square().mean() / values.float().square().mean().clamp_min(1e-12)).item()
                        )
                    restored.append(decoded.to(values.dtype))
                    payload_bytes += size
                psnr, mae, mse = measure(evaluation, restored)
                restored_kv = sum(project_neox_kv(model, restored), [])
                kv_psnr, kv_mae, _ = measure(original_kv, restored_kv)
                record = {
                    "experiment": "direct_state", "mode": mode,
                    "nominal_bits": nominal_bits,
                    "coarse_bits": coarse_bits if mode != "direct" else None,
                    "detail_bits": detail_bits if mode != "direct" else None,
                    "compression_ratio": original_bytes / payload_bytes,
                    "payload_bytes": payload_bytes, "residual_psnr": psnr,
                    "residual_mae": mae, "residual_mse": mse,
                    "projected_kv_psnr": kv_psnr, "projected_kv_mae": kv_mae,
                    "mean_relative_detail_energy": (
                        sum(detail_energy) / len(detail_energy) if detail_energy else None
                    ),
                }
                records.append(record)
                print(
                    f"{mode:20s} budget={nominal_bits}b alloc={coarse_bits}/{detail_bits} "
                    f"ratio={record['compression_ratio']:.2f}x "
                    f"res={psnr:.2f}dB kv={kv_psnr:.2f}dB detail={record['mean_relative_detail_energy']}",
                    flush=True,
                )

    # Fold the actual decoder-visible adjacent prediction error. Fold specs are
    # fit on calibration deltas, never on evaluation tokens.
    config = GIHKCCV2Config(
        similarity_threshold=0.0, max_keyframe_span=len(evaluation) + 1,
        prediction_mode="adjacent",
    )
    for delta_bits in (2, 3):
        direct_stack = compress_predictive_stack(evaluation, 8, delta_bits, config)
        direct_restored = decompress_predictive_stack(direct_stack)
        direct_kv = sum(project_neox_kv(model, direct_restored), [])
        psnr, mae, mse = measure(evaluation, direct_restored)
        kv_psnr, kv_mae, _ = measure(original_kv, direct_kv)
        records.append({
            "experiment": "adjacent_closed_loop", "mode": "direct",
            "anchor_bits": 8, "nominal_bits": delta_bits,
            "coarse_bits": None, "detail_bits": None,
            "compression_ratio": original_bytes / direct_stack.compressed_bytes,
            "payload_bytes": direct_stack.compressed_bytes,
            "residual_psnr": psnr, "residual_mae": mae, "residual_mse": mse,
            "projected_kv_psnr": kv_psnr, "projected_kv_mae": kv_mae,
            "mean_relative_detail_energy": None,
        })
        print(f"GIHKCC direct {delta_bits}b ratio={original_bytes/direct_stack.compressed_bytes:.2f}x "
              f"res={psnr:.2f}dB kv={kv_psnr:.2f}dB", flush=True)
        allocations = [(delta_bits, delta_bits), (delta_bits + 1, delta_bits - 1)]
        calibration_deltas = [
            calibration[i] - calibration[i - 1] for i in range(1, len(calibration))
        ]
        for mode in modes[1:]:
            specs = [fit_fold(delta, mode, args.seed + i) for i, delta in enumerate(calibration_deltas, 1)]
            for coarse_bits, detail_bits in allocations:
                restored = []
                anchor, anchor_size = quantize(evaluation[0], 8)
                restored.append(anchor.half())
                payload_bytes = anchor_size
                detail_energy = []
                for layer_index in range(1, len(evaluation)):
                    prediction_error = evaluation[layer_index] - restored[-1]
                    coarse, detail = fold(prediction_error, specs[layer_index - 1])
                    decoded_coarse, coarse_size = quantize(coarse, coarse_bits)
                    decoded_detail, detail_size = quantize(detail, detail_bits)
                    decoded_error = unfold(
                        decoded_coarse, decoded_detail, specs[layer_index - 1]
                    )
                    restored.append((restored[-1].float() + decoded_error).half())
                    payload_bytes += coarse_size + detail_size + 8
                    detail_energy.append(
                        (detail.square().mean() / prediction_error.float().square().mean().clamp_min(1e-12)).item()
                    )
                psnr, mae, mse = measure(evaluation, restored)
                restored_kv = sum(project_neox_kv(model, restored), [])
                kv_psnr, kv_mae, _ = measure(original_kv, restored_kv)
                record = {
                    "experiment": "adjacent_closed_loop", "mode": mode,
                    "anchor_bits": 8, "nominal_bits": delta_bits,
                    "coarse_bits": coarse_bits, "detail_bits": detail_bits,
                    "compression_ratio": original_bytes / payload_bytes,
                    "payload_bytes": payload_bytes, "residual_psnr": psnr,
                    "residual_mae": mae, "residual_mse": mse,
                    "projected_kv_psnr": kv_psnr, "projected_kv_mae": kv_mae,
                    "mean_relative_detail_energy": sum(detail_energy) / len(detail_energy),
                }
                records.append(record)
                print(f"GIHKCC {mode} {coarse_bits}/{detail_bits} ratio={record['compression_ratio']:.2f}x "
                      f"res={psnr:.2f}dB kv={kv_psnr:.2f}dB", flush=True)
    report = {
        "model": args.model, "calibration_tokens": args.calibration_tokens,
        "evaluation_tokens": evaluation[0].shape[0], "records": records,
        "complete": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
