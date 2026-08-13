"""Test whether learned residual bases improve fixed-budget compression."""

from __future__ import annotations

import argparse

import torch

from residual_repacking import (
    allocate_mixed_bits, fit_repacking_basis, quantize_repacked,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--calibration", type=int, default=128)
    parser.add_argument("--average-bits", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    text = ("Compression should preserve information that downstream readers can see. " * 100)
    token_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=args.tokens).input_ids.to(args.device)
    captured, hooks = [], []
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, arguments: captured.append(arguments[0][0].detach().cpu())
        ))
    with torch.no_grad():
        model(token_ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    split = min(args.calibration, captured[0].shape[0] // 2)
    print(f"model={args.model} tokens={captured[0].shape[0]} calibration={split} "
          f"evaluation={captured[0].shape[0] - split} avg_bits={args.average_bits}")
    print(f"{'method':10} {'residual MSE':>14} {'QKV MSE':>14} {'ratio':>9}")
    totals = {name: [0.0, 0.0, 0, 0] for name in ("identity", "random", "pca", "reader")}

    for layer_idx in range(1, len(captured)):
        calibration_delta = captured[layer_idx][:split] - captured[layer_idx - 1][:split]
        evaluation_delta = captured[layer_idx][split:] - captured[layer_idx - 1][split:]
        layer = model.gpt_neox.layers[layer_idx]
        weight = layer.attention.query_key_value.weight.detach().float().cpu()
        reader_gram = weight.T @ weight
        truth_qkv = evaluation_delta.float() @ weight.T
        original_bytes = evaluation_delta.numel() * 2
        for name in totals:
            basis = fit_repacking_basis(calibration_delta, name, reader_gram)
            bits = allocate_mixed_bits(basis.importance, args.average_bits)
            decoded, payload_bytes = quantize_repacked(evaluation_delta.half(), basis, bits)
            residual_mse = (decoded.float() - evaluation_delta.float()).square().mean().item()
            projected = decoded.float() @ weight.T
            qkv_mse = (projected - truth_qkv).square().mean().item()
            accumulator = totals[name]
            accumulator[0] += residual_mse
            accumulator[1] += qkv_mse
            accumulator[2] += original_bytes
            accumulator[3] += payload_bytes

    count = len(captured) - 1
    for name, (residual_mse, qkv_mse, original, payload) in totals.items():
        print(f"{name:10} {residual_mse/count:14.7g} {qkv_mse/count:14.7g} "
              f"{original/payload:8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

