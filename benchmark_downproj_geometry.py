"""Measure whether Qwen down_proj geometry exposes a useful packing basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def effective_rank(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.double().clamp_min(0)
    probabilities = values / values.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return entropy.exp().item()


def top_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    _, vectors = torch.linalg.eigh(matrix.double())
    return vectors[:, -rank:].float()


def covariance(samples: torch.Tensor) -> torch.Tensor:
    samples = samples.float()
    centered = samples - samples.mean(0, keepdim=True)
    return centered.T @ centered / max(1, samples.shape[0] - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--calibration", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--layers", default="5,11,17,23")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    selected = [int(value) for value in args.layers.split(",")]
    coefficients, messages, hooks = {}, {}, []
    for layer_idx in selected:
        module = model.model.layers[layer_idx].mlp.down_proj
        hooks.append(module.register_forward_pre_hook(
            lambda _module, call, index=layer_idx: coefficients.setdefault(
                index, call[0].detach().cpu().squeeze(0)
            )
        ))
        hooks.append(module.register_forward_hook(
            lambda _module, _call, output, index=layer_idx: messages.setdefault(
                index, output.detach().cpu().squeeze(0)
            )
        ))
    text = ("Models route features through learned projections and communicate useful updates. " * 100)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=args.tokens).input_ids.to(args.device)
    with torch.no_grad():
        model(ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    split = min(args.calibration, ids.shape[1] // 2)
    records = []
    print(f"model={args.model} tokens={ids.shape[1]} calibration={split} "
          f"holdout={ids.shape[1]-split} rank={args.rank}")
    print(f"{'layer':>5} {'basis':>12} {'msg MSE':>12} {'LN MSE':>12} "
          f"{'reader MSE':>12} {'eff rank':>10}")
    for layer_idx in selected:
        layer = model.model.layers[layer_idx]
        coeff_cal = coefficients[layer_idx][:split].float()
        message_cal = messages[layer_idx][:split].float()
        message_test = messages[layer_idx][split:].float()
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        next_layer = model.model.layers[min(layer_idx + 1, len(model.model.layers) - 1)]
        norm = next_layer.input_layernorm
        readers = [next_layer.self_attn.q_proj, next_layer.self_attn.k_proj,
                   next_layer.self_attn.v_proj]

        coefficient_variance = coeff_cal.var(0, unbiased=False)
        raw_gram = down @ down.T
        weighted = (down * coefficient_variance.sqrt().unsqueeze(0))
        weighted_gram = weighted @ weighted.T
        message_covariance = covariance(message_cal)
        identity = torch.eye(down.shape[0])
        bases = {
            "identity": identity[:, :args.rank],
            "down_svd": top_basis(raw_gram, args.rank),
            "act_down": top_basis(weighted_gram, args.rank),
            "message_pca": top_basis(message_covariance, args.rank),
        }
        eigenvalues = torch.linalg.eigvalsh(message_covariance.double())
        rank_value = effective_rank(eigenvalues)
        dtype = next(next_layer.parameters()).dtype
        device = next(next_layer.parameters()).device
        with torch.no_grad():
            target_norm = norm(message_test.to(device=device, dtype=dtype)).float()
            target_reader = torch.cat([
                reader(target_norm.to(dtype)).float() for reader in readers
            ], dim=-1)
        for name, basis in bases.items():
            restored = (message_test @ basis) @ basis.T
            with torch.no_grad():
                restored_norm = norm(restored.to(device=device, dtype=dtype)).float()
                restored_reader = torch.cat([
                    reader(restored_norm.to(dtype)).float() for reader in readers
                ], dim=-1)
            record = {
                "layer": layer_idx, "basis": name, "rank": args.rank,
                "message_mse": (restored - message_test).square().mean().item(),
                "normalized_mse": (restored_norm.cpu() - target_norm.cpu()).square().mean().item(),
                "reader_mse": (restored_reader.cpu() - target_reader.cpu()).square().mean().item(),
                "message_effective_rank": rank_value,
            }
            records.append(record)
            print(f"{layer_idx:5d} {name:>12} {record['message_mse']:12.5g} "
                  f"{record['normalized_mse']:12.5g} {record['reader_mse']:12.5g} "
                  f"{rank_value:10.2f}")
    report = {"model": args.model, "tokens": ids.shape[1], "calibration": split,
              "rank": args.rank, "records": records}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
