"""Scan MLP packing geometry across every Qwen layer under live intervention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_downproj_geometry import covariance, effective_rank, top_basis


def capture_downproj(model, token_ids):
    coefficients, messages, hooks = {}, {}, []
    for layer_idx, layer in enumerate(model.model.layers):
        module = layer.mlp.down_proj
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
    with torch.no_grad():
        logits = model(token_ids, use_cache=False).logits.float().cpu()
    for hook in hooks:
        hook.remove()
    return coefficients, messages, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--calibration-tokens", type=int, default=256)
    parser.add_argument("--evaluation-tokens", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    calibration_text = (
        "Science compares explanations against measurements. Software decomposes problems into explicit operations. "
        "History records how institutions and ideas change. Dialogue requires tracking intent and context. " * 80
    )
    evaluation_text = (
        "Gardens adapt to climate through soil, water, and careful observation. Music develops expectation through rhythm and harmony. "
        "A useful summary preserves causal structure while removing repetition. " * 80
    )
    calibration_ids = tokenizer(
        calibration_text, return_tensors="pt", truncation=True,
        max_length=args.calibration_tokens,
    ).input_ids.to(args.device)
    evaluation_ids = tokenizer(
        evaluation_text, return_tensors="pt", truncation=True,
        max_length=args.evaluation_tokens,
    ).input_ids.to(args.device)
    calibration_coefficients, calibration_messages, _ = capture_downproj(
        model, calibration_ids
    )
    _, evaluation_messages, baseline_logits = capture_downproj(model, evaluation_ids)
    target = evaluation_ids[:, 1:].cpu()
    baseline_nll = F.cross_entropy(
        baseline_logits[:, :-1].reshape(-1, baseline_logits.shape[-1]),
        target.reshape(-1), reduction="mean",
    ).item()
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} layers={len(model.model.layers)} rank={args.rank} "
          f"cal={calibration_ids.shape[1]} eval={evaluation_ids.shape[1]}")
    print(f"{'layer':>5} {'basis':>12} {'msg MSE':>12} {'KL':>11} "
          f"{'PPL change':>12} {'top1':>9} {'eff rank':>10}")
    for layer_idx, layer in enumerate(model.model.layers):
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        variance = calibration_coefficients[layer_idx].float().var(0, unbiased=False)
        weighted = down * variance.sqrt().unsqueeze(0)
        permutation = torch.randperm(variance.numel(), generator=generator)
        shuffled = down * variance[permutation].sqrt().unsqueeze(0)
        message_covariance = covariance(calibration_messages[layer_idx])
        bases = {
            "down_svd": top_basis(down @ down.T, args.rank),
            "act_down": top_basis(weighted @ weighted.T, args.rank),
            "shuffled": top_basis(shuffled @ shuffled.T, args.rank),
            "message_pca": top_basis(message_covariance, args.rank),
        }
        eigenvalues = torch.linalg.eigvalsh(message_covariance.double())
        rank_value = effective_rank(eigenvalues)
        message_test = evaluation_messages[layer_idx].float()
        for basis_name, basis in bases.items():
            restored = (message_test @ basis) @ basis.T
            message_mse = (restored - message_test).square().mean().item()
            module = layer.mlp.down_proj
            hook = module.register_forward_hook(
                lambda _module, _call, output, local_basis=basis: (
                    (output.float() @ local_basis.to(output.device))
                    @ local_basis.to(output.device).T
                ).to(output.dtype)
            )
            with torch.no_grad():
                modified_logits = model(evaluation_ids, use_cache=False).logits.float().cpu()
            hook.remove()
            modified_nll = F.cross_entropy(
                modified_logits[:, :-1].reshape(-1, modified_logits.shape[-1]),
                target.reshape(-1), reduction="mean",
            ).item()
            baseline_log_probs = F.log_softmax(baseline_logits[:, :-1], dim=-1)
            modified_log_probs = F.log_softmax(modified_logits[:, :-1], dim=-1)
            kl = F.kl_div(
                modified_log_probs, baseline_log_probs.exp(), reduction="batchmean"
            ).item() / max(1, evaluation_ids.shape[1] - 1)
            agreement = (
                baseline_logits[:, :-1].argmax(-1)
                == modified_logits[:, :-1].argmax(-1)
            ).float().mean().item()
            record = {
                "layer": layer_idx, "basis": basis_name, "rank": args.rank,
                "message_mse": message_mse, "mean_kl": kl,
                "ppl_change": math.exp(modified_nll - baseline_nll) - 1,
                "top1_agreement": agreement,
                "message_effective_rank": rank_value,
            }
            records.append(record)
            print(f"{layer_idx:5d} {basis_name:>12} {message_mse:12.5g} "
                  f"{kl:11.6f} {record['ppl_change']:12.2%} "
                  f"{agreement:9.2%} {rank_value:10.2f}", flush=True)
            checkpoint = {
                "model": args.model, "rank": args.rank, "seed": args.seed,
                "calibration_tokens": calibration_ids.shape[1],
                "evaluation_tokens": evaluation_ids.shape[1],
                "baseline_ppl": math.exp(baseline_nll), "complete": False,
                "records": records,
            }
            output.write_text(
                json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
            )
    report = {
        "model": args.model, "rank": args.rank, "seed": args.seed,
        "calibration_tokens": calibration_ids.shape[1],
        "evaluation_tokens": evaluation_ids.shape[1],
        "baseline_ppl": math.exp(baseline_nll), "records": records,
    }
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
