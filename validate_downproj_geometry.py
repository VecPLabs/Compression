"""Intervene on Qwen down_proj outputs using calibrated low-rank geometries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_downproj_geometry import covariance, top_basis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--calibration-tokens", type=int, default=64)
    parser.add_argument("--evaluation-tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
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
    calibration_text = ("Models organize activated features into useful messages for later computation. " * 80)
    evaluation_text = ("A separate evaluation passage tests whether learned geometry preserves predictions. " * 80)
    calibration_ids = tokenizer(
        calibration_text, return_tensors="pt", truncation=True,
        max_length=args.calibration_tokens,
    ).input_ids.to(args.device)
    evaluation_ids = tokenizer(
        evaluation_text, return_tensors="pt", truncation=True,
        max_length=args.evaluation_tokens,
    ).input_ids.to(args.device)
    with torch.no_grad():
        model(calibration_ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    bases = {"act_down": {}, "message_pca": {}}
    for layer_idx in selected:
        layer = model.model.layers[layer_idx]
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        variance = coefficients[layer_idx].float().var(0, unbiased=False)
        weighted = down * variance.sqrt().unsqueeze(0)
        bases["act_down"][layer_idx] = top_basis(
            weighted @ weighted.T, args.rank
        )
        bases["message_pca"][layer_idx] = top_basis(
            covariance(messages[layer_idx]), args.rank
        )
    with torch.no_grad():
        baseline_logits = model(evaluation_ids, use_cache=False).logits.float()
    target = evaluation_ids[:, 1:]
    baseline_nll = F.cross_entropy(
        baseline_logits[:, :-1].reshape(-1, baseline_logits.shape[-1]),
        target.reshape(-1), reduction="mean",
    ).item()
    records = []
    print(f"model={args.model} rank={args.rank} layers={selected}")
    print(f"{'basis':>12} {'scope':>12} {'PPL change':>12} {'KL':>12} {'top1':>10}")
    scopes = [(f"layer-{layer_idx}", [layer_idx]) for layer_idx in selected]
    scopes.append(("all", selected))
    for basis_name, by_layer in bases.items():
        for scope_name, scope_layers in scopes:
            intervention_hooks = []
            for layer_idx in scope_layers:
                basis = by_layer[layer_idx]
                module = model.model.layers[layer_idx].mlp.down_proj
                intervention_hooks.append(module.register_forward_hook(
                    lambda _module, _call, output, local_basis=basis: (
                        (output.float() @ local_basis.to(output.device))
                        @ local_basis.to(output.device).T
                    ).to(output.dtype)
                ))
            with torch.no_grad():
                modified_logits = model(evaluation_ids, use_cache=False).logits.float()
            for hook in intervention_hooks:
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
                "basis": basis_name, "scope": scope_name,
                "ppl_change": math.exp(modified_nll - baseline_nll) - 1,
                "mean_kl": kl, "top1_agreement": agreement,
            }
            records.append(record)
            print(f"{basis_name:>12} {scope_name:>12} {record['ppl_change']:12.2%} "
                  f"{kl:12.6f} {agreement:10.2%}")
    report = {"model": args.model, "rank": args.rank, "layers": selected,
              "baseline_ppl": math.exp(baseline_nll), "records": records}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
