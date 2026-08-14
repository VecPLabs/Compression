"""Profile and validate layer-specific MLP message rank allocation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_downproj_geometry import top_basis
from scan_downproj_geometry import capture_downproj, model_layers, output_projection


CALIBRATION_TEXT = (
    "Science compares explanations against measurements. Software decomposes "
    "problems into explicit operations. History records how institutions and "
    "ideas change. Dialogue requires tracking intent and context. " * 100
)
PROFILE_TEXT = (
    "Gardens adapt to climate through soil, water, and careful observation. "
    "Music develops expectation through rhythm and harmony. A useful summary "
    "preserves causal structure while removing repetition. " * 100
)
VALIDATION_TEXT = (
    "Navigation combines a destination with a changing map of nearby choices. "
    "Economic decisions balance limited resources against uncertain outcomes. "
    "Clear explanations connect evidence to conclusions without hiding assumptions. " * 100
)


def metrics(baseline_logits, modified_logits, token_ids):
    target = token_ids[:, 1:].cpu()
    baseline = baseline_logits[:, :-1]
    modified = modified_logits[:, :-1]
    baseline_nll = F.cross_entropy(
        baseline.reshape(-1, baseline.shape[-1]), target.reshape(-1)
    ).item()
    modified_nll = F.cross_entropy(
        modified.reshape(-1, modified.shape[-1]), target.reshape(-1)
    ).item()
    kl = F.kl_div(
        F.log_softmax(modified, dim=-1), F.softmax(baseline, dim=-1),
        reduction="batchmean",
    ).item() / max(1, token_ids.shape[1] - 1)
    agreement = (baseline.argmax(-1) == modified.argmax(-1)).float().mean().item()
    return {
        "mean_kl": kl,
        "ppl_change": math.exp(modified_nll - baseline_nll) - 1,
        "top1_agreement": agreement,
    }


def run_with_bases(model, token_ids, bases):
    hooks = []
    for layer, basis in zip(model_layers(model), bases):
        projection = output_projection(layer)
        hooks.append(projection.register_forward_hook(
            lambda _module, _call, output, local_basis=basis: (
                (output.float() @ local_basis.to(output.device))
                @ local_basis.to(output.device).T
            ).to(output.dtype)
        ))
    with torch.no_grad():
        logits = model(token_ids, use_cache=False).logits.float().cpu()
    for hook in hooks:
        hook.remove()
    return logits


def allocate(profile, ranks, budget):
    """Exact multiple-choice knapsack minimizing additive single-layer KL."""
    states = {0: (0.0, [])}
    for layer_records in profile:
        next_states = {}
        by_rank = {record["rank"]: record["mean_kl"] for record in layer_records}
        for used, (cost, choices) in states.items():
            for rank in ranks:
                total = used + rank
                if total > budget:
                    continue
                candidate = (cost + by_rank[rank], choices + [rank])
                if total not in next_states or candidate[0] < next_states[total][0]:
                    next_states[total] = candidate
        states = next_states
    feasible = [(used, value) for used, value in states.items() if used <= budget]
    used, (cost, choices) = min(feasible, key=lambda item: (item[1][0], -item[0]))
    return choices, used, cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--calibration-tokens", type=int, default=256)
    parser.add_argument("--profile-tokens", type=int, default=64)
    parser.add_argument("--validation-tokens", type=int, default=64)
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--average-ranks", default="32,64,96")
    parser.add_argument("--profile-input", help="Reuse profile records from a prior run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    average_ranks = [int(value) for value in args.average_ranks.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    encode = lambda text, length: tokenizer(
        text, return_tensors="pt", truncation=True, max_length=length
    ).input_ids.to(args.device)
    calibration_ids = encode(CALIBRATION_TEXT, args.calibration_tokens)
    profile_ids = encode(PROFILE_TEXT, args.profile_tokens)
    validation_ids = encode(VALIDATION_TEXT, args.validation_tokens)

    coefficients, _, _ = capture_downproj(model, calibration_ids)
    _, _, profile_baseline = capture_downproj(model, profile_ids)
    _, _, validation_baseline = capture_downproj(model, validation_ids)
    layers = model_layers(model)
    max_rank = max(ranks)
    full_bases = []
    for index, layer in enumerate(layers):
        weight = output_projection(layer).weight.detach().float().cpu()
        variance = coefficients[index].float().var(0, unbiased=False)
        weighted = weight * variance.sqrt().unsqueeze(0)
        # top_basis returns the selected eigenvectors in ascending eigenvalue
        # order. Reverse them so every prefix is the correct top-r subspace.
        full_bases.append(top_basis(weighted @ weighted.T, max_rank).flip(1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_input:
        prior = json.loads(Path(args.profile_input).read_text(encoding="utf-8"))
        if prior["ranks"] != ranks or len(prior["profile"]) != len(layers):
            raise ValueError("Reused profile ranks or layer count do not match")
        profile = prior["profile"]
    else:
        profile = []
        for index, layer in enumerate(layers):
            records = []
            projection = output_projection(layer)
            for rank in ranks:
                basis = full_bases[index][:, :rank]
                hook = projection.register_forward_hook(
                    lambda _module, _call, value, local_basis=basis: (
                        (value.float() @ local_basis.to(value.device))
                        @ local_basis.to(value.device).T
                    ).to(value.dtype)
                )
                with torch.no_grad():
                    logits = model(profile_ids, use_cache=False).logits.float().cpu()
                hook.remove()
                record = {"layer": index, "rank": rank}
                record.update(metrics(profile_baseline, logits, profile_ids))
                records.append(record)
                print(f"profile layer={index:02d} rank={rank:03d} "
                      f"kl={record['mean_kl']:.6f}", flush=True)
            profile.append(records)

    validations = []
    for average_rank in average_ranks:
        budget = average_rank * len(layers)
        allocation, used, predicted_kl = allocate(profile, ranks, budget)
        dynamic_logits = run_with_bases(
            model, validation_ids,
            [basis[:, :rank] for basis, rank in zip(full_bases, allocation)],
        )
        dynamic = {
            "strategy": "dynamic", "average_rank_budget": average_rank,
            "total_rank": used, "allocation": allocation,
            "profile_additive_kl": predicted_kl,
        }
        dynamic.update(metrics(validation_baseline, dynamic_logits, validation_ids))
        validations.append(dynamic)

        if average_rank <= max_rank:
            uniform_logits = run_with_bases(
                model, validation_ids,
                [basis[:, :average_rank] for basis in full_bases],
            )
            uniform = {
                "strategy": "uniform", "average_rank_budget": average_rank,
                "total_rank": budget, "allocation": [average_rank] * len(layers),
            }
            uniform.update(metrics(validation_baseline, uniform_logits, validation_ids))
            validations.append(uniform)
        print(f"validate avg_rank={average_rank} dynamic_kl={dynamic['mean_kl']:.6f}",
              flush=True)

    report = {
        "model": args.model, "ranks": ranks, "average_rank_budgets": average_ranks,
        "calibration_tokens": calibration_ids.shape[1],
        "profile_tokens": profile_ids.shape[1],
        "validation_tokens": validation_ids.shape[1],
        "profile": profile, "validation": validations, "complete": True,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
