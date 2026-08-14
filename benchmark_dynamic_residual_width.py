"""Test oracle sequence-by-layer dynamic residual width on Pythia-style models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark_downproj_geometry import top_basis
from benchmark_dynamic_mlp_compression import allocate, metrics
from scan_downproj_geometry import model_layers


ORACLE_TEXT = (
    "Scientific models become useful when their predictions survive comparison "
    "with observations. Engineers improve systems by measuring bottlenecks, "
    "testing alternatives, and preserving the constraints that matter. " * 100
)
TRANSFER_TEXT = (
    "The old library stood beside the river, where generations of readers had "
    "left notes in the margins. A careful story makes later choices follow from "
    "what its characters previously learned and misunderstood. " * 100
)


def reader_gram(layer):
    """Residual-input geometry induced by the layer's linear readers."""
    matrices = []
    if hasattr(layer, "attention") and hasattr(layer.attention, "query_key_value"):
        matrices.append(layer.attention.query_key_value.weight.detach().float().cpu())
    elif hasattr(layer, "self_attn"):
        for name in ("q_proj", "k_proj", "v_proj"):
            matrices.append(getattr(layer.self_attn, name).weight.detach().float().cpu())
    else:
        raise ValueError(f"Unsupported attention module in {type(layer).__name__}")
    mlp = layer.mlp
    if hasattr(mlp, "dense_h_to_4h"):
        matrices.append(mlp.dense_h_to_4h.weight.detach().float().cpu())
    else:
        for name in ("gate_proj", "up_proj"):
            matrices.append(getattr(mlp, name).weight.detach().float().cpu())
    width = matrices[0].shape[1]
    gram = torch.zeros((width, width), dtype=torch.float32)
    for matrix in matrices:
        gram.add_(matrix.T @ matrix)
    return gram


def run_with_widths(model, token_ids, bases, widths):
    hooks = []
    for layer, basis, width in zip(model_layers(model), bases, widths):
        if width >= basis.shape[1]:
            continue
        local = basis[:, :width]

        def project(_module, call, local_basis=local):
            hidden = call[0]
            projected = (
                (hidden.float() @ local_basis.to(hidden.device))
                @ local_basis.to(hidden.device).T
            ).to(hidden.dtype)
            return (projected, *call[1:])

        hooks.append(layer.register_forward_pre_hook(project))
    with torch.no_grad():
        logits = model(token_ids, use_cache=False).logits.float().cpu()
    for hook in hooks:
        hook.remove()
    return logits


def profile_sequence(model, token_ids, baseline, bases, ranks):
    profile = []
    layers = model_layers(model)
    full_width = bases[0].shape[1]
    for index, layer in enumerate(layers):
        records = []
        for rank in ranks:
            widths = [full_width] * len(layers)
            widths[index] = rank
            logits = run_with_widths(model, token_ids, bases, widths)
            record = {"layer": index, "rank": rank}
            record.update(metrics(baseline, logits, token_ids))
            records.append(record)
            print(f"profile layer={index:02d} rank={rank:04d} "
                  f"kl={record['mean_kl']:.6f}", flush=True)
        profile.append(records)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--ranks", default="256,512,768,1024")
    parser.add_argument("--average-ranks", default="512,768,896,960")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    encode = lambda text: tokenizer(
        text, return_tensors="pt", truncation=True, max_length=args.tokens
    ).input_ids.to(args.device)
    oracle_ids, transfer_ids = encode(ORACLE_TEXT), encode(TRANSFER_TEXT)
    with torch.no_grad():
        oracle_baseline = model(oracle_ids, use_cache=False).logits.float().cpu()
        transfer_baseline = model(transfer_ids, use_cache=False).logits.float().cpu()
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    average_ranks = [int(value) for value in args.average_ranks.split(",")]
    layers = model_layers(model)
    full_width = layers[0].hidden_size if hasattr(layers[0], "hidden_size") else model.config.hidden_size
    if ranks[-1] != full_width:
        raise ValueError("Ranks must include full residual width")
    bases = [top_basis(reader_gram(layer), full_width).flip(1) for layer in layers]
    profile = profile_sequence(model, oracle_ids, oracle_baseline, bases, ranks)

    validation = []
    for average_rank in average_ranks:
        budget = average_rank * len(layers)
        allocation, used, predicted = allocate(profile, ranks, budget)
        for passage, token_ids, baseline in (
            ("oracle", oracle_ids, oracle_baseline),
            ("transfer", transfer_ids, transfer_baseline),
        ):
            dynamic_logits = run_with_widths(model, token_ids, bases, allocation)
            dynamic = {
                "strategy": "dynamic", "passage": passage,
                "average_rank_budget": average_rank, "total_rank": used,
                "allocation": allocation, "profile_additive_kl": predicted,
            }
            dynamic.update(metrics(baseline, dynamic_logits, token_ids))
            validation.append(dynamic)
            uniform_logits = run_with_widths(
                model, token_ids, bases, [average_rank] * len(layers)
            )
            uniform = {
                "strategy": "uniform", "passage": passage,
                "average_rank_budget": average_rank,
                "total_rank": average_rank * len(layers),
                "allocation": [average_rank] * len(layers),
            }
            uniform.update(metrics(baseline, uniform_logits, token_ids))
            validation.append(uniform)
        print(f"validated average_rank={average_rank}", flush=True)

    report = {
        "model": args.model, "tokens": args.tokens, "ranks": ranks,
        "average_rank_budgets": average_ranks, "basis": "next_reader_weight_gram",
        "profile_passage": "oracle", "profile": profile,
        "validation": validation, "complete": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
