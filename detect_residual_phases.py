"""Detect candidate transformer-depth phase boundaries from layer updates."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean(0, keepdim=True)
    right = right.float() - right.float().mean(0, keepdim=True)
    cross = torch.linalg.norm(left.T @ right).square()
    denominator = (
        torch.linalg.norm(left.T @ left) * torch.linalg.norm(right.T @ right)
    ).clamp_min(1e-12)
    return (cross / denominator).item()


def choose_boundaries(scores, layers: int, phases: int, minimum: int):
    candidates = range(minimum, layers - minimum + 1)
    best = None
    for selected in itertools.combinations(candidates, phases - 1):
        edges = (0,) + selected + (layers,)
        if min(b - a for a, b in zip(edges[:-1], edges[1:])) < minimum:
            continue
        score = sum(scores[index - 1]["score"] for index in selected)
        if best is None or score > best[0]:
            best = score, edges
    return list(best[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--minimum-layers", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    text = ("Models identify context, develop alternatives, produce answers, and refine outputs. " * 100)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=args.tokens).input_ids.to(args.device)
    inputs, outputs, hooks = [], [], []
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, call: inputs.append(call[0][0].detach().cpu())
        ))
        hooks.append(layer.register_forward_hook(
            lambda _module, _call, output: outputs.append(output[0].detach().cpu())
        ))
    with torch.no_grad():
        model(ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    updates = [output.float() - input_.float()
               for input_, output in zip(inputs, outputs)]
    scores = []
    for index in range(len(updates) - 1):
        similarity = linear_cka(updates[index], updates[index + 1])
        left_norm = updates[index].norm(dim=-1).mean().item()
        right_norm = updates[index + 1].norm(dim=-1).mean().item()
        norm_shift = abs(torch.log(torch.tensor(right_norm / left_norm)).item())
        scores.append({
            "boundary": index + 1,
            "cka": similarity,
            "norm_shift": norm_shift,
            "score": 1 - similarity + 0.25 * norm_shift,
        })
    boundaries = choose_boundaries(
        scores, len(updates), args.phases, args.minimum_layers
    )
    report = {"model": args.model, "tokens": args.tokens,
              "boundaries": boundaries, "scores": scores}
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
