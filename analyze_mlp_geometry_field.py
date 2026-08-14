"""Measure depth-, prompt-, and gradient-conditioned MLP write geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_downproj_geometry import covariance, top_basis
from scan_downproj_geometry import model_layers, output_projection


DOMAINS = {
    "science": (
        "Experiments distinguish competing explanations through controlled evidence. "
        "Physical models connect measurements with mechanisms and uncertainty. " * 80
    ),
    "narrative": (
        "The traveler reached the quiet station and remembered why the journey began. "
        "Each conversation changed what the characters expected to happen next. " * 80
    ),
    "code": (
        "A function transforms explicit inputs into outputs while preserving its contract. "
        "Tests isolate failures and make assumptions visible to future maintainers. " * 80
    ),
}

EVALUATION_TEXT = (
    "Cities coordinate transportation, energy, communication, and public services. "
    "Good plans respond to new evidence without losing sight of their original goals. " * 80
)


def capture_geometry(model, token_ids):
    coefficients, messages, gradients, hooks = {}, {}, {}, []
    for index, layer in enumerate(model_layers(model)):
        projection = output_projection(layer)
        hooks.append(projection.register_forward_pre_hook(
            lambda _module, call, i=index: coefficients.setdefault(i, call[0])
        ))

        def save_output(_module, _call, output, i=index):
            messages[i] = output
            output.retain_grad()

        hooks.append(projection.register_forward_hook(save_output))
    model.zero_grad(set_to_none=True)
    logits = model(token_ids, use_cache=False).logits.float()
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), token_ids[:, 1:].reshape(-1)
    )
    loss.backward()
    for index, message in messages.items():
        gradients[index] = message.grad.detach().cpu()
        messages[index] = message.detach().cpu()
        coefficients[index] = coefficients[index].detach().cpu()
    for hook in hooks:
        hook.remove()
    model.zero_grad(set_to_none=True)
    return coefficients, messages, gradients


def subspace_similarity(left, right):
    rank = min(left.shape[1], right.shape[1])
    return ((left.T @ right).square().sum() / rank).item()


def joint_basis(write_covariance, fisher, rank):
    values, vectors = torch.linalg.eigh(write_covariance.double())
    positive = values.clamp_min(0)
    root = (vectors * positive.sqrt().unsqueeze(0)) @ vectors.T
    interaction = root @ fisher.double() @ root
    directions = top_basis(interaction.float(), rank).double()
    candidates = root @ directions
    basis, _ = torch.linalg.qr(candidates, mode="reduced")
    return basis[:, :rank].float()


def intervention_metrics(model, token_ids, baseline_logits, layer, basis):
    projection = output_projection(layer)
    hook = projection.register_forward_hook(
        lambda _module, _call, output: (
            (output.float() @ basis.to(output.device)) @ basis.to(output.device).T
        ).to(output.dtype)
    )
    with torch.no_grad():
        modified = model(token_ids, use_cache=False).logits.float().cpu()
    hook.remove()
    baseline = baseline_logits[:, :-1]
    changed = modified[:, :-1]
    target = token_ids[:, 1:].cpu()
    baseline_nll = F.cross_entropy(
        baseline.reshape(-1, baseline.shape[-1]), target.reshape(-1)
    ).item()
    changed_nll = F.cross_entropy(
        changed.reshape(-1, changed.shape[-1]), target.reshape(-1)
    ).item()
    kl = F.kl_div(
        F.log_softmax(changed, dim=-1), F.softmax(baseline, dim=-1),
        reduction="batchmean",
    ).item() / max(1, token_ids.shape[1] - 1)
    return {
        "mean_kl": kl,
        "ppl_change": math.exp(changed_nll - baseline_nll) - 1,
        "top1_agreement": (
            baseline.argmax(-1) == changed.argmax(-1)
        ).float().mean().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--evaluation-tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--phase-boundaries", default="5,9,20")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    encode = lambda text, length: tokenizer(
        text, return_tensors="pt", truncation=True, max_length=length
    ).input_ids.to(args.device)
    layers = model_layers(model)
    per_domain = {}
    domain_names = list(DOMAINS)
    domain_ids = torch.cat([encode(DOMAINS[name], args.tokens) for name in domain_names])
    all_coefficients, all_messages, all_gradients = capture_geometry(model, domain_ids)
    for domain_index, name in enumerate(domain_names):
        coefficients = {
            layer: values[domain_index] for layer, values in all_coefficients.items()
        }
        messages = {
            layer: values[domain_index] for layer, values in all_messages.items()
        }
        gradients = {
            layer: values[domain_index] for layer, values in all_gradients.items()
        }
        bases = []
        for index, layer in enumerate(layers):
            weight = output_projection(layer).weight.detach().float().cpu()
            variance = coefficients[index].float().var(0, unbiased=False)
            write_cov = (weight * variance.sqrt().unsqueeze(0)) @ (
                weight * variance.sqrt().unsqueeze(0)
            ).T
            bases.append(top_basis(write_cov, args.rank))
        per_domain[name] = {
            "coefficients": coefficients, "messages": messages,
            "gradients": gradients, "active_bases": bases,
            "adjacent_similarity": [
                subspace_similarity(bases[i], bases[i + 1])
                for i in range(len(bases) - 1)
            ],
        }
        print(f"captured domain={name}", flush=True)

    prompt_similarity = []
    for layer_index in range(len(layers)):
        similarities = []
        for left in range(len(domain_names)):
            for right in range(left + 1, len(domain_names)):
                similarities.append(subspace_similarity(
                    per_domain[domain_names[left]]["active_bases"][layer_index],
                    per_domain[domain_names[right]]["active_bases"][layer_index],
                ))
        prompt_similarity.append(sum(similarities) / len(similarities))

    # Pool domain-conditioned statistics without mixing evaluation text.
    candidate_bases = []
    for index, layer in enumerate(layers):
        weight = output_projection(layer).weight.detach().float().cpu()
        coefficient_samples = torch.cat([
            per_domain[name]["coefficients"][index] for name in domain_names
        ])
        message_samples = torch.cat([
            per_domain[name]["messages"][index] for name in domain_names
        ])
        gradient_samples = torch.cat([
            per_domain[name]["gradients"][index] for name in domain_names
        ])
        variance = coefficient_samples.float().var(0, unbiased=False)
        weighted = weight * variance.sqrt().unsqueeze(0)
        write_cov = weighted @ weighted.T
        fisher = gradient_samples.float().T @ gradient_samples.float()
        candidate_bases.append({
            "active_down": top_basis(write_cov, args.rank),
            "message_pca": top_basis(covariance(message_samples), args.rank),
            "fisher": top_basis(fisher, args.rank),
            "joint": joint_basis(write_cov, fisher, args.rank),
        })

    eval_ids = encode(EVALUATION_TEXT, args.evaluation_tokens)
    with torch.no_grad():
        baseline_logits = model(eval_ids, use_cache=False).logits.float().cpu()
    interventions = []
    for index, layer in enumerate(layers):
        for basis_name, basis in candidate_bases[index].items():
            result = {"layer": index, "basis": basis_name}
            result.update(intervention_metrics(
                model, eval_ids, baseline_logits, layer, basis
            ))
            interventions.append(result)
            print(f"layer={index:02d} basis={basis_name:11s} "
                  f"kl={result['mean_kl']:.6f}", flush=True)

    boundaries = {int(value) for value in args.phase_boundaries.split(",") if value}
    adjacency = []
    for boundary in range(1, len(layers)):
        values = [
            per_domain[name]["adjacent_similarity"][boundary - 1]
            for name in domain_names
        ]
        adjacency.append({
            "boundary": boundary, "is_phase_boundary": boundary in boundaries,
            "mean_similarity": sum(values) / len(values),
            "domain_similarity": dict(zip(domain_names, values)),
        })

    serializable_domains = {
        name: {"adjacent_similarity": data["adjacent_similarity"]}
        for name, data in per_domain.items()
    }
    report = {
        "model": args.model, "rank": args.rank, "tokens": args.tokens,
        "evaluation_tokens": eval_ids.shape[1],
        "phase_boundaries": sorted(boundaries), "domains": serializable_domains,
        "adjacency": adjacency, "prompt_similarity_by_layer": prompt_similarity,
        "interventions": interventions, "complete": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
