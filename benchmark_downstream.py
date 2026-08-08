"""Quick downstream validation with a live GIHKCC-compressed KV cache."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F

from validate_autoregressive import (
    append_neox_cache,
    build_neox_cache,
    capture_layer_inputs,
    compress_residual_history,
)


def prepare_context_caches(model, context_ids, bits=2):
    context_len = context_ids.numel()
    seed_len = max(1, context_len - 1)
    seed = context_ids[:seed_len].unsqueeze(0)

    with capture_layer_inputs(
        model, "residual", storage_device=context_ids.device
    ) as captured:
        with torch.no_grad():
            native = model(seed, use_cache=True)
    native_cache = native.past_key_values
    _, decoded = compress_residual_history(
        captured, bits, "residual", model, prediction_mode="adjacent"
    )
    compressed_cache = build_neox_cache(model, decoded, "residual")
    return native_cache, compressed_cache, seed_len


def score_prepared(
    model, context_ids, continuation_ids, native_cache,
    compressed_cache, seed_len, bits=2,
):
    """Score one continuation from independent copies of prepared caches."""
    native_cache = copy.deepcopy(native_cache)
    compressed_cache = copy.deepcopy(compressed_cache)
    all_ids = torch.cat([context_ids, continuation_ids])
    context_len = context_ids.numel()

    native_ll = compressed_ll = 0.0
    native_exact = compressed_exact = True
    scored = 0
    # Feed the final context token first, ensuring every scored prediction uses
    # the compressed prefix cache rather than the uncompressed prefill logits.
    for position in range(seed_len, all_ids.numel()):
        input_id = all_ids[position:position + 1].view(1, 1)
        with torch.no_grad():
            native = model(input_id, past_key_values=native_cache, use_cache=True)
        native_cache = native.past_key_values
        with capture_layer_inputs(
            model, "residual", storage_device=context_ids.device
        ) as current:
            with torch.no_grad():
                compressed = model(
                    input_id, past_key_values=compressed_cache, use_cache=True
                )

        _, decoded_current = compress_residual_history(
            current, bits, "residual", model, prediction_mode="adjacent"
        )
        compressed_cache.crop(position)
        append_neox_cache(
            model, compressed_cache, decoded_current, "residual",
            start_position=position,
        )

        target_position = position + 1
        if target_position >= context_len and target_position < all_ids.numel():
            target = all_ids[target_position]
            native_logp = F.log_softmax(native.logits[0, -1].float(), dim=-1)
            compressed_logp = F.log_softmax(
                compressed.logits[0, -1].float(), dim=-1
            )
            native_ll += native_logp[target].item()
            compressed_ll += compressed_logp[target].item()
            native_exact &= native_logp.argmax().item() == target.item()
            compressed_exact &= compressed_logp.argmax().item() == target.item()
            scored += 1
    return native_ll, compressed_ll, native_exact, compressed_exact, scored


def score_continuation(model, context_ids, continuation_ids, bits=2):
    """Return native/compressed log likelihood and greedy exact matches."""
    native_cache, compressed_cache, seed_len = prepare_context_caches(
        model, context_ids, bits
    )
    return score_prepared(
        model, context_ids, continuation_ids, native_cache,
        compressed_cache, seed_len, bits,
    )


def load_model(name, device, revision):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=dtype
    ).to(device)
    model.eval()
    return model, tokenizer


def wilson_interval(correct, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return (center - radius, center + radius)


def run_lambada(model, tokenizer, device, limit, bits, seed, offset):
    from datasets import load_dataset

    dataset = load_dataset("EleutherAI/lambada_openai", split="test")
    native_exact = compressed_exact = agreement = valid = 0
    shuffled = dataset.shuffle(seed=seed)
    stop = min(offset + limit, len(shuffled))
    sample = shuffled.select(range(offset, stop))
    for row in sample:
        text = row["text"].strip()
        split = text.rfind(" ")
        if split <= 0:
            continue
        context = tokenizer(text[:split], return_tensors="pt").input_ids[0].to(device)
        target = tokenizer(text[split:], add_special_tokens=False,
                           return_tensors="pt").input_ids[0].to(device)
        _, _, native_match, compressed_match, count = score_continuation(
            model, context, target, bits
        )
        native_exact += int(native_match)
        compressed_exact += int(compressed_match)
        agreement += int(native_match == compressed_match)
        valid += int(count > 0)
    return {
        "dataset": "EleutherAI/lambada_openai",
        "split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "examples": valid,
        "native_correct": native_exact,
        "native_exact": native_exact / max(valid, 1),
        "native_95ci": wilson_interval(native_exact, valid),
        "compressed_correct": compressed_exact,
        "compressed_exact": compressed_exact / max(valid, 1),
        "compressed_95ci": wilson_interval(compressed_exact, valid),
        "correctness_agreement_count": agreement,
        "correctness_agreement": agreement / max(valid, 1),
    }


def run_hellaswag(model, tokenizer, device, limit, bits, seed, offset):
    from datasets import load_dataset

    dataset = load_dataset("Rowan/hellaswag", split="validation")
    native_correct = compressed_correct = choice_agreement = 0
    shuffled = dataset.shuffle(seed=seed)
    stop = min(offset + limit, len(shuffled))
    total = max(0, stop - offset)
    for row in shuffled.select(range(offset, stop)):
        context_text = (row["ctx_a"] + " " + row["ctx_b"]).strip()
        context = tokenizer(context_text, return_tensors="pt").input_ids[0].to(device)
        native_cache, compressed_cache, seed_len = prepare_context_caches(
            model, context, bits
        )
        native_scores = []
        compressed_scores = []
        for ending in row["endings"]:
            continuation = tokenizer(
                " " + ending, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)
            native_ll, compressed_ll, _, _, count = score_prepared(
                model, context, continuation, native_cache,
                compressed_cache, seed_len, bits,
            )
            native_scores.append(native_ll / max(count, 1))
            compressed_scores.append(compressed_ll / max(count, 1))
        label = int(row["label"])
        native_choice = torch.tensor(native_scores).argmax().item()
        compressed_choice = torch.tensor(compressed_scores).argmax().item()
        native_correct += int(native_choice == label)
        compressed_correct += int(compressed_choice == label)
        choice_agreement += int(native_choice == compressed_choice)
    return {
        "dataset": "Rowan/hellaswag",
        "split": "validation",
        "dataset_fingerprint": dataset._fingerprint,
        "examples": total,
        "native_correct": native_correct,
        "native_accuracy": native_correct / max(total, 1),
        "native_95ci": wilson_interval(native_correct, total),
        "compressed_correct": compressed_correct,
        "compressed_accuracy": compressed_correct / max(total, 1),
        "compressed_95ci": wilson_interval(compressed_correct, total),
        "choice_agreement_count": choice_agreement,
        "choice_agreement": choice_agreement / max(total, 1),
    }


def run_arc(model, tokenizer, device, limit, bits, seed, offset, subset):
    from datasets import load_dataset

    config = "ARC-Easy" if subset == "arc_easy" else "ARC-Challenge"
    dataset = load_dataset("allenai/ai2_arc", config, split="validation")
    shuffled = dataset.shuffle(seed=seed)
    stop = min(offset + limit, len(shuffled))
    sample = shuffled.select(range(offset, stop))
    native_correct = compressed_correct = choice_agreement = 0
    for row in sample:
        context_text = f"Question: {row['question']}\nAnswer:"
        context = tokenizer(context_text, return_tensors="pt").input_ids[0].to(device)
        native_cache, compressed_cache, seed_len = prepare_context_caches(
            model, context, bits
        )
        native_scores = []
        compressed_scores = []
        for choice in row["choices"]["text"]:
            continuation = tokenizer(
                " " + choice, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)
            native_ll, compressed_ll, _, _, count = score_prepared(
                model, context, continuation, native_cache,
                compressed_cache, seed_len, bits,
            )
            native_scores.append(native_ll / max(count, 1))
            compressed_scores.append(compressed_ll / max(count, 1))
        labels = row["choices"]["label"]
        label = labels.index(row["answerKey"])
        native_choice = torch.tensor(native_scores).argmax().item()
        compressed_choice = torch.tensor(compressed_scores).argmax().item()
        native_correct += int(native_choice == label)
        compressed_correct += int(compressed_choice == label)
        choice_agreement += int(native_choice == compressed_choice)
    total = len(sample)
    return {
        "dataset": "allenai/ai2_arc",
        "config": config,
        "split": "validation",
        "dataset_fingerprint": dataset._fingerprint,
        "examples": total,
        "native_correct": native_correct,
        "native_accuracy": native_correct / max(total, 1),
        "native_95ci": wilson_interval(native_correct, total),
        "compressed_correct": compressed_correct,
        "compressed_accuracy": compressed_correct / max(total, 1),
        "compressed_95ci": wilson_interval(compressed_correct, total),
        "choice_agreement_count": choice_agreement,
        "choice_agreement": choice_agreement / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument(
        "--revision", default="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
    )
    parser.add_argument(
        "--task",
        choices=("lambada", "hellaswag", "arc_easy", "arc_challenge"),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", help="Write metadata and results as JSON")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch build")
    model, tokenizer = load_model(args.model, args.device, args.revision)
    if args.task == "lambada":
        result = run_lambada(
            model, tokenizer, args.device, args.limit, args.bits,
            args.seed, args.offset,
        )
    elif args.task == "hellaswag":
        result = run_hellaswag(
            model, tokenizer, args.device, args.limit, args.bits,
            args.seed, args.offset,
        )
    else:
        result = run_arc(
            model, tokenizer, args.device, args.limit, args.bits,
            args.seed, args.offset, args.task,
        )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": args.revision,
        "task": args.task,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "bits": args.bits,
        "prediction": "adjacent",
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "result": result,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
