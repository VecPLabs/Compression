"""Quick downstream validation with a live GIHKCC-compressed KV cache."""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from validate_autoregressive import (
    append_neox_cache,
    build_neox_cache,
    capture_layer_inputs,
    compress_residual_history,
)


def score_continuation(model, context_ids, continuation_ids, bits=2):
    """Return native/compressed log likelihood and compressed greedy match."""
    all_ids = torch.cat([context_ids, continuation_ids])
    context_len = context_ids.numel()
    seed_len = max(1, context_len - 1)
    seed = all_ids[:seed_len].unsqueeze(0)

    with capture_layer_inputs(model, "residual") as captured:
        with torch.no_grad():
            native = model(seed, use_cache=True)
    native_cache = native.past_key_values
    _, decoded = compress_residual_history(
        captured, bits, "residual", model, prediction_mode="adjacent"
    )
    compressed_cache = build_neox_cache(model, decoded, "residual")

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
        with capture_layer_inputs(model, "residual") as current:
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


def load_model(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def run_lambada(model, tokenizer, device, limit, bits):
    from datasets import load_dataset

    dataset = load_dataset("EleutherAI/lambada_openai", split="test")
    native_exact = compressed_exact = agreement = valid = 0
    for row in dataset.select(range(min(limit, len(dataset)))):
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
        "examples": valid,
        "native_exact": native_exact / max(valid, 1),
        "compressed_exact": compressed_exact / max(valid, 1),
        "correctness_agreement": agreement / max(valid, 1),
    }


def run_hellaswag(model, tokenizer, device, limit, bits):
    from datasets import load_dataset

    dataset = load_dataset("Rowan/hellaswag", split="validation")
    native_correct = compressed_correct = choice_agreement = 0
    total = min(limit, len(dataset))
    for row in dataset.select(range(total)):
        context_text = (row["ctx_a"] + " " + row["ctx_b"]).strip()
        context = tokenizer(context_text, return_tensors="pt").input_ids[0].to(device)
        native_scores = []
        compressed_scores = []
        for ending in row["endings"]:
            continuation = tokenizer(
                " " + ending, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)
            native_ll, compressed_ll, _, _, count = score_continuation(
                model, context, continuation, bits
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
        "examples": total,
        "native_accuracy": native_correct / max(total, 1),
        "compressed_accuracy": compressed_correct / max(total, 1),
        "choice_agreement": choice_agreement / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--task", choices=("lambada", "hellaswag"), required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch build")
    model, tokenizer = load_model(args.model, args.device)
    result = (
        run_lambada(model, tokenizer, args.device, args.limit, args.bits)
        if args.task == "lambada"
        else run_hellaswag(model, tokenizer, args.device, args.limit, args.bits)
    )
    print(f"{args.task} ({args.model}, adjacent {args.bits}-bit): {result}")


if __name__ == "__main__":
    main()
