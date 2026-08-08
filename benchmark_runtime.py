"""Measure CUDA runtime and memory of native and reference compressed caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from validate_autoregressive import (
    append_neox_cache,
    build_neox_cache,
    capture_layer_inputs,
    compress_residual_history,
    native_kv_bytes,
)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_native(model, tokens, prefix, steps):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        output = model(tokens[:, :prefix], use_cache=True)
    cache = output.past_key_values
    del output
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    for position in range(prefix, prefix + steps):
        with torch.inference_mode():
            output = model(
                tokens[:, position:position + 1],
                past_key_values=cache,
                use_cache=True,
            )
        cache = output.past_key_values
    synchronize()
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "tokens_per_second": steps / elapsed,
        "resident_bytes_above_model": torch.cuda.memory_allocated() - base,
        "generation_peak_bytes_above_model":
            torch.cuda.max_memory_allocated() - base,
    }


def run_reference_compressed(model, tokens, prefix, steps, bits):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    with capture_layer_inputs(model, "residual") as captured:
        with torch.inference_mode():
            native_prefill = model(tokens[:, :prefix], use_cache=True)
    stack, decoded = compress_residual_history(
        captured, bits, "residual", model, prediction_mode="adjacent"
    )
    cache = build_neox_cache(model, decoded, "residual")
    del native_prefill, decoded, captured
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    for position in range(prefix, prefix + steps):
        token = tokens[:, position:position + 1]
        with capture_layer_inputs(model, "residual") as current:
            with torch.inference_mode():
                output = model(token, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        _, decoded_current = compress_residual_history(
            current, bits, "residual", model, prediction_mode="adjacent"
        )
        cache.crop(position)
        append_neox_cache(
            model, cache, decoded_current, "residual", start_position=position
        )
    synchronize()
    elapsed = time.perf_counter() - started
    final_tokens = prefix + steps
    return {
        "seconds": elapsed,
        "tokens_per_second": steps / elapsed,
        "resident_bytes_above_model": torch.cuda.memory_allocated() - base,
        "generation_peak_bytes_above_model":
            torch.cuda.max_memory_allocated() - base,
        "estimated_persistent_payload_bytes": stack.compressed_bytes
            * final_tokens // prefix,
        "fp16_kv_bytes": native_kv_bytes(model, final_tokens),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.model_max_length = 10**9
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16
    ).to("cuda").eval()
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
    )
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, return_tensors="pt").input_ids[:, :
        args.prefix + args.steps].to("cuda")

    native = run_native(model, tokens, args.prefix, args.steps)
    torch.cuda.empty_cache()
    compressed = run_reference_compressed(
        model, tokens, args.prefix, args.steps, args.bits
    )
    report = {
        "model": args.model,
        "model_revision": args.revision,
        "prefix": args.prefix,
        "steps": args.steps,
        "bits": args.bits,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "native": native,
        "reference_compressed": compressed,
        "throughput_ratio": (
            compressed["tokens_per_second"] / native["tokens_per_second"]
        ),
        "warning": (
            "The reference path materializes a standard FP16 K/V cache; "
            "payload compression is not resident VRAM compression."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
