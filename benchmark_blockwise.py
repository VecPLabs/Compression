"""Validate packed blockwise attention against materialized attention."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from blockwise_attention import (
    PackedResidualStack,
    neox_blockwise_attention,
    neox_materialized_attention,
)
from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack
from validate_autoregressive import capture_layer_inputs


def measure(call):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started, (
        torch.cuda.max_memory_allocated() - base
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output")
    args = parser.parse_args()

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
    tokens = tokenizer(text, return_tensors="pt").input_ids[:, :args.tokens].to("cuda")
    with capture_layer_inputs(model, "residual", storage_device="cuda") as residuals:
        with torch.inference_mode():
            model(tokens, use_cache=False)

    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=64,
        prediction_mode="adjacent",
    )
    reference = compress_predictive_stack(residuals, 8, args.bits, config)
    packed = PackedResidualStack(reference)
    layer_idx = args.layer if args.layer >= 0 else len(residuals) - 1
    query_residual = residuals[layer_idx][-1]
    position = args.tokens - 1

    # Warm rotation caches before measuring transient attention allocations.
    neox_blockwise_attention(
        model, packed, layer_idx, query_residual, position, args.block_size
    )
    materialized, materialized_seconds, materialized_peak = measure(
        lambda: neox_materialized_attention(
            model, packed, layer_idx, query_residual, position
        )
    )
    blockwise, blockwise_seconds, blockwise_peak = measure(
        lambda: neox_blockwise_attention(
            model, packed, layer_idx, query_residual, position, args.block_size
        )
    )
    difference = materialized.float() - blockwise.float()
    report = {
        "model": args.model,
        "model_revision": args.revision,
        "tokens": args.tokens,
        "bits": args.bits,
        "layer": layer_idx,
        "block_size": args.block_size,
        "packed_resident_bytes": packed.resident_bytes,
        "accounted_payload_bytes": reference.compressed_bytes,
        "max_attention_error": difference.abs().max().item(),
        "mean_attention_error": difference.abs().mean().item(),
        "materialized_seconds": materialized_seconds,
        "blockwise_seconds": blockwise_seconds,
        "materialized_peak_temporary_bytes": materialized_peak,
        "blockwise_peak_temporary_bytes": blockwise_peak,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
