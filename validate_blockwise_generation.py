"""Validate end-to-end generation with no materialized historical K/V cache."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from blockwise_attention import PackedResidualStack
from blockwise_generation import PackedNeoXController, patch_neox_attention
from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack
from validate_autoregressive import capture_layer_inputs, native_kv_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
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
    tokens = tokenizer(text, return_tensors="pt").input_ids[:, :
        args.prefix + args.steps + 1].to("cuda")

    prefix_ids = tokens[:, :args.prefix]
    with capture_layer_inputs(model, "residual", storage_device="cuda") as prefix_residuals:
        with torch.inference_mode():
            prefix_output = model(prefix_ids, use_cache=True)
    baseline_cache = prefix_output.past_key_values
    baseline_nll = 0.0
    baseline_logits = []
    torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    for offset in range(args.steps):
        position = args.prefix + offset
        with torch.inference_mode():
            output = model(
                tokens[:, position:position + 1],
                past_key_values=baseline_cache, use_cache=True,
            )
        baseline_cache = output.past_key_values
        logits = output.logits[:, -1].float()
        baseline_logits.append(logits.cpu())
        baseline_nll += F.cross_entropy(
            logits, tokens[:, position + 1], reduction="sum"
        ).item()
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started
    del baseline_cache, prefix_output, output
    torch.cuda.empty_cache()

    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=64,
        prediction_mode="adjacent",
    )
    prefix_stack = compress_predictive_stack(
        prefix_residuals, 8, args.bits, config
    )
    controller = PackedNeoXController(
        model, PackedResidualStack(prefix_stack), args.bits, args.block_size
    )
    del prefix_residuals, prefix_stack
    torch.cuda.empty_cache()
    compressed_nll = kl_total = 0.0
    top_matches = 0
    compressed_base_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    compressed_started = time.perf_counter()
    with patch_neox_attention(controller):
        for offset in range(args.steps):
            position = args.prefix + offset
            position_ids = torch.tensor([[position]], device="cuda")
            with capture_layer_inputs(
                model, "residual", storage_device="cuda"
            ) as current_residuals:
                with torch.inference_mode():
                    output = model(
                        tokens[:, position:position + 1],
                        position_ids=position_ids,
                        use_cache=False,
                    )
            logits = output.logits[:, -1].float()
            target = tokens[:, position + 1]
            compressed_nll += F.cross_entropy(
                logits, target, reduction="sum"
            ).item()
            base = baseline_logits[offset].to(logits.device)
            top_matches += int(base.argmax(-1).item() == logits.argmax(-1).item())
            kl_total += F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(base, dim=-1), reduction="batchmean",
            ).item()
            controller.append_token(current_residuals)
            del current_residuals
    torch.cuda.synchronize()
    compressed_seconds = time.perf_counter() - compressed_started

    final_tokens = args.prefix + args.steps
    report = {
        "model": args.model,
        "model_revision": args.revision,
        "prefix": args.prefix,
        "steps": args.steps,
        "bits": args.bits,
        "block_size": args.block_size,
        "baseline_ppl": math.exp(baseline_nll / args.steps),
        "compressed_ppl": math.exp(compressed_nll / args.steps),
        "mean_kl": kl_total / args.steps,
        "top1_agreement": top_matches / args.steps,
        "baseline_tokens_per_second": args.steps / baseline_seconds,
        "compressed_tokens_per_second": args.steps / compressed_seconds,
        "packed_resident_bytes": controller.packed.resident_bytes,
        "fp16_kv_bytes": native_kv_bytes(model, final_tokens),
        "resident_ratio": native_kv_bytes(model, final_tokens)
            / controller.packed.resident_bytes,
        "compressed_peak_bytes": torch.cuda.max_memory_allocated(),
        "compressed_base_bytes": compressed_base_bytes,
        "compressed_peak_above_base_bytes":
            torch.cuda.max_memory_allocated() - compressed_base_bytes,
        "historical_kv_materialized": False,
    }
    report["relative_ppl_change"] = (
        report["compressed_ppl"] / report["baseline_ppl"] - 1
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
