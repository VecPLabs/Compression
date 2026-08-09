"""Validate projected quantized K/V serving against native FP16 generation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from quantized_kv_generation import QuantizedKVController, patch_neox_quantized_kv
from validate_autoregressive import native_kv_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--scheme", choices=("symmetric", "kivi"), default="kivi")
    parser.add_argument("--key-bits", type=int, default=8)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--direct-packed-attention", action="store_true")
    parser.add_argument("--hot-window", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.model_max_length = 10**9
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16
    ).to("cuda").eval()
    all_tokens = torch.load(
        args.token_cache, map_location="cpu", weights_only=True
    )
    required = args.prefix + args.steps + 1
    if all_tokens.shape[1] < args.token_offset + required:
        raise ValueError("token cache does not contain the requested window")
    tokens = all_tokens[
        :, args.token_offset:args.token_offset + required
    ].to("cuda")

    prefix_ids = tokens[:, :args.prefix]
    with torch.inference_mode():
        prefix_output = model(prefix_ids, use_cache=True)
    baseline_cache = prefix_output.past_key_values
    baseline_logits = []
    baseline_nll = 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
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
    baseline_seconds = time.perf_counter() - started
    del baseline_cache, prefix_output, output
    torch.cuda.empty_cache()

    controller = QuantizedKVController(
        model, bits=args.bits, hot_window=args.hot_window,
        block_size=args.block_size,
        scheme=args.scheme, key_bits=args.key_bits,
        value_bits=args.value_bits, page_size=args.page_size,
        direct_packed_attention=args.direct_packed_attention,
    )
    with patch_neox_quantized_kv(controller):
        for position in range(args.prefix):
            position_ids = torch.tensor([[position]], device="cuda")
            with torch.inference_mode():
                model(
                    tokens[:, position:position + 1],
                    position_ids=position_ids, use_cache=False,
                )

        compressed_nll = kl_total = 0.0
        top_matches = 0
        compressed_base_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for offset in range(args.steps):
            position = args.prefix + offset
            position_ids = torch.tensor([[position]], device="cuda")
            with torch.inference_mode():
                output = model(
                    tokens[:, position:position + 1],
                    position_ids=position_ids, use_cache=False,
                )
            logits = output.logits[:, -1].float()
            target = tokens[:, position + 1]
            compressed_nll += F.cross_entropy(
                logits, target, reduction="sum"
            ).item()
            base = baseline_logits[offset].to(logits.device)
            top_matches += int(base.argmax(-1).item() == logits.argmax(-1).item())
            kl_total += F.kl_div(
                F.log_softmax(logits, dim=-1), F.softmax(base, dim=-1),
                reduction="batchmean",
            ).item()
        torch.cuda.synchronize()
        compressed_seconds = time.perf_counter() - started

    final_tokens = args.prefix + args.steps
    fp16_bytes = native_kv_bytes(model, final_tokens)
    report = {
        "model": args.model,
        "model_revision": args.revision,
        "prefix": args.prefix,
        "steps": args.steps,
        "token_offset": args.token_offset,
        "bits": args.bits,
        "scheme": args.scheme,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "page_size": args.page_size,
        "direct_packed_attention": args.direct_packed_attention,
        "hot_window": args.hot_window,
        "block_size": args.block_size,
        "baseline_ppl": math.exp(baseline_nll / args.steps),
        "compressed_ppl": math.exp(compressed_nll / args.steps),
        "relative_ppl_change": math.exp(compressed_nll / args.steps)
            / math.exp(baseline_nll / args.steps) - 1,
        "mean_kl": kl_total / args.steps,
        "top1_agreement": top_matches / args.steps,
        "baseline_tokens_per_second": args.steps / baseline_seconds,
        "compressed_tokens_per_second": args.steps / compressed_seconds,
        "cache_resident_bytes": controller.resident_bytes,
        "cold_resident_bytes": controller.cold_resident_bytes,
        "hot_resident_bytes": controller.hot_resident_bytes,
        "fp16_kv_bytes": fp16_bytes,
        "resident_ratio": fp16_bytes / controller.resident_bytes,
        "cold_tokens": controller.cold_tokens,
        "hot_tokens": controller.hot_tokens,
        "compressed_peak_bytes": torch.cuda.max_memory_allocated(),
        "compressed_base_bytes": compressed_base_bytes,
        "compressed_peak_above_base_bytes":
            torch.cuda.max_memory_allocated() - compressed_base_bytes,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
