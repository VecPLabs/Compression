"""Validate end-to-end generation with no materialized historical K/V cache."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    parser.add_argument(
        "--prefix-cache",
        help="Load or create a reusable tensor-only packed-prefix checkpoint",
    )
    parser.add_argument(
        "--phase-log",
        help="Append phase timings immediately for diagnosing long runs",
    )
    parser.add_argument("--dataset-arrow", help="Read a local Arrow dataset directly")
    parser.add_argument("--token-cache", help="Load or create tokenized held-out text")
    parser.add_argument(
        "--profile-subphases", action="store_true",
        help="Synchronize around compressed forward and append for diagnosis",
    )
    parser.add_argument(
        "--fused-projection", action="store_true",
        help="Fold inverse rotation and LayerNorm affine into historical QKV",
    )
    parser.add_argument(
        "--fused-attention", action="store_true",
        help="Use fused SDPA while the complete history fits one block",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    phase_seconds = {}
    phase_log = Path(args.phase_log) if args.phase_log else None
    if phase_log:
        phase_log.parent.mkdir(parents=True, exist_ok=True)
        phase_log.write_text("", encoding="utf-8")

    def phase(name, started):
        elapsed = time.perf_counter() - started
        phase_seconds[name] = elapsed
        print(f"[phase] {name}: {elapsed:.3f}s", file=sys.stderr, flush=True)
        if phase_log:
            with phase_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"phase": name, "seconds": elapsed}) + "\n")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.model_max_length = 10**9
    phase("tokenizer_load", started)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16
    ).to("cuda").eval()
    phase("model_load", started)
    token_cache = Path(args.token_cache) if args.token_cache else None
    if token_cache and token_cache.exists():
        started = time.perf_counter()
        all_tokens = torch.load(token_cache, map_location="cpu", weights_only=True)
        phase("token_cache_load", started)
    else:
        started = time.perf_counter()
        if args.dataset_arrow:
            from datasets import Dataset
            dataset = Dataset.from_file(args.dataset_arrow)
        else:
            dataset = load_dataset(
                "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
            )
        phase("dataset_load", started)
        started = time.perf_counter()
        text = "\n".join(row["text"] for row in dataset if row["text"].strip())
        all_tokens = tokenizer(text, return_tensors="pt").input_ids.cpu()
        phase("dataset_tokenize", started)
        if token_cache:
            started = time.perf_counter()
            token_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(all_tokens, token_cache)
            phase("token_cache_save", started)
    required_tokens = args.prefix + args.steps + 1
    if all_tokens.shape[1] < required_tokens:
        raise ValueError(
            f"token cache has {all_tokens.shape[1]} tokens; {required_tokens} required"
        )
    tokens = all_tokens[:, :required_tokens].to("cuda")

    prefix_ids = tokens[:, :args.prefix]
    cache_path = Path(args.prefix_cache) if args.prefix_cache else None
    cached_packed = None
    if cache_path and cache_path.exists():
        started = time.perf_counter()
        saved = torch.load(cache_path, map_location="cuda", weights_only=True)
        expected = {
            "model": args.model, "revision": args.revision,
            "prefix": args.prefix, "bits": args.bits,
            "prefix_token_ids": prefix_ids.cpu().reshape(-1).tolist(),
        }
        if saved.get("metadata") != expected:
            raise ValueError(
                f"prefix cache metadata mismatch: {saved.get('metadata')} != {expected}"
            )
        cached_packed = PackedResidualStack.from_checkpoint(saved["packed"])
        phase("prefix_cache_load", started)

    started = time.perf_counter()
    if cached_packed is None:
        with capture_layer_inputs(
            model, "residual", storage_device="cuda"
        ) as prefix_residuals:
            with torch.inference_mode():
                prefix_output = model(prefix_ids, use_cache=True)
        phase("prefix_forward_and_capture", started)
    else:
        with torch.inference_mode():
            prefix_output = model(prefix_ids, use_cache=True)
        phase("prefix_forward", started)
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
    phase_seconds["baseline_generation"] = baseline_seconds
    print(
        f"[phase] baseline_generation: {baseline_seconds:.3f}s",
        file=sys.stderr, flush=True,
    )
    del baseline_cache, prefix_output, output
    torch.cuda.empty_cache()

    if cached_packed is None:
        started = time.perf_counter()
        config = GIHKCCV2Config(
            similarity_threshold=0.0,
            max_keyframe_span=64,
            prediction_mode="adjacent",
        )
        prefix_stack = compress_predictive_stack(
            prefix_residuals, 8, args.bits, config
        )
        cached_packed = PackedResidualStack(prefix_stack)
        phase("prefix_compression", started)
        if cache_path:
            started = time.perf_counter()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "metadata": {
                    "model": args.model, "revision": args.revision,
                    "prefix": args.prefix, "bits": args.bits,
                    "prefix_token_ids": prefix_ids.cpu().reshape(-1).tolist(),
                },
                "packed": cached_packed.checkpoint(),
            }, cache_path)
            phase("prefix_cache_save", started)
        del prefix_residuals, prefix_stack
    controller = PackedNeoXController(
        model, cached_packed, args.bits, args.block_size,
        reserve_tokens=args.prefix + args.steps,
        fused_projection=args.fused_projection,
        fused_attention=args.fused_attention,
    )
    torch.cuda.empty_cache()
    compressed_nll = kl_total = 0.0
    top_matches = 0
    forward_profile_seconds = 0.0
    append_profile_seconds = 0.0
    compressed_base_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    compressed_started = time.perf_counter()
    with patch_neox_attention(controller):
        for offset in range(args.steps):
            position = args.prefix + offset
            position_ids = torch.tensor([[position]], device="cuda")
            if args.profile_subphases:
                torch.cuda.synchronize()
                subphase_started = time.perf_counter()
            with capture_layer_inputs(
                model, "residual", storage_device="cuda"
            ) as current_residuals:
                with torch.inference_mode():
                    output = model(
                        tokens[:, position:position + 1],
                        position_ids=position_ids,
                        use_cache=False,
                    )
            if args.profile_subphases:
                torch.cuda.synchronize()
                forward_profile_seconds += time.perf_counter() - subphase_started
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
            if args.profile_subphases:
                torch.cuda.synchronize()
                subphase_started = time.perf_counter()
            controller.append_token(current_residuals)
            if args.profile_subphases:
                torch.cuda.synchronize()
                append_profile_seconds += time.perf_counter() - subphase_started
            del current_residuals
    torch.cuda.synchronize()
    compressed_seconds = time.perf_counter() - compressed_started
    phase_seconds["compressed_generation"] = compressed_seconds
    print(
        f"[phase] compressed_generation: {compressed_seconds:.3f}s",
        file=sys.stderr, flush=True,
    )

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
        "packed_logical_bytes": controller.packed.logical_bytes,
        "packed_capacity_tokens": controller.packed.capacity_tokens,
        "fused_projection": args.fused_projection,
        "fused_projection_bytes": controller.fused_projection_bytes,
        "fused_attention": args.fused_attention,
        "fp16_kv_bytes": native_kv_bytes(model, final_tokens),
        "resident_ratio": native_kv_bytes(model, final_tokens)
            / controller.packed.resident_bytes,
        "compressed_peak_bytes": torch.cuda.max_memory_allocated(),
        "compressed_base_bytes": compressed_base_bytes,
        "compressed_peak_above_base_bytes":
            torch.cuda.max_memory_allocated() - compressed_base_bytes,
        "historical_kv_materialized": False,
        "phase_seconds": phase_seconds,
    }
    if args.profile_subphases:
        report["profiled_compressed_subphases"] = {
            "forward_seconds": forward_profile_seconds,
            "append_compression_seconds": append_profile_seconds,
            "forward_fraction": forward_profile_seconds / compressed_seconds,
            "append_fraction": append_profile_seconds / compressed_seconds,
            "synchronization_changes_throughput": True,
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
