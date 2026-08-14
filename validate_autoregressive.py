"""Live autoregressive validation for residual-native GIHKCC on GPT-NeoX."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb as apply_qwen2_rotary_pos_emb,
)

from gihkcc_v2 import (
    GIHKCCV2Config,
    PredictiveEntry,
    PredictiveStack,
    compress_predictive_stack,
    decompress_predictive_stack,
)
from residual_repacking import (
    compress_adjacent_repacked, compress_blockwise_repacked,
    compress_message_block,
)
from residual_folding import (
    compress_folded_adjacent, decompress_folded_adjacent, fit_fold,
)
from turboquant_paper import paper_turboquant_compress, paper_turboquant_decompress


HELD_OUT_TEXT = """
The history of scientific measurement is also a history of learning what must
be preserved and what may safely be discarded. Early astronomers recorded the
positions of visible stars with simple instruments, yet those observations
were precise enough to reveal patterns spanning centuries. Modern instruments
produce vastly more information, but the central problem remains unchanged:
measure the signal faithfully while controlling noise, cost, and complexity.

Compression becomes especially valuable when a system repeatedly produces
related states. Instead of storing every state independently, a representation
can preserve a reference and encode only the changes. The usefulness of such a
scheme depends not merely on numerical reconstruction error, but on whether the
downstream computation continues to make the same decisions.
"""


def decoder_layers(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError(f"unsupported model architecture: {model.__class__.__name__}")


@contextmanager
def capture_layer_inputs(
    model, capture_point: str = "residual", storage_device="cpu"
):
    captured = []
    hooks = []
    for layer in decoder_layers(model):
        module = (
            layer if capture_point == "residual"
            else (
                layer.attention.query_key_value
                if hasattr(layer, "attention")
                else layer.self_attn.q_proj
            )
        )
        hooks.append(module.register_forward_pre_hook(
            lambda hooked_module, arguments: captured.append(
                arguments[0][0].detach().to(storage_device)
            )
        ))
    try:
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def capture_neox_communication(model, storage_device="cpu"):
    """Capture the layer-0 anchor and each attention/MLP residual write."""
    inputs, attention, mlp, hooks = [], [], [], []
    for layer in decoder_layers(model):
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, args: inputs.append(
                args[0][0].detach().to(storage_device)
            )
        ))
        hooks.append(layer.post_attention_dropout.register_forward_hook(
            lambda _module, _args, output: attention.append(
                output[0].detach().to(storage_device)
            )
        ))
        hooks.append(layer.post_mlp_dropout.register_forward_hook(
            lambda _module, _args, output: mlp.append(
                output[0].detach().to(storage_device)
            )
        ))
    try:
        yield inputs, attention, mlp
    finally:
        for hook in hooks:
            hook.remove()


def interleave_messages(attention, mlp, combined: bool = False):
    messages = []
    for attention_write, mlp_write in zip(attention, mlp):
        if combined:
            messages.append(attention_write + mlp_write)
        else:
            messages.extend([attention_write, mlp_write])
    return messages


def build_neox_cache(
    model, residuals: List[torch.Tensor], capture_point: str = "residual"
) -> DynamicCache:
    """Regenerate RoPE-applied K/V exactly as GPTNeoXAttention.forward does."""
    cache = DynamicCache(config=model.config)
    append_neox_cache(model, cache, residuals, capture_point, start_position=0)
    return cache


def append_neox_cache(
    model, cache: DynamicCache, residuals: List[torch.Tensor],
    capture_point: str = "residual", start_position: int = 0,
) -> None:
    """Project decoded residuals and append their K/V at absolute positions."""
    for layer_idx, (layer, residual) in enumerate(zip(decoder_layers(model), residuals)):
        dtype = next(layer.parameters()).dtype
        device = next(layer.parameters()).device
        hidden = residual.unsqueeze(0).to(device=device, dtype=dtype)
        if capture_point == "residual":
            hidden = layer.input_layernorm(hidden)
        positions = torch.arange(
            start_position, start_position + residual.shape[0],
            device=hidden.device,
        ).unsqueeze(0)
        if hasattr(layer, "attention"):
            qkv = layer.attention.query_key_value(hidden)
            qkv = qkv.view(
                1, residual.shape[0], model.config.num_attention_heads,
                3 * layer.attention.head_size,
            ).transpose(1, 2)
            query, key, value = qkv.chunk(3, dim=-1)
            cos, sin = model.gpt_neox.rotary_emb(hidden, position_ids=positions)
            _, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            attention = layer.self_attn
            head_dim = attention.head_dim
            query = attention.q_proj(hidden).view(
                1, residual.shape[0], model.config.num_attention_heads, head_dim
            ).transpose(1, 2)
            key = attention.k_proj(hidden).view(
                1, residual.shape[0], model.config.num_key_value_heads, head_dim
            ).transpose(1, 2)
            value = attention.v_proj(hidden).view(
                1, residual.shape[0], model.config.num_key_value_heads, head_dim
            ).transpose(1, 2)
            cos, sin = model.model.rotary_emb(hidden, position_ids=positions)
            _, key = apply_qwen2_rotary_pos_emb(query, key, cos, sin)
        cache.update(key.detach(), value.detach(), layer_idx)


def compress_residual_history(
    residuals, delta_bits, capture_point: str = "residual", model=None,
    ln_aware_candidates: int = 1, ln_aware_seeds=None,
    prediction_mode: str = "anchor", repacking: str | None = None,
    repacking_bases=None, repacking_block_size: int | None = None,
    repacking_protected_fraction: float | None = None,
    repacking_boundaries: list[int] | None = None,
    folding_allocation=None, folding_specs=None,
):
    if folding_allocation is not None:
        if capture_point != "residual":
            raise ValueError("folding requires residual capture")
        if len(folding_allocation) != len(residuals) - 1:
            raise ValueError("folding allocation must cover every delta layer")
        if folding_specs is None:
            folding_specs = [
                fit_fold(
                    residuals[index] - residuals[index - 1],
                    "correlation_lifting", 1234 + index,
                )
                for index in range(1, len(residuals))
            ]
        stack = compress_folded_adjacent(
            residuals, folding_specs, folding_allocation, anchor_bits=8
        )
        return stack, decompress_folded_adjacent(stack)
    if repacking:
        if capture_point != "residual" or not isinstance(delta_bits, int):
            raise ValueError("repacking requires residual capture and uniform bits")
        grams = []
        for layer in decoder_layers(model):
            weight = layer.attention.query_key_value.weight.detach().float().cpu()
            grams.append(weight.T @ weight)
        if repacking_block_size or repacking_boundaries:
            return compress_blockwise_repacked(
                residuals, delta_bits, grams,
                repacking_block_size or len(residuals),
                repacking, repacking_bases,
                protected_fraction=repacking_protected_fraction,
                boundaries=repacking_boundaries,
            )
        return compress_adjacent_repacked(
            residuals, delta_bits, grams, repacking, repacking_bases,
            protected_fraction=repacking_protected_fraction,
        )
    if ln_aware_candidates > 1 or ln_aware_seeds is not None:
        if capture_point != "residual" or not isinstance(delta_bits, int):
            raise ValueError("LayerNorm-aware search requires uniform residual bits")
        return compress_layernorm_aware(
            model, residuals, delta_bits, ln_aware_candidates,
            ln_aware_seeds,
        )
    if capture_point == "preprojection":
        if not isinstance(delta_bits, int):
            raise ValueError("preprojection mode currently requires uniform bits")
        # Normalized projection inputs are not cross-layer predictors. Encode
        # each layer independently while retaining the residual→K/V multiplier.
        config = GIHKCCV2Config(similarity_threshold=1.1, max_keyframe_span=1)
        anchor_bits = delta_bits
    else:
        config = GIHKCCV2Config(
            similarity_threshold=0.0,
            max_keyframe_span=64,
            prediction_mode=prediction_mode,
        )
        anchor_bits = 8
    stack = compress_predictive_stack(
        residuals, anchor_bits, delta_bits, config
    )
    return stack, decompress_predictive_stack(stack)


def compress_layernorm_aware(
    model, residuals, delta_bits: int, candidates: int, locked_seeds=None
):
    """Choose each delta rotation by error after the consumer LayerNorm."""
    stack = PredictiveStack(num_layers=len(residuals))
    reconstructed = [None] * len(residuals)
    anchor = paper_turboquant_compress(residuals[0], 8, rotation_seed=42)
    reconstructed[0] = paper_turboquant_decompress(anchor)
    stack.entries.append(PredictiveEntry(0, None, anchor))

    for layer_idx in range(1, len(residuals)):
        prediction = reconstructed[0]
        delta = residuals[layer_idx] - prediction
        layer = decoder_layers(model)[layer_idx]
        dtype = next(layer.parameters()).dtype
        device = next(layer.parameters()).device
        with torch.no_grad():
            target = layer.input_layernorm(
                residuals[layer_idx].unsqueeze(0).to(device=device, dtype=dtype)
            )
        best_payload = best_restored = None
        best_error = math.inf
        candidate_seeds = (
            [locked_seeds[layer_idx]] if locked_seeds is not None
            else [42 + candidate_idx * 997 for candidate_idx in range(candidates)]
        )
        for seed in candidate_seeds:
            payload = paper_turboquant_compress(
                delta, delta_bits, rotation_seed=seed
            )
            restored = prediction + paper_turboquant_decompress(payload)
            with torch.no_grad():
                normalized = layer.input_layernorm(
                    restored.unsqueeze(0).to(device=device, dtype=dtype)
                )
                error = (target - normalized).float().square().mean().item()
            if error < best_error:
                best_error = error
                best_payload = payload
                best_restored = restored
        stack.entries.append(PredictiveEntry(layer_idx, 0, best_payload))
        reconstructed[layer_idx] = best_restored
    return stack, reconstructed


def native_kv_bytes(model, tokens: int) -> int:
    heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    return model.config.num_hidden_layers * 2 * heads * head_dim * tokens * 2


def parity_check(
    model, tokens: torch.Tensor, prefix: int, capture_point: str = "residual"
) -> None:
    prefix_ids = tokens[:, :prefix]
    next_id = tokens[:, prefix:prefix + 1]
    with capture_layer_inputs(model, capture_point) as residuals:
        with torch.no_grad():
            native_prefix = model(prefix_ids, use_cache=True)
    rebuilt = build_neox_cache(model, residuals, capture_point)
    with torch.no_grad():
        native_next = model(next_id, past_key_values=native_prefix.past_key_values,
                            use_cache=True).logits[:, -1]
        rebuilt_next = model(next_id, past_key_values=rebuilt, use_cache=True).logits[:, -1]
    difference = native_next.float() - rebuilt_next.float()
    print("Parity gate (lossless residual -> live RoPE cache)")
    print(f"  max logit error: {difference.abs().max().item():.8f}")
    print(f"  mean logit error: {difference.abs().mean().item():.8f}")
    print(f"  top-token match: {native_next.argmax(-1).item() == rebuilt_next.argmax(-1).item()}")
    if difference.abs().max().item() > 1e-4:
        raise RuntimeError("lossless rebuilt cache failed logit parity")


def evaluate(
    model, tokens: torch.Tensor, prefix: int, steps: int, delta_bits,
    capture_point: str = "residual", ln_aware_candidates: int = 1,
    prediction_mode: str = "anchor",
    incremental: bool = False, repacking: str | None = None,
    repacking_block_size: int | None = None,
    repacking_protected_fraction: float | None = None,
    repacking_boundaries: list[int] | None = None,
    folding_allocation=None,
):
    prefix_ids = tokens[:, :prefix]
    with capture_layer_inputs(model, capture_point) as captured:
        with torch.no_grad():
            baseline_output = model(prefix_ids, use_cache=True)
    baseline_cache = baseline_output.past_key_values
    residual_history = [tensor.clone() for tensor in captured]

    compressed_stack, decoded = compress_residual_history(
        residual_history, delta_bits, capture_point, model, ln_aware_candidates,
        prediction_mode=prediction_mode, repacking=repacking,
        repacking_block_size=repacking_block_size,
        repacking_protected_fraction=repacking_protected_fraction,
        repacking_boundaries=repacking_boundaries,
        folding_allocation=folding_allocation,
    )
    locked_repacking_bases = getattr(compressed_stack, "bases", None)
    locked_folding_specs = getattr(compressed_stack, "specs", None)
    locked_ln_seeds = None
    if ln_aware_candidates > 1:
        locked_ln_seeds = [
            entry.payload.rotation_seed for entry in compressed_stack.entries
        ]
    compressed_cache = build_neox_cache(model, decoded, capture_point)
    baseline_nll = compressed_nll = 0.0
    top_matches = 0
    kl_total = 0.0

    available = min(steps, tokens.shape[1] - prefix - 1)
    for offset in range(available):
        position = prefix + offset
        input_id = tokens[:, position:position + 1]
        target = tokens[:, position + 1]

        with torch.no_grad():
            baseline = model(input_id, past_key_values=baseline_cache, use_cache=True)
        baseline_cache = baseline.past_key_values

        with capture_layer_inputs(model, capture_point) as current_residuals:
            with torch.no_grad():
                compressed = model(input_id, past_key_values=compressed_cache, use_cache=True)

        baseline_logits = baseline.logits[:, -1].float()
        compressed_logits = compressed.logits[:, -1].float()
        baseline_nll += F.cross_entropy(baseline_logits, target, reduction="sum").item()
        compressed_nll += F.cross_entropy(compressed_logits, target, reduction="sum").item()
        top_matches += int(baseline_logits.argmax(-1).item() == compressed_logits.argmax(-1).item())
        baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
        compressed_log_probs = F.log_softmax(compressed_logits, dim=-1)
        kl_total += F.kl_div(
            compressed_log_probs, baseline_log_probs.exp(), reduction="batchmean"
        ).item()

        for layer_idx, current in enumerate(current_residuals):
            residual_history[layer_idx] = torch.cat(
                [residual_history[layer_idx], current], dim=0
            )
        if incremental:
            _, decoded_current = compress_residual_history(
                current_residuals, delta_bits, capture_point, model,
                ln_aware_candidates, locked_ln_seeds, prediction_mode,
                repacking, locked_repacking_bases, repacking_block_size,
                repacking_protected_fraction,
                repacking_boundaries,
                folding_allocation, locked_folding_specs,
            )
            # The model appended an uncompressed current-token entry in place.
            # Remove it and replace it with K/V projected from the decoded state.
            compressed_cache.crop(position)
            append_neox_cache(
                model, compressed_cache, decoded_current, capture_point,
                start_position=position,
            )
        else:
            compressed_stack, decoded = compress_residual_history(
                residual_history, delta_bits, capture_point, model,
                ln_aware_candidates, locked_ln_seeds, prediction_mode,
                repacking, locked_repacking_bases, repacking_block_size,
                repacking_protected_fraction,
                repacking_boundaries,
                folding_allocation, locked_folding_specs,
            )
            compressed_cache = build_neox_cache(model, decoded, capture_point)

    if incremental:
        compressed_stack, _ = compress_residual_history(
            residual_history, delta_bits, capture_point, model,
            ln_aware_candidates, locked_ln_seeds, prediction_mode,
            repacking, locked_repacking_bases, repacking_block_size,
            repacking_protected_fraction,
            repacking_boundaries,
            folding_allocation, locked_folding_specs,
        )
    ratio = native_kv_bytes(model, prefix + available) / compressed_stack.compressed_bytes
    return {
        "tokens": available,
        "baseline_ppl": math.exp(baseline_nll / available),
        "compressed_ppl": math.exp(compressed_nll / available),
        "mean_kl": kl_total / available,
        "top1_agreement": top_matches / available,
        "persistent_ratio": ratio,
    }


def evaluate_message_repacking(
    model, tokens: torch.Tensor, prefix: int, steps: int,
    bits: int, method: str,
):
    """Validate joint attention/MLP-message coding in incremental inference."""
    prefix_ids = tokens[:, :prefix]
    with capture_neox_communication(model) as (inputs, attention, mlp):
        with torch.no_grad():
            baseline_output = model(prefix_ids, use_cache=True)
    baseline_cache = baseline_output.past_key_values
    anchor_history = inputs[0].clone()
    combined = method.endswith("_combined")
    fit_method = method.removesuffix("_combined")
    writes_per_layer = 1 if combined else 2
    message_history = [item.clone() for item in interleave_messages(
        attention, mlp, combined
    )]
    compressed_stack, decoded = compress_message_block(
        anchor_history, message_history, bits, fit_method,
        writes_per_layer=writes_per_layer,
    )
    basis = compressed_stack.basis
    compressed_cache = build_neox_cache(model, decoded)
    baseline_nll = compressed_nll = kl_total = 0.0
    top_matches = 0
    available = min(steps, tokens.shape[1] - prefix - 1)
    for offset in range(available):
        position = prefix + offset
        input_id = tokens[:, position:position + 1]
        target = tokens[:, position + 1]
        with torch.no_grad():
            baseline = model(input_id, past_key_values=baseline_cache, use_cache=True)
        baseline_cache = baseline.past_key_values
        with capture_neox_communication(model) as (inputs, attention, mlp):
            with torch.no_grad():
                compressed = model(
                    input_id, past_key_values=compressed_cache, use_cache=True
                )
        baseline_logits = baseline.logits[:, -1].float()
        compressed_logits = compressed.logits[:, -1].float()
        baseline_nll += F.cross_entropy(baseline_logits, target, reduction="sum").item()
        compressed_nll += F.cross_entropy(compressed_logits, target, reduction="sum").item()
        top_matches += int(
            baseline_logits.argmax(-1).item() == compressed_logits.argmax(-1).item()
        )
        baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
        compressed_log_probs = F.log_softmax(compressed_logits, dim=-1)
        kl_total += F.kl_div(
            compressed_log_probs, baseline_log_probs.exp(), reduction="batchmean"
        ).item()
        current_messages = interleave_messages(attention, mlp, combined)
        _, decoded_current = compress_message_block(
            inputs[0], current_messages, bits, fit_method, basis,
            writes_per_layer,
        )
        compressed_cache.crop(position)
        append_neox_cache(
            model, compressed_cache, decoded_current, start_position=position
        )
        anchor_history = torch.cat([anchor_history, inputs[0]], dim=0)
        for index, message in enumerate(current_messages):
            message_history[index] = torch.cat(
                [message_history[index], message], dim=0
            )
    compressed_stack, _ = compress_message_block(
        anchor_history, message_history, bits, fit_method, basis,
        writes_per_layer,
    )
    ratio = native_kv_bytes(model, prefix + available) / compressed_stack.compressed_bytes
    return {
        "tokens": available,
        "baseline_ppl": math.exp(baseline_nll / available),
        "compressed_ppl": math.exp(compressed_nll / available),
        "mean_kl": kl_total / available,
        "top1_agreement": top_matches / available,
        "persistent_ratio": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--revision")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Inference device; auto selects CUDA when available",
    )
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--output", help="Write metadata and metrics as JSON")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument(
        "--layer-bits",
        help="Comma-separated per-layer allocation; layer 0 is the anchor",
    )
    parser.add_argument("--text-file")
    parser.add_argument(
        "--wikitext", action="store_true",
        help="Use the standard WikiText-2 raw validation split",
    )
    parser.add_argument(
        "--capture-point", choices=("residual", "preprojection"),
        default="residual",
    )
    parser.add_argument(
        "--ln-aware-candidates", type=int, default=1,
        help="Try N rotation seeds per delta and minimize post-LayerNorm MSE",
    )
    parser.add_argument(
        "--prediction", choices=("anchor", "adjacent", "middle_out"),
        default="anchor",
        help="Use shared-anchor, forward-adjacent, or bidirectional prediction",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Compress and replace only each new token instead of rebuilding history",
    )
    parser.add_argument(
        "--repacking", choices=(
            "identity", "random", "pca", "reader", "reader_closedloop",
        ),
        help="Repack adjacent deltas before mixed-bit quantization",
    )
    parser.add_argument(
        "--repacking-block-size", type=int,
        help="Share one basis within N-layer depth blocks and anchor each block",
    )
    parser.add_argument(
        "--repacking-protected-fraction", type=float,
        help="Fraction of most important basis directions stored at 8 bits",
    )
    parser.add_argument(
        "--repacking-boundaries",
        help="Comma-separated layer starts for phase-aligned repacking",
    )
    parser.add_argument(
        "--message-repacking", choices=(
            "identity", "pca", "prefix", "identity_combined",
            "pca_combined", "prefix_combined",
        ),
        help="Jointly compress attention/MLP writes across message depth",
    )
    parser.add_argument(
        "--folding-artifact",
        help="Projection-aware folding result JSON supplying a fixed allocation",
    )
    parser.add_argument(
        "--folding-target-bits", type=int, choices=(2, 3), default=3,
        help="Target-bit allocation to select from --folding-artifact",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    load_kwargs = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    tokenizer.model_max_length = 10**9
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation has no CUDA support"
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, **load_kwargs
    ).to(device)
    model.eval()
    if args.wikitext:
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    else:
        text = open(args.text_file, encoding="utf-8").read() if args.text_file else HELD_OUT_TEXT
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
    needed = args.token_offset + args.prefix + args.steps + 1
    if tokens.shape[1] < needed:
        repeats = math.ceil(needed / tokens.shape[1])
        tokens = tokens.repeat(1, repeats)
    tokens = tokens[:, args.token_offset:needed]

    parity_check(model, tokens, args.prefix, args.capture_point)
    allocation = (
        [int(value) for value in args.layer_bits.split(",")]
        if args.layer_bits else args.bits
    )
    if isinstance(allocation, list) and len(allocation) != len(decoder_layers(model)):
        raise ValueError("--layer-bits must contain one value per model layer")
    repacking_boundaries = (
        [int(value) for value in args.repacking_boundaries.split(",")]
        if args.repacking_boundaries else None
    )
    folding_allocation = None
    if args.folding_artifact:
        folding_report = json.loads(
            Path(args.folding_artifact).read_text(encoding="utf-8")
        )
        matches = [
            record for record in folding_report["records"]
            if record["strategy"] == "projection_aware_folded"
            and record["target_bits"] == args.folding_target_bits
        ]
        if len(matches) != 1:
            raise ValueError("folding artifact does not contain one matching allocation")
        folding_allocation = [tuple(pair) for pair in matches[0]["allocation"]]
    if args.message_repacking:
        if not isinstance(allocation, int):
            raise ValueError("message repacking requires uniform --bits")
        result = evaluate_message_repacking(
            model, tokens, args.prefix, args.steps, allocation,
            args.message_repacking,
        )
    else:
        result = evaluate(
            model, tokens, args.prefix, args.steps, allocation, args.capture_point,
            args.ln_aware_candidates, args.prediction, args.incremental,
            args.repacking, args.repacking_block_size,
            args.repacking_protected_fraction,
            repacking_boundaries,
            folding_allocation,
        )
    relative = result["compressed_ppl"] / result["baseline_ppl"] - 1
    print(f"\nTeacher-forced autoregressive validation ({result['tokens']} tokens)")
    print(f"  capture point:           {args.capture_point}")
    print(f"  device:                  {device}")
    print(f"  LN-aware candidates:     {args.ln_aware_candidates}")
    print(f"  prediction:              {args.prediction}")
    print(f"  incremental:             {args.incremental}")
    print(f"  repacking:               {args.repacking or 'none'}")
    print(f"  repacking block size:    {args.repacking_block_size or 'none'}")
    print(f"  protected fraction:      {args.repacking_protected_fraction if args.repacking_protected_fraction is not None else 'none'}")
    print(f"  repacking boundaries:    {repacking_boundaries or 'none'}")
    print(f"  message repacking:       {args.message_repacking or 'none'}")
    print(f"  projection-aware fold:   {args.folding_target_bits if folding_allocation else 'none'}")
    print(f"  FP16-cache baseline PPL: {result['baseline_ppl']:.4f}")
    label = args.layer_bits or f"{args.bits}-bit"
    print(f"  GIHKCC {label} PPL:    {result['compressed_ppl']:.4f}")
    print(f"  relative PPL change:     {relative:+.2%}")
    print(f"  mean logit KL:           {result['mean_kl']:.6f}")
    print(f"  top-1 agreement:         {result['top1_agreement']:.2%}")
    print(f"  persistent-cache ratio: {result['persistent_ratio']:.2f}x")
    if args.output:
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "model_revision": args.revision,
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "split": "validation",
            "token_offset": args.token_offset,
            "prefix": args.prefix,
            "steps": args.steps,
            "bits": allocation,
            "prediction": args.prediction,
            "capture_point": args.capture_point,
            "incremental": args.incremental,
            "repacking": args.repacking,
            "repacking_block_size": args.repacking_block_size,
            "repacking_protected_fraction": args.repacking_protected_fraction,
            "repacking_boundaries": repacking_boundaries,
            "message_repacking": args.message_repacking,
            "folding_artifact": args.folding_artifact,
            "folding_target_bits": (
                args.folding_target_bits if folding_allocation else None
            ),
            "device": device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "result": result,
            "relative_ppl_change": relative,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
