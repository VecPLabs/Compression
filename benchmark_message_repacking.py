"""Compress attention/MLP writes jointly across transformer depth."""

from __future__ import annotations

import argparse

import torch

from residual_repacking import (
    allocate_mixed_bits, fit_repacking_basis, quantize_repacked,
)


def capture_neox_messages(model, token_ids):
    inputs, outputs, attention, mlp, hooks = [], [], [], [], []
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, args: inputs.append(args[0][0].detach().cpu())
        ))
        hooks.append(layer.register_forward_hook(
            lambda _module, _args, result: outputs.append(result.detach().cpu())
        ))
        hooks.append(layer.post_attention_dropout.register_forward_hook(
            lambda _module, _args, result: attention.append(result[0].detach().cpu())
        ))
        hooks.append(layer.post_mlp_dropout.register_forward_hook(
            lambda _module, _args, result: mlp.append(result[0].detach().cpu())
        ))
    with torch.no_grad():
        model(token_ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    messages = []
    for layer_idx in range(len(inputs)):
        messages.extend([attention[layer_idx], mlp[layer_idx]])
        expected = inputs[layer_idx] + attention[layer_idx] + mlp[layer_idx]
        if not torch.allclose(expected, outputs[layer_idx], atol=0.1, rtol=1e-3):
            error = (expected - outputs[layer_idx]).abs().max().item()
            raise RuntimeError(
                f"captured writes do not reconstruct layer output: {error}"
            )
    return inputs, messages


def message_samples(messages, token_slice):
    stack = torch.stack([message[token_slice].float() for message in messages])
    # Every token/feature coordinate is one observation across message depth.
    return stack.permute(1, 2, 0).reshape(-1, stack.shape[0])


def reconstruct_inputs(anchor, decoded_messages, layer_count):
    current = anchor.float()
    restored = []
    for layer_idx in range(layer_count):
        restored.append(current)
        current = current + decoded_messages[2 * layer_idx].float()
        current = current + decoded_messages[2 * layer_idx + 1].float()
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--calibration", type=int, default=128)
    parser.add_argument("--bits", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    text = ("Layers transform information and pass their contributions forward. " * 100)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=args.tokens).input_ids.to(args.device)
    inputs, messages = capture_neox_messages(model, ids)
    split = min(args.calibration, inputs[0].shape[0] // 2)
    calibration = message_samples(messages, slice(0, split))
    evaluation = message_samples(messages, slice(split, None))
    message_count = len(messages)
    layer_count = len(inputs)
    # Prefix sums are the actual consumers of the message sequence.
    prefix = torch.zeros(layer_count, message_count)
    for layer_idx in range(layer_count):
        prefix[layer_idx, :2 * layer_idx] = 1
    prefix_gram = prefix.T @ prefix

    print(f"model={args.model} layers={layer_count} messages={message_count} "
          f"calibration={split} evaluation={inputs[0].shape[0]-split}")
    print(f"{'axis basis':12} {'message MSE':>14} {'state MSE':>14} "
          f"{'QKV MSE':>14} {'ratio':>9}")
    truth_inputs = [item[split:].float() for item in inputs]
    original_bytes = sum(message[split:].numel() * 2 for message in messages)
    for method in ("identity", "pca", "prefix"):
        basis = fit_repacking_basis(
            calibration, "reader" if method == "prefix" else method,
            prefix_gram if method == "prefix" else None,
        )
        bits = allocate_mixed_bits(basis.importance, args.bits)
        decoded, payload = quantize_repacked(evaluation.half(), basis, bits)
        decoded_stack = decoded.reshape(
            inputs[0].shape[0] - split, inputs[0].shape[1], message_count
        ).permute(2, 0, 1)
        truth_stack = torch.stack([message[split:].float() for message in messages])
        message_mse = (decoded_stack.float() - truth_stack).square().mean().item()
        restored_inputs = reconstruct_inputs(
            inputs[0][split:], list(decoded_stack), layer_count
        )
        state_mse = torch.stack([
            (restored - truth).square().mean()
            for restored, truth in zip(restored_inputs, truth_inputs)
        ]).mean().item()
        qkv_errors = []
        for layer_idx, (restored, truth) in enumerate(zip(restored_inputs, truth_inputs)):
            weight = model.gpt_neox.layers[layer_idx].attention.query_key_value.weight.detach().float().cpu()
            qkv_errors.append(((restored @ weight.T) - (truth @ weight.T)).square().mean())
        qkv_mse = torch.stack(qkv_errors).mean().item()
        print(f"{method:12} {message_mse:14.7g} {state_mse:14.7g} "
              f"{qkv_mse:14.7g} {original_bytes/payload:8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
