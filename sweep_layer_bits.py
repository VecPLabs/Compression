"""Measure GPT-NeoX residual-delta sensitivity one layer at a time."""

from __future__ import annotations

import argparse
import json

from validate_autoregressive import evaluate, parity_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", default="results/layer-bit-sweep.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="float32")
    model.eval()
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
    )
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, return_tensors="pt").input_ids[
        :, :args.prefix + args.steps + 1
    ]
    parity_check(model, tokens, args.prefix)

    layers = model.config.num_hidden_layers
    baseline_bits = [8] + [4] * (layers - 1)
    configurations = [("all-4bit", baseline_bits)]
    for layer in range(1, layers):
        for bits in (3, 2):
            allocation = baseline_bits.copy()
            allocation[layer] = bits
            configurations.append((f"L{layer}={bits}b", allocation))

    results = []
    print(f"\n{'Config':12} {'Ratio':>8} {'PPL':>10} {'ΔPPL':>9} {'KL':>10} {'Top1':>8}")
    print("-" * 62)
    for name, allocation in configurations:
        result = evaluate(model, tokens, args.prefix, args.steps, allocation)
        relative = result["compressed_ppl"] / result["baseline_ppl"] - 1
        result.update(name=name, allocation=allocation, relative_ppl=relative)
        results.append(result)
        print(
            f"{name:12} {result['persistent_ratio']:7.2f}x "
            f"{result['compressed_ppl']:10.3f} {relative:+8.2%} "
            f"{result['mean_kl']:10.5f} {result['top1_agreement']:8.2%}"
        )

    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

