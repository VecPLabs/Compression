"""Aggregate repeated quantized K/V validation JSON files by configuration."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summary(values):
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = {}
    for name in args.inputs:
        path = Path(name)
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (
            report["key_bits"], report["value_bits"], report["hot_window"],
            report["page_size"], report["steps"],
        )
        groups.setdefault(key, []).append((path.name, report))

    aggregate = {"configurations": []}
    for key, runs in sorted(groups.items(), reverse=True):
        first = runs[0][1]
        aggregate["configurations"].append({
            "key_bits": key[0], "value_bits": key[1],
            "hot_window": key[2], "page_size": key[3], "steps": key[4],
            "repeat_count": len(runs),
            "token_offsets": sorted({
                report.get("token_offset", 0) for _, report in runs
            }),
            "source_files": [name for name, _ in runs],
            "resident_ratio": first["resident_ratio"],
            "relative_ppl_change": summary([
                report["relative_ppl_change"] for _, report in runs
            ]),
            "mean_kl": summary([
                report["mean_kl"] for _, report in runs
            ]),
            "top1_agreement": summary([
                report["top1_agreement"] for _, report in runs
            ]),
            "compressed_tokens_per_second": summary([
                report["compressed_tokens_per_second"] for _, report in runs
            ]),
            "baseline_tokens_per_second": summary([
                report["baseline_tokens_per_second"] for _, report in runs
            ]),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
