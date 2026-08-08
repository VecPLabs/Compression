"""Comparable synthetic benchmarks for compression methods and valid stacks."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Tuple

import torch

from gihkcc import GIHKCCConfig, compress_kv_cache, decompress_kv_cache
from gihkcc_v2 import (
    GIHKCCV2Config,
    compress_kv_cache_v2,
    decompress_kv_cache_v2,
)
from kvtc import KVTCConfig, kvtc_compress_all_deltas, kvtc_decompress_all_deltas
from pca_layer import PCAConfig, pca_compress_layers, pca_decompress_layers
from ternary import (
    quint5_compress_residuals,
    quint5_decompress_residuals,
    xnor_compress_residuals,
    xnor_decompress_residuals,
)
from turboquant import (
    TurboQuantConfig,
    turboquant_compress_list,
    turboquant_decompress_list,
)
from turboquant_paper import (
    paper_turboquant_compress_list,
    paper_turboquant_decompress_list,
)


TensorList = List[torch.Tensor]


@dataclass
class Result:
    method: str
    compressed_bytes: int
    ratio: float
    psnr_db: float
    mae: float
    relative_error: float
    inner_product_nrmse: float
    inner_product_bias: float
    compress_ms: float
    decompress_ms: float


def tensor_bytes(tensors: TensorList) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def measure(original: TensorList, reconstructed: TensorList) -> Tuple[float, float, float]:
    squared_error = absolute_error = signal = 0.0
    elements = 0
    for source, restored in zip(original, reconstructed):
        delta = source.float() - restored.float()
        squared_error += delta.square().sum().item()
        absolute_error += delta.abs().sum().item()
        signal += source.float().square().sum().item()
        elements += source.numel()
    mse = squared_error / elements
    mae = absolute_error / elements
    psnr = 10.0 * math.log10(signal / squared_error) if squared_error else math.inf
    relative_error = math.sqrt(squared_error / signal) if signal else 0.0
    return psnr, mae, relative_error


def measure_inner_products(
    original: TensorList, reconstructed: TensorList, seed: int = 1234
) -> Tuple[float, float]:
    """Measure attention-like dot products against fixed random query vectors."""
    generator = torch.Generator().manual_seed(seed)
    squared_error = signal = signed_error = 0.0
    count = 0
    for source, restored in zip(original, reconstructed):
        vectors = source.float().reshape(-1, source.shape[-1])
        restored_vectors = restored.float().reshape_as(vectors)
        query = torch.randn(source.shape[-1], generator=generator)
        query = query / query.norm()
        expected = vectors @ query
        observed = restored_vectors @ query
        error = observed - expected
        squared_error += error.square().sum().item()
        signal += expected.square().sum().item()
        signed_error += error.sum().item()
        count += error.numel()
    nrmse = math.sqrt(squared_error / signal) if signal else 0.0
    signal_rms = math.sqrt(signal / count) if count else 0.0
    normalized_bias = (signed_error / count) / signal_rms if signal_rms else 0.0
    return nrmse, normalized_bias


def generate_cache(
    layers: int, heads: int, tokens: int, head_dim: int, seed: int
) -> Tuple[TensorList, TensorList]:
    """Generate correlated, low-rank layer evolution with periodic shifts."""
    def stack(stack_seed: int) -> TensorList:
        generator = torch.Generator().manual_seed(stack_seed)
        current = torch.randn(heads, tokens, head_dim, generator=generator).half()
        output = [current.clone()]
        rank = min(8, head_dim)
        for layer in range(1, layers):
            scale = 0.25 if layer % 8 == 0 else 0.03
            left = torch.randn(heads * tokens, rank, generator=generator) * scale
            right = torch.randn(rank, head_dim, generator=generator)
            update = (left @ right).reshape(heads, tokens, head_dim).half()
            current = current + update
            output.append(current.clone())
        return output

    return stack(seed), stack(seed + 1000)


def timed(fn: Callable):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def record(
    name: str,
    original: TensorList,
    reconstructed: TensorList,
    compressed_bytes: int,
    compress_ms: float,
    decompress_ms: float,
) -> Result:
    psnr, mae, relative_error = measure(original, reconstructed)
    ip_nrmse, ip_bias = measure_inner_products(original, reconstructed)
    original_bytes = tensor_bytes(original)
    return Result(
        method=name,
        compressed_bytes=compressed_bytes,
        ratio=original_bytes / compressed_bytes,
        psnr_db=psnr,
        mae=mae,
        relative_error=relative_error,
        inner_product_nrmse=ip_nrmse,
        inner_product_bias=ip_bias,
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
    )


def delta_entries(cache):
    entries = []
    for side in (cache.keys_l2, cache.values_l2):
        entries.extend(side.keyframe_deltas)
        entries.extend(side.l1_deltas)
    return entries


def anchor_bytes(cache) -> int:
    return sum(
        frame.data.numel() * frame.data.element_size()
        for side in (cache.keys_l2, cache.values_l2)
        for frame in side.super_keyframes
    )


def benchmark_fold_codec(keys, values, codec: str, quality: float) -> Result:
    original = keys + values
    config = GIHKCCConfig(l1_snr_threshold=0.92, l1_max_keyframe_span=8, l2_enabled=True)
    cache, fold_ms = timed(lambda: compress_kv_cache(keys, values, config))
    entries = delta_entries(cache)
    deltas = [entry.delta for entry in entries]

    if codec == "kvtc":
        codec_config = KVTCConfig(energy_retention=quality, coeff_quant_bits=8)
        (encoded, stats), codec_ms = timed(
            lambda: kvtc_compress_all_deltas(deltas, codec_config)
        )
        restored_deltas, decode_ms = timed(lambda: kvtc_decompress_all_deltas(encoded))
        size = anchor_bytes(cache) + stats["compressed_bytes"]
        label = f"GIHKCC + KVTC e={quality:g}"
    elif codec == "turboquant":
        bits = int(quality)
        # QJL payload reconstruction is not implemented yet, so do not count
        # or advertise correction data that the decoder cannot apply.
        codec_config = TurboQuantConfig(target_bits=bits, qjl_enabled=False)
        (encoded, stats), codec_ms = timed(
            lambda: turboquant_compress_list(deltas, codec_config)
        )
        restored_deltas, decode_ms = timed(lambda: turboquant_decompress_list(encoded))
        size = anchor_bytes(cache) + stats["compressed_bytes"]
        label = f"GIHKCC + TurboQuant {bits}b"
    else:
        raise ValueError(codec)

    restored_cache = copy.deepcopy(cache)
    for entry, restored in zip(delta_entries(restored_cache), restored_deltas):
        entry.delta = restored
    (restored_keys, restored_values), unfold_ms = timed(
        lambda: decompress_kv_cache(restored_cache)
    )
    return record(
        label, original, restored_keys + restored_values, size,
        fold_ms + codec_ms, decode_ms + unfold_ms,
    )


def benchmark_gihkcc_v2(
    keys: TensorList, values: TensorList, config: GIHKCCV2Config, label: str
) -> Result:
    original = keys + values
    cache, compress_ms = timed(lambda: compress_kv_cache_v2(keys, values, config))
    (restored_keys, restored_values), decompress_ms = timed(
        lambda: decompress_kv_cache_v2(cache)
    )
    return record(
        label, original, restored_keys + restored_values, cache.compressed_bytes,
        compress_ms, decompress_ms,
    )


def benchmark_fold_pca_turboquant(
    keys: TensorList, values: TensorList, variance: float, bits: int
) -> Result:
    original = keys + values
    config = GIHKCCConfig(l1_snr_threshold=0.92, l1_max_keyframe_span=8, l2_enabled=True)
    cache, fold_ms = timed(lambda: compress_kv_cache(keys, values, config))
    entries = delta_entries(cache)
    deltas = [entry.delta for entry in entries]

    pca_config = PCAConfig(variance_threshold=variance, per_head=False)
    (pca_encoded, _), pca_ms = timed(lambda: pca_compress_layers(deltas, pca_config))
    coefficients = [item.coefficients for item in pca_encoded]
    tq_config = TurboQuantConfig(target_bits=bits, qjl_enabled=False)
    (tq_encoded, tq_stats), tq_ms = timed(
        lambda: turboquant_compress_list(coefficients, tq_config)
    )
    restored_coefficients, tq_decode_ms = timed(
        lambda: turboquant_decompress_list(tq_encoded)
    )
    restored_pca = copy.deepcopy(pca_encoded)
    for item, coefficients_tensor in zip(restored_pca, restored_coefficients):
        item.coefficients = coefficients_tensor
    restored_deltas, pca_decode_ms = timed(lambda: pca_decompress_layers(restored_pca))

    restored_cache = copy.deepcopy(cache)
    for entry, restored in zip(delta_entries(restored_cache), restored_deltas):
        entry.delta = restored
    (restored_keys, restored_values), unfold_ms = timed(
        lambda: decompress_kv_cache(restored_cache)
    )
    basis_bytes = sum(
        item.compressed_bytes
        - item.coefficients.numel() * item.coefficients.element_size()
        for item in pca_encoded
    )
    size = anchor_bytes(cache) + basis_bytes + tq_stats["compressed_bytes"]
    return record(
        f"GIHKCC + PCA {variance:g} + TQ {bits}b", original,
        restored_keys + restored_values, size, fold_ms + pca_ms + tq_ms,
        tq_decode_ms + pca_decode_ms + unfold_ms,
    )


def benchmark_fold_paper_turboquant(
    keys: TensorList, values: TensorList, bits: int
) -> Result:
    original = keys + values
    config = GIHKCCConfig(l1_snr_threshold=0.92, l1_max_keyframe_span=8, l2_enabled=True)
    cache, fold_ms = timed(lambda: compress_kv_cache(keys, values, config))
    entries = delta_entries(cache)
    deltas = [entry.delta for entry in entries]
    (encoded, encoded_bytes), codec_ms = timed(
        lambda: paper_turboquant_compress_list(deltas, bits, inner_product=False)
    )
    restored_deltas, decode_ms = timed(lambda: paper_turboquant_decompress_list(encoded))
    restored_cache = copy.deepcopy(cache)
    for entry, restored in zip(delta_entries(restored_cache), restored_deltas):
        entry.delta = restored
    (restored_keys, restored_values), unfold_ms = timed(
        lambda: decompress_kv_cache(restored_cache)
    )
    return record(
        f"GIHKCC + Paper TQ {bits}b MSE", original,
        restored_keys + restored_values, anchor_bytes(cache) + encoded_bytes,
        fold_ms + codec_ms, decode_ms + unfold_ms,
    )


def benchmark_turboquant(tensors: TensorList, bits: int) -> Result:
    config = TurboQuantConfig(target_bits=bits, qjl_enabled=False)
    (encoded, stats), compress_ms = timed(lambda: turboquant_compress_list(tensors, config))
    restored, decompress_ms = timed(lambda: turboquant_decompress_list(encoded))
    return record(
        f"TurboQuant {bits}b", tensors, restored, stats["compressed_bytes"],
        compress_ms, decompress_ms,
    )


def benchmark_paper_turboquant(
    tensors: TensorList, bits: int, inner_product: bool
) -> Result:
    (encoded, compressed_bytes), compress_ms = timed(
        lambda: paper_turboquant_compress_list(
            tensors, bits, inner_product=inner_product
        )
    )
    restored, decompress_ms = timed(lambda: paper_turboquant_decompress_list(encoded))
    objective = "product" if inner_product else "MSE"
    return record(
        f"Paper TurboQuant {bits}b {objective}", tensors, restored,
        compressed_bytes, compress_ms, decompress_ms,
    )


def benchmark_pca_turboquant(tensors: TensorList, variance: float, bits: int) -> Result:
    pca_config = PCAConfig(variance_threshold=variance, per_head=False)
    (pca_encoded, _), pca_ms = timed(lambda: pca_compress_layers(tensors, pca_config))
    coefficients = [item.coefficients for item in pca_encoded]
    tq_config = TurboQuantConfig(target_bits=bits, qjl_enabled=False)
    (tq_encoded, tq_stats), tq_ms = timed(
        lambda: turboquant_compress_list(coefficients, tq_config)
    )
    restored_coefficients, tq_decode_ms = timed(
        lambda: turboquant_decompress_list(tq_encoded)
    )
    restored_pca = copy.deepcopy(pca_encoded)
    for item, coefficients_tensor in zip(restored_pca, restored_coefficients):
        item.coefficients = coefficients_tensor
    restored, pca_decode_ms = timed(lambda: pca_decompress_layers(restored_pca))
    basis_bytes = sum(
        item.compressed_bytes
        - item.coefficients.numel() * item.coefficients.element_size()
        for item in pca_encoded
    )
    return record(
        f"PCA {variance:g} + TurboQuant {bits}b", tensors, restored,
        basis_bytes + tq_stats["compressed_bytes"], pca_ms + tq_ms,
        tq_decode_ms + pca_decode_ms,
    )


def benchmark_pca(tensors: TensorList, variance: float) -> Result:
    config = PCAConfig(variance_threshold=variance, per_head=False)
    (encoded, stats), compress_ms = timed(lambda: pca_compress_layers(tensors, config))
    restored, decompress_ms = timed(lambda: pca_decompress_layers(encoded))
    return record(
        f"PCA variance={variance:g}", tensors, restored, stats["compressed_bytes"],
        compress_ms, decompress_ms,
    )


def benchmark_ternary(keys: TensorList, values: TensorList) -> Result:
    start = time.perf_counter()
    encoded_keys = xnor_compress_residuals(keys)
    encoded_values = xnor_compress_residuals(values)
    compress_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    restored = xnor_decompress_residuals(encoded_keys) + xnor_decompress_residuals(encoded_values)
    decompress_ms = (time.perf_counter() - start) * 1000.0
    size = encoded_keys.total_compressed_bytes + encoded_values.total_compressed_bytes
    return record("XNOR ternary chain", keys + values, restored, size, compress_ms, decompress_ms)


def benchmark_xnor_levels(keys: TensorList, values: TensorList, levels: int) -> Result:
    start = time.perf_counter()
    encoded_keys = quint5_compress_residuals(keys, levels=levels)
    encoded_values = quint5_compress_residuals(values, levels=levels)
    compress_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    restored = (
        quint5_decompress_residuals(encoded_keys, levels=levels)
        + quint5_decompress_residuals(encoded_values, levels=levels)
    )
    decompress_ms = (time.perf_counter() - start) * 1000.0
    size = encoded_keys.total_compressed_bytes + encoded_values.total_compressed_bytes
    return record(
        f"XNOR {2 * levels + 1}-level chain", keys + values, restored, size,
        compress_ms, decompress_ms,
    )


def print_results(results: List[Result], original_bytes: int) -> None:
    print(f"\nOriginal: {original_bytes / 1024**2:.2f} MiB")
    print(f"{'Method':32} {'Ratio':>8} {'PSNR':>9} {'MAE':>11} {'IP nRMSE':>10} {'IP bias':>9} {'Enc ms':>9} {'Dec ms':>9}")
    print("-" * 116)
    for item in sorted(results, key=lambda result: result.ratio):
        psnr = "inf" if math.isinf(item.psnr_db) else f"{item.psnr_db:.2f}"
        print(
            f"{item.method:32} {item.ratio:7.2f}x {psnr:>8} "
            f"{item.mae:11.6f} {item.inner_product_nrmse:10.4f} "
            f"{item.inner_product_bias:9.4f} "
            f"{item.compress_ms:9.1f} {item.decompress_ms:9.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default=None, help="Optional result file")
    args = parser.parse_args()

    torch.set_num_threads(1)
    keys, values = generate_cache(args.layers, args.heads, args.tokens, args.head_dim, args.seed)
    tensors = keys + values
    results = [
        benchmark_pca(tensors, 0.99),
        benchmark_pca_turboquant(tensors, 0.99, 4),
        benchmark_turboquant(tensors, 4),
        benchmark_turboquant(tensors, 8),
        benchmark_paper_turboquant(tensors, 3, False),
        benchmark_paper_turboquant(tensors, 4, False),
        benchmark_paper_turboquant(tensors, 4, True),
        benchmark_ternary(keys, values),
        benchmark_xnor_levels(keys, values, 2),
        benchmark_xnor_levels(keys, values, 4),
        benchmark_xnor_levels(keys, values, 8),
        benchmark_fold_codec(keys, values, "kvtc", 0.99),
        benchmark_fold_codec(keys, values, "kvtc", 0.999),
        benchmark_fold_codec(keys, values, "turboquant", 4),
        benchmark_fold_codec(keys, values, "turboquant", 8),
        benchmark_fold_paper_turboquant(keys, values, 3),
        benchmark_fold_paper_turboquant(keys, values, 4),
        benchmark_gihkcc_v2(
            keys, values,
            GIHKCCV2Config(
                key_anchor_bits=8, key_delta_bits=4,
                value_anchor_bits=8, value_delta_bits=4,
            ),
            "GIHKCC v2 closed-loop 4/4b",
        ),
        benchmark_gihkcc_v2(
            keys, values,
            GIHKCCV2Config(
                key_anchor_bits=8, key_delta_bits=4,
                value_anchor_bits=8, value_delta_bits=3,
            ),
            "GIHKCC v2 asymmetric K4/V3",
        ),
        benchmark_gihkcc_v2(
            keys, values,
            GIHKCCV2Config(
                key_anchor_bits=8, key_delta_bits=3,
                value_anchor_bits=8, value_delta_bits=3,
            ),
            "GIHKCC v2 closed-loop 3/3b",
        ),
        benchmark_gihkcc_v2(
            keys, values,
            GIHKCCV2Config(
                key_anchor_bits=8, key_delta_bits=2,
                value_anchor_bits=8, value_delta_bits=2,
            ),
            "GIHKCC v2 closed-loop 2/2b",
        ),
        benchmark_gihkcc_v2(
            keys, values,
            GIHKCCV2Config(
                key_anchor_bits=8, key_delta_bits=2,
                value_anchor_bits=8, value_delta_bits=1,
            ),
            "GIHKCC v2 asymmetric K2/V1",
        ),
        benchmark_fold_pca_turboquant(keys, values, 0.95, 4),
        benchmark_fold_pca_turboquant(keys, values, 0.99, 4),
    ]
    print_results(results, tensor_bytes(tensors))
    if args.json:
        payload = {"shape": vars(args), "original_bytes": tensor_bytes(tensors),
                   "results": [asdict(result) for result in results]}
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
