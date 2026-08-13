"""Orthogonal residual repacking and importance-aware mixed-bit quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from turboquant_paper import (
    normal_lloyd_max, paper_turboquant_compress, paper_turboquant_decompress,
)


@dataclass
class RepackingBasis:
    matrix: torch.Tensor
    importance: torch.Tensor
    name: str


@dataclass
class RepackedPredictiveStack:
    compressed_bytes: int
    bases: list[RepackingBasis | None]


@dataclass
class MessageRepackingStack:
    compressed_bytes: int
    basis: RepackingBasis


def _covariance(samples: torch.Tensor) -> torch.Tensor:
    samples = samples.float()
    centered = samples - samples.mean(0, keepdim=True)
    return centered.T @ centered / max(1, samples.shape[0] - 1)


def fit_repacking_basis(
    samples: torch.Tensor,
    method: str,
    reader_gram: torch.Tensor | None = None,
    seed: int = 42,
) -> RepackingBasis:
    """Fit an orthogonal basis using calibration samples only.

    Rows are observations and columns are residual-stream coordinates.  The
    returned matrix uses ``z = x @ matrix`` and ``x = z @ matrix.T``.
    """
    covariance = _covariance(samples).double()
    dim = covariance.shape[0]
    if method == "identity":
        matrix = torch.eye(dim, dtype=torch.float64)
    elif method == "random":
        generator = torch.Generator().manual_seed(seed)
        matrix, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator,
                                                dtype=torch.float64))
    elif method == "pca":
        _, matrix = torch.linalg.eigh(covariance)
        matrix = matrix.flip(1)
    elif method == "reader":
        if reader_gram is None:
            raise ValueError("reader method requires reader_gram")
        sensitivity = reader_gram.detach().cpu().double()
        # A symmetric joint energy/visibility operator. Its eigenvectors form
        # an orthogonal, cheaply invertible coordinate system.
        joint = (covariance @ sensitivity + sensitivity @ covariance) / 2
        _, matrix = torch.linalg.eigh(joint)
        matrix = matrix.flip(1)
    else:
        raise ValueError(f"unknown repacking method: {method}")

    transformed_covariance = matrix.T @ covariance @ matrix
    variance = transformed_covariance.diag().clamp_min(0)
    if reader_gram is None:
        importance = variance
    else:
        sensitivity = reader_gram.detach().cpu().double()
        visibility = (matrix.T @ sensitivity @ matrix).diag().clamp_min(0)
        importance = variance * visibility
    return RepackingBasis(matrix.float(), importance.float(), method)


def allocate_mixed_bits(
    importance: torch.Tensor, average_bits: float = 2.0,
    low_bits: int = 0, high_bits: int | None = None,
) -> torch.Tensor:
    """Allocate two precision levels while matching an average bit budget."""
    high_bits = high_bits or max(3, math.ceil(average_bits))
    fraction_high = (average_bits - low_bits) / (high_bits - low_bits)
    if not 0 <= fraction_high <= 1:
        raise ValueError("average_bits must lie between low_bits and high_bits")
    count = round(importance.numel() * fraction_high)
    bits = torch.full_like(importance, low_bits, dtype=torch.int64)
    if count:
        bits[torch.topk(importance, count).indices] = high_bits
    return bits


def allocate_three_band_bits(
    importance: torch.Tensor, average_bits: float,
    protected_fraction: float, protected_bits: int = 8,
    quantized_bits: int = 3,
) -> torch.Tensor:
    """Assign important coordinates to protected, quantized, or dropped bands."""
    if not 0 <= protected_fraction <= 1:
        raise ValueError("protected_fraction must be between zero and one")
    count = importance.numel()
    protected = round(count * protected_fraction)
    remaining_budget = average_bits * count - protected_bits * protected
    quantized = round(remaining_budget / quantized_bits)
    if remaining_budget < 0 or protected + quantized > count:
        raise ValueError("three-band allocation cannot meet the requested budget")
    order = torch.argsort(importance, descending=True)
    bits = torch.zeros(count, dtype=torch.int64, device=importance.device)
    bits[order[:protected]] = protected_bits
    bits[order[protected:protected + quantized]] = quantized_bits
    return bits


def _bits_for_basis(
    basis: RepackingBasis, average_bits: float,
    protected_fraction: float | None,
) -> torch.Tensor:
    if protected_fraction is None:
        return allocate_mixed_bits(basis.importance, average_bits)
    return allocate_three_band_bits(
        basis.importance, average_bits, protected_fraction
    )


def quantize_repacked(
    tensor: torch.Tensor, basis: RepackingBasis, bits: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Repack, groupwise symmetric-quantize, and invert the transform."""
    shape = tensor.shape
    vectors = tensor.float().reshape(-1, shape[-1])
    matrix = basis.matrix.to(vectors.device)
    transformed = vectors @ matrix
    restored = torch.empty_like(transformed)
    payload_bits = 0
    metadata_bytes = 0
    for width in bits.unique(sorted=True).tolist():
        columns = torch.where(bits.to(vectors.device) == width)[0]
        group = transformed[:, columns]
        if width == 0:
            restored[:, columns] = 0
            continue
        norms = group.norm(dim=1, keepdim=True)
        unit = group / norms.clamp_min(1e-12)
        centroids = normal_lloyd_max(int(width)).to(group.device)
        centroids = centroids / max(1, group.shape[1]) ** 0.5
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        codes = torch.bucketize(unit.contiguous(), boundaries)
        restored[:, columns] = centroids[codes] * norms
        payload_bits += group.numel() * int(width)
        metadata_bytes += vectors.shape[0] * 2  # one FP16 scale per row/group
    decoded = (restored @ matrix.T).reshape(shape).to(tensor.dtype)
    return decoded, (payload_bits + 7) // 8 + metadata_bytes


def compress_adjacent_repacked(
    states: list[torch.Tensor], delta_bits: float,
    reader_grams: list[torch.Tensor], method: str = "reader",
    bases: list[RepackingBasis | None] | None = None,
    calibration_tokens: int = 32,
    protected_fraction: float | None = None,
) -> tuple[RepackedPredictiveStack, list[torch.Tensor]]:
    """Closed-loop adjacent coding in learned, reusable orthogonal bases."""
    if not states:
        return RepackedPredictiveStack(0, []), []
    if bases is None:
        fit_method = "reader" if method == "reader_closedloop" else method
        bases = [None]
        for layer_idx in range(1, len(states)):
            samples = states[layer_idx] - states[layer_idx - 1]
            samples = samples[:min(calibration_tokens, samples.shape[0])]
            bases.append(fit_repacking_basis(
                samples, fit_method, reader_grams[layer_idx]
            ))
        if method == "reader_closedloop":
            calibration = [state[:min(calibration_tokens, state.shape[0])]
                           for state in states]
            # Fixed-point calibration: expose each layer to the error produced
            # by the current decoder-visible predecessor, then refit. Three
            # rounds were enough for this cheap diagnostic to settle.
            for _ in range(3):
                anchor = paper_turboquant_compress(calibration[0], 8)
                reconstructed_calibration = [paper_turboquant_decompress(anchor)]
                updated = [None]
                for layer_idx in range(1, len(calibration)):
                    delta = calibration[layer_idx] - reconstructed_calibration[-1]
                    updated_basis = fit_repacking_basis(
                        delta, "reader", reader_grams[layer_idx]
                    )
                    bits = _bits_for_basis(
                        updated_basis, delta_bits, protected_fraction
                    )
                    restored_delta, _ = quantize_repacked(
                        delta, updated_basis, bits
                    )
                    reconstructed_calibration.append(
                        reconstructed_calibration[-1] + restored_delta
                    )
                    updated.append(updated_basis)
                bases = updated
    anchor = paper_turboquant_compress(states[0], 8)
    reconstructed = [paper_turboquant_decompress(anchor)]
    size = anchor.compressed_bytes + 8
    for layer_idx in range(1, len(states)):
        delta = states[layer_idx] - reconstructed[layer_idx - 1]
        bits = _bits_for_basis(
            bases[layer_idx], delta_bits, protected_fraction
        )
        restored_delta, payload_bytes = quantize_repacked(
            delta, bases[layer_idx], bits
        )
        reconstructed.append(reconstructed[layer_idx - 1] + restored_delta)
        size += payload_bytes + 8
    return RepackedPredictiveStack(size, bases), reconstructed


def compress_blockwise_repacked(
    states: list[torch.Tensor], delta_bits: float,
    reader_grams: list[torch.Tensor], block_size: int,
    method: str = "reader",
    bases: list[RepackingBasis | None] | None = None,
    calibration_tokens: int = 32,
    protected_fraction: float | None = None,
    boundaries: list[int] | None = None,
) -> tuple[RepackedPredictiveStack, list[torch.Tensor]]:
    """Share one basis within each depth block and reset at block boundaries."""
    if block_size < 2 and boundaries is None:
        raise ValueError("block_size must be at least 2")
    if not states:
        return RepackedPredictiveStack(0, []), []
    if boundaries is None:
        boundaries = list(range(0, len(states), block_size)) + [len(states)]
    else:
        boundaries = sorted(set(boundaries + [0, len(states)]))
    segments = list(zip(boundaries[:-1], boundaries[1:]))
    starts = {start for start, _ in segments}
    if bases is None:
        bases = [None] * len(states)
        for start, stop in segments:
            if stop - start == 1:
                continue
            samples = torch.cat([
                (states[layer_idx] - states[layer_idx - 1])[
                    :min(calibration_tokens, states[layer_idx].shape[0])
                ]
                for layer_idx in range(start + 1, stop)
            ])
            gram = torch.stack(reader_grams[start + 1:stop]).mean(0)
            basis = fit_repacking_basis(samples, method, gram)
            for layer_idx in range(start + 1, stop):
                bases[layer_idx] = basis

    reconstructed: list[torch.Tensor] = []
    size = 0
    for layer_idx, state in enumerate(states):
        if layer_idx in starts:
            anchor = paper_turboquant_compress(state, 8)
            reconstructed.append(paper_turboquant_decompress(anchor))
            size += anchor.compressed_bytes + 8
            continue
        delta = state - reconstructed[layer_idx - 1]
        basis = bases[layer_idx]
        bits = _bits_for_basis(basis, delta_bits, protected_fraction)
        restored_delta, payload_bytes = quantize_repacked(delta, basis, bits)
        reconstructed.append(reconstructed[layer_idx - 1] + restored_delta)
        size += payload_bytes + 8
    return RepackedPredictiveStack(size, bases), reconstructed


def compress_message_block(
    anchor_state: torch.Tensor, messages: list[torch.Tensor],
    average_bits: float, method: str = "pca",
    basis: RepackingBasis | None = None,
    writes_per_layer: int = 2,
) -> tuple[MessageRepackingStack, list[torch.Tensor]]:
    """Jointly code attention/MLP writes along the message-depth axis."""
    message_count = len(messages)
    layer_count = message_count // writes_per_layer
    stack = torch.stack([message.float() for message in messages])
    samples = stack.permute(1, 2, 0).reshape(-1, message_count)
    if basis is None:
        gram = None
        fit_method = method
        if method == "prefix":
            prefix = torch.zeros(layer_count, message_count)
            for layer_idx in range(layer_count):
                prefix[layer_idx, :writes_per_layer * layer_idx] = 1
            gram = prefix.T @ prefix
            fit_method = "reader"
        basis = fit_repacking_basis(samples, fit_method, gram)
    bits = allocate_mixed_bits(basis.importance, average_bits)
    decoded, payload_bytes = quantize_repacked(
        samples.to(anchor_state.dtype), basis, bits
    )
    decoded_messages = decoded.reshape(
        anchor_state.shape[0], anchor_state.shape[1], message_count
    ).permute(2, 0, 1)
    anchor = paper_turboquant_compress(anchor_state, 8)
    current = paper_turboquant_decompress(anchor).float()
    residuals = []
    for layer_idx in range(layer_count):
        residuals.append(current.to(anchor_state.dtype))
        start = writes_per_layer * layer_idx
        for write_idx in range(start, start + writes_per_layer):
            current = current + decoded_messages[write_idx].float()
    size = anchor.compressed_bytes + payload_bytes + 8 * (message_count + 1)
    return MessageRepackingStack(size, basis), residuals
