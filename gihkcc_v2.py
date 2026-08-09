"""GIHKCC v2: closed-loop predictive coding with real encoded payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch

from gihkcc import compute_statistical_snr
from turboquant_paper import (
    PaperTurboQuantCompressed,
    normal_lloyd_max,
    paper_turboquant_compress,
    paper_turboquant_decompress,
)
from turboquant import get_rotation_matrix


@dataclass
class GIHKCCV2Config:
    similarity_threshold: float = 0.92
    max_keyframe_span: int = 8
    key_anchor_bits: int = 8
    key_delta_bits: int = 4
    value_anchor_bits: int = 8
    value_delta_bits: int = 3
    prediction_mode: str = "anchor"


@dataclass
class PredictiveEntry:
    layer_idx: int
    reference_layer: int | None
    payload: PaperTurboQuantCompressed

    @property
    def is_anchor(self) -> bool:
        return self.reference_layer is None


@dataclass
class PredictiveStack:
    entries: List[PredictiveEntry] = field(default_factory=list)
    num_layers: int = 0

    @property
    def compressed_bytes(self) -> int:
        return sum(entry.payload.compressed_bytes + 8 for entry in self.entries)


@dataclass
class GIHKCCV2Cache:
    keys: PredictiveStack
    values: PredictiveStack
    config: GIHKCCV2Config
    original_bytes: int

    @property
    def compressed_bytes(self) -> int:
        return self.keys.compressed_bytes + self.values.compressed_bytes

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes


def compress_predictive_stack(
    states: List[torch.Tensor], anchor_bits: int,
    delta_bits: int | Sequence[int],
    config: GIHKCCV2Config,
) -> PredictiveStack:
    if not states:
        return PredictiveStack()

    if config.prediction_mode not in {"anchor", "adjacent", "middle_out"}:
        raise ValueError(
            "prediction_mode must be 'anchor', 'adjacent', or 'middle_out'"
        )

    result = PredictiveStack(num_layers=len(states))
    reconstructed = [None] * len(states)

    if config.prediction_mode == "middle_out" and len(states) > 1:
        # Decode both endpoints first, then walk inward using decoder-visible
        # references on each side. Entries are stored in dependency order so
        # the generic decoder never needs the original tensors.
        for layer_idx in (0, len(states) - 1):
            payload = paper_turboquant_compress(states[layer_idx], anchor_bits)
            reconstructed[layer_idx] = paper_turboquant_decompress(payload)
            result.entries.append(PredictiveEntry(layer_idx, None, payload))

        midpoint = (len(states) - 1) // 2
        decode_order = list(range(1, midpoint + 1))
        decode_order += list(range(len(states) - 2, midpoint, -1))
        for layer_idx in decode_order:
            reference_layer = (
                layer_idx - 1 if layer_idx <= midpoint else layer_idx + 1
            )
            prediction = reconstructed[reference_layer]
            residual = states[layer_idx] - prediction
            layer_bits = (
                delta_bits[layer_idx] if isinstance(delta_bits, Sequence)
                else delta_bits
            )
            payload = paper_turboquant_compress(residual, layer_bits)
            reconstructed[layer_idx] = (
                prediction + paper_turboquant_decompress(payload)
            )
            result.entries.append(
                PredictiveEntry(layer_idx, reference_layer, payload)
            )
        return result

    current_anchor = 0
    layers_since_anchor = 0

    for layer_idx, state in enumerate(states):
        if layer_idx == 0:
            is_anchor = True
        else:
            similarity = compute_statistical_snr(state, states[current_anchor])
            layers_since_anchor += 1
            is_anchor = (
                similarity < config.similarity_threshold
                or layers_since_anchor >= config.max_keyframe_span
            )

        if is_anchor:
            payload = paper_turboquant_compress(state, anchor_bits)
            restored = paper_turboquant_decompress(payload)
            result.entries.append(PredictiveEntry(layer_idx, None, payload))
            reconstructed[layer_idx] = restored
            current_anchor = layer_idx
            layers_since_anchor = 0
        else:
            # Closed-loop prediction: encode the error from the decoder-visible
            # reference, not from the unavailable full-precision reference.
            reference_layer = (
                layer_idx - 1
                if config.prediction_mode == "adjacent"
                else current_anchor
            )
            prediction = reconstructed[reference_layer]
            residual = state - prediction
            layer_bits = (
                delta_bits[layer_idx] if isinstance(delta_bits, Sequence)
                else delta_bits
            )
            payload = paper_turboquant_compress(residual, layer_bits)
            restored = prediction + paper_turboquant_decompress(payload)
            result.entries.append(PredictiveEntry(layer_idx, reference_layer, payload))
            reconstructed[layer_idx] = restored

    return result


def compress_adjacent_shared_rotation(
    states: List[torch.Tensor], anchor_bits: int, delta_bits: int,
    rotation_seed: int = 42,
) -> PredictiveStack:
    """Encode a closed-loop adjacent chain with one batched forward rotation."""
    if not states:
        return PredictiveStack()
    shapes = {tuple(state.shape) for state in states}
    devices = {state.device for state in states}
    if len(shapes) != 1 or len(devices) != 1:
        raise ValueError("shared-rotation encoding requires equal shapes and devices")
    shape = tuple(states[0].shape)
    dim = shape[-1]
    vectors = torch.stack(
        [state.float().reshape(-1, dim) for state in states]
    )
    rotation = get_rotation_matrix(
        dim, seed=rotation_seed, device=str(states[0].device)
    )
    rotated_states = vectors @ rotation.T
    result = PredictiveStack(num_layers=len(states))
    reconstructed_rotated = None

    for layer_idx, rotated in enumerate(rotated_states):
        bits = anchor_bits if layer_idx == 0 else delta_bits
        residual_rotated = (
            rotated if layer_idx == 0 else rotated - reconstructed_rotated
        )
        norms = residual_rotated.norm(dim=-1, keepdim=True)
        centroids = normal_lloyd_max(bits).to(rotated.device) / dim**0.5
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        unit_rotated = residual_rotated / norms.clamp_min(1e-12)
        indices = torch.bucketize(
            unit_rotated.contiguous(), boundaries
        ).to(torch.uint8)
        quantized_rotated = centroids[indices.long()] * norms
        reconstructed_rotated = (
            quantized_rotated if layer_idx == 0
            else reconstructed_rotated + quantized_rotated
        )
        payload = PaperTurboQuantCompressed(
            indices=indices,
            norms=norms.half(),
            original_shape=shape,
            original_dtype=states[layer_idx].dtype,
            mse_bits=bits,
            rotation_seed=rotation_seed,
        )
        result.entries.append(PredictiveEntry(
            layer_idx, None if layer_idx == 0 else layer_idx - 1, payload
        ))
    return result


def compress_kv_cache_v2(
    keys: List[torch.Tensor], values: List[torch.Tensor],
    config: GIHKCCV2Config | None = None,
) -> GIHKCCV2Cache:
    config = config or GIHKCCV2Config()
    if len(keys) != len(values):
        raise ValueError("keys and values must contain the same number of layers")
    original_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in keys + values
    )
    return GIHKCCV2Cache(
        keys=compress_predictive_stack(
            keys, config.key_anchor_bits, config.key_delta_bits, config
        ),
        values=compress_predictive_stack(
            values, config.value_anchor_bits, config.value_delta_bits, config
        ),
        config=config,
        original_bytes=original_bytes,
    )


def decompress_predictive_stack(stack: PredictiveStack) -> List[torch.Tensor]:
    restored = [None] * stack.num_layers
    for entry in stack.entries:
        decoded = paper_turboquant_decompress(entry.payload)
        if entry.reference_layer is not None:
            decoded = restored[entry.reference_layer] + decoded
        restored[entry.layer_idx] = decoded
    return restored


def decompress_kv_cache_v2(cache: GIHKCCV2Cache):
    return decompress_predictive_stack(cache.keys), decompress_predictive_stack(cache.values)
