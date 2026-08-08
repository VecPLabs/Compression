"""Packed residual payloads and blockwise GPT-NeoX attention prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gihkcc_v2 import PredictiveStack
from turboquant import get_rotation_matrix
from turboquant_paper import PaperTurboQuantCompressed, normal_lloyd_max
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Densely pack unsigned quantizer indices into a byte tensor."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be between 1 and 8")
    flat = indices.reshape(-1).to(torch.int64)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=indices.device)
    positions = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    bit_positions = positions * bits
    byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
    shifts = bit_positions.remainder(8)
    byte_count = math.ceil(flat.numel() * bits / 8)
    packed = torch.zeros(byte_count + 1, dtype=torch.int64, device=flat.device)
    packed.scatter_add_(0, byte_indices, (flat << shifts).bitwise_and(255))
    overflow = shifts + bits > 8
    if overflow.any():
        packed.scatter_add_(
            0, byte_indices[overflow] + 1,
            flat[overflow] >> (8 - shifts[overflow]),
        )
    return packed[:byte_count].to(torch.uint8)


def unpack_indices(
    packed: torch.Tensor, bits: int, start: int, count: int
) -> torch.Tensor:
    """Decode an arbitrary flat range without unpacking the whole payload."""
    positions = torch.arange(start, start + count, device=packed.device)
    bit_positions = positions * bits
    byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
    shifts = bit_positions.remainder(8)
    padded = torch.cat(
        [packed, torch.zeros(1, dtype=torch.uint8, device=packed.device)]
    ).to(torch.int64)
    values = padded[byte_indices] >> shifts
    crosses = shifts + bits > 8
    if crosses.any():
        values[crosses] |= padded[byte_indices[crosses] + 1] << (
            8 - shifts[crosses]
        )
    return values.bitwise_and((1 << bits) - 1).to(torch.uint8)


@dataclass
class PackedPayload:
    packed_indices: torch.Tensor
    norms: torch.Tensor
    vector_count: int
    dim: int
    bits: int
    rotation_seed: int

    @classmethod
    def from_reference(cls, payload: PaperTurboQuantCompressed):
        if payload.qjl_signs is not None:
            raise ValueError("packed prototype currently supports MSE payloads only")
        return cls(
            pack_indices(payload.indices, payload.mse_bits),
            payload.norms.reshape(-1, 1),
            payload.vector_count,
            payload.original_shape[-1],
            payload.mse_bits,
            payload.rotation_seed,
        )

    @property
    def resident_bytes(self):
        return (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.norms.numel() * self.norms.element_size()
        )

    def decode_vectors(self, start: int, end: int) -> torch.Tensor:
        count = end - start
        flat_start = start * self.dim
        indices = unpack_indices(
            self.packed_indices, self.bits, flat_start, count * self.dim
        ).reshape(count, self.dim)
        centroids = normal_lloyd_max(self.bits).to(indices.device) / math.sqrt(self.dim)
        rotation = get_rotation_matrix(
            self.dim, seed=self.rotation_seed, device=str(indices.device)
        )
        unit = centroids[indices.long()] @ rotation
        return unit * self.norms[start:end].float()


@dataclass
class PackedEntry:
    layer_idx: int
    reference_layer: int | None
    payload: PackedPayload


class PackedResidualStack:
    def __init__(self, stack: PredictiveStack):
        self.num_layers = stack.num_layers
        self.entries = {
            entry.layer_idx: PackedEntry(
                entry.layer_idx,
                entry.reference_layer,
                PackedPayload.from_reference(entry.payload),
            )
            for entry in stack.entries
        }

    @property
    def resident_bytes(self):
        return sum(entry.payload.resident_bytes for entry in self.entries.values())

    @property
    def tokens(self):
        return next(iter(self.entries.values())).payload.vector_count

    def decode_layer_block(self, layer_idx: int, start: int, end: int):
        restored = {}
        for index in range(layer_idx + 1):
            entry = self.entries[index]
            decoded = entry.payload.decode_vectors(start, end)
            if entry.reference_layer is not None:
                decoded = restored[entry.reference_layer] + decoded
            restored[index] = decoded
        return restored[layer_idx]

    def append(self, other: "PackedResidualStack"):
        """Append independently encoded token vectors to this packed stream."""
        if self.num_layers != other.num_layers:
            raise ValueError("layer counts differ")
        for layer_idx, entry in self.entries.items():
            incoming = other.entries[layer_idx]
            if entry.reference_layer != incoming.reference_layer:
                raise ValueError("predictive topology differs")
            left = entry.payload
            right = incoming.payload
            if (
                left.dim != right.dim
                or left.bits != right.bits
                or left.rotation_seed != right.rotation_seed
                or (left.vector_count * left.dim * left.bits) % 8 != 0
            ):
                raise ValueError("payloads cannot be byte-aligned for append")
            left.packed_indices = torch.cat(
                [left.packed_indices, right.packed_indices]
            )
            left.norms = torch.cat([left.norms, right.norms], dim=0)
            left.vector_count += right.vector_count


@torch.inference_mode()
def neox_blockwise_attention(
    model, packed: PackedResidualStack, layer_idx: int,
    query_residual: torch.Tensor, query_position: int, block_size: int = 64,
):
    """Compute one-token attention without materializing full historical K/V."""
    layer = model.gpt_neox.layers[layer_idx]
    device = next(layer.parameters()).device
    dtype = next(layer.parameters()).dtype
    query_hidden = layer.input_layernorm(
        query_residual.reshape(1, 1, -1).to(device=device, dtype=dtype)
    )
    heads = model.config.num_attention_heads
    qkv = layer.attention.query_key_value(query_hidden).view(
        1, 1, heads, 3 * layer.attention.head_size
    ).transpose(1, 2)
    query, _, _ = qkv.chunk(3, dim=-1)
    query_pos = torch.tensor([[query_position]], device=device)
    cos, sin = model.gpt_neox.rotary_emb(query_hidden, position_ids=query_pos)
    query, _ = apply_rotary_pos_emb(query, query, cos, sin)

    running_max = torch.full(
        (1, heads, 1, 1), -torch.inf, device=device, dtype=torch.float32
    )
    running_sum = torch.zeros_like(running_max)
    accumulator = torch.zeros(
        (1, heads, 1, layer.attention.head_size),
        device=device, dtype=torch.float32,
    )
    for start in range(0, packed.tokens, block_size):
        end = min(start + block_size, packed.tokens)
        residual = packed.decode_layer_block(layer_idx, start, end)
        hidden = layer.input_layernorm(
            residual.unsqueeze(0).to(device=device, dtype=dtype)
        )
        block_qkv = layer.attention.query_key_value(hidden).view(
            1, end - start, heads, 3 * layer.attention.head_size
        ).transpose(1, 2)
        block_query, key, value = block_qkv.chunk(3, dim=-1)
        positions = torch.arange(start, end, device=device).unsqueeze(0)
        cos, sin = model.gpt_neox.rotary_emb(hidden, position_ids=positions)
        _, key = apply_rotary_pos_emb(block_query, key, cos, sin)
        scores = (
            torch.matmul(query.float(), key.float().transpose(2, 3))
            * layer.attention.scaling
        )
        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        previous_scale = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max)
        accumulator = (
            accumulator * previous_scale
            + torch.matmul(weights, value.float())
        )
        running_sum = running_sum * previous_scale + weights.sum(
            dim=-1, keepdim=True
        )
        running_max = new_max
    return (accumulator / running_sum).to(dtype).transpose(1, 2).contiguous()


@torch.inference_mode()
def neox_materialized_attention(
    model, packed: PackedResidualStack, layer_idx: int,
    query_residual: torch.Tensor, query_position: int,
):
    """Reference attention over the same decoded packed payload."""
    layer = model.gpt_neox.layers[layer_idx]
    device = next(layer.parameters()).device
    dtype = next(layer.parameters()).dtype
    heads = model.config.num_attention_heads
    query_hidden = layer.input_layernorm(
        query_residual.reshape(1, 1, -1).to(device=device, dtype=dtype)
    )
    query_qkv = layer.attention.query_key_value(query_hidden).view(
        1, 1, heads, 3 * layer.attention.head_size
    ).transpose(1, 2)
    query, _, _ = query_qkv.chunk(3, dim=-1)
    query_pos = torch.tensor([[query_position]], device=device)
    cos, sin = model.gpt_neox.rotary_emb(query_hidden, position_ids=query_pos)
    query, _ = apply_rotary_pos_emb(query, query, cos, sin)

    residual = packed.decode_layer_block(layer_idx, 0, packed.tokens)
    hidden = layer.input_layernorm(
        residual.unsqueeze(0).to(device=device, dtype=dtype)
    )
    qkv = layer.attention.query_key_value(hidden).view(
        1, packed.tokens, heads, 3 * layer.attention.head_size
    ).transpose(1, 2)
    block_query, key, value = qkv.chunk(3, dim=-1)
    positions = torch.arange(packed.tokens, device=device).unsqueeze(0)
    cos, sin = model.gpt_neox.rotary_emb(hidden, position_ids=positions)
    _, key = apply_rotary_pos_emb(block_query, key, cos, sin)
    scores = torch.matmul(query.float(), key.float().transpose(2, 3))
    scores = scores * layer.attention.scaling
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value.float()).to(dtype).transpose(
        1, 2
    ).contiguous()
