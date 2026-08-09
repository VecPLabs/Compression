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

    def decode_pre_rotation(self, start: int, end: int) -> torch.Tensor:
        count = end - start
        centroids = normal_lloyd_max(self.bits).to(
            self.packed_indices.device
        ) / math.sqrt(self.dim)
        if self.packed_indices.is_cuda:
            try:
                from triton_kernels import packed_dequantize
                return packed_dequantize(
                    self.packed_indices, self.norms, centroids,
                    start, count, self.dim, self.bits,
                )
            except (ImportError, RuntimeError):
                pass
        flat_start = start * self.dim
        indices = unpack_indices(
            self.packed_indices, self.bits, flat_start, count * self.dim
        ).reshape(count, self.dim)
        return centroids[indices.long()] * self.norms[start:end].float()

    def decode_vectors(self, start: int, end: int) -> torch.Tensor:
        pre_rotation = self.decode_pre_rotation(start, end)
        rotation = get_rotation_matrix(
            self.dim, seed=self.rotation_seed,
            device=str(self.packed_indices.device),
        ).to(pre_rotation.dtype)
        return (pre_rotation @ rotation).to(self.norms.dtype)


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
        self.fused_delta_indices = None
        self.fused_delta_norms = None
        self.fused_anchor_indices = None
        self.fused_anchor_norms = None
        self.capacity_tokens = None
        self._vector_count = None

    def checkpoint(self):
        """Return a tensor-only representation suitable for ``torch.save``."""
        return {
            "format_version": 1,
            "num_layers": self.num_layers,
            "entries": [
                {
                    "layer_idx": entry.layer_idx,
                    "reference_layer": entry.reference_layer,
                    "packed_indices": entry.payload.packed_indices.clone(),
                    "norms": entry.payload.norms.clone(),
                    "vector_count": entry.payload.vector_count,
                    "dim": entry.payload.dim,
                    "bits": entry.payload.bits,
                    "rotation_seed": entry.payload.rotation_seed,
                }
                for entry in self.entries.values()
            ],
        }

    @classmethod
    def from_checkpoint(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("unsupported packed residual checkpoint version")
        packed = cls.__new__(cls)
        packed.num_layers = state["num_layers"]
        packed.entries = {}
        for item in state["entries"]:
            payload = PackedPayload(
                item["packed_indices"], item["norms"], item["vector_count"],
                item["dim"], item["bits"], item["rotation_seed"],
            )
            packed.entries[item["layer_idx"]] = PackedEntry(
                item["layer_idx"], item["reference_layer"], payload
            )
        packed.fused_delta_indices = None
        packed.fused_delta_norms = None
        packed.fused_anchor_indices = None
        packed.fused_anchor_norms = None
        packed.capacity_tokens = None
        packed._vector_count = None
        return packed

    def enable_fused_chain_storage(self, capacity_tokens: int | None = None):
        """Store adjacent deltas as layer-major matrices for fused decoding."""
        delta_entries = [self.entries[index] for index in range(1, self.num_layers)]
        if any(entry.reference_layer != entry.layer_idx - 1 for entry in delta_entries):
            raise ValueError("fused chain storage requires adjacent prediction")
        if len({entry.payload.bits for entry in delta_entries}) > 1:
            raise ValueError("fused chain storage requires uniform delta bit widths")
        delta_indices = torch.stack(
            [entry.payload.packed_indices for entry in delta_entries]
        ).contiguous()
        delta_norms = torch.stack(
            [entry.payload.norms.reshape(-1) for entry in delta_entries]
        ).contiguous()
        anchor = self.entries[0].payload
        self._vector_count = anchor.vector_count
        capacity = capacity_tokens or self._vector_count
        if capacity < self._vector_count:
            raise ValueError("capacity cannot be smaller than the existing stream")
        self.capacity_tokens = capacity
        anchor_bytes = anchor.dim * anchor.bits // 8
        delta = delta_entries[0].payload
        delta_bytes = delta.dim * delta.bits // 8
        self.fused_anchor_indices = torch.empty(
            capacity * anchor_bytes, dtype=anchor.packed_indices.dtype,
            device=anchor.packed_indices.device,
        )
        self.fused_anchor_norms = torch.empty(
            capacity, dtype=anchor.norms.dtype, device=anchor.norms.device,
        )
        self.fused_delta_indices = torch.empty(
            self.num_layers - 1, capacity * delta_bytes,
            dtype=delta_indices.dtype, device=delta_indices.device,
        )
        self.fused_delta_norms = torch.empty(
            self.num_layers - 1, capacity,
            dtype=delta_norms.dtype, device=delta_norms.device,
        )
        self.fused_anchor_indices[:anchor.packed_indices.numel()].copy_(
            anchor.packed_indices
        )
        self.fused_anchor_norms[:self._vector_count].copy_(anchor.norms.reshape(-1))
        self.fused_delta_indices[:, :delta_indices.shape[1]].copy_(delta_indices)
        self.fused_delta_norms[:, :self._vector_count].copy_(delta_norms)
        self._bind_fused_views()

    def _bind_fused_views(self):
        anchor = self.entries[0].payload
        anchor_bytes = anchor.dim * anchor.bits // 8
        anchor.packed_indices = self.fused_anchor_indices[
            :self._vector_count * anchor_bytes
        ]
        anchor.norms = self.fused_anchor_norms[:self._vector_count].reshape(-1, 1)
        anchor.vector_count = self._vector_count
        for row, index in enumerate(range(1, self.num_layers)):
            payload = self.entries[index].payload
            bytes_per_token = payload.dim * payload.bits // 8
            payload.packed_indices = self.fused_delta_indices[
                row, :self._vector_count * bytes_per_token
            ]
            payload.norms = self.fused_delta_norms[
                row, :self._vector_count
            ].reshape(-1, 1)
            payload.vector_count = self._vector_count

    def _grow_fused_storage(self, capacity_tokens: int):
        logical = self.checkpoint()
        restored = PackedResidualStack.from_checkpoint(logical)
        restored.enable_fused_chain_storage(capacity_tokens)
        self.__dict__.update(restored.__dict__)

    @property
    def resident_bytes(self):
        if self.fused_delta_indices is not None:
            tensors = (
                self.fused_anchor_indices, self.fused_anchor_norms,
                self.fused_delta_indices, self.fused_delta_norms,
            )
            return sum(t.numel() * t.element_size() for t in tensors)
        return sum(entry.payload.resident_bytes for entry in self.entries.values())

    @property
    def logical_bytes(self):
        return sum(entry.payload.resident_bytes for entry in self.entries.values())

    @property
    def tokens(self):
        if self._vector_count is not None:
            return self._vector_count
        return next(iter(self.entries.values())).payload.vector_count

    def decode_layer_block(self, layer_idx: int, start: int, end: int):
        combined = self.decode_layer_pre_rotation_block(layer_idx, start, end)
        if combined is not None:
            exemplar = self.entries[layer_idx].payload
            rotation = get_rotation_matrix(
                exemplar.dim, seed=exemplar.rotation_seed,
                device=str(exemplar.packed_indices.device),
            ).to(combined.dtype)
            return (combined @ rotation).to(exemplar.norms.dtype)

        restored = {}
        for index in range(layer_idx + 1):
            entry = self.entries[index]
            decoded = entry.payload.decode_vectors(start, end)
            if entry.reference_layer is not None:
                decoded = restored[entry.reference_layer] + decoded
            restored[index] = decoded
        return restored[layer_idx]

    def decode_layer_pre_rotation_block(
        self, layer_idx: int, start: int, end: int,
    ):
        """Decode a chain before its shared inverse rotation, when possible."""
        ancestry = []
        index = layer_idx
        while index is not None:
            entry = self.entries[index]
            ancestry.append(entry)
            index = entry.reference_layer
        seeds = {entry.payload.rotation_seed for entry in ancestry}
        dims = {entry.payload.dim for entry in ancestry}
        if len(seeds) == 1 and len(dims) == 1:
            exemplar = ancestry[0].payload
            combined = None
            if self.fused_delta_indices is not None and exemplar.packed_indices.is_cuda:
                try:
                    from triton_kernels import packed_chain_dequantize
                    anchor = self.entries[0].payload
                    delta = self.entries[1].payload
                    anchor_centroids = normal_lloyd_max(anchor.bits).to(
                        anchor.packed_indices.device
                    ) / math.sqrt(anchor.dim)
                    delta_centroids = normal_lloyd_max(delta.bits).to(
                        delta.packed_indices.device
                    ) / math.sqrt(delta.dim)
                    combined = packed_chain_dequantize(
                        anchor.packed_indices, anchor.norms, anchor_centroids,
                        self.fused_delta_indices, self.fused_delta_norms,
                        delta_centroids, start, end - start, anchor.dim,
                        anchor.bits, delta.bits, layer_idx,
                    )
                except (ImportError, RuntimeError):
                    pass
            if combined is None:
                for entry in reversed(ancestry):
                    decoded = entry.payload.decode_pre_rotation(start, end)
                    combined = decoded if combined is None else combined + decoded
            return combined
        return None

    def append(self, other: "PackedResidualStack"):
        """Append independently encoded token vectors to this packed stream."""
        if self.num_layers != other.num_layers:
            raise ValueError("layer counts differ")
        if self.fused_delta_indices is not None:
            anchor = self.entries[0].payload
            incoming_anchor = other.entries[0].payload
            if (
                anchor.bits != incoming_anchor.bits
                or anchor.dim != incoming_anchor.dim
                or anchor.rotation_seed != incoming_anchor.rotation_seed
                or (anchor.vector_count * anchor.dim * anchor.bits) % 8 != 0
            ):
                raise ValueError("anchor payload cannot be byte-aligned for append")
            incoming_count = incoming_anchor.vector_count
            new_count = self._vector_count + incoming_count
            if new_count > self.capacity_tokens:
                self._grow_fused_storage(max(new_count, self.capacity_tokens * 2))
                anchor = self.entries[0].payload
            incoming_indices = torch.stack([
                other.entries[index].payload.packed_indices
                for index in range(1, self.num_layers)
            ])
            incoming_norms = torch.stack([
                other.entries[index].payload.norms.reshape(-1)
                for index in range(1, self.num_layers)
            ])
            anchor_bytes = anchor.dim * anchor.bits // 8
            delta = self.entries[1].payload
            delta_bytes = delta.dim * delta.bits // 8
            anchor_start = self._vector_count * anchor_bytes
            delta_start = self._vector_count * delta_bytes
            self.fused_anchor_indices[
                anchor_start:new_count * anchor_bytes
            ].copy_(incoming_anchor.packed_indices)
            self.fused_anchor_norms[self._vector_count:new_count].copy_(
                incoming_anchor.norms.reshape(-1)
            )
            self.fused_delta_indices[
                :, delta_start:new_count * delta_bytes
            ].copy_(incoming_indices)
            self.fused_delta_norms[:, self._vector_count:new_count].copy_(
                incoming_norms
            )
            self._vector_count = new_count
            self._bind_fused_views()
            return
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
