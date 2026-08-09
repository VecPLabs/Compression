"""Autoregressive GPT-NeoX attention with a projected quantized K/V cache."""

from __future__ import annotations

from contextlib import contextmanager
import math
from types import MethodType

import torch
import torch.nn.functional as F

from blockwise_attention import pack_indices, unpack_indices
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


class PackedKVStream:
    """Interleaved K/V vectors packed as [K0,V0,K1,V1,...]."""

    def __init__(self, bits: int):
        self.bits = bits
        self.packed_indices = None
        self.scales = None
        self.heads = None
        self.head_size = None

    @property
    def tokens(self):
        return (
            0 if self.scales is None else self.scales.shape[0]
        )

    @property
    def resident_bytes(self):
        if self.packed_indices is None:
            return 0
        return (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def append(self, key: torch.Tensor, value: torch.Tensor):
        if self.heads is None:
            self.heads = key.shape[-2]
            self.head_size = key.shape[-1]
        key_heads = key.reshape(self.heads, self.head_size)
        value_heads = value.reshape(self.heads, self.head_size)
        vectors = torch.stack([key_heads, value_heads])
        qmax = (1 << (self.bits - 1)) - 1
        scales = vectors.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
        codes = torch.round(vectors.float() / scales).clamp(-qmax, qmax)
        codes = (codes + qmax).to(torch.uint8)
        incoming = pack_indices(codes, self.bits)
        self.packed_indices = (
            incoming if self.packed_indices is None
            else torch.cat([self.packed_indices, incoming])
        )
        self.scales = (
            scales.half().unsqueeze(0) if self.scales is None
            else torch.cat([self.scales, scales.half().unsqueeze(0)], dim=0)
        )

    def decode(self, start: int, end: int):
        values_per_token = 2 * self.heads * self.head_size
        codes = unpack_indices(
            self.packed_indices, self.bits, start * values_per_token,
            (end - start) * values_per_token,
        ).reshape(end - start, 2, self.heads, self.head_size)
        qmax = (1 << (self.bits - 1)) - 1
        return (codes.float() - qmax) * self.scales[start:end].float()


class KIVIPage:
    """Pagewise per-channel keys and per-token/per-head values."""

    def __init__(self, keys, values, key_bits, value_bits):
        self.tokens, self.heads, self.head_size = keys.shape
        self.key_bits = key_bits
        self.value_bits = value_bits

        key_qmax = (1 << key_bits) - 1
        key_min = keys.float().amin(dim=0)
        key_max = keys.float().amax(dim=0)
        key_scale = ((key_max - key_min) / key_qmax).clamp_min(1e-12)
        key_codes = torch.round(
            (keys.float() - key_min.unsqueeze(0)) / key_scale.unsqueeze(0)
        ).clamp(0, key_qmax).to(torch.uint8)
        self.key_packed = pack_indices(key_codes, key_bits)
        self.key_min = key_min.half()
        self.key_scale = key_scale.half()

        value_qmax = (1 << (value_bits - 1)) - 1
        value_scale = (
            values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            / value_qmax
        )
        value_codes = torch.round(values.float() / value_scale).clamp(
            -value_qmax, value_qmax
        )
        self.value_packed = pack_indices(
            (value_codes + value_qmax).to(torch.uint8), value_bits
        )
        self.value_scale = value_scale.half()

    @property
    def resident_bytes(self):
        tensors = (
            self.key_packed, self.key_min, self.key_scale,
            self.value_packed, self.value_scale,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def decode(self):
        shape = (self.tokens, self.heads, self.head_size)
        count = self.tokens * self.heads * self.head_size
        key_codes = unpack_indices(
            self.key_packed, self.key_bits, 0, count
        ).reshape(shape).float()
        keys = (
            key_codes * self.key_scale.float().unsqueeze(0)
            + self.key_min.float().unsqueeze(0)
        )
        value_qmax = (1 << (self.value_bits - 1)) - 1
        value_codes = unpack_indices(
            self.value_packed, self.value_bits, 0, count
        ).reshape(shape).float()
        values = (
            value_codes - value_qmax
        ) * self.value_scale.float()
        return torch.stack([keys, values], dim=1)


class PagedKIVIStream:
    def __init__(self, key_bits=8, value_bits=4, page_size=32):
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.page_size = page_size
        self.page_count = 0
        self.key_packed = None
        self.key_min = None
        self.key_scale = None
        self.value_packed = None
        self.value_scale = None
        self.heads = None
        self.head_size = None
        self.pending_keys = []
        self.pending_values = []

    @property
    def tokens(self):
        return self.page_count * self.page_size + len(self.pending_keys)

    @property
    def resident_bytes(self):
        page_tensors = (
            self.key_packed, self.key_min, self.key_scale,
            self.value_packed, self.value_scale,
        )
        pages = sum(
            tensor.numel() * tensor.element_size()
            for tensor in page_tensors if tensor is not None
        )
        pending = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.pending_keys + self.pending_values
        )
        return pages + pending

    def append(self, key, value):
        if self.heads is None:
            self.heads, self.head_size = key.shape
        self.pending_keys.append(key.detach())
        self.pending_values.append(value.detach())
        if len(self.pending_keys) == self.page_size:
            page = KIVIPage(
                torch.stack(self.pending_keys), torch.stack(self.pending_values),
                self.key_bits, self.value_bits,
            )
            self.key_packed = (
                page.key_packed if self.key_packed is None
                else torch.cat([self.key_packed, page.key_packed])
            )
            self.value_packed = (
                page.value_packed if self.value_packed is None
                else torch.cat([self.value_packed, page.value_packed])
            )
            for name in ("key_min", "key_scale", "value_scale"):
                incoming = getattr(page, name).unsqueeze(0)
                current = getattr(self, name)
                setattr(self, name, incoming if current is None else torch.cat([
                    current, incoming
                ], dim=0))
            self.page_count += 1
            self.pending_keys.clear()
            self.pending_values.clear()

    def decode(self, start, end):
        chunks = []
        if self.page_count:
            shape = (
                self.page_count, self.page_size, self.heads, self.head_size
            )
            count = math.prod(shape)
            key_codes = unpack_indices(
                self.key_packed, self.key_bits, 0, count
            ).reshape(shape).float()
            keys = (
                key_codes * self.key_scale.float().unsqueeze(1)
                + self.key_min.float().unsqueeze(1)
            )
            value_qmax = (1 << (self.value_bits - 1)) - 1
            value_codes = unpack_indices(
                self.value_packed, self.value_bits, 0, count
            ).reshape(shape).float()
            values = (
                value_codes - value_qmax
            ) * self.value_scale.float()
            chunks.append(torch.stack([keys, values], dim=2).reshape(
                self.page_count * self.page_size, 2,
                self.heads, self.head_size,
            ))
        if self.pending_keys:
            chunks.append(torch.stack([
                torch.stack(self.pending_keys), torch.stack(self.pending_values)
            ], dim=1))
        return torch.cat(chunks, dim=0)[start:end]


class QuantizedKVController:
    def __init__(
        self, model, bits=4, hot_window=0, block_size=256,
        scheme="symmetric", key_bits=8, value_bits=4, page_size=32,
        direct_packed_attention=False,
    ):
        self.model = model
        self.bits = bits
        self.hot_window = hot_window
        self.block_size = block_size
        layers = len(model.gpt_neox.layers)
        self.scheme = scheme
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.page_size = page_size
        self.direct_packed_attention = direct_packed_attention
        if scheme == "kivi":
            self.cold = [
                PagedKIVIStream(key_bits, value_bits, page_size)
                for _ in range(layers)
            ]
        elif scheme == "symmetric":
            self.cold = [PackedKVStream(bits) for _ in range(layers)]
        else:
            raise ValueError("scheme must be 'symmetric' or 'kivi'")
        self.hot_keys = [[] for _ in range(layers)]
        self.hot_values = [[] for _ in range(layers)]

    @property
    def resident_bytes(self):
        return self.cold_resident_bytes + self.hot_resident_bytes

    @property
    def cold_resident_bytes(self):
        return sum(stream.resident_bytes for stream in self.cold)

    @property
    def hot_resident_bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.hot_keys + self.hot_values for tensor in layer
        )

    @property
    def cold_tokens(self):
        return self.cold[0].tokens

    @property
    def hot_tokens(self):
        return len(self.hot_keys[0])

    def _append(self, layer_idx, key, value):
        keys = self.hot_keys[layer_idx]
        values = self.hot_values[layer_idx]
        keys.append(key.detach().reshape(key.shape[-3], key.shape[-1]))
        values.append(value.detach().reshape(value.shape[-3], value.shape[-1]))
        if len(keys) > self.hot_window:
            self.cold[layer_idx].append(keys.pop(0), values.pop(0))

    @torch.inference_mode()
    def attention(self, module, hidden_states, position_embeddings):
        if hidden_states.shape[:2] != (1, 1):
            raise ValueError("quantized K/V prototype supports batch=1, query=1")
        layer_idx = module.layer_idx
        heads = self.model.config.num_attention_heads
        head_size = module.head_size
        qkv = module.query_key_value(hidden_states).view(
            1, 1, heads, 3 * head_size
        ).transpose(1, 2)
        query, current_key, current_value = qkv.chunk(3, dim=-1)
        cos, sin = position_embeddings
        query, current_key = apply_rotary_pos_emb(
            query, current_key, cos, sin
        )

        stream = self.cold[layer_idx]
        if (
            self.direct_packed_attention
            and isinstance(stream, PagedKIVIStream)
            and stream.page_count > 0
        ):
            from triton_kernels import paged_kv_attention

            cold_count = stream.page_count * stream.page_size
            cold_max, cold_sum, cold_accumulator = paged_kv_attention(
                query.reshape(heads, head_size).contiguous(),
                stream.key_packed, stream.key_min, stream.key_scale,
                stream.value_packed, stream.value_scale,
                cold_count, module.scaling, heads, head_size,
                stream.page_size, stream.key_bits, stream.value_bits,
            )
            combined_cold_max = cold_max.amax(dim=0)
            page_weights = torch.exp(
                cold_max - combined_cold_max.unsqueeze(0)
            )
            cold_sum = torch.sum(cold_sum * page_weights, dim=0)
            cold_accumulator = torch.sum(
                cold_accumulator * page_weights.unsqueeze(-1), dim=0
            )
            cold_max = combined_cold_max
            exact_keys = stream.pending_keys + self.hot_keys[layer_idx]
            exact_values = stream.pending_values + self.hot_values[layer_idx]
            exact_keys = exact_keys + [current_key.reshape(heads, head_size)]
            exact_values = exact_values + [current_value.reshape(heads, head_size)]
            key = torch.stack(exact_keys).float().transpose(0, 1)
            value = torch.stack(exact_values).float().transpose(0, 1)
            scores = torch.sum(
                query.reshape(heads, 1, head_size).float() * key, dim=-1
            ) * module.scaling
            exact_max = scores.amax(dim=-1)
            exact_weights = torch.exp(scores - exact_max.unsqueeze(-1))
            exact_sum = exact_weights.sum(dim=-1)
            exact_accumulator = torch.sum(
                exact_weights.unsqueeze(-1) * value, dim=1
            )
            maximum = torch.maximum(cold_max, exact_max)
            cold_weight = torch.exp(cold_max - maximum)
            exact_weight = torch.exp(exact_max - maximum)
            denominator = cold_sum * cold_weight + exact_sum * exact_weight
            accumulator = (
                cold_accumulator * cold_weight.unsqueeze(-1)
                + exact_accumulator * exact_weight.unsqueeze(-1)
            )
            output = (accumulator / denominator.unsqueeze(-1)).to(
                hidden_states.dtype
            ).reshape(1, heads, 1, head_size)
            self._append(layer_idx, current_key, current_value)
            output = output.transpose(1, 2).reshape(1, 1, -1).contiguous()
            return module.dense(output), None

        keys = []
        values = []
        for start in range(0, stream.tokens, self.block_size):
            end = min(start + self.block_size, stream.tokens)
            decoded = stream.decode(start, end).to(hidden_states.dtype)
            keys.append(decoded[:, 0])
            values.append(decoded[:, 1])
        if self.hot_keys[layer_idx]:
            keys.append(torch.stack(self.hot_keys[layer_idx]).reshape(-1, heads, head_size))
            values.append(
                torch.stack(self.hot_values[layer_idx]).reshape(-1, heads, head_size)
            )
        keys.append(current_key[0].transpose(0, 1))
        values.append(current_value[0].transpose(0, 1))
        key = torch.cat(keys, dim=0).transpose(0, 1).unsqueeze(0)
        value = torch.cat(values, dim=0).transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False,
            scale=module.scaling,
        )
        self._append(layer_idx, current_key, current_value)
        output = output.transpose(1, 2).reshape(1, 1, -1).contiguous()
        return module.dense(output), None


@contextmanager
def patch_neox_quantized_kv(controller: QuantizedKVController):
    originals = []
    for attention in (layer.attention for layer in controller.model.gpt_neox.layers):
        originals.append((attention, attention.forward))

        def forward(
            module, hidden_states, attention_mask=None, layer_past=None,
            position_embeddings=None, **kwargs,
        ):
            return controller.attention(module, hidden_states, position_embeddings)

        attention.forward = MethodType(forward, attention)
    try:
        yield
    finally:
        for attention, original in originals:
            attention.forward = original
