"""End-to-end GPT-NeoX generation using packed blockwise residual attention."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from blockwise_attention import PackedResidualStack
from gihkcc_v2 import compress_adjacent_shared_rotation
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from turboquant import get_rotation_matrix


def build_fused_normalized_projection(layer, rotation_seed=42, kv_only=False):
    """Fold inverse rotation, LayerNorm affine, and QKV weight together."""
    norm = layer.input_layernorm
    linear = layer.attention.query_key_value
    dim = linear.in_features
    rotation = get_rotation_matrix(
        dim, seed=rotation_seed, device=str(linear.weight.device)
    ).to(torch.float32)
    weight = linear.weight.float()
    bias = linear.bias.float() if linear.bias is not None else None
    if kv_only:
        head_size = layer.attention.head_size
        heads = weight.shape[0] // (3 * head_size)
        selected = torch.cat([
            torch.arange(
                head * 3 * head_size + head_size,
                head * 3 * head_size + 3 * head_size,
                device=weight.device,
            )
            for head in range(heads)
        ])
        weight = weight[selected]
        if bias is not None:
            bias = bias[selected]
    weight_t = weight.T
    gamma_weight = norm.weight.float().unsqueeze(1) * weight_t
    matrix = (rotation @ gamma_weight).to(linear.weight.dtype)
    mean_weight = norm.weight.float() @ weight_t
    offset = norm.bias.float() @ weight_t
    if bias is not None:
        offset = offset + bias
    mean_vector = rotation @ torch.full(
        (dim,), 1.0 / dim, device=rotation.device
    )
    epsilon = getattr(norm, "variance_epsilon", norm.eps)
    return matrix, mean_weight, offset, mean_vector, epsilon


def fused_normalized_projection(pre_rotation, parameters):
    """Project LayerNorm(pre_rotation @ R) without materializing that residual."""
    matrix, mean_weight, offset, mean_vector, epsilon = parameters
    values = pre_rotation.float()
    mean = (values @ mean_vector).unsqueeze(-1)
    variance = values.square().mean(dim=-1, keepdim=True) - mean.square()
    inverse_std = torch.rsqrt(variance.clamp_min(0) + epsilon)
    projected = values.to(matrix.dtype) @ matrix
    projected = (
        projected.float() - mean * mean_weight.unsqueeze(0)
    ) * inverse_std + offset.unsqueeze(0)
    return projected.to(matrix.dtype)


class PackedNeoXController:
    def __init__(
        self, model, packed: PackedResidualStack, bits=2, block_size=256,
        reserve_tokens=None, fused_projection=False, fused_attention=False,
    ):
        self.model = model
        self.packed = packed
        self.bits = bits
        self.block_size = block_size
        self.packed.enable_fused_chain_storage(reserve_tokens)
        self.fused_projection = fused_projection
        self.fused_attention = fused_attention
        self.fused_projection_parameters = None
        if fused_projection:
            self.fused_projection_parameters = [
                build_fused_normalized_projection(layer, kv_only=True)
                for layer in self.model.gpt_neox.layers
            ]

    @property
    def fused_projection_bytes(self):
        if self.fused_projection_parameters is None:
            return 0
        return sum(
            tensor.numel() * tensor.element_size()
            for parameters in self.fused_projection_parameters
            for tensor in parameters[:4]
        )

    @torch.inference_mode()
    def attention(self, module, hidden_states, position_embeddings):
        layer_idx = module.layer_idx
        batch, query_length, _ = hidden_states.shape
        if batch != 1 or query_length != 1:
            raise ValueError("prototype supports batch=1, one-token decoding")
        heads = self.model.config.num_attention_heads
        qkv = module.query_key_value(hidden_states).view(
            1, 1, heads, 3 * module.head_size
        ).transpose(1, 2)
        query, current_key, current_value = qkv.chunk(3, dim=-1)
        cos, sin = position_embeddings
        query, current_key = apply_rotary_pos_emb(
            query, current_key, cos, sin
        )

        layer = self.model.gpt_neox.layers[layer_idx]
        dtype = hidden_states.dtype

        def project_history(start, end):
            if self.fused_projection:
                pre_rotation = self.packed.decode_layer_pre_rotation_block(
                    layer_idx, start, end
                )
                projected = fused_normalized_projection(
                    pre_rotation, self.fused_projection_parameters[layer_idx]
                )
                history_kv = projected.unsqueeze(0).view(
                    1, end - start, heads, 2 * module.head_size
                ).transpose(1, 2)
                key, value = history_kv.chunk(2, dim=-1)
                rotary_input = hidden_states
            else:
                residual = self.packed.decode_layer_block(layer_idx, start, end)
                normalized = layer.input_layernorm(
                    residual.unsqueeze(0).to(dtype=dtype)
                )
                history_qkv = module.query_key_value(normalized).view(
                    1, end - start, heads, 3 * module.head_size
                ).transpose(1, 2)
                _, key, value = history_qkv.chunk(3, dim=-1)
                rotary_input = normalized
            positions = torch.arange(
                start, end, device=hidden_states.device
            ).unsqueeze(0)
            block_cos, block_sin = self.model.gpt_neox.rotary_emb(
                rotary_input, position_ids=positions
            )
            _, key = apply_rotary_pos_emb(key, key, block_cos, block_sin)
            return key, value

        if self.fused_attention and self.packed.tokens <= self.block_size:
            key, value = project_history(0, self.packed.tokens)
            key = torch.cat([key, current_key], dim=2)
            value = torch.cat([value, current_value], dim=2)
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False,
                scale=module.scaling,
            )
            output = output.transpose(1, 2).reshape(1, 1, -1).contiguous()
            return module.dense(output), None

        running_max = torch.full(
            (1, heads, 1, 1), -torch.inf,
            device=hidden_states.device, dtype=torch.float32,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (1, heads, 1, module.head_size),
            device=hidden_states.device, dtype=torch.float32,
        )

        def accumulate(key, value):
            nonlocal running_max, running_sum, accumulator
            scores = torch.matmul(
                query.float(), key.float().transpose(2, 3)
            ) * module.scaling
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            previous_scale = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max)
            accumulator = accumulator * previous_scale + torch.matmul(
                weights, value.float()
            )
            running_sum = running_sum * previous_scale + weights.sum(
                dim=-1, keepdim=True
            )
            running_max = new_max

        for start in range(0, self.packed.tokens, self.block_size):
            end = min(start + self.block_size, self.packed.tokens)
            key, value = project_history(start, end)
            accumulate(key, value)
        accumulate(current_key, current_value)
        output = (accumulator / running_sum).to(dtype)
        output = output.transpose(1, 2).reshape(1, 1, -1).contiguous()
        return module.dense(output), None

    def append_token(self, residuals):
        stack = compress_adjacent_shared_rotation(residuals, 8, self.bits)
        self.packed.append(PackedResidualStack(stack))


@contextmanager
def patch_neox_attention(controller: PackedNeoXController):
    originals = []
    for attention in (layer.attention for layer in controller.model.gpt_neox.layers):
        originals.append((attention, attention.forward))

        def forward(
            module, hidden_states, attention_mask=None, layer_past=None,
            position_embeddings=None, **kwargs,
        ):
            return controller.attention(
                module, hidden_states, position_embeddings
            )

        attention.forward = MethodType(forward, attention)
    try:
        yield
    finally:
        for attention, original in originals:
            attention.forward = original
