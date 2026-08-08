"""End-to-end GPT-NeoX generation using packed blockwise residual attention."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from blockwise_attention import PackedResidualStack
from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


class PackedNeoXController:
    def __init__(self, model, packed: PackedResidualStack, bits=2, block_size=256):
        self.model = model
        self.packed = packed
        self.bits = bits
        self.block_size = block_size
        self.config = GIHKCCV2Config(
            similarity_threshold=0.0,
            max_keyframe_span=64,
            prediction_mode="adjacent",
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

        layer = self.model.gpt_neox.layers[layer_idx]
        dtype = hidden_states.dtype
        for start in range(0, self.packed.tokens, self.block_size):
            end = min(start + self.block_size, self.packed.tokens)
            residual = self.packed.decode_layer_block(layer_idx, start, end)
            normalized = layer.input_layernorm(residual.unsqueeze(0).to(dtype=dtype))
            history_qkv = module.query_key_value(normalized).view(
                1, end - start, heads, 3 * module.head_size
            ).transpose(1, 2)
            history_query, key, value = history_qkv.chunk(3, dim=-1)
            positions = torch.arange(
                start, end, device=hidden_states.device
            ).unsqueeze(0)
            block_cos, block_sin = self.model.gpt_neox.rotary_emb(
                normalized, position_ids=positions
            )
            _, key = apply_rotary_pos_emb(
                history_query, key, block_cos, block_sin
            )
            accumulate(key, value)
        accumulate(current_key, current_value)
        output = (accumulator / running_sum).to(dtype)
        output = output.transpose(1, 2).reshape(1, 1, -1).contiguous()
        return module.dense(output), None

    def append_token(self, residuals):
        stack = compress_predictive_stack(
            residuals, 8, self.bits, self.config
        )
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
