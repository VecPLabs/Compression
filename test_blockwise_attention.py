import os
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).parent / ".triton-cache"))

import torch

from blockwise_attention import pack_indices, unpack_indices


def test_bitpacked_indices_roundtrip_and_ranges():
    torch.manual_seed(13)
    for bits in (2, 3, 4):
        original = torch.randint(0, 1 << bits, (137,), dtype=torch.uint8)
        packed = pack_indices(original, bits)
        restored = unpack_indices(packed, bits, 0, original.numel())
        partial = unpack_indices(packed, bits, 17, 83)
        assert torch.equal(restored, original)
        assert torch.equal(partial, original[17:100])
        assert packed.numel() == (original.numel() * bits + 7) // 8


def test_packed_residual_stream_appends_token_payloads():
    from blockwise_attention import PackedResidualStack
    from gihkcc_v2 import (
        GIHKCCV2Config,
        compress_predictive_stack,
        decompress_predictive_stack,
    )

    torch.manual_seed(14)
    prefix = [torch.randn(3, 16), torch.randn(3, 16)]
    token = [torch.randn(1, 16), torch.randn(1, 16)]
    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=8,
        prediction_mode="adjacent",
    )
    packed = PackedResidualStack(
        compress_predictive_stack(prefix, 8, 2, config)
    )
    packed.enable_fused_chain_storage(capacity_tokens=4)
    delta_storage = packed.fused_delta_indices.data_ptr()
    packed.append(PackedResidualStack(
        compress_predictive_stack(token, 8, 2, config)
    ))
    assert packed.tokens == 4
    assert packed.fused_delta_indices.data_ptr() == delta_storage
    assert packed.capacity_tokens == 4
    assert packed.logical_bytes == packed.resident_bytes
    restored = packed.decode_layer_block(1, 0, 4)
    full = [torch.cat([prefix[i], token[i]]) for i in range(2)]
    expected = decompress_predictive_stack(
        compress_predictive_stack(full, 8, 2, config)
    )[1]
    assert torch.allclose(
        restored.float(), expected.float(), atol=2e-3, rtol=2e-3
    )


def test_fused_cuda_chain_matches_reference_for_24_layers():
    if not torch.cuda.is_available():
        return
    from blockwise_attention import PackedResidualStack
    from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack

    torch.manual_seed(15)
    residuals = [torch.randn(4, 16, device="cuda") for _ in range(24)]
    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=64,
        prediction_mode="adjacent",
    )
    packed = PackedResidualStack(
        compress_predictive_stack(residuals, 8, 2, config)
    )
    expected = packed.decode_layer_block(23, 0, 4)
    packed.enable_fused_chain_storage()
    actual = packed.decode_layer_block(23, 0, 4)
    assert torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)


def test_packed_checkpoint_roundtrip():
    from blockwise_attention import PackedResidualStack
    from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack

    torch.manual_seed(16)
    residuals = [torch.randn(3, 16), torch.randn(3, 16)]
    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=8,
        prediction_mode="adjacent",
    )
    packed = PackedResidualStack(
        compress_predictive_stack(residuals, 8, 2, config)
    )
    restored = PackedResidualStack.from_checkpoint(packed.checkpoint())
    assert restored.resident_bytes == packed.resident_bytes
    assert torch.equal(
        restored.decode_layer_block(1, 0, 3),
        packed.decode_layer_block(1, 0, 3),
    )


def test_shared_rotation_adjacent_encoder_matches_reference():
    from gihkcc_v2 import (
        GIHKCCV2Config,
        compress_adjacent_shared_rotation,
        compress_predictive_stack,
        decompress_predictive_stack,
    )

    torch.manual_seed(17)
    states = [torch.randn(2, 16) for _ in range(4)]
    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=64,
        prediction_mode="adjacent",
    )
    reference = decompress_predictive_stack(
        compress_predictive_stack(states, 8, 2, config)
    )
    optimized = decompress_predictive_stack(
        compress_adjacent_shared_rotation(states, 8, 2)
    )
    for expected, actual in zip(reference, optimized):
        assert torch.allclose(actual.float(), expected.float(), atol=3e-3, rtol=3e-3)


def test_fused_normalized_projection_matches_layer_path():
    from blockwise_generation import (
        build_fused_normalized_projection,
        fused_normalized_projection,
    )
    from turboquant import get_rotation_matrix

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_size = 4
            self.query_key_value = torch.nn.Linear(16, 48)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(16)
            self.attention = Attention()

    torch.manual_seed(18)
    layer = Layer()
    pre_rotation = torch.randn(5, 16)
    rotation = get_rotation_matrix(16, seed=42, device="cpu")
    expected = layer.attention.query_key_value(
        layer.input_layernorm(pre_rotation @ rotation)
    )
    actual = fused_normalized_projection(
        pre_rotation, build_fused_normalized_projection(layer)
    )
    assert torch.allclose(actual.float(), expected.float(), atol=2e-5, rtol=2e-5)
    selected = torch.tensor([
        index
        for head in range(4)
        for index in range(head * 12 + 4, head * 12 + 12)
    ])
    actual_kv = fused_normalized_projection(
        pre_rotation, build_fused_normalized_projection(layer, kv_only=True)
    )
    assert torch.allclose(
        actual_kv.float(), expected[:, selected].float(), atol=2e-5, rtol=2e-5
    )


def test_single_block_sdpa_matches_explicit_attention():
    import torch.nn.functional as F

    torch.manual_seed(19)
    query = torch.randn(1, 4, 1, 16)
    key = torch.randn(1, 4, 33, 16)
    value = torch.randn(1, 4, 33, 16)
    scale = 16**-0.5
    expected = torch.softmax(
        query @ key.transpose(2, 3) * scale, dim=-1
    ) @ value
    actual = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, scale=scale
    )
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_projected_kv_stream_preserves_order_and_int8_accuracy():
    from quantized_kv_generation import PackedKVStream

    torch.manual_seed(20)
    stream = PackedKVStream(bits=8)
    expected = []
    for _ in range(3):
        key = torch.randn(4, 16)
        value = torch.randn(4, 16)
        stream.append(key, value)
        expected.append(torch.stack([key, value]))
    restored = stream.decode(0, 3)
    expected = torch.stack(expected)
    assert stream.tokens == 3
    assert restored.shape == (3, 2, 4, 16)
    assert torch.allclose(restored, expected, atol=2e-2, rtol=2e-2)


def test_kivi_page_preserves_order_and_expected_precision():
    from quantized_kv_generation import PagedKIVIStream

    torch.manual_seed(21)
    stream = PagedKIVIStream(key_bits=8, value_bits=4, page_size=4)
    expected = []
    for _ in range(6):
        key = torch.randn(3, 8)
        value = torch.randn(3, 8)
        stream.append(key, value)
        expected.append(torch.stack([key, value]))
    restored = stream.decode(0, 6)
    expected = torch.stack(expected)
    assert stream.tokens == 6
    assert stream.page_count == 1
    assert len(stream.pending_keys) == 2
    assert torch.allclose(restored[:, 0], expected[:, 0], atol=2e-2, rtol=2e-2)
    assert torch.allclose(restored[:, 1], expected[:, 1], atol=3e-1, rtol=2e-1)


def test_direct_packed_page_attention_matches_decoded_attention():
    if not torch.cuda.is_available():
        return
    from quantized_kv_generation import PagedKIVIStream
    from triton_kernels import paged_kv_attention

    torch.manual_seed(22)
    stream = PagedKIVIStream(key_bits=6, value_bits=4, page_size=4)
    for _ in range(4):
        stream.append(torch.randn(4, 16, device="cuda"),
                      torch.randn(4, 16, device="cuda"))
    query = torch.randn(4, 16, device="cuda")
    maximum, denominator, accumulator = paged_kv_attention(
        query, stream.key_packed, stream.key_min, stream.key_scale,
        stream.value_packed, stream.value_scale, 4, 16**-0.5,
        4, 16, 4, 6, 4,
    )
    actual = (accumulator / denominator.unsqueeze(-1)).squeeze(0)
    decoded = stream.decode(0, 4)
    keys = decoded[:, 0].transpose(0, 1)
    values = decoded[:, 1].transpose(0, 1)
    weights = torch.softmax(
        torch.sum(query[:, None] * keys, dim=-1) * 16**-0.5, dim=-1
    )
    expected = torch.sum(weights.unsqueeze(-1) * values, dim=1)
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)
