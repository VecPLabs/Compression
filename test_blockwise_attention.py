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
    from gihkcc_v2 import GIHKCCV2Config, compress_predictive_stack

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
    packed.append(PackedResidualStack(
        compress_predictive_stack(token, 8, 2, config)
    ))
    assert packed.tokens == 4
    assert packed.decode_layer_block(1, 3, 4).shape == (1, 16)
