import torch

from gihkcc_v2 import GIHKCCV2Config, compress_kv_cache_v2, decompress_kv_cache_v2


def test_v2_roundtrip_and_real_payload_ratio():
    torch.manual_seed(7)
    base = torch.randn(2, 32, 64, dtype=torch.float16)
    keys = [base + torch.randn_like(base) * 0.02 for _ in range(8)]
    values = [base + torch.randn_like(base) * 0.03 for _ in range(8)]
    cache = compress_kv_cache_v2(keys, values)
    restored_keys, restored_values = decompress_kv_cache_v2(cache)
    assert len(restored_keys) == len(keys)
    assert len(restored_values) == len(values)
    assert cache.ratio > 2.0
    assert all(torch.isfinite(tensor).all() for tensor in restored_keys + restored_values)


def test_closed_loop_entries_reference_decoded_anchors():
    torch.manual_seed(8)
    base = torch.randn(1, 16, 32, dtype=torch.float16)
    states = [base + layer * 0.01 for layer in range(4)]
    cache = compress_kv_cache_v2(
        states, states,
        GIHKCCV2Config(similarity_threshold=0.8, max_keyframe_span=8),
    )
    assert cache.keys.entries[0].is_anchor
    assert all(entry.reference_layer == 0 for entry in cache.keys.entries[1:])


def test_predictive_stack_accepts_per_layer_bits():
    from gihkcc_v2 import compress_predictive_stack

    torch.manual_seed(9)
    base = torch.randn(1, 8, 16, dtype=torch.float16)
    states = [base + layer * 0.02 for layer in range(4)]
    config = GIHKCCV2Config(similarity_threshold=0.0, max_keyframe_span=8)
    stack = compress_predictive_stack(states, 8, [8, 2, 3, 4], config)
    assert [entry.payload.mse_bits for entry in stack.entries] == [8, 2, 3, 4]


def test_adjacent_closed_loop_entries_form_a_chain():
    from gihkcc_v2 import compress_predictive_stack, decompress_predictive_stack

    torch.manual_seed(10)
    base = torch.randn(1, 8, 16, dtype=torch.float16)
    states = [base + layer * 0.02 for layer in range(5)]
    config = GIHKCCV2Config(
        similarity_threshold=0.0,
        max_keyframe_span=8,
        prediction_mode="adjacent",
    )
    stack = compress_predictive_stack(states, 8, 3, config)
    restored = decompress_predictive_stack(stack)
    assert [entry.reference_layer for entry in stack.entries] == [None, 0, 1, 2, 3]
    assert all(torch.isfinite(tensor).all() for tensor in restored)


def test_middle_out_entries_decode_from_both_endpoints():
    from gihkcc_v2 import compress_predictive_stack, decompress_predictive_stack

    torch.manual_seed(11)
    base = torch.randn(1, 8, 16, dtype=torch.float16)
    states = [base + layer * 0.02 for layer in range(6)]
    config = GIHKCCV2Config(prediction_mode="middle_out")
    stack = compress_predictive_stack(states, 8, 2, config)
    restored = decompress_predictive_stack(stack)
    references = {
        entry.layer_idx: entry.reference_layer for entry in stack.entries
    }
    assert references == {0: None, 5: None, 1: 0, 2: 1, 4: 5, 3: 4}
    assert all(torch.isfinite(tensor).all() for tensor in restored)
