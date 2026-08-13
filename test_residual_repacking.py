import torch

from residual_repacking import (
    allocate_mixed_bits, allocate_three_band_bits, compress_adjacent_repacked,
    compress_blockwise_repacked, compress_message_block, fit_repacking_basis,
    quantize_repacked,
)


def test_repacking_basis_is_orthogonal_and_budgeted():
    samples = torch.randn(64, 12)
    gram = torch.randn(12, 12)
    gram = gram.T @ gram
    for method in ("identity", "random", "pca", "reader"):
        basis = fit_repacking_basis(samples, method, gram)
        assert torch.allclose(basis.matrix.T @ basis.matrix, torch.eye(12), atol=1e-5)
        bits = allocate_mixed_bits(basis.importance, 2.0)
        assert bits.float().mean().item() == 2.0


def test_repacking_round_trip_shape_and_finite_values():
    samples = torch.randn(32, 12)
    basis = fit_repacking_basis(samples[:16], "pca")
    bits = allocate_mixed_bits(basis.importance)
    restored, size = quantize_repacked(samples[16:], basis, bits)
    assert restored.shape == samples[16:].shape
    assert torch.isfinite(restored).all()
    assert size > 0


def test_closed_loop_repacking_returns_reusable_bases():
    states = [torch.randn(16, 12) for _ in range(4)]
    readers = [torch.eye(12) for _ in states]
    stack, restored = compress_adjacent_repacked(
        states, 2, readers, method="reader_closedloop", calibration_tokens=8
    )
    reused, restored_reused = compress_adjacent_repacked(
        states, 2, readers, method="reader_closedloop", bases=stack.bases
    )
    assert len(restored) == len(states)
    assert all(torch.isfinite(item).all() for item in restored)
    assert all(torch.equal(a, b) for a, b in zip(restored, restored_reused))
    assert reused.compressed_bytes == stack.compressed_bytes


def test_blockwise_repacking_shares_bases_and_resets():
    states = [torch.randn(16, 12) for _ in range(6)]
    readers = [torch.eye(12) for _ in states]
    stack, restored = compress_blockwise_repacked(
        states, 2, readers, block_size=3, method="reader",
        calibration_tokens=8,
    )
    assert stack.bases[0] is None and stack.bases[3] is None
    assert stack.bases[1] is stack.bases[2]
    assert stack.bases[4] is stack.bases[5]
    assert len(restored) == len(states)
    reused, restored_reused = compress_blockwise_repacked(
        states, 2, readers, block_size=3, method="reader", bases=stack.bases
    )
    assert reused.compressed_bytes == stack.compressed_bytes
    assert all(torch.equal(a, b) for a, b in zip(restored, restored_reused))


def test_three_band_allocation_matches_budget_and_order():
    importance = torch.arange(100, dtype=torch.float32)
    bits = allocate_three_band_bits(importance, 2.0, 0.10)
    assert bits.float().mean().item() == 2.0
    assert torch.all(bits[-10:] == 8)
    assert set(bits.tolist()) == {0, 3, 8}


def test_message_block_reconstructs_every_layer_input():
    anchor = torch.randn(8, 12)
    messages = [torch.randn(8, 12) for _ in range(6)]
    stack, residuals = compress_message_block(
        anchor, messages, 2, "prefix", writes_per_layer=1
    )
    assert len(residuals) == 6
    assert stack.compressed_bytes > 0
    assert all(torch.isfinite(item).all() for item in residuals)
