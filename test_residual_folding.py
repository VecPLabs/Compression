import pytest
import torch

from residual_folding import (
    compress_folded_adjacent, decompress_folded_adjacent,
    fit_fold, fold, fold_spec_from_dict, fold_spec_to_dict, unfold,
)


@pytest.mark.parametrize(
    "mode", ["adjacent_haar", "random_lifting", "correlation_lifting"]
)
def test_fold_round_trip(mode):
    values = torch.randn(12, 16)
    spec = fit_fold(values[:6], mode)
    coarse, detail = fold(values[6:], spec)
    restored = unfold(coarse, detail, spec)
    assert torch.allclose(restored, values[6:], atol=1e-5)


def test_folded_predictive_stack_round_trip_has_real_payload():
    states = [torch.randn(8, 16) for _ in range(3)]
    specs = [fit_fold(states[i] - states[i - 1], "adjacent_haar") for i in (1, 2)]
    stack = compress_folded_adjacent(states, specs, [(4, 4), (4, 4)])
    restored = decompress_folded_adjacent(stack)
    assert len(restored) == len(states)
    assert stack.compressed_bytes > 0


def test_fold_spec_serialization_round_trip():
    values = torch.randn(12, 16)
    spec = fit_fold(values, "correlation_lifting")
    restored = fold_spec_from_dict(fold_spec_to_dict(spec))
    assert torch.equal(restored.permutation, spec.permutation)
    assert torch.allclose(restored.alpha, spec.alpha)
    assert restored.update == spec.update
