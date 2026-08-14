import pytest
import torch

from residual_folding import fit_fold, fold, unfold


@pytest.mark.parametrize(
    "mode", ["adjacent_haar", "random_lifting", "correlation_lifting"]
)
def test_fold_round_trip(mode):
    values = torch.randn(12, 16)
    spec = fit_fold(values[:6], mode)
    coarse, detail = fold(values[6:], spec)
    restored = unfold(coarse, detail, spec)
    assert torch.allclose(restored, values[6:], atol=1e-5)
