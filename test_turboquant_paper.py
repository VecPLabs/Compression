import torch

from turboquant_paper import (
    normal_lloyd_max,
    paper_turboquant_compress,
    paper_turboquant_decompress,
)


def test_lloyd_max_codebooks_are_symmetric():
    for bits in (1, 2, 3, 4):
        centroids = normal_lloyd_max(bits)
        assert torch.all(centroids[1:] > centroids[:-1])
        assert torch.allclose(centroids, -centroids.flip(0), atol=1e-5)


def test_paper_turboquant_roundtrip_shape_and_budget():
    source = torch.randn(4, 32, 64, dtype=torch.float16)
    compressed = paper_turboquant_compress(source, bits=4)
    restored = paper_turboquant_decompress(compressed)
    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    assert compressed.original_bytes / compressed.compressed_bytes > 3.5


def test_qjl_path_reconstructs_finite_values():
    source = torch.randn(2, 16, 32, dtype=torch.float16)
    compressed = paper_turboquant_compress(source, bits=4, inner_product=True)
    restored = paper_turboquant_decompress(compressed)
    assert compressed.qjl_signs is not None
    assert torch.isfinite(restored).all()

