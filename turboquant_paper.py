"""Paper-faithful reference implementation of TurboQuant (arXiv:2504.19874).

This favors clarity and measurable correctness over production kernel speed.
Random matrices and Lloyd-Max codebooks are global parameters and therefore
are not counted in the per-cache payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import torch

from turboquant import get_rotation_matrix


@lru_cache(maxsize=None)
def normal_lloyd_max(bits: int, iterations: int = 200) -> torch.Tensor:
    """Return MSE-optimal centroids for a standard normal distribution."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be between 1 and 8")
    count = 1 << bits
    normal = torch.distributions.Normal(0.0, 1.0)
    probabilities = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    centroids = normal.icdf(probabilities).double()
    root_two_pi = math.sqrt(2.0 * math.pi)

    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        lower = torch.cat([torch.tensor([-torch.inf]), boundaries])
        upper = torch.cat([boundaries, torch.tensor([torch.inf])])
        cdf_lower = normal.cdf(lower)
        cdf_upper = normal.cdf(upper)
        pdf_lower = torch.exp(-lower.square() / 2) / root_two_pi
        pdf_upper = torch.exp(-upper.square() / 2) / root_two_pi
        pdf_lower[0] = 0
        pdf_upper[-1] = 0
        updated = (pdf_lower - pdf_upper) / (cdf_upper - cdf_lower).clamp_min(1e-15)
        if torch.max(torch.abs(updated - centroids)) < 1e-12:
            centroids = updated
            break
        centroids = updated
    return centroids.float()


@lru_cache(maxsize=None)
def qjl_matrix(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dim, dim, generator=generator)


@dataclass
class PaperTurboQuantCompressed:
    indices: torch.Tensor
    norms: torch.Tensor
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    mse_bits: int
    rotation_seed: int
    qjl_signs: torch.Tensor | None = None
    residual_norms: torch.Tensor | None = None
    qjl_seed: int = 1042

    @property
    def vector_count(self) -> int:
        return self.indices.numel() // self.original_shape[-1]

    @property
    def compressed_bytes(self) -> int:
        dim = self.original_shape[-1]
        bits = self.indices.numel() * self.mse_bits
        if self.qjl_signs is not None:
            bits += self.vector_count * dim
        # FP16 input norms, plus FP16 residual norms for the QJL path.
        metadata = self.vector_count * 2
        if self.residual_norms is not None:
            metadata += self.vector_count * 2
        return math.ceil(bits / 8) + metadata + 24

    @property
    def original_bytes(self) -> int:
        return math.prod(self.original_shape) * torch.tensor([], dtype=self.original_dtype).element_size()


def _mse_encode_unit(unit: torch.Tensor, bits: int, seed: int):
    dim = unit.shape[-1]
    rotation = get_rotation_matrix(dim, seed=seed, device=str(unit.device))
    rotated = unit @ rotation.T
    centroids = normal_lloyd_max(bits).to(unit.device) / math.sqrt(dim)
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    indices = torch.bucketize(rotated.contiguous(), boundaries).to(torch.uint8)
    restored_rotated = centroids[indices.long()]
    return indices, restored_rotated @ rotation


def paper_turboquant_compress(
    tensor: torch.Tensor,
    bits: int = 4,
    *,
    inner_product: bool = False,
    rotation_seed: int = 42,
    qjl_seed: int = 1042,
) -> PaperTurboQuantCompressed:
    """Compress vectors along the last dimension using Algorithms 1 or 2."""
    if inner_product and bits < 2:
        raise ValueError("inner-product TurboQuant needs at least 2 total bits")
    vectors = tensor.float().reshape(-1, tensor.shape[-1])
    norms = vectors.norm(dim=-1, keepdim=True)
    unit = vectors / norms.clamp_min(1e-12)
    mse_bits = bits - 1 if inner_product else bits
    indices, mse_unit = _mse_encode_unit(unit, mse_bits, rotation_seed)

    qjl_signs = residual_norms = None
    if inner_product:
        residual = unit - mse_unit
        residual_norms = residual.norm(dim=-1, keepdim=True)
        residual_unit = residual / residual_norms.clamp_min(1e-12)
        sketch = qjl_matrix(tensor.shape[-1], qjl_seed).to(vectors.device)
        qjl_signs = torch.where(residual_unit @ sketch.T >= 0, 1, -1).to(torch.int8)

    return PaperTurboQuantCompressed(
        indices=indices,
        norms=norms.half(),
        original_shape=tuple(tensor.shape),
        original_dtype=tensor.dtype,
        mse_bits=mse_bits,
        rotation_seed=rotation_seed,
        qjl_signs=qjl_signs,
        residual_norms=residual_norms.half() if residual_norms is not None else None,
        qjl_seed=qjl_seed,
    )


def paper_turboquant_decompress(compressed: PaperTurboQuantCompressed) -> torch.Tensor:
    dim = compressed.original_shape[-1]
    centroids = normal_lloyd_max(compressed.mse_bits).to(compressed.indices.device) / math.sqrt(dim)
    rotation = get_rotation_matrix(dim, seed=compressed.rotation_seed,
                                   device=str(compressed.indices.device))
    mse_unit = centroids[compressed.indices.long()] @ rotation
    restored_unit = mse_unit
    if compressed.qjl_signs is not None:
        sketch = qjl_matrix(dim, compressed.qjl_seed).to(compressed.indices.device)
        correction = (
            math.sqrt(math.pi / 2.0) / dim
            * (compressed.qjl_signs.float() @ sketch)
            * compressed.residual_norms.float()
        )
        restored_unit = restored_unit + correction
    restored = restored_unit * compressed.norms.float()
    return restored.reshape(compressed.original_shape).to(compressed.original_dtype)


def paper_turboquant_compress_list(
    tensors: List[torch.Tensor], bits: int, *, inner_product: bool = False
):
    encoded = [
        paper_turboquant_compress(tensor, bits, inner_product=inner_product)
        for tensor in tensors
    ]
    return encoded, sum(item.compressed_bytes for item in encoded)


def paper_turboquant_decompress_list(encoded: List[PaperTurboQuantCompressed]):
    return [paper_turboquant_decompress(item) for item in encoded]

