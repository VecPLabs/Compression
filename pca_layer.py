"""
Per-Layer PCA — Feature Decorrelation Within Layers

Unlike NVIDIA's KVTC which concatenates across layers and runs PCA on
the combined vector, this module runs PCA *independently per layer* on
the feature dimensions. This is orthogonal to GIHKCC:

  - GIHKCC exploits inter-layer redundancy (depth axis)
  - Per-layer PCA exploits intra-layer feature correlation (width axis)

The PCA basis can be computed on calibration data (like KVTC) or
estimated on-the-fly from the current cache. We support both modes.

VecP Labs LLC | vecplabs.com | Patent Pending
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch

logger = logging.getLogger("gihkcc.pca")


@dataclass
class PCAConfig:
    """Configuration for per-layer PCA."""

    # Number of principal components to retain. If None, use variance_threshold.
    n_components: Optional[int] = None

    # Fraction of total variance to retain. Only used if n_components is None.
    # 0.95 means keep enough components to explain 95% of variance.
    variance_threshold: float = 0.95

    # Minimum components to keep (safety floor).
    min_components: int = 8

    # Whether to compute PCA per-head or on concatenated heads.
    # per_head=True: PCA on [seq_len, head_dim] independently per head
    # per_head=False: PCA on [seq_len, num_heads * head_dim] (more aggressive)
    per_head: bool = True


@dataclass
class PCABasis:
    """Stored PCA basis for a single layer (or head)."""
    components: torch.Tensor   # [n_components, feature_dim] — the V^T matrix
    mean: torch.Tensor         # [feature_dim] — mean vector
    explained_variance: torch.Tensor  # [n_components]
    n_components: int
    feature_dim: int
    variance_explained_ratio: float  # Total fraction of variance retained


@dataclass
class PCACompressedLayer:
    """A single layer's KV entry in PCA space."""
    # Projected coefficients: [num_heads, seq_len, n_components] or [seq_len, n_components]
    coefficients: torch.Tensor

    # PCA basis (needed for reconstruction)
    basis: PCABasis

    # Per-head bases (when per_head=True, stores one basis per head)
    all_bases: Optional[List[PCABasis]] = None

    # Original metadata
    original_shape: Tuple[int, ...] = ()
    original_dtype: torch.dtype = torch.float16
    per_head: bool = True

    @property
    def compressed_bytes(self) -> int:
        coeff_bytes = self.coefficients.nelement() * self.coefficients.element_size()
        # Basis is shared overhead (amortized across tokens)
        basis_bytes = (self.basis.components.nelement() * 4 +
                       self.basis.mean.nelement() * 4 + 32)
        return coeff_bytes + basis_bytes

    @property
    def original_bytes(self) -> int:
        n = 1
        for s in self.original_shape:
            n *= s
        return n * 2  # float16

    @property
    def ratio(self) -> float:
        cb = self.compressed_bytes
        return self.original_bytes / cb if cb > 0 else 1.0

    @property
    def dimensionality_reduction(self) -> float:
        """How much the feature dimension was reduced."""
        return self.basis.feature_dim / self.basis.n_components


def compute_pca_basis(
    data: torch.Tensor,
    config: PCAConfig,
) -> PCABasis:
    """
    Compute PCA basis from data matrix.

    Args:
        data: [n_samples, feature_dim] — each row is one observation.
        config: PCA configuration.

    Returns:
        PCABasis
    """
    data_f32 = data.float()
    n_samples, feature_dim = data_f32.shape

    # Center
    mean = data_f32.mean(dim=0)
    centered = data_f32 - mean

    # SVD (economy mode — only compute as many as we need)
    # For large matrices, randomized SVD would be faster
    max_components = min(n_samples, feature_dim)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    # Explained variance
    explained_var = (S ** 2) / (n_samples - 1)
    total_var = explained_var.sum().item()

    # Determine number of components
    if config.n_components is not None:
        n_comp = min(config.n_components, max_components)
    else:
        # Keep enough for variance_threshold
        cumvar = explained_var.cumsum(0) / total_var
        n_comp = (cumvar < config.variance_threshold).sum().item() + 1
        n_comp = min(n_comp, max_components)

    n_comp = max(n_comp, config.min_components)
    n_comp = min(n_comp, max_components)

    var_ratio = explained_var[:n_comp].sum().item() / total_var if total_var > 0 else 1.0

    return PCABasis(
        components=Vt[:n_comp],  # [n_comp, feature_dim]
        mean=mean,
        explained_variance=explained_var[:n_comp],
        n_components=n_comp,
        feature_dim=feature_dim,
        variance_explained_ratio=var_ratio,
    )


def pca_project(
    data: torch.Tensor,
    basis: PCABasis,
) -> torch.Tensor:
    """
    Project data into PCA space.

    Args:
        data: [n_samples, feature_dim]
        basis: PCA basis

    Returns:
        [n_samples, n_components] — projected coefficients
    """
    centered = data.float() - basis.mean
    return centered @ basis.components.T  # [n, d] @ [d, k] → [n, k]


def pca_reconstruct(
    coefficients: torch.Tensor,
    basis: PCABasis,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Reconstruct data from PCA coefficients.

    Args:
        coefficients: [n_samples, n_components]
        basis: PCA basis

    Returns:
        [n_samples, feature_dim] — reconstructed data
    """
    return (coefficients.float() @ basis.components + basis.mean).to(dtype)


# ---------------------------------------------------------------------------
# Per-layer compression
# ---------------------------------------------------------------------------

def pca_compress_layer(
    kv_layer: torch.Tensor,
    config: Optional[PCAConfig] = None,
) -> PCACompressedLayer:
    """
    Compress a single layer's KV entry via PCA.

    Args:
        kv_layer: [num_heads, seq_len, head_dim] — one layer's keys or values.
        config: PCA configuration.

    Returns:
        PCACompressedLayer
    """
    if config is None:
        config = PCAConfig()

    original_shape = tuple(kv_layer.shape)
    original_dtype = kv_layer.dtype

    if config.per_head:
        # PCA independently per head
        num_heads, seq_len, head_dim = kv_layer.shape
        all_coeffs = []
        # Use first head's basis as representative (they share structure)
        # Actually, compute per-head for accuracy
        bases = []
        for h in range(num_heads):
            head_data = kv_layer[h]  # [seq_len, head_dim]
            basis = compute_pca_basis(head_data, config)
            coeffs = pca_project(head_data, basis)  # [seq_len, n_comp]
            all_coeffs.append(coeffs)
            bases.append(basis)

        # For simplicity, use the first head's basis as the stored one
        # (in production, you'd store per-head bases or a shared one)
        # Here we store coefficients with the max n_components across heads
        max_comp = max(b.n_components for b in bases)

        # Pad coefficients to uniform size
        padded = []
        for coeffs, basis in zip(all_coeffs, bases):
            if coeffs.shape[1] < max_comp:
                pad = torch.zeros(seq_len, max_comp - coeffs.shape[1],
                                  dtype=coeffs.dtype, device=coeffs.device)
                coeffs = torch.cat([coeffs, pad], dim=1)
            padded.append(coeffs)

        stacked = torch.stack(padded)  # [num_heads, seq_len, max_comp]

        return PCACompressedLayer(
            coefficients=stacked.to(original_dtype),
            basis=bases[0],  # Representative (for stats)
            all_bases=bases,  # All per-head bases for reconstruction
            original_shape=original_shape,
            original_dtype=original_dtype,
            per_head=True,
        )
    else:
        # Concatenate heads and PCA on full hidden dim
        num_heads, seq_len, head_dim = kv_layer.shape
        flat = kv_layer.transpose(0, 1).reshape(seq_len, num_heads * head_dim)
        # [seq_len, hidden_dim]

        basis = compute_pca_basis(flat, config)
        coeffs = pca_project(flat, basis)  # [seq_len, n_comp]

        return PCACompressedLayer(
            coefficients=coeffs.to(original_dtype),
            basis=basis,
            original_shape=original_shape,
            original_dtype=original_dtype,
            per_head=False,
        )


def pca_decompress_layer(
    compressed: PCACompressedLayer,
) -> torch.Tensor:
    """Reconstruct layer KV from PCA compressed representation."""
    if compressed.per_head:
        num_heads, seq_len, _ = compressed.original_shape
        head_dim = compressed.original_shape[2]

        heads = []
        for h in range(num_heads):
            # Use per-head basis if available, else fall back to shared basis
            if compressed.all_bases and h < len(compressed.all_bases):
                basis = compressed.all_bases[h]
            else:
                basis = compressed.basis
            n_comp = basis.n_components
            coeffs = compressed.coefficients[h, :, :n_comp]  # [seq_len, n_comp]
            recon = pca_reconstruct(coeffs, basis, dtype=compressed.original_dtype)
            heads.append(recon)

        return torch.stack(heads)  # [num_heads, seq_len, head_dim]
    else:
        num_heads, seq_len, head_dim = compressed.original_shape
        coeffs = compressed.coefficients  # [seq_len, n_comp]
        flat = pca_reconstruct(coeffs, compressed.basis, dtype=compressed.original_dtype)
        # [seq_len, hidden_dim] → [seq_len, num_heads, head_dim] → [num_heads, seq_len, head_dim]
        return flat.reshape(seq_len, num_heads, head_dim).transpose(0, 1)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def pca_compress_layers(
    kv_layers: List[torch.Tensor],
    config: Optional[PCAConfig] = None,
) -> Tuple[List[PCACompressedLayer], Dict[str, Any]]:
    """Compress a list of layer KV entries."""
    if config is None:
        config = PCAConfig()

    results = []
    total_orig = 0
    total_comp = 0
    total_dim_reduction = 0.0

    for layer in kv_layers:
        c = pca_compress_layer(layer, config)
        results.append(c)
        total_orig += c.original_bytes
        total_comp += c.compressed_bytes
        total_dim_reduction += c.dimensionality_reduction

    n = len(kv_layers) if kv_layers else 1
    stats = {
        "num_layers": len(kv_layers),
        "original_bytes": total_orig,
        "compressed_bytes": total_comp,
        "ratio": round(total_orig / total_comp, 2) if total_comp > 0 else 0,
        "mean_dim_reduction": round(total_dim_reduction / n, 2),
        "variance_threshold": config.variance_threshold,
    }
    return results, stats


def pca_decompress_layers(
    compressed: List[PCACompressedLayer],
) -> List[torch.Tensor]:
    """Decompress a list of PCA-compressed layers."""
    return [pca_decompress_layer(c) for c in compressed]
