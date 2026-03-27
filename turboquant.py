"""
TurboQuant — Minimal Implementation for GIHKCC Integration

Implements the core of Google's TurboQuant (ICLR 2026):
  1. PolarQuant: Random orthogonal rotation → scalar quantization
  2. QJL: 1-bit Johnson-Lindenstrauss correction on quantization residual

The random rotation makes marginals approximately Gaussian (by CLT in
high dimensions), which means a simple uniform scalar quantizer per
coordinate is near-optimal. This is the key insight — no calibration,
no codebooks, no dataset-specific tuning.

VecP Labs LLC | vecplabs.com | Patent Pending
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("gihkcc.turboquant")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant compression."""

    # Target bits per value. Paper shows 3-3.5 bits is zero-loss.
    target_bits: float = 3.0

    # Enable QJL residual correction (1 extra bit for unbiased inner products).
    qjl_enabled: bool = True

    # Random seed for reproducible rotation matrices.
    # CRITICAL: must be the same for compress and decompress.
    rotation_seed: int = 42

    # Cache rotation matrices across calls (saves recomputation).
    cache_rotations: bool = True


# ---------------------------------------------------------------------------
# Rotation matrix generation
# ---------------------------------------------------------------------------

_rotation_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}


def get_rotation_matrix(
    dim: int,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate a random orthogonal matrix via QR decomposition of a
    Gaussian random matrix. This is the Pi matrix from the paper.

    Cached by (dim, seed, device) to avoid regeneration.
    """
    cache_key = (dim, seed, str(device))
    if cache_key in _rotation_cache:
        cached = _rotation_cache[cache_key]
        if cached.dtype == dtype:
            return cached

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=dtype)
    Pi, _ = torch.linalg.qr(G)

    # Move to target device
    Pi = Pi.to(device=device, dtype=dtype)
    _rotation_cache[cache_key] = Pi
    return Pi


# ---------------------------------------------------------------------------
# Scalar Quantizer — uniform symmetric
# ---------------------------------------------------------------------------

def uniform_quantize(
    x: torch.Tensor,
    bits: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-channel uniform symmetric quantization.

    For each coordinate (last dim), compute scale from the range and
    quantize to `bits` levels.

    Args:
        x: [..., dim] tensor to quantize.
        bits: Bit width per value.

    Returns:
        (codes, scales, zeros) where:
          codes: int8/int16 quantized values, same shape as x
          scales: [dim] per-coordinate scale factors
          zeros: [dim] per-coordinate zero points
    """
    n_levels = (1 << bits)
    qmin = -(n_levels // 2)
    qmax = n_levels // 2 - 1

    # Per-coordinate range (over all non-last dims)
    flat = x.reshape(-1, x.shape[-1]).float()
    vmin = flat.min(dim=0).values  # [dim]
    vmax = flat.max(dim=0).values  # [dim]

    scale = (vmax - vmin) / (qmax - qmin)
    scale = scale.clamp(min=1e-10)  # Avoid division by zero
    zero = vmin

    # Quantize
    codes = torch.clamp(
        torch.round((x.float() - zero) / scale + qmin),
        qmin, qmax,
    ).to(torch.int8 if bits <= 8 else torch.int16)

    return codes, scale, zero


def uniform_dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: int = 3,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct from uniform quantized codes."""
    n_levels = (1 << bits)
    qmin = -(n_levels // 2)
    return ((codes.float() - qmin) * scale + zero).to(dtype)


# ---------------------------------------------------------------------------
# QJL — Quantized Johnson-Lindenstrauss (1-bit residual correction)
# ---------------------------------------------------------------------------

@dataclass
class QJLCorrection:
    """1-bit sketch of quantization residual for unbiased inner products."""
    signs: torch.Tensor  # packed uint8 bitmap of sign(Pi_qjl @ residual)
    residual_norm: torch.Tensor  # scalar norm per vector for rescaling
    projection_seed: int = 0
    dim: int = 0
    num_vectors: int = 0

    @property
    def bytes(self) -> int:
        return self.signs.numel() + self.residual_norm.numel() * 4 + 16


def qjl_encode(
    residual: torch.Tensor,
    seed: int = 137,
) -> QJLCorrection:
    """
    Compute 1-bit QJL sketch of the quantization residual.

    For each vector in residual [..., dim]:
      1. Project with random Gaussian matrix
      2. Store only the signs (1 bit each)
      3. Store the residual norm for rescaling

    This allows correction of inner product bias during attention.
    """
    shape = residual.shape
    dim = shape[-1]
    flat = residual.reshape(-1, dim).float()
    n_vec = flat.shape[0]

    # Random projection (same dim → dim for maximal correction)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # Use a smaller projection for efficiency: dim → dim/4
    proj_dim = max(dim // 4, 8)
    P = torch.randn(dim, proj_dim, generator=gen, dtype=torch.float32)
    P = P / math.sqrt(proj_dim)  # Normalize
    P = P.to(flat.device)

    # Project
    projected = flat @ P  # [n_vec, proj_dim]

    # Store signs as packed bits
    sign_bits = (projected > 0).to(torch.uint8)
    # Pack 8 bits per byte
    pad = (8 - proj_dim * n_vec % 8) % 8
    sign_flat = sign_bits.flatten()
    if pad > 0:
        sign_flat = torch.cat([sign_flat, torch.zeros(pad, dtype=torch.uint8)])
    packed = torch.zeros(sign_flat.numel() // 8, dtype=torch.uint8)
    for bit in range(8):
        packed |= sign_flat[bit::8] << (7 - bit)

    # Residual norms for rescaling
    norms = flat.norm(dim=-1)  # [n_vec]

    return QJLCorrection(
        signs=packed,
        residual_norm=norms,
        projection_seed=seed,
        dim=dim,
        num_vectors=n_vec,
    )


# ---------------------------------------------------------------------------
# TurboQuant compressed representation
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantCompressed:
    """Compressed tensor via TurboQuant."""
    # Quantized codes in rotated space
    codes: torch.Tensor       # int8
    scales: torch.Tensor      # [dim] per-coordinate
    zeros: torch.Tensor       # [dim] per-coordinate

    # QJL correction (optional)
    qjl: Optional[QJLCorrection] = None

    # Metadata
    original_shape: Tuple[int, ...] = ()
    original_dtype: torch.dtype = torch.float16
    target_bits: int = 3
    rotation_seed: int = 42

    @property
    def compressed_bytes(self) -> int:
        """Estimated storage with proper bit-packing."""
        # Codes: packed at target_bits per element
        n_elements = self.codes.numel()
        code_bytes = math.ceil(n_elements * self.target_bits / 8)
        # Per-coordinate scales and zeros: stored once per feature dim
        scale_bytes = self.scales.numel() * 4  # float32
        zero_bytes = self.zeros.numel() * 4
        qjl_bytes = self.qjl.bytes if self.qjl else 0
        return code_bytes + scale_bytes + zero_bytes + qjl_bytes + 32

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


# ---------------------------------------------------------------------------
# Compress / Decompress
# ---------------------------------------------------------------------------

def turboquant_compress(
    x: torch.Tensor,
    config: Optional[TurboQuantConfig] = None,
) -> TurboQuantCompressed:
    """
    Compress a tensor using TurboQuant.

    Pipeline: random rotation → per-coordinate quantization → (optional QJL)

    Args:
        x: Tensor of any shape. Last dimension is the feature dimension
           that gets rotated. Typically [num_heads, seq_len, head_dim]
           or [seq_len, hidden_dim].
        config: TurboQuant configuration.

    Returns:
        TurboQuantCompressed
    """
    if config is None:
        config = TurboQuantConfig()

    original_shape = tuple(x.shape)
    original_dtype = x.dtype
    dim = x.shape[-1]
    bits = int(config.target_bits)

    # Step 1: Random orthogonal rotation
    Pi = get_rotation_matrix(dim, seed=config.rotation_seed,
                              device=str(x.device), dtype=torch.float32)
    x_f32 = x.float()
    # Rotate: apply Pi to last dimension
    # x_rot[..., i] = sum_j Pi[i,j] * x[..., j]
    x_rot = x_f32 @ Pi.T  # [..., dim] @ [dim, dim] → [..., dim]

    # Step 2: Per-coordinate uniform quantization
    codes, scales, zeros = uniform_quantize(x_rot, bits=bits)

    # Step 3: Optional QJL on quantization residual
    qjl = None
    if config.qjl_enabled:
        x_rot_recon = uniform_dequantize(codes, scales, zeros, bits=bits, dtype=torch.float32)
        residual = x_rot - x_rot_recon
        qjl = qjl_encode(residual, seed=config.rotation_seed + 1000)

    return TurboQuantCompressed(
        codes=codes,
        scales=scales,
        zeros=zeros,
        qjl=qjl,
        original_shape=original_shape,
        original_dtype=original_dtype,
        target_bits=bits,
        rotation_seed=config.rotation_seed,
    )


def turboquant_decompress(
    compressed: TurboQuantCompressed,
) -> torch.Tensor:
    """
    Decompress a TurboQuant-compressed tensor.

    Inverse: dequantize → inverse rotation (Pi^T since Pi is orthogonal)
    """
    bits = compressed.target_bits
    dim = compressed.original_shape[-1]

    # Step 1: Dequantize in rotated space
    x_rot = uniform_dequantize(
        compressed.codes, compressed.scales, compressed.zeros,
        bits=bits, dtype=torch.float32,
    )

    # Step 2: Inverse rotation
    Pi = get_rotation_matrix(dim, seed=compressed.rotation_seed,
                              device=str(x_rot.device), dtype=torch.float32)
    # Inverse of orthogonal matrix is its transpose
    x_recon = x_rot @ Pi  # [..., dim] @ [dim, dim] → [..., dim]

    return x_recon.to(compressed.original_dtype)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def turboquant_compress_list(
    tensors: List[torch.Tensor],
    config: Optional[TurboQuantConfig] = None,
) -> Tuple[List[TurboQuantCompressed], Dict[str, Any]]:
    """Compress a list of tensors (e.g., per-layer KV entries)."""
    if config is None:
        config = TurboQuantConfig()

    results = []
    total_orig = 0
    total_comp = 0

    for t in tensors:
        c = turboquant_compress(t, config)
        results.append(c)
        total_orig += c.original_bytes
        total_comp += c.compressed_bytes

    stats = {
        "num_tensors": len(tensors),
        "original_bytes": total_orig,
        "compressed_bytes": total_comp,
        "ratio": round(total_orig / total_comp, 2) if total_comp > 0 else 0,
        "target_bits": config.target_bits,
        "qjl_enabled": config.qjl_enabled,
    }
    return results, stats


def turboquant_decompress_list(
    compressed: List[TurboQuantCompressed],
) -> List[torch.Tensor]:
    """Decompress a list of TurboQuant-compressed tensors."""
    return [turboquant_decompress(c) for c in compressed]
