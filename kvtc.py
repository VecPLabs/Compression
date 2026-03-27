"""
KVTC — KV Cache Transform Coding

The missing middle-out compression stage. Sits between the fold hierarchy
(L1/L2/L3) and storage, applying frequency-domain transform coding to
delta tensors.

Pipeline: delta → block DCT → energy threshold → quantize sparse → RLE → bytes
Inverse:  bytes → RLE → dequantize sparse → IDCT → delta

Design rationale:
  Delta tensors from the fold stages are small perturbations on a smooth
  residual stream. In the frequency domain, this means most energy concentrates
  in low-frequency DCT coefficients. Zeroing the high-frequency tail and
  run-length encoding the resulting sparse tensor gives 10-20x on top of
  the structural compression from folding.

  This is exactly what H.264 does to motion-compensated residuals — DCT,
  quantize, zigzag scan, RLE, CABAC. We skip CABAC (diminishing returns
  for our tensor shapes) but the core is identical.

VecP Labs LLC | vecplabs.com | Patent Pending (USPTO 63/931,565)
"""

from __future__ import annotations

import math
import struct
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("gihkcc.kvtc")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KVTCConfig:
    """Configuration for the transform coding stage."""

    # --- DCT ---
    # Block size for blocked DCT. The delta tensor is reshaped into blocks
    # of this size along the last dimension. Must divide head_dim evenly.
    # Smaller blocks = faster but less spectral resolution.
    # 0 = full-length DCT (no blocking).
    block_size: int = 0

    # --- Energy thresholding ---
    # Fraction of total spectral energy to retain. 0.95 means keep enough
    # coefficients to preserve 95% of the signal energy. Higher = more
    # fidelity, less compression. Range: [0.80, 0.999].
    energy_retention: float = 0.95

    # Alternative: fixed top-k coefficients per block. If > 0, overrides
    # energy_retention. Useful for deterministic compression ratios.
    top_k_coefficients: int = 0

    # --- Quantization ---
    # Bit width for surviving coefficients after thresholding.
    coeff_quant_bits: int = 8

    # --- Entropy coding ---
    # Enable run-length encoding of zero runs in the sparse coefficient tensor.
    rle_enabled: bool = True

    # Enable Huffman coding on top of RLE. Marginal gains (~10-20%) but
    # adds decode complexity. Off by default for latency-sensitive paths.
    huffman_enabled: bool = False


# ---------------------------------------------------------------------------
# DCT / IDCT — Type-II DCT via matrix multiplication
# ---------------------------------------------------------------------------
# We use explicit DCT matrices rather than scipy.fft.dctn because:
# 1. Stays on GPU (no CPU round-trip)
# 2. Works with torch autograd if we ever need gradients
# 3. Block DCT is just a matmul on reshaped tensors

_dct_matrix_cache: Dict[Tuple[int, str], torch.Tensor] = {}


def _get_dct_matrix(n: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute the Type-II DCT matrix of size n×n.
    C[k,i] = sqrt(2/n) * cos(pi * k * (2i+1) / (2n)), with C[0,:] scaled by 1/sqrt(2).
    """
    cache_key = (n, str(device))
    if cache_key in _dct_matrix_cache:
        cached = _dct_matrix_cache[cache_key]
        if cached.dtype == dtype:
            return cached

    i = torch.arange(n, dtype=dtype, device=device)
    k = torch.arange(n, dtype=dtype, device=device)

    # Outer product: k * (2i + 1)
    C = torch.cos(math.pi * k.unsqueeze(1) * (2 * i.unsqueeze(0) + 1) / (2 * n))
    C *= math.sqrt(2.0 / n)
    C[0, :] /= math.sqrt(2.0)

    _dct_matrix_cache[cache_key] = C
    return C


def dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply Type-II DCT along a single dimension via matrix multiplication.
    """
    n = x.shape[dim]
    C = _get_dct_matrix(n, device=str(x.device), dtype=x.dtype)

    # Move target dim to last position for matmul
    x_moved = x.transpose(dim, -1) if dim != -1 else x
    result = x_moved @ C.T  # C @ x would be [k, i] @ [i] = [k], but we batch via right-mul
    # Actually: X @ C^T applies DCT: result[..., k] = sum_i X[..., i] * C[k, i]
    if dim != -1:
        result = result.transpose(dim, -1)
    return result


def idct_1d(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply Type-II inverse DCT (Type-III DCT) along a single dimension.
    Since C is orthogonal, C^{-1} = C^T.
    """
    n = X.shape[dim]
    C = _get_dct_matrix(n, device=str(X.device), dtype=X.dtype)

    X_moved = X.transpose(dim, -1) if dim != -1 else X
    # Inverse: x = C^T @ X (since C is orthogonal)
    result = X_moved @ C  # X @ C applies C^T on the right: result[..., i] = sum_k X[..., k] * C[k, i]
    if dim != -1:
        result = result.transpose(dim, -1)
    return result


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply 2D DCT (last two dimensions)."""
    return dct_1d(dct_1d(x, dim=-1), dim=-2)


def idct_2d(X: torch.Tensor) -> torch.Tensor:
    """Apply 2D inverse DCT (last two dimensions)."""
    return idct_1d(idct_1d(X, dim=-1), dim=-2)


# ---------------------------------------------------------------------------
# Energy-based coefficient thresholding
# ---------------------------------------------------------------------------

def energy_threshold(
    coeffs: torch.Tensor,
    retention: float = 0.95,
) -> Tuple[torch.Tensor, int, float]:
    """
    Zero out DCT coefficients that contribute least to signal energy,
    retaining `retention` fraction of total energy.

    Energy of a coefficient = coefficient^2 (Parseval's theorem: energy
    is preserved under orthogonal transforms).

    Args:
        coeffs: DCT coefficient tensor (any shape).
        retention: Fraction of energy to keep, in [0, 1].

    Returns:
        (thresholded_coeffs, num_nonzero, threshold_value)
    """
    flat = coeffs.flatten().float()
    energies = flat ** 2
    total_energy = energies.sum().item()

    if total_energy < 1e-12:
        # All-zero delta — nothing to compress
        return torch.zeros_like(coeffs), 0, 0.0

    # Sort energies descending, find cutoff
    sorted_energies, sorted_indices = energies.sort(descending=True)
    cumulative = sorted_energies.cumsum(0)
    target = total_energy * retention

    # Find the index where we exceed the target energy
    mask = cumulative <= target
    num_keep = mask.sum().item() + 1  # +1 because we want to include the crossing point
    num_keep = min(num_keep, len(flat))

    # Get the threshold value (smallest energy we keep)
    threshold = sorted_energies[num_keep - 1].item()
    threshold_amp = math.sqrt(max(threshold, 0))

    # Zero out below threshold
    result = coeffs.clone()
    result[coeffs.abs() < threshold_amp] = 0

    num_nonzero = (result != 0).sum().item()
    sparsity = 1.0 - (num_nonzero / result.numel())

    logger.debug(
        f"Energy threshold: retain={retention:.2%}, "
        f"kept {num_nonzero}/{result.numel()} coeffs ({sparsity:.1%} sparse), "
        f"threshold_amp={threshold_amp:.6f}"
    )

    return result, num_nonzero, threshold_amp


def topk_threshold(
    coeffs: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, int, float]:
    """
    Keep only the top-k coefficients by absolute value. Deterministic
    compression ratio: exactly k non-zeros per tensor.
    """
    flat = coeffs.flatten()
    if k >= flat.numel():
        return coeffs.clone(), flat.numel(), 0.0

    abs_flat = flat.abs()
    topk_vals, topk_idx = abs_flat.topk(k)
    threshold_amp = topk_vals[-1].item() if k > 0 else float("inf")

    result = torch.zeros_like(coeffs)
    result_flat = result.flatten()
    result_flat[topk_idx] = flat[topk_idx]
    result = result_flat.reshape(coeffs.shape)

    return result, k, threshold_amp


# ---------------------------------------------------------------------------
# Sparse coefficient quantization
# ---------------------------------------------------------------------------

@dataclass
class SparseQuantizedTensor:
    """
    Compact representation of a sparse, quantized coefficient tensor.

    Stores only the non-zero positions and their quantized values.
    """
    # Flat indices of non-zero coefficients
    indices: torch.Tensor       # int32

    # Quantized values at those indices
    values: torch.Tensor        # int8 (or int16 for higher bit depths)

    # Quantization parameters
    scale: float
    zero_point: float
    quant_bits: int

    # Original tensor shape (for reconstruction)
    original_shape: Tuple[int, ...]

    # Stats
    num_nonzero: int
    total_elements: int
    sparsity: float

    @property
    def compressed_bytes(self) -> int:
        """Estimated storage size."""
        idx_bytes = self.indices.nelement() * 4  # int32
        val_bytes = self.indices.nelement() * max(self.quant_bits // 8, 1)
        overhead = 32  # scale, zp, shape, etc.
        return idx_bytes + val_bytes + overhead

    @property
    def original_bytes(self) -> int:
        """Size of original uncompressed tensor at float16."""
        return self.total_elements * 2

    @property
    def ratio(self) -> float:
        cb = self.compressed_bytes
        return self.original_bytes / cb if cb > 0 else 1.0


def sparse_quantize(
    sparse_coeffs: torch.Tensor,
    bits: int = 8,
) -> SparseQuantizedTensor:
    """
    Quantize a sparse tensor — only encode non-zero values.
    """
    flat = sparse_coeffs.flatten()
    nonzero_mask = flat != 0
    indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
    values = flat[nonzero_mask].float()

    if values.numel() == 0:
        return SparseQuantizedTensor(
            indices=torch.empty(0, dtype=torch.int32),
            values=torch.empty(0, dtype=torch.int8),
            scale=1.0,
            zero_point=0.0,
            quant_bits=bits,
            original_shape=tuple(sparse_coeffs.shape),
            num_nonzero=0,
            total_elements=flat.numel(),
            sparsity=1.0,
        )

    vmin = values.min().item()
    vmax = values.max().item()
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    scale = (vmax - vmin) / (qmax - qmin) if vmax != vmin else 1.0
    zero_point = vmin

    quantized = torch.clamp(
        torch.round((values - zero_point) / scale + qmin),
        qmin, qmax,
    ).to(torch.int8)

    num_nz = indices.numel()
    return SparseQuantizedTensor(
        indices=indices,
        values=quantized,
        scale=scale,
        zero_point=zero_point,
        quant_bits=bits,
        original_shape=tuple(sparse_coeffs.shape),
        num_nonzero=num_nz,
        total_elements=flat.numel(),
        sparsity=1.0 - (num_nz / flat.numel()),
    )


def sparse_dequantize(sqt: SparseQuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Reconstruct sparse coefficient tensor from quantized representation.
    """
    result = torch.zeros(sqt.total_elements, dtype=dtype)

    if sqt.num_nonzero == 0:
        return result.reshape(sqt.original_shape)

    qmin = -(1 << (sqt.quant_bits - 1))
    dequantized = (sqt.values.float() - qmin) * sqt.scale + sqt.zero_point

    result[sqt.indices.long()] = dequantized.to(dtype)
    return result.reshape(sqt.original_shape)


# ---------------------------------------------------------------------------
# Run-Length Encoding for sparse quantized coefficients
# ---------------------------------------------------------------------------

@dataclass
class RLEEncoded:
    """
    Run-length encoded sparse quantized tensor.

    Encodes alternating (zero_run_length, nonzero_value) pairs in the
    zigzag-scanned coefficient order. This is the same structure H.264
    uses for DCT block coding.
    """
    # Packed pairs: [(run, value), (run, value), ...]
    runs: torch.Tensor       # int16: zero-run lengths
    values: torch.Tensor     # int8: quantized non-zero values

    # Quantization params (carried through from SparseQuantizedTensor)
    scale: float
    zero_point: float
    quant_bits: int
    original_shape: Tuple[int, ...]
    total_elements: int

    @property
    def compressed_bytes(self) -> int:
        """Storage: 2 bytes per run + 1 byte per value + overhead."""
        return self.runs.nelement() * 2 + self.values.nelement() * 1 + 32

    @property
    def original_bytes(self) -> int:
        return self.total_elements * 2  # float16 baseline

    @property
    def ratio(self) -> float:
        cb = self.compressed_bytes
        return self.original_bytes / cb if cb > 0 else 1.0


def rle_encode(sqt: SparseQuantizedTensor) -> RLEEncoded:
    """
    Run-length encode a sparse quantized tensor.

    Scans the full flat tensor, emitting (zero_run_length, value) pairs
    for each non-zero coefficient. Long zero runs between non-zeros
    compress heavily.
    """
    if sqt.num_nonzero == 0:
        return RLEEncoded(
            runs=torch.empty(0, dtype=torch.int16),
            values=torch.empty(0, dtype=torch.int8),
            scale=sqt.scale,
            zero_point=sqt.zero_point,
            quant_bits=sqt.quant_bits,
            original_shape=sqt.original_shape,
            total_elements=sqt.total_elements,
        )

    # Sort indices to scan in order
    sorted_order = sqt.indices.argsort()
    sorted_indices = sqt.indices[sorted_order]
    sorted_values = sqt.values[sorted_order]

    runs = []
    values = []
    prev_pos = 0

    for i in range(sqt.num_nonzero):
        pos = sorted_indices[i].item()
        zero_run = pos - prev_pos
        runs.append(zero_run)
        values.append(sorted_values[i].item())
        prev_pos = pos + 1

    return RLEEncoded(
        runs=torch.tensor(runs, dtype=torch.int16),
        values=torch.tensor(values, dtype=torch.int8),
        scale=sqt.scale,
        zero_point=sqt.zero_point,
        quant_bits=sqt.quant_bits,
        original_shape=sqt.original_shape,
        total_elements=sqt.total_elements,
    )


def rle_decode(rle: RLEEncoded) -> SparseQuantizedTensor:
    """Reconstruct SparseQuantizedTensor from RLE."""
    if rle.runs.numel() == 0:
        return SparseQuantizedTensor(
            indices=torch.empty(0, dtype=torch.int32),
            values=torch.empty(0, dtype=torch.int8),
            scale=rle.scale,
            zero_point=rle.zero_point,
            quant_bits=rle.quant_bits,
            original_shape=rle.original_shape,
            num_nonzero=0,
            total_elements=rle.total_elements,
            sparsity=1.0,
        )

    indices = []
    pos = 0
    for i in range(rle.runs.numel()):
        pos += rle.runs[i].item()
        indices.append(pos)
        pos += 1

    num_nz = len(indices)
    return SparseQuantizedTensor(
        indices=torch.tensor(indices, dtype=torch.int32),
        values=rle.values.clone(),
        scale=rle.scale,
        zero_point=rle.zero_point,
        quant_bits=rle.quant_bits,
        original_shape=rle.original_shape,
        num_nonzero=num_nz,
        total_elements=rle.total_elements,
        sparsity=1.0 - (num_nz / rle.total_elements) if rle.total_elements > 0 else 1.0,
    )


# ---------------------------------------------------------------------------
# Bitmap + Packed Values — optimal for moderate-to-high sparsity
# ---------------------------------------------------------------------------
# At S% sparsity, cost = ceil(N/8) bytes (bitmap) + (1-S)*N bytes (values)
#
# Sparsity  | bytes/element | ratio vs float16
# 30%       | 0.825         | 2.4x
# 50%       | 0.625         | 3.2x
# 70%       | 0.425         | 4.7x
# 90%       | 0.225         | 8.9x
#
# Always better than float16 regardless of sparsity (bitmap overhead is tiny).
# Always better than sparse index-based storage below ~85% sparsity.

@dataclass
class BitmapPackedTensor:
    """
    Bitmap + packed values encoding for quantized coefficient tensors.

    Storage:
      - bitmap: 1 bit per element (0 = zero, 1 = non-zero)
      - values: packed int8 values for all non-zero positions
      - scale + zero_point: quantization parameters
    """
    bitmap: torch.Tensor         # uint8, ceil(N/8) bytes
    packed_values: torch.Tensor  # int8, num_nonzero entries

    scale: float
    zero_point: float
    quant_bits: int

    original_shape: Tuple[int, ...]
    total_elements: int
    num_nonzero: int

    @property
    def compressed_bytes(self) -> int:
        bitmap_bytes = self.bitmap.numel()
        values_bytes = self.packed_values.numel()
        overhead = 32  # scale, zp, shape metadata
        return bitmap_bytes + values_bytes + overhead

    @property
    def original_bytes(self) -> int:
        return self.total_elements * 2  # float16

    @property
    def ratio(self) -> float:
        cb = self.compressed_bytes
        return self.original_bytes / cb if cb > 0 else 1.0

    @property
    def sparsity(self) -> float:
        return 1.0 - (self.num_nonzero / self.total_elements) if self.total_elements > 0 else 0.0


def bitmap_pack_encode(
    coeffs: torch.Tensor,
    bits: int = 8,
) -> BitmapPackedTensor:
    """
    Quantize all coefficients, then encode as bitmap + packed non-zero values.

    1. Quantize entire tensor to int8 (or int4)
    2. Create bitmap: 1 bit per element (non-zero = 1)
    3. Pack only non-zero quantized values

    This gives guaranteed compression since bitmap costs only 0.125 bytes/element.
    """
    flat = coeffs.flatten().float()
    total = flat.numel()

    # Quantize
    vmin = flat.min().item()
    vmax = flat.max().item()
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    scale = (vmax - vmin) / (qmax - qmin) if vmax != vmin else 1.0
    zero_point = vmin

    quantized = torch.clamp(
        torch.round((flat - zero_point) / scale + qmin),
        qmin, qmax,
    ).to(torch.int8)

    # Map exact zeros in the input to the nearest quantized zero
    q_zero = int(round((0.0 - zero_point) / scale + qmin))
    q_zero = max(qmin, min(qmax, q_zero))
    quantized[flat == 0] = q_zero

    # Build bitmap
    nonzero_mask = quantized != q_zero
    num_nonzero = nonzero_mask.sum().item()

    # Pack bitmap into bytes (8 elements per byte)
    bitmap_bits = nonzero_mask.to(torch.uint8)
    # Pad to multiple of 8
    pad_len = (8 - total % 8) % 8
    if pad_len > 0:
        bitmap_bits = torch.cat([bitmap_bits, torch.zeros(pad_len, dtype=torch.uint8)])

    # Pack 8 bits into each byte
    bitmap_bytes = bitmap_bits.reshape(-1, 8)
    bitmap = torch.zeros(bitmap_bytes.shape[0], dtype=torch.uint8)
    for bit in range(8):
        bitmap |= (bitmap_bytes[:, bit] << (7 - bit))

    # Pack non-zero values
    packed_values = quantized[nonzero_mask]

    return BitmapPackedTensor(
        bitmap=bitmap,
        packed_values=packed_values,
        scale=scale,
        zero_point=zero_point,
        quant_bits=bits,
        original_shape=tuple(coeffs.shape),
        total_elements=total,
        num_nonzero=num_nonzero,
    )


def bitmap_pack_decode(bpt: BitmapPackedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct coefficients from bitmap-packed representation."""
    qmin = -(1 << (bpt.quant_bits - 1))
    total = bpt.total_elements

    # Unpack bitmap
    nonzero_mask = torch.zeros(total, dtype=torch.bool)
    for byte_idx in range(bpt.bitmap.numel()):
        byte_val = bpt.bitmap[byte_idx].item()
        for bit in range(8):
            elem_idx = byte_idx * 8 + bit
            if elem_idx < total:
                nonzero_mask[elem_idx] = bool((byte_val >> (7 - bit)) & 1)

    # Reconstruct quantized tensor
    q_zero = int(round((0.0 - bpt.zero_point) / bpt.scale + qmin))
    quantized = torch.full((total,), q_zero, dtype=torch.int8)
    quantized[nonzero_mask] = bpt.packed_values

    # Dequantize
    result = (quantized.float() - qmin) * bpt.scale + bpt.zero_point
    return result.to(dtype).reshape(bpt.original_shape)


# ---------------------------------------------------------------------------
# Full KVTC Pipeline — Transform → Threshold → Dense Quantize → RLE
# ---------------------------------------------------------------------------

@dataclass
class KVTCCompressedDelta:
    """
    A single delta tensor after full transform coding.
    This is the final compressed representation.
    """
    # Bitmap + packed values payload (primary path)
    bitmap_packed: Optional[BitmapPackedTensor] = None

    # Legacy sparse path (kept for backward compat)
    rle: Optional[RLEEncoded] = None
    sparse: Optional[SparseQuantizedTensor] = None

    # Original tensor metadata
    original_shape: Tuple[int, ...] = ()
    original_dtype: torch.dtype = torch.float16

    # Transform stats
    energy_retained: float = 0.0
    sparsity: float = 0.0
    dct_mode: str = "1d"  # "1d" or "2d"

    @property
    def compressed_bytes(self) -> int:
        if self.bitmap_packed is not None:
            return self.bitmap_packed.compressed_bytes
        elif self.rle is not None:
            return self.rle.compressed_bytes
        elif self.sparse is not None:
            return self.sparse.compressed_bytes
        return 0

    @property
    def original_bytes(self) -> int:
        total_elem = 1
        for s in self.original_shape:
            total_elem *= s
        return total_elem * 2  # float16

    @property
    def ratio(self) -> float:
        cb = self.compressed_bytes
        return self.original_bytes / cb if cb > 0 else 1.0


def kvtc_compress_delta(
    delta: torch.Tensor,
    config: Optional[KVTCConfig] = None,
) -> KVTCCompressedDelta:
    """
    Full KVTC compression of a single delta tensor.

    Pipeline: DCT → energy threshold → bitmap-pack quantize

    Args:
        delta: Raw delta tensor from L1/L2 fold. Any shape, typically
               [num_heads, seq_len, head_dim].
        config: KVTC configuration.

    Returns:
        KVTCCompressedDelta — compact compressed representation.
    """
    if config is None:
        config = KVTCConfig()

    original_shape = tuple(delta.shape)
    original_dtype = delta.dtype

    # --- Step 1: DCT ---
    delta_f32 = delta.float()

    if delta_f32.dim() >= 2:
        coeffs = dct_2d(delta_f32)
        dct_mode = "2d"
    else:
        coeffs = dct_1d(delta_f32)
        dct_mode = "1d"

    # --- Step 2: Energy thresholding ---
    if config.top_k_coefficients > 0:
        sparse_coeffs, num_nz, thresh = topk_threshold(coeffs, config.top_k_coefficients)
    else:
        sparse_coeffs, num_nz, thresh = energy_threshold(coeffs, config.energy_retention)

    sparsity = 1.0 - (num_nz / coeffs.numel()) if coeffs.numel() > 0 else 1.0

    # --- Step 3: Bitmap-packed quantization ---
    bitmap_packed = bitmap_pack_encode(sparse_coeffs, bits=config.coeff_quant_bits)

    return KVTCCompressedDelta(
        bitmap_packed=bitmap_packed,
        original_shape=original_shape,
        original_dtype=original_dtype,
        energy_retained=config.energy_retention,
        sparsity=sparsity,
        dct_mode=dct_mode,
    )


def kvtc_decompress_delta(
    compressed: KVTCCompressedDelta,
) -> torch.Tensor:
    """
    Reconstruct delta tensor from KVTC compressed representation.

    Inverse pipeline: bitmap-unpack dequantize → IDCT
    """
    # --- Step 1: Decode + dequantize ---
    if compressed.bitmap_packed is not None:
        sparse_coeffs = bitmap_pack_decode(compressed.bitmap_packed, dtype=torch.float32)
    elif compressed.rle is not None:
        sqt = rle_decode(compressed.rle)
        sparse_coeffs = sparse_dequantize(sqt, dtype=torch.float32)
    elif compressed.sparse is not None:
        sparse_coeffs = sparse_dequantize(compressed.sparse, dtype=torch.float32)
    else:
        return torch.zeros(compressed.original_shape, dtype=compressed.original_dtype)

    # --- Step 2: Inverse DCT ---
    if compressed.dct_mode == "2d":
        delta = idct_2d(sparse_coeffs)
    else:
        delta = idct_1d(sparse_coeffs)

    return delta.to(compressed.original_dtype)


# ---------------------------------------------------------------------------
# Batch operations — compress all deltas in an L1/L2 cache
# ---------------------------------------------------------------------------

def kvtc_compress_all_deltas(
    deltas: List[torch.Tensor],
    config: Optional[KVTCConfig] = None,
) -> Tuple[List[KVTCCompressedDelta], Dict[str, Any]]:
    """
    Apply KVTC to a list of delta tensors (e.g., all L1 deltas).

    Returns:
        (compressed_deltas, stats_dict)
    """
    if config is None:
        config = KVTCConfig()

    results = []
    total_original = 0
    total_compressed = 0
    total_sparsity = 0.0

    for delta in deltas:
        c = kvtc_compress_delta(delta, config)
        results.append(c)
        total_original += c.original_bytes
        total_compressed += c.compressed_bytes
        total_sparsity += c.sparsity

    n = len(deltas) if deltas else 1
    stats = {
        "num_deltas": len(deltas),
        "original_bytes": total_original,
        "compressed_bytes": total_compressed,
        "ratio": round(total_original / total_compressed, 2) if total_compressed > 0 else 0,
        "mean_sparsity": round(total_sparsity / n, 4),
        "energy_retention": config.energy_retention,
        "coeff_quant_bits": config.coeff_quant_bits,
    }

    return results, stats


def kvtc_decompress_all_deltas(
    compressed: List[KVTCCompressedDelta],
) -> List[torch.Tensor]:
    """Reconstruct all delta tensors from KVTC compressed representations."""
    return [kvtc_decompress_delta(c) for c in compressed]
