"""
Ternary Residual Stream Compression

Two approaches:
  1. XNOR Ternary: Ternary quantize residuals, compute deltas as XNOR-like
     sparse difference, store as bitmap + sparse nonzeros.
  2. Hierarchical Ternary Fold: Multi-level fold on ternary residuals —
     root anchor → L2 super-keyframe deltas → L1 layer deltas.
     Sparsity compounds at each level.

Both exploit the 0.95-0.99 cosine similarity in the residual stream
proven in tonight's Cerberus test.

VecP Labs LLC | vecplabs.com | Patent Pending
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch

logger = logging.getLogger("gihkcc.ternary")


# ═══════════════════════════════════════════════════════════════════════════
# Ternary Quantization Core
# ═══════════════════════════════════════════════════════════════════════════

def ternary_quantize(
    x: torch.Tensor,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Quantize a tensor to ternary {-1, 0, 1}.

    Uses a magnitude threshold: values above threshold → +1/-1 (by sign),
    values below → 0. Default threshold is 0.7 × mean(|x|), following
    the TWN (Ternary Weight Networks) strategy.

    Args:
        x: Input tensor, any shape.
        threshold: Absolute threshold for zero. If None, uses 0.7 * mean(|x|).

    Returns:
        (ternary, scale) where:
          ternary: int8 tensor with values in {-1, 0, 1}
          scale: float scaling factor for reconstruction (mean of |nonzero elements|)
    """
    x_f32 = x.float()

    if threshold is None:
        threshold = 0.7 * x_f32.abs().mean().item()

    ternary = torch.zeros_like(x_f32, dtype=torch.int8)
    pos_mask = x_f32 > threshold
    neg_mask = x_f32 < -threshold
    ternary[pos_mask] = 1
    ternary[neg_mask] = -1

    # Scale factor: mean magnitude of non-zero entries
    nonzero = x_f32.abs()[pos_mask | neg_mask]
    scale = nonzero.mean().item() if nonzero.numel() > 0 else 1.0

    return ternary, scale


def quint5_quantize(
    x: torch.Tensor,
    levels: int = 2,
) -> Tuple[torch.Tensor, float]:
    """
    Quantize a tensor to (2*levels+1) values: {-levels, ..., -1, 0, 1, ..., levels}.

    levels=1 → ternary-like {-1,0,1} (3 values, 1.58 bits)
    levels=2 → {-2,-1,0,1,2} (5 values, 2.32 bits)
    levels=3 → {-3..3} (7 values, 2.81 bits)
    levels=6 → {-6..6} (13 values, 3.70 bits)

    Uses mean-absolute-value scaling: typical values map to ±1,
    large values saturate at ±levels.

    Returns:
        (quantized, scale)
    """
    x_f32 = x.float()

    mean_abs = x_f32.abs().mean().item()
    if mean_abs < 1e-10:
        return torch.zeros_like(x_f32, dtype=torch.int8), 1.0

    scale = mean_abs  # One step = mean absolute value

    quantized = torch.clamp(torch.round(x_f32 / scale), -levels, levels).to(torch.int8)

    return quantized, scale


def quint5_dequantize(
    quantized: torch.Tensor,
    scale: float,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct from 5-level representation."""
    return (quantized.float() * scale).to(dtype)


def ternary_dequantize(
    ternary: torch.Tensor,
    scale: float,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct from ternary representation."""
    return (ternary.float() * scale).to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
# XNOR Ternary Delta Encoding
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class XNORDelta:
    """
    Delta between two ternary tensors.

    When two ternary values agree, delta = 0.
    When they disagree, delta values depend on quantization levels.
    For ternary {-1,0,1}: deltas in {-2,-1,1,2} (4 values, 2 bits).
    For levels=N: deltas in {-2N..2N} (4N values, ceil(log2(4N)) bits).

    Storage: zero_bitmap (1 bit/element) + nonzero values (variable bits).
    """
    # Bitmap: 1 = nonzero delta, 0 = agreement
    # Packed into uint8 (8 elements per byte)
    zero_bitmap: torch.Tensor  # uint8

    # Nonzero delta values
    nonzero_values: torch.Tensor  # int8

    # Scales for reconstruction
    scale_current: float
    scale_reference: float

    # Shape metadata
    original_shape: Tuple[int, ...]
    total_elements: int
    num_nonzero: int

    # Quantization level (for correct byte accounting)
    quant_levels: int = 1  # 1=ternary, 2=quint5, etc.

    @property
    def _bits_per_nonzero(self) -> float:
        """Bits needed per nonzero delta value."""
        n_possible = max(4 * self.quant_levels, 4)  # {-2L..2L} minus 0
        return math.ceil(math.log2(n_possible)) if n_possible > 1 else 1

    @property
    def agreement_rate(self) -> float:
        return 1.0 - (self.num_nonzero / self.total_elements) if self.total_elements > 0 else 1.0

    @property
    def compressed_bytes(self) -> int:
        bitmap_bytes = self.zero_bitmap.numel()  # 1 bit/element packed into bytes
        bpnz = self._bits_per_nonzero
        nonzero_bytes = math.ceil(self.num_nonzero * bpnz / 8)
        overhead = 16  # scales + shape
        return bitmap_bytes + nonzero_bytes + overhead

    @property
    def original_bits_per_element(self) -> float:
        """Bits per element for this delta (lower = more compressible)."""
        if self.total_elements == 0:
            return 0
        nz_frac = self.num_nonzero / self.total_elements
        return 1.0 + nz_frac * self._bits_per_nonzero

    @property
    def full_ternary_bytes(self) -> int:
        """Size if stored as full quantized entry."""
        n_values = 2 * self.quant_levels + 1
        bits = math.log2(n_values) if n_values > 1 else 1
        return math.ceil(self.total_elements * bits / 8)


def xnor_ternary_delta(
    current: torch.Tensor,
    reference: torch.Tensor,
    scale_current: float,
    scale_reference: float,
    levels: int = 1,
) -> XNORDelta:
    """
    Compute the XNOR-like delta between two quantized tensors.
    levels=1 for ternary, levels=2 for quint5, etc.
    """
    total = current.numel()
    flat_c = current.flatten().int()
    flat_r = reference.flatten().int()

    delta = flat_c - flat_r
    nonzero_mask = delta != 0
    num_nz = nonzero_mask.sum().item()

    # Pack bitmap — vectorized
    bitmap_bits = nonzero_mask.to(torch.uint8)
    pad = (8 - total % 8) % 8
    if pad > 0:
        bitmap_bits = torch.cat([bitmap_bits, torch.zeros(pad, dtype=torch.uint8)])
    bitmap_bits = bitmap_bits.reshape(-1, 8)
    bitmap = torch.zeros(bitmap_bits.shape[0], dtype=torch.uint8)
    for bit in range(8):
        bitmap |= bitmap_bits[:, bit] << (7 - bit)

    nz_values = delta[nonzero_mask].to(torch.int8)

    return XNORDelta(
        zero_bitmap=bitmap,
        nonzero_values=nz_values,
        scale_current=scale_current,
        scale_reference=scale_reference,
        original_shape=tuple(current.shape),
        total_elements=total,
        num_nonzero=num_nz,
        quant_levels=levels,
    )


def xnor_reconstruct(
    delta: XNORDelta,
    reference: torch.Tensor,
    clamp_range: int = 1,
) -> torch.Tensor:
    """Reconstruct quantized tensor from reference + XNOR delta.
    clamp_range=1 for ternary {-1,0,1}, clamp_range=2 for quint5 {-2,-1,0,1,2}.
    """
    flat_r = reference.flatten().int()

    # Unpack bitmap — vectorized
    total = delta.total_elements
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    byte_expanded = delta.zero_bitmap.unsqueeze(1)  # [N_bytes, 1]
    nz_mask = ((byte_expanded >> shifts) & 1).bool().flatten()[:total]

    # Apply delta
    result = flat_r.clone()
    result[nz_mask] += delta.nonzero_values.int()

    # Clamp to valid range
    result = result.clamp(-clamp_range, clamp_range).to(torch.int8)
    return result.reshape(delta.original_shape)


# ═══════════════════════════════════════════════════════════════════════════
# Approach 1: XNOR Flat Delta Chain
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class XNORCompressedResiduals:
    """Flat XNOR delta chain: anchor + sequential deltas."""
    # Anchor (first residual, full ternary)
    anchor_ternary: torch.Tensor  # int8 {-1, 0, 1}
    anchor_scale: float

    # Sequential deltas
    deltas: List[XNORDelta]

    # Original float residuals' scales (for dequantization)
    layer_scales: List[float]

    # Stats
    n_layers: int = 0
    quant_levels: int = 1

    @property
    def total_compressed_bytes(self) -> int:
        bits_per_value = math.log2(2 * self.quant_levels + 1)
        anchor_bytes = math.ceil(self.anchor_ternary.numel() * bits_per_value / 8) + 8
        delta_bytes = sum(d.compressed_bytes for d in self.deltas)
        return anchor_bytes + delta_bytes

    @property
    def mean_agreement(self) -> float:
        if not self.deltas:
            return 1.0
        return sum(d.agreement_rate for d in self.deltas) / len(self.deltas)

    @property
    def mean_bits_per_element(self) -> float:
        if not self.deltas:
            return 1.58
        total_bits = 1.58  # anchor
        total_bits += sum(d.original_bits_per_element for d in self.deltas)
        return total_bits / (1 + len(self.deltas))


def xnor_compress_residuals(
    residuals: List[torch.Tensor],
    threshold: Optional[float] = None,
) -> XNORCompressedResiduals:
    """
    Compress a list of residual states using XNOR ternary delta chain.

    Each residual is ternary-quantized, then stored as a delta from
    the previous layer's ternary representation.
    """
    n = len(residuals)
    if n == 0:
        return XNORCompressedResiduals(
            anchor_ternary=torch.empty(0, dtype=torch.int8),
            anchor_scale=0, deltas=[], layer_scales=[], n_layers=0,
        )

    # Ternary quantize all residuals
    ternaries = []
    scales = []
    for r in residuals:
        t, s = ternary_quantize(r, threshold=threshold)
        ternaries.append(t)
        scales.append(s)

    # First is anchor
    anchor = ternaries[0]
    anchor_scale = scales[0]

    # Rest are deltas from previous
    deltas = []
    for i in range(1, n):
        d = xnor_ternary_delta(
            ternaries[i], ternaries[i - 1],
            scale_current=scales[i],
            scale_reference=scales[i - 1],
            levels=1,
        )
        deltas.append(d)

    return XNORCompressedResiduals(
        anchor_ternary=anchor,
        anchor_scale=anchor_scale,
        deltas=deltas,
        layer_scales=scales,
        n_layers=n,
        quant_levels=1,
    )


def xnor_decompress_residuals(
    compressed: XNORCompressedResiduals,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Reconstruct float residuals from XNOR compressed chain."""
    results = []

    # Anchor
    current_ternary = compressed.anchor_ternary
    results.append(ternary_dequantize(
        current_ternary, compressed.layer_scales[0], dtype,
    ))

    # Chain reconstruct
    for i, delta in enumerate(compressed.deltas):
        current_ternary = xnor_reconstruct(delta, current_ternary)
        results.append(ternary_dequantize(
            current_ternary, compressed.layer_scales[i + 1], dtype,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Sigma-Delta Noise Shaping — Error-Corrected Quantization Chain
# ═══════════════════════════════════════════════════════════════════════════

def sigmadelta_compress_residuals(
    residuals: List[torch.Tensor],
    levels: int = 1,
    threshold: Optional[float] = None,
) -> XNORCompressedResiduals:
    """
    Compress residuals with sigma-delta noise shaping.

    Instead of quantizing each residual independently, feed the
    quantization error from layer i into layer i+1's quantization.
    The errors zigzag around zero instead of drifting.

    levels=1: ternary {-1,0,1}
    levels=2: quint5 {-2..2}
    levels=N: {-N..N}

    Uses the same XNORCompressedResiduals format — the delta chain
    structure is identical, just the quantized values are smarter.
    """
    n = len(residuals)
    if n == 0:
        return XNORCompressedResiduals(
            anchor_ternary=torch.empty(0, dtype=torch.int8),
            anchor_scale=0, deltas=[], layer_scales=[], n_layers=0,
        )

    quantized_layers = []
    scales = []
    accumulated_error = torch.zeros_like(residuals[0].float())

    for i, r in enumerate(residuals):
        # Add accumulated error from previous layers to this layer's input
        # This is the sigma-delta feedback: the quantizer sees the raw value
        # plus the correction term, so it compensates for previous errors
        corrected = r.float() + accumulated_error

        # Quantize the corrected value
        if levels == 1:
            q, s = ternary_quantize(corrected.to(r.dtype), threshold=threshold)
            # Dequantize to compute error
            recon = ternary_dequantize(q, s, dtype=torch.float32)
        else:
            q, s = quint5_quantize(corrected.to(r.dtype), levels=levels)
            recon = quint5_dequantize(q, s, dtype=torch.float32)

        quantized_layers.append(q)
        scales.append(s)

        # Update accumulated error: what the quantizer couldn't represent
        # This is the "sigma" part — integrating the error
        accumulated_error = corrected - recon

    # Now build the delta chain from the noise-shaped quantized values
    anchor = quantized_layers[0]
    anchor_scale = scales[0]

    deltas = []
    for i in range(1, n):
        d = xnor_ternary_delta(
            quantized_layers[i], quantized_layers[i - 1],
            scale_current=scales[i],
            scale_reference=scales[i - 1],
            levels=levels,
        )
        deltas.append(d)

    return XNORCompressedResiduals(
        anchor_ternary=anchor,
        anchor_scale=anchor_scale,
        deltas=deltas,
        layer_scales=scales,
        n_layers=n,
        quant_levels=levels,
    )


def sigmadelta_decompress_residuals(
    compressed: XNORCompressedResiduals,
    levels: int = 1,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """
    Reconstruct from sigma-delta compressed chain.

    Identical to normal decompression — the noise shaping only affects
    the *encoding* decisions, not the decode path. The stored values
    are still quantized+delta, they're just smarter quantized values.
    """
    results = []
    current = compressed.anchor_ternary

    if levels == 1:
        results.append(ternary_dequantize(current, compressed.layer_scales[0], dtype))
    else:
        results.append(quint5_dequantize(current, compressed.layer_scales[0], dtype))

    for i, delta in enumerate(compressed.deltas):
        current = xnor_reconstruct(delta, current, clamp_range=levels)
        if levels == 1:
            results.append(ternary_dequantize(current, compressed.layer_scales[i + 1], dtype))
        else:
            results.append(quint5_dequantize(current, compressed.layer_scales[i + 1], dtype))

    return results

def quint5_compress_residuals(
    residuals: List[torch.Tensor],
    levels: int = 2,
) -> XNORCompressedResiduals:
    """
    Compress residuals using N-level quantization + XNOR delta chain.
    levels=2 → {-2..2}, levels=6 → {-6..6}, etc.
    """
    n = len(residuals)
    if n == 0:
        return XNORCompressedResiduals(
            anchor_ternary=torch.empty(0, dtype=torch.int8),
            anchor_scale=0, deltas=[], layer_scales=[], n_layers=0,
        )

    quantized = []
    scales = []
    for r in residuals:
        q, s = quint5_quantize(r, levels=levels)
        quantized.append(q)
        scales.append(s)

    anchor = quantized[0]
    anchor_scale = scales[0]

    deltas = []
    for i in range(1, n):
        d = xnor_ternary_delta(
            quantized[i], quantized[i - 1],
            scale_current=scales[i],
            scale_reference=scales[i - 1],
            levels=levels,
        )
        deltas.append(d)

    return XNORCompressedResiduals(
        anchor_ternary=anchor,
        anchor_scale=anchor_scale,
        deltas=deltas,
        layer_scales=scales,
        n_layers=n,
        quant_levels=levels,
    )


def quint5_decompress_residuals(
    compressed: XNORCompressedResiduals,
    levels: int = 2,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Reconstruct float residuals from N-level XNOR compressed chain."""
    results = []

    current = compressed.anchor_ternary
    results.append(quint5_dequantize(current, compressed.layer_scales[0], dtype))

    for i, delta in enumerate(compressed.deltas):
        current = xnor_reconstruct(delta, current, clamp_range=levels)
        results.append(quint5_dequantize(current, compressed.layer_scales[i + 1], dtype))

    return results


def quint5_hierarchical_compress(
    residuals: List[torch.Tensor],
    group_size: int = 4,
    levels: int = 2,
) -> HierarchicalCompressedResiduals:
    """Hierarchical fold using N-level quantization."""
    n = len(residuals)
    if n == 0:
        return HierarchicalCompressedResiduals(
            levels=[], root_ternary=torch.empty(0, dtype=torch.int8),
            root_scale=0, layer_scales=[], n_layers=0,
        )

    quantized = []
    scales = []
    for r in residuals:
        q, s = quint5_quantize(r, levels=levels)
        quantized.append(q)
        scales.append(s)

    # Level 1: Group layers, delta from group leader
    skf_indices = []
    l1_deltas = []
    for group_start in range(0, n, group_size):
        group_end = min(group_start + group_size, n)
        skf_indices.append(group_start)
        for i in range(group_start + 1, group_end):
            d = xnor_ternary_delta(
                quantized[i], quantized[group_start],
                scale_current=scales[i], scale_reference=scales[group_start],
                levels=levels,
            )
            l1_deltas.append(d)

    level1 = HierarchicalLevel(
        name="L1_layer",
        keyframe_indices=skf_indices,
        deltas=l1_deltas,
        keyframe_ternaries=[quantized[i] for i in skf_indices],
        keyframe_scales=[scales[i] for i in skf_indices],
        quant_levels=levels,
    )

    # Level 2: Delta-encode super-keyframes from root
    root_idx = skf_indices[0]
    root = quantized[root_idx]
    root_scale = scales[root_idx]
    l2_deltas = []
    for skf_i in skf_indices[1:]:
        d = xnor_ternary_delta(
            quantized[skf_i], root,
            scale_current=scales[skf_i], scale_reference=root_scale,
            levels=levels,
        )
        l2_deltas.append(d)

    level2 = HierarchicalLevel(
        name="L2_skf",
        keyframe_indices=[skf_indices[0]],
        deltas=l2_deltas,
        keyframe_ternaries=[root],
        keyframe_scales=[root_scale],
        quant_levels=levels,
    )

    return HierarchicalCompressedResiduals(
        levels=[level1, level2],
        root_ternary=root,
        root_scale=root_scale,
        layer_scales=scales,
        n_layers=n,
        group_size=group_size,
        quant_levels=levels,
    )


def quint5_hierarchical_decompress(
    compressed: HierarchicalCompressedResiduals,
    levels: int = 2,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Reconstruct from N-level hierarchical fold."""
    n = compressed.n_layers
    level1 = compressed.levels[0]
    level2 = compressed.levels[1]

    # Reconstruct SKFs from root
    skf_q = [compressed.root_ternary.clone()]
    for d in level2.deltas:
        recon = xnor_reconstruct(d, compressed.root_ternary, clamp_range=levels)
        skf_q.append(recon)

    # Reconstruct all layers from SKFs
    group_size = compressed.group_size
    all_q = [None] * n
    l1_idx = 0
    for skf_group_idx, group_start in enumerate(level1.keyframe_indices):
        all_q[group_start] = skf_q[skf_group_idx]
        group_end = min(group_start + group_size, n)
        for i in range(group_start + 1, group_end):
            all_q[i] = xnor_reconstruct(level1.deltas[l1_idx], skf_q[skf_group_idx], clamp_range=levels)
            l1_idx += 1

    return [quint5_dequantize(all_q[i], compressed.layer_scales[i], dtype) for i in range(n)]

@dataclass
class HierarchicalLevel:
    """One level of the hierarchical fold."""
    name: str
    keyframe_indices: List[int]
    deltas: List[XNORDelta]
    keyframe_ternaries: List[torch.Tensor]
    keyframe_scales: List[float]
    quant_levels: int = 1  # 1=ternary, 2=quint5, etc.

    @property
    def bytes(self) -> int:
        n_values = 2 * self.quant_levels + 1
        bits_per = math.log2(n_values) if n_values > 1 else 1
        kf_bytes = sum(
            math.ceil(t.numel() * bits_per / 8) + 8
            for t in self.keyframe_ternaries
        )
        delta_bytes = sum(d.compressed_bytes for d in self.deltas)
        return kf_bytes + delta_bytes

    @property
    def mean_agreement(self) -> float:
        if not self.deltas:
            return 1.0
        return sum(d.agreement_rate for d in self.deltas) / len(self.deltas)


@dataclass
class HierarchicalCompressedResiduals:
    """Multi-level hierarchical fold."""
    levels: List[HierarchicalLevel]
    root_ternary: torch.Tensor
    root_scale: float
    layer_scales: List[float]
    n_layers: int = 0
    group_size: int = 4
    quant_levels: int = 1

    @property
    def total_compressed_bytes(self) -> int:
        n_values = 2 * self.quant_levels + 1
        bits_per = math.log2(n_values) if n_values > 1 else 1
        root_bytes = math.ceil(self.root_ternary.numel() * bits_per / 8) + 8
        level_bytes = sum(l.bytes for l in self.levels)
        return root_bytes + level_bytes

    def summary(self) -> Dict[str, Any]:
        result = {
            "n_layers": self.n_layers,
            "n_levels": len(self.levels),
            "total_bytes": self.total_compressed_bytes,
        }
        for l in self.levels:
            result[f"{l.name}_kf"] = len(l.keyframe_indices)
            result[f"{l.name}_deltas"] = len(l.deltas)
            result[f"{l.name}_agreement"] = round(l.mean_agreement, 4)
            result[f"{l.name}_bytes"] = l.bytes
        return result


def hierarchical_compress_residuals(
    residuals: List[torch.Tensor],
    group_size: int = 4,
    threshold: Optional[float] = None,
) -> HierarchicalCompressedResiduals:
    """
    Compress residuals with hierarchical ternary fold.

    Level 1 (Layer fold): Group every `group_size` layers. First in each
      group is a super-keyframe. Rest are XNOR deltas from the SKF.

    Level 2 (SKF fold): The super-keyframes are themselves delta-encoded
      from the root anchor. Sparsity compounds.

    Root: Single full ternary residual (the first SKF).
    """
    n = len(residuals)
    if n == 0:
        return HierarchicalCompressedResiduals(
            levels=[], root_ternary=torch.empty(0, dtype=torch.int8),
            root_scale=0, layer_scales=[], n_layers=0,
        )

    # Ternary quantize all residuals
    ternaries = []
    scales = []
    for r in residuals:
        t, s = ternary_quantize(r, threshold=threshold)
        ternaries.append(t)
        scales.append(s)

    # ── Level 1: Group layers, delta from group leader ──────────────
    skf_indices = []  # Indices of super-keyframes
    l1_deltas = []    # Deltas from group leader

    for group_start in range(0, n, group_size):
        group_end = min(group_start + group_size, n)
        # First in group is super-keyframe
        skf_indices.append(group_start)

        # Rest are deltas from group leader
        for i in range(group_start + 1, group_end):
            d = xnor_ternary_delta(
                ternaries[i], ternaries[group_start],
                scale_current=scales[i],
                scale_reference=scales[group_start],
                levels=1,
            )
            l1_deltas.append(d)

    level1 = HierarchicalLevel(
        name="L1_layer",
        keyframe_indices=skf_indices,
        deltas=l1_deltas,
        keyframe_ternaries=[ternaries[i] for i in skf_indices],
        keyframe_scales=[scales[i] for i in skf_indices],
        quant_levels=1,
    )

    # ── Level 2: Delta-encode the super-keyframes from root ────────
    root_idx = skf_indices[0]
    root_ternary = ternaries[root_idx]
    root_scale = scales[root_idx]

    l2_deltas = []
    l2_kf_indices = [0]  # Root is the only L2 keyframe

    for i, skf_i in enumerate(skf_indices[1:], 1):
        d = xnor_ternary_delta(
            ternaries[skf_i], root_ternary,
            scale_current=scales[skf_i],
            scale_reference=root_scale,
            levels=1,
        )
        l2_deltas.append(d)

    level2 = HierarchicalLevel(
        name="L2_skf",
        keyframe_indices=[skf_indices[0]],
        deltas=l2_deltas,
        keyframe_ternaries=[root_ternary],
        keyframe_scales=[root_scale],
        quant_levels=1,
    )

    return HierarchicalCompressedResiduals(
        levels=[level1, level2],
        root_ternary=root_ternary,
        root_scale=root_scale,
        layer_scales=scales,
        n_layers=n,
        group_size=group_size,
        quant_levels=1,
    )


def hierarchical_decompress_residuals(
    compressed: HierarchicalCompressedResiduals,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Reconstruct float residuals from hierarchical ternary fold."""
    n = compressed.n_layers
    level1 = compressed.levels[0]  # Layer fold
    level2 = compressed.levels[1]  # SKF fold

    # Step 1: Reconstruct super-keyframes from root + L2 deltas
    skf_ternaries = [compressed.root_ternary.clone()]  # Root is first SKF
    for d in level2.deltas:
        recon = xnor_reconstruct(d, compressed.root_ternary)
        skf_ternaries.append(recon)

    # Step 2: Reconstruct all layers from SKFs + L1 deltas
    group_size = compressed.group_size
    all_ternaries = [None] * n

    # Place SKFs
    l1_delta_idx = 0
    for skf_group_idx, group_start in enumerate(level1.keyframe_indices):
        all_ternaries[group_start] = skf_ternaries[skf_group_idx]
        group_end = min(group_start + group_size, n)

        # Reconstruct group members from SKF + L1 deltas
        for i in range(group_start + 1, group_end):
            all_ternaries[i] = xnor_reconstruct(
                level1.deltas[l1_delta_idx],
                skf_ternaries[skf_group_idx],
            )
            l1_delta_idx += 1

    # Step 3: Dequantize all ternaries to float
    results = []
    for i in range(n):
        results.append(ternary_dequantize(
            all_ternaries[i], compressed.layer_scales[i], dtype,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis Helpers
# ═══════════════════════════════════════════════════════════════════════════

def analyze_ternary_stats(
    residuals: List[torch.Tensor],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyze ternary quantization and inter-layer agreement statistics."""
    ternaries = []
    scales = []
    sparsities = []

    for r in residuals:
        t, s = ternary_quantize(r, threshold=threshold)
        ternaries.append(t)
        scales.append(s)
        sparsities.append((t == 0).float().mean().item())

    # Adjacent-layer agreement
    agreements = []
    for i in range(len(ternaries) - 1):
        agree = (ternaries[i] == ternaries[i + 1]).float().mean().item()
        agreements.append(agree)

    return {
        "n_layers": len(residuals),
        "mean_sparsity": sum(sparsities) / len(sparsities),
        "mean_scale": sum(scales) / len(scales),
        "agreements": agreements,
        "mean_agreement": sum(agreements) / len(agreements) if agreements else 0,
        "min_agreement": min(agreements) if agreements else 0,
        "max_agreement": max(agreements) if agreements else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Even/Odd Wavelet Fold — Adaptive Precision by Magnitude Band
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvenOddCompressed:
    """
    Even/odd wavelet decomposition of quantized residual stream.

    Evens (coarse grid) → fold by //2, delta-encode between layers.
    Odds (detail) → store as sparse corrections.
    """
    coarse_anchor: torch.Tensor
    coarse_deltas: List[XNORDelta]
    parity_bitmaps: List[torch.Tensor]  # uint8 packed
    odd_values: List[torch.Tensor]  # sparse int8

    layer_scales: List[float]
    original_shapes: List[Tuple[int, ...]]
    levels: int = 6
    n_layers: int = 0

    @property
    def total_compressed_bytes(self) -> int:
        coarse_levels = self.levels // 2
        coarse_bits = math.log2(2 * coarse_levels + 1) if coarse_levels > 0 else 1
        anchor_bytes = math.ceil(self.coarse_anchor.numel() * coarse_bits / 8) + 8
        delta_bytes = sum(d.compressed_bytes for d in self.coarse_deltas)
        parity_bytes = sum(b.numel() for b in self.parity_bitmaps)
        odd_bits = max(math.ceil(math.log2(max(self.levels, 2))), 1)
        odd_bytes = sum(math.ceil(v.numel() * odd_bits / 8) + 4 for v in self.odd_values)
        return anchor_bytes + delta_bytes + parity_bytes + odd_bytes + 32

    @property
    def coarse_agreement(self) -> float:
        if not self.coarse_deltas:
            return 1.0
        return sum(d.agreement_rate for d in self.coarse_deltas) / len(self.coarse_deltas)


def evenodd_compress_residuals(
    residuals: List[torch.Tensor],
    levels: int = 6,
) -> EvenOddCompressed:
    """
    Compress residuals with even/odd wavelet fold.

    1. Quantize to ±levels
    2. Evens → fold //2 → coarse ±(levels//2), delta-encode between layers
    3. Odds → sparse corrections with parity bitmap
    """
    n = len(residuals)
    coarse_levels = levels // 2

    scales = []
    parity_bitmaps = []
    odd_values_list = []
    coarse_list = []

    for r in residuals:
        q, s = quint5_quantize(r, levels=levels)
        scales.append(s)

        flat = q.flatten()
        is_odd = (flat.int().abs() % 2 == 1)

        # Pack parity bitmap
        parity_bits = is_odd.to(torch.uint8)
        pad = (8 - parity_bits.numel() % 8) % 8
        if pad > 0:
            parity_bits = torch.cat([parity_bits, torch.zeros(pad, dtype=torch.uint8)])
        pb = parity_bits.reshape(-1, 8)
        bitmap = torch.zeros(pb.shape[0], dtype=torch.uint8)
        for bit in range(8):
            bitmap |= pb[:, bit] << (7 - bit)
        parity_bitmaps.append(bitmap)

        # Extract odd values
        odd_values_list.append(flat[is_odd].to(torch.int8))

        # Fold evens
        coarse = flat.clone()
        coarse[is_odd] = 0
        coarse = (coarse.int() // 2).to(torch.int8).reshape(q.shape)
        coarse_list.append(coarse)

    # Delta-encode coarse grid
    coarse_anchor = coarse_list[0]
    coarse_deltas = []
    for i in range(1, n):
        d = xnor_ternary_delta(
            coarse_list[i], coarse_list[i - 1],
            scale_current=scales[i], scale_reference=scales[i - 1],
            levels=coarse_levels,
        )
        coarse_deltas.append(d)

    return EvenOddCompressed(
        coarse_anchor=coarse_anchor,
        coarse_deltas=coarse_deltas,
        parity_bitmaps=parity_bitmaps,
        odd_values=odd_values_list,
        layer_scales=scales,
        original_shapes=[tuple(r.shape) for r in residuals],
        levels=levels,
        n_layers=n,
    )


def evenodd_decompress_residuals(
    compressed: EvenOddCompressed,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Reconstruct from even/odd wavelet fold."""
    n = compressed.n_layers
    coarse_levels = compressed.levels // 2

    # Reconstruct coarse chain
    coarse_list = [compressed.coarse_anchor]
    current = compressed.coarse_anchor
    for d in compressed.coarse_deltas:
        current = xnor_reconstruct(d, current, clamp_range=coarse_levels)
        coarse_list.append(current)

    results = []
    for i in range(n):
        coarse = coarse_list[i].flatten().int()
        full = coarse * 2  # Unfold evens

        # Unpack parity
        total = full.numel()
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
        byte_exp = compressed.parity_bitmaps[i].unsqueeze(1)
        is_odd = ((byte_exp >> shifts) & 1).bool().flatten()[:total]

        # Re-insert odds
        full[is_odd] = compressed.odd_values[i].int()
        full = full.clamp(-compressed.levels, compressed.levels)

        scale = compressed.layer_scales[i]
        results.append((full.float() * scale).to(dtype).reshape(compressed.original_shapes[i]))

    return results
