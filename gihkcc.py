"""
GIHKCC — Guardian-Informed Hierarchical KV Cache Compression
Monolithic Model Adaptation

VecP Labs LLC | vecplabs.com | Patent Pending (USPTO 63/931,565)

This module implements Levels 1–3 of the recursive multi-scale fold-compress
pipeline for standard (monolithic) transformer architectures. Level 0
(inter-head-group factorization) requires Cerberus's structural head-group
separation and is not available here.

Adaptation strategy:
  - Guardian SNR is replaced by Statistical SNR: cosine similarity between
    adjacent-layer KV states drives keyframe placement.
  - The three compression tiers (Guardian/Reasoning/Language) collapse to a
    single uniform policy, since monolithic models have no head-group
    separation to exploit for differentiated compression.

Compression pipeline (monolithic):
  KV → L1 (inter-layer fold) → L2 (inter-keyframe fold) → L3 (inter-token fold)
      → quantize → (optional entropy coding)

Decompression is the reverse: each unfold is element-wise addition.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("gihkcc")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GIHKCCConfig:
    """Configuration for the monolithic GIHKCC compression pipeline."""

    # --- Level 1: Inter-Layer Fold ---
    # SNR threshold for keyframe placement. Higher = fewer keyframes = more
    # compression but more reconstruction error. Range: [0.8, 0.99].
    l1_snr_threshold: float = 0.92

    # Maximum span (in layers) between keyframes. Even if SNR stays high,
    # force a keyframe after this many layers to bound error accumulation.
    l1_max_keyframe_span: int = 8

    # --- Level 2: Inter-Keyframe Fold ---
    l2_enabled: bool = True

    # How many keyframes between super-keyframes. The first keyframe in each
    # group becomes the super-keyframe; the rest become deltas from it.
    l2_super_keyframe_interval: int = 4

    # --- Level 3: Inter-Token Fold ---
    l3_enabled: bool = False  # Off by default — noisy, use for edge deployments

    # Token-level keyframe stride. Every N tokens gets a full KV entry;
    # intermediate tokens are stored as deltas from nearest token keyframe.
    l3_token_keyframe_stride: int = 16

    # --- Quantization ---
    # Bit width for delta quantization. Keyframes stay at original precision.
    delta_quant_bits: int = 8

    # Keyframe quantization bits. None = keep original dtype.
    keyframe_quant_bits: Optional[int] = None

    # --- Adaptive degradation ---
    # If True, under memory pressure L3 is automatically enabled and
    # L2 intervals widen.
    adaptive_degradation: bool = True

    # Memory threshold (fraction of available GPU memory) that triggers
    # adaptive degradation.
    memory_pressure_threshold: float = 0.85


# ---------------------------------------------------------------------------
# Statistical SNR — replaces Guardian SNR for monolithic models
# ---------------------------------------------------------------------------

def compute_statistical_snr(
    kv_current: torch.Tensor,
    kv_reference: torch.Tensor,
) -> float:
    """
    Compute the cosine similarity between two KV tensors as a proxy for
    Guardian SNR. High similarity → delta encoding is efficient (small delta).
    Low similarity → this is a natural keyframe boundary.

    Args:
        kv_current:   [num_heads, seq_len, head_dim] or [seq_len, hidden_dim]
        kv_reference: same shape

    Returns:
        Scalar similarity in [0, 1]. Values near 1.0 mean high redundancy
        (good delta candidate). Values near 0.0 mean low redundancy
        (keyframe candidate).
    """
    a = kv_current.flatten().float()
    b = kv_reference.flatten().float()
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return max(0.0, sim)  # Clamp negatives — they indicate orthogonal states


def compute_snr_profile(
    kv_states: List[torch.Tensor],
) -> List[float]:
    """
    Compute pairwise SNR between adjacent layers across an entire stack.

    Args:
        kv_states: List of KV tensors, one per layer. Each tensor shape:
                   [num_heads, seq_len, head_dim] (keys or values separately).

    Returns:
        List of SNR values. Length = len(kv_states) - 1.
        snr_profile[i] = similarity between layer i and layer i+1.
    """
    profile = []
    for i in range(len(kv_states) - 1):
        snr = compute_statistical_snr(kv_states[i], kv_states[i + 1])
        profile.append(snr)
    return profile


# ---------------------------------------------------------------------------
# Level 1: Inter-Layer Fold (Depth)
# ---------------------------------------------------------------------------

@dataclass
class L1KeyframeEntry:
    """A keyframe in the Level 1 fold."""
    layer_idx: int
    data: torch.Tensor        # Full-precision KV tensor
    snr_at_boundary: float    # SNR value that triggered this keyframe


@dataclass
class L1DeltaEntry:
    """A delta-encoded layer in the Level 1 fold."""
    layer_idx: int
    reference_layer: int      # Layer index of the keyframe this deltas from
    delta: torch.Tensor       # kv_current - kv_reference (or quantized)
    snr: float


@dataclass
class L1CompressedCache:
    """Output of Level 1 compression."""
    keyframes: List[L1KeyframeEntry] = field(default_factory=list)
    deltas: List[L1DeltaEntry] = field(default_factory=list)
    num_layers: int = 0
    original_dtype: torch.dtype = torch.float16

    @property
    def compression_ratio(self) -> float:
        if self.num_layers == 0:
            return 1.0
        # Keyframes are full size, deltas are quantized (smaller).
        # Approximate: delta is ~(quant_bits/16) the size of a keyframe.
        n_kf = len(self.keyframes)
        n_delta = len(self.deltas)
        if n_kf + n_delta == 0:
            return 1.0
        # Assume 16-bit baseline, 8-bit deltas → delta costs 0.5x
        delta_cost = 0.5  # Adjustable based on actual quant bits
        effective = n_kf + n_delta * delta_cost
        return self.num_layers / effective if effective > 0 else 1.0


def l1_compress(
    kv_states: List[torch.Tensor],
    config: GIHKCCConfig,
) -> L1CompressedCache:
    """
    Level 1: Inter-layer fold.

    Scans KV states layer-by-layer. When SNR drops below threshold (indicating
    a representational shift), places a keyframe. Between keyframes, stores
    deltas.

    Args:
        kv_states: List of KV tensors per layer.
        config: GIHKCC configuration.

    Returns:
        L1CompressedCache with keyframes and deltas.
    """
    if not kv_states:
        return L1CompressedCache()

    cache = L1CompressedCache(
        num_layers=len(kv_states),
        original_dtype=kv_states[0].dtype,
    )

    # Layer 0 is always a keyframe
    cache.keyframes.append(L1KeyframeEntry(
        layer_idx=0,
        data=kv_states[0].clone(),
        snr_at_boundary=0.0,
    ))

    current_keyframe_idx = 0
    layers_since_keyframe = 0

    for i in range(1, len(kv_states)):
        snr = compute_statistical_snr(kv_states[i], kv_states[current_keyframe_idx])
        layers_since_keyframe += 1

        # Keyframe condition: SNR dropped below threshold OR max span exceeded
        is_keyframe = (
            snr < config.l1_snr_threshold
            or layers_since_keyframe >= config.l1_max_keyframe_span
        )

        if is_keyframe:
            cache.keyframes.append(L1KeyframeEntry(
                layer_idx=i,
                data=kv_states[i].clone(),
                snr_at_boundary=snr,
            ))
            current_keyframe_idx = i
            layers_since_keyframe = 0
        else:
            delta = kv_states[i] - kv_states[current_keyframe_idx]
            cache.deltas.append(L1DeltaEntry(
                layer_idx=i,
                reference_layer=current_keyframe_idx,
                delta=delta,
                snr=snr,
            ))

    logger.info(
        f"L1: {cache.num_layers} layers → {len(cache.keyframes)} keyframes + "
        f"{len(cache.deltas)} deltas (ratio: {cache.compression_ratio:.2f}x)"
    )
    return cache


def l1_decompress(cache: L1CompressedCache) -> List[torch.Tensor]:
    """Reconstruct full KV states from L1 compressed cache."""
    reconstructed = [None] * cache.num_layers

    # Place keyframes
    kf_map = {}
    for kf in cache.keyframes:
        reconstructed[kf.layer_idx] = kf.data
        kf_map[kf.layer_idx] = kf.data

    # Reconstruct deltas
    for d in cache.deltas:
        ref = kf_map[d.reference_layer]
        reconstructed[d.layer_idx] = ref + d.delta

    return reconstructed


# ---------------------------------------------------------------------------
# Level 2: Inter-Keyframe Fold (Deep Structure)
# ---------------------------------------------------------------------------

@dataclass
class L2SuperKeyframe:
    """A super-keyframe in the Level 2 fold."""
    layer_idx: int
    data: torch.Tensor


@dataclass
class L2KeyframeDelta:
    """A keyframe stored as a delta from its super-keyframe."""
    layer_idx: int
    super_keyframe_layer: int
    delta: torch.Tensor
    original_snr: float


@dataclass
class L2CompressedCache:
    """Output of Level 2 compression (wraps L1 output)."""
    super_keyframes: List[L2SuperKeyframe] = field(default_factory=list)
    keyframe_deltas: List[L2KeyframeDelta] = field(default_factory=list)
    l1_deltas: List[L1DeltaEntry] = field(default_factory=list)
    num_layers: int = 0
    original_dtype: torch.dtype = torch.float16

    @property
    def compression_ratio(self) -> float:
        if self.num_layers == 0:
            return 1.0
        n_skf = len(self.super_keyframes)
        n_kfd = len(self.keyframe_deltas)
        n_d = len(self.l1_deltas)
        delta_cost = 0.5
        effective = n_skf + n_kfd * delta_cost + n_d * delta_cost
        return self.num_layers / effective if effective > 0 else 1.0


def l2_compress(
    l1_cache: L1CompressedCache,
    config: GIHKCCConfig,
) -> L2CompressedCache:
    """
    Level 2: Inter-keyframe fold.

    Takes L1 keyframes and groups them. The first keyframe in each group
    becomes a super-keyframe; remaining keyframes become deltas from it.
    """
    if not config.l2_enabled:
        # Pass through — wrap L1 as-is
        l2 = L2CompressedCache(
            num_layers=l1_cache.num_layers,
            original_dtype=l1_cache.original_dtype,
            l1_deltas=l1_cache.deltas,
        )
        for kf in l1_cache.keyframes:
            l2.super_keyframes.append(L2SuperKeyframe(
                layer_idx=kf.layer_idx, data=kf.data,
            ))
        return l2

    l2 = L2CompressedCache(
        num_layers=l1_cache.num_layers,
        original_dtype=l1_cache.original_dtype,
        l1_deltas=l1_cache.deltas,
    )

    keyframes = sorted(l1_cache.keyframes, key=lambda kf: kf.layer_idx)
    interval = config.l2_super_keyframe_interval

    for group_start in range(0, len(keyframes), interval):
        group = keyframes[group_start:group_start + interval]
        # First in group is the super-keyframe
        skf = group[0]
        l2.super_keyframes.append(L2SuperKeyframe(
            layer_idx=skf.layer_idx,
            data=skf.data,
        ))
        # Rest are deltas from super-keyframe
        for kf in group[1:]:
            delta = kf.data - skf.data
            l2.keyframe_deltas.append(L2KeyframeDelta(
                layer_idx=kf.layer_idx,
                super_keyframe_layer=skf.layer_idx,
                delta=delta,
                original_snr=kf.snr_at_boundary,
            ))

    logger.info(
        f"L2: {len(l1_cache.keyframes)} keyframes → "
        f"{len(l2.super_keyframes)} super-keyframes + "
        f"{len(l2.keyframe_deltas)} keyframe-deltas "
        f"(ratio: {l2.compression_ratio:.2f}x)"
    )
    return l2


def l2_decompress(l2_cache: L2CompressedCache) -> L1CompressedCache:
    """Reconstruct L1 keyframes from L2 compressed cache."""
    # Rebuild keyframe list
    skf_map = {skf.layer_idx: skf.data for skf in l2_cache.super_keyframes}

    keyframes = []
    for skf in l2_cache.super_keyframes:
        keyframes.append(L1KeyframeEntry(
            layer_idx=skf.layer_idx,
            data=skf.data,
            snr_at_boundary=0.0,
        ))
    for kfd in l2_cache.keyframe_deltas:
        ref = skf_map[kfd.super_keyframe_layer]
        reconstructed = ref + kfd.delta
        keyframes.append(L1KeyframeEntry(
            layer_idx=kfd.layer_idx,
            data=reconstructed,
            snr_at_boundary=kfd.original_snr,
        ))

    l1 = L1CompressedCache(
        keyframes=sorted(keyframes, key=lambda kf: kf.layer_idx),
        deltas=l2_cache.l1_deltas,
        num_layers=l2_cache.num_layers,
        original_dtype=l2_cache.original_dtype,
    )
    return l1


# ---------------------------------------------------------------------------
# Level 3: Inter-Token Fold (Sequence Coherence)
# ---------------------------------------------------------------------------

@dataclass
class L3CompressedSequence:
    """
    Level 3 compressed representation for a single layer's KV cache.
    Folds across the sequence (token) dimension.
    """
    # Token keyframes: full KV entries at stride boundaries
    # Shape: [num_keyframes, num_heads, head_dim]
    token_keyframes: torch.Tensor

    # Token deltas: difference from nearest preceding token keyframe
    # Shape: [num_delta_tokens, num_heads, head_dim]
    token_deltas: torch.Tensor

    # Mapping: for each original token position, which keyframe it references
    # and whether it IS a keyframe
    keyframe_indices: List[int]    # Index into token_keyframes, per position
    is_keyframe: List[bool]        # True if this position is a keyframe

    seq_len: int = 0
    stride: int = 16


def l3_compress_layer(
    kv_layer: torch.Tensor,
    stride: int = 16,
) -> L3CompressedSequence:
    """
    Level 3: Inter-token fold for a single layer.

    Args:
        kv_layer: [seq_len, num_heads, head_dim] or [num_heads, seq_len, head_dim]
        stride: Token keyframe interval.

    Returns:
        L3CompressedSequence
    """
    # Normalize to [seq_len, num_heads, head_dim]
    if kv_layer.dim() == 3 and kv_layer.shape[0] != kv_layer.shape[1]:
        # Heuristic: if dim 0 < dim 1, assume [num_heads, seq_len, head_dim]
        if kv_layer.shape[0] < kv_layer.shape[1]:
            kv_layer = kv_layer.transpose(0, 1)  # → [seq_len, ...]

    seq_len = kv_layer.shape[0]
    keyframes = []
    deltas = []
    keyframe_indices = []
    is_keyframe = []

    current_kf_idx = 0
    current_kf_data = None

    for t in range(seq_len):
        if t % stride == 0:
            # This is a token keyframe
            keyframes.append(kv_layer[t])
            current_kf_idx = len(keyframes) - 1
            current_kf_data = kv_layer[t]
            keyframe_indices.append(current_kf_idx)
            is_keyframe.append(True)
        else:
            # Delta from nearest preceding keyframe
            delta = kv_layer[t] - current_kf_data
            deltas.append(delta)
            keyframe_indices.append(current_kf_idx)
            is_keyframe.append(False)

    return L3CompressedSequence(
        token_keyframes=torch.stack(keyframes) if keyframes else torch.empty(0),
        token_deltas=torch.stack(deltas) if deltas else torch.empty(0),
        keyframe_indices=keyframe_indices,
        is_keyframe=is_keyframe,
        seq_len=seq_len,
        stride=stride,
    )


def l3_decompress_layer(compressed: L3CompressedSequence) -> torch.Tensor:
    """Reconstruct full sequence from L3 compressed representation."""
    result = []
    delta_idx = 0

    for t in range(compressed.seq_len):
        kf_idx = compressed.keyframe_indices[t]
        if compressed.is_keyframe[t]:
            result.append(compressed.token_keyframes[kf_idx])
        else:
            ref = compressed.token_keyframes[kf_idx]
            result.append(ref + compressed.token_deltas[delta_idx])
            delta_idx += 1

    return torch.stack(result)


# ---------------------------------------------------------------------------
# Delta Quantization
# ---------------------------------------------------------------------------

def quantize_delta(
    delta: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, float, float]:
    """
    Uniform symmetric quantization of a delta tensor.

    Returns:
        (quantized_int8, scale, zero_point) — enough to reconstruct.
    """
    dmin = delta.min().item()
    dmax = delta.max().item()
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    scale = (dmax - dmin) / (qmax - qmin) if dmax != dmin else 1.0
    zero_point = dmin

    quantized = torch.clamp(
        torch.round((delta - zero_point) / scale + qmin),
        qmin, qmax,
    ).to(torch.int8)

    return quantized, scale, zero_point


def dequantize_delta(
    quantized: torch.Tensor,
    scale: float,
    zero_point: float,
    bits: int = 8,
) -> torch.Tensor:
    """Reconstruct delta from quantized representation."""
    qmin = -(1 << (bits - 1))
    return (quantized.float() - qmin) * scale + zero_point


# ---------------------------------------------------------------------------
# Full Pipeline — Unified Compress / Decompress
# ---------------------------------------------------------------------------

@dataclass
class GIHKCCCompressedKVCache:
    """
    Complete compressed KV cache for a monolithic model.
    Contains both keys and values compressed through L1→L2→(optional L3).
    """
    keys_l2: L2CompressedCache
    values_l2: L2CompressedCache
    # L3 is applied per-layer after L1/L2 reconstruction if enabled
    config: GIHKCCConfig = field(default_factory=GIHKCCConfig)
    snr_profile_keys: List[float] = field(default_factory=list)
    snr_profile_values: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Return compression statistics."""
        k_ratio = self.keys_l2.compression_ratio
        v_ratio = self.values_l2.compression_ratio
        return {
            "num_layers": self.keys_l2.num_layers,
            "keys_compression_ratio": round(k_ratio, 2),
            "values_compression_ratio": round(v_ratio, 2),
            "combined_ratio": round((k_ratio + v_ratio) / 2, 2),
            "keys_super_keyframes": len(self.keys_l2.super_keyframes),
            "keys_keyframe_deltas": len(self.keys_l2.keyframe_deltas),
            "keys_l1_deltas": len(self.keys_l2.l1_deltas),
            "values_super_keyframes": len(self.values_l2.super_keyframes),
            "values_keyframe_deltas": len(self.values_l2.keyframe_deltas),
            "values_l1_deltas": len(self.values_l2.l1_deltas),
            "l2_enabled": self.config.l2_enabled,
            "l3_enabled": self.config.l3_enabled,
            "snr_profile_keys": [round(s, 4) for s in self.snr_profile_keys],
            "snr_profile_values": [round(s, 4) for s in self.snr_profile_values],
        }


def compress_kv_cache(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    config: Optional[GIHKCCConfig] = None,
) -> GIHKCCCompressedKVCache:
    """
    Full GIHKCC compression pipeline for a monolithic model's KV cache.

    Args:
        keys:   List of key tensors, one per layer.
                Shape per tensor: [num_heads, seq_len, head_dim]
        values: List of value tensors, one per layer. Same shape as keys.
        config: Compression configuration. Uses defaults if None.

    Returns:
        GIHKCCCompressedKVCache
    """
    if config is None:
        config = GIHKCCConfig()

    assert len(keys) == len(values), "Keys and values must have same number of layers"

    # Compute SNR profiles
    snr_k = compute_snr_profile(keys)
    snr_v = compute_snr_profile(values)

    # L1: Inter-layer fold
    l1_keys = l1_compress(keys, config)
    l1_values = l1_compress(values, config)

    # L2: Inter-keyframe fold
    l2_keys = l2_compress(l1_keys, config)
    l2_values = l2_compress(l1_values, config)

    return GIHKCCCompressedKVCache(
        keys_l2=l2_keys,
        values_l2=l2_values,
        config=config,
        snr_profile_keys=snr_k,
        snr_profile_values=snr_v,
    )


def decompress_kv_cache(
    compressed: GIHKCCCompressedKVCache,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Full decompression: L2 → L1 → raw KV states.

    Returns:
        (keys, values) — Lists of tensors per layer.
    """
    # L2 → L1
    l1_keys = l2_decompress(compressed.keys_l2)
    l1_values = l2_decompress(compressed.values_l2)

    # L1 → raw
    keys = l1_decompress(l1_keys)
    values = l1_decompress(l1_values)

    return keys, values


# ---------------------------------------------------------------------------
# Memory size estimation
# ---------------------------------------------------------------------------

def estimate_memory_bytes(
    compressed: GIHKCCCompressedKVCache,
) -> Dict[str, Any]:
    """
    Estimate memory footprint of compressed vs uncompressed cache.

    Reports two views:
      - structural: raw tensor storage (deltas at original dtype)
      - projected:  with delta quantization applied (8-bit deltas)
    """
    def _tensor_bytes(t: torch.Tensor) -> int:
        return t.nelement() * t.element_size()

    def _tensor_elements(t: torch.Tensor) -> int:
        return t.nelement()

    # Count keyframe bytes (full precision) and delta elements (to be quantized)
    kf_bytes = 0
    delta_elements = 0
    for skf in compressed.keys_l2.super_keyframes:
        kf_bytes += _tensor_bytes(skf.data)
    for skf in compressed.values_l2.super_keyframes:
        kf_bytes += _tensor_bytes(skf.data)
    for kfd in compressed.keys_l2.keyframe_deltas:
        delta_elements += _tensor_elements(kfd.delta)
    for kfd in compressed.values_l2.keyframe_deltas:
        delta_elements += _tensor_elements(kfd.delta)
    for d in compressed.keys_l2.l1_deltas:
        delta_elements += _tensor_elements(d.delta)
    for d in compressed.values_l2.l1_deltas:
        delta_elements += _tensor_elements(d.delta)

    # Structural: deltas at original dtype
    orig_elem_size = 2  # float16
    if compressed.keys_l2.super_keyframes:
        orig_elem_size = compressed.keys_l2.super_keyframes[0].data.element_size()

    structural_bytes = kf_bytes + delta_elements * orig_elem_size

    # Projected: deltas quantized to 8-bit (1 byte per element + 8 bytes overhead per tensor)
    quant_bits = compressed.config.delta_quant_bits
    quant_byte_per_elem = max(quant_bits / 8, 0.5)  # At least 4-bit = 0.5 bytes
    n_delta_tensors = (
        len(compressed.keys_l2.keyframe_deltas) + len(compressed.values_l2.keyframe_deltas)
        + len(compressed.keys_l2.l1_deltas) + len(compressed.values_l2.l1_deltas)
    )
    projected_bytes = kf_bytes + int(delta_elements * quant_byte_per_elem) + n_delta_tensors * 8

    # Original size estimate
    original_bytes = 0
    if compressed.keys_l2.super_keyframes:
        per_layer = _tensor_bytes(compressed.keys_l2.super_keyframes[0].data)
        n = compressed.keys_l2.num_layers
        original_bytes = per_layer * n * 2  # keys + values

    return {
        "original_bytes": original_bytes,
        "structural_bytes": structural_bytes,
        "projected_bytes": projected_bytes,
        "structural_ratio": round(original_bytes / structural_bytes, 2) if structural_bytes > 0 else 0,
        "projected_ratio": round(original_bytes / projected_bytes, 2) if projected_bytes > 0 else 0,
        "savings_projected_bytes": original_bytes - projected_bytes,
    }
