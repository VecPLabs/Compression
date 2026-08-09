"""Optional Triton kernels for packed GIHKCC payload decoding."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional accelerator
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _packed_dequant_kernel(
        packed_ptr, norms_ptr, centroids_ptr, output_ptr,
        start_element, element_count,
        dim: tl.constexpr, bits: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < element_count
        global_elements = start_element + offsets
        bit_positions = global_elements * bits
        byte_indices = bit_positions // 8
        shifts = bit_positions % 8
        low = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.int32)
        high = tl.load(
            packed_ptr + byte_indices + 1,
            mask=mask & (shifts + bits > 8), other=0,
        ).to(tl.int32)
        values = (low >> shifts) | (high << (8 - shifts))
        values = values & ((1 << bits) - 1)
        vector_indices = offsets // dim
        norms = tl.load(norms_ptr + vector_indices).to(tl.float32)
        centroids = tl.load(centroids_ptr + values).to(tl.float32)
        tl.store(output_ptr + offsets, centroids * norms, mask=mask)


    @triton.jit
    def _packed_chain_dequant_kernel(
        anchor_packed_ptr, anchor_norms_ptr, anchor_centroids_ptr,
        delta_packed_ptr, delta_norms_ptr, delta_centroids_ptr, output_ptr,
        start_element, element_count, vector_start,
        packed_stride, norm_stride,
        dim: tl.constexpr, anchor_bits: tl.constexpr,
        delta_bits: tl.constexpr, target_layer,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < element_count
        global_elements = start_element + offsets
        vector_indices = vector_start + offsets // dim

        anchor_bit_positions = global_elements * anchor_bits
        anchor_bytes = anchor_bit_positions // 8
        anchor_shifts = anchor_bit_positions % 8
        anchor_low = tl.load(
            anchor_packed_ptr + anchor_bytes, mask=mask, other=0
        ).to(tl.int32)
        anchor_high = tl.load(
            anchor_packed_ptr + anchor_bytes + 1,
            mask=mask & (anchor_shifts + anchor_bits > 8), other=0,
        ).to(tl.int32)
        anchor_values = (
            (anchor_low >> anchor_shifts)
            | (anchor_high << (8 - anchor_shifts))
        ) & ((1 << anchor_bits) - 1)
        result = tl.load(anchor_centroids_ptr + anchor_values).to(tl.float32)
        result *= tl.load(anchor_norms_ptr + vector_indices).to(tl.float32)

        delta_bit_positions = global_elements * delta_bits
        delta_bytes = delta_bit_positions // 8
        delta_shifts = delta_bit_positions % 8
        for layer in tl.range(0, target_layer, loop_unroll_factor=1):
            row = layer * packed_stride
            low = tl.load(
                delta_packed_ptr + row + delta_bytes, mask=mask, other=0
            ).to(tl.int32)
            high = tl.load(
                delta_packed_ptr + row + delta_bytes + 1,
                mask=mask & (delta_shifts + delta_bits > 8), other=0,
            ).to(tl.int32)
            values = (
                (low >> delta_shifts) | (high << (8 - delta_shifts))
            ) & ((1 << delta_bits) - 1)
            centroids = tl.load(delta_centroids_ptr + values).to(tl.float32)
            norms = tl.load(
                delta_norms_ptr + layer * norm_stride + vector_indices
            ).to(tl.float32)
            result += centroids * norms
        tl.store(output_ptr + offsets, result, mask=mask)


    @triton.jit
    def _paged_kv_attention_kernel(
        query_ptr, key_packed_ptr, key_min_ptr, key_scale_ptr,
        value_packed_ptr, value_scale_ptr,
        max_ptr, sum_ptr, accumulator_ptr,
        scaling,
        heads: tl.constexpr, head_size: tl.constexpr,
        page_size: tl.constexpr, key_bits: tl.constexpr,
        value_bits: tl.constexpr, BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head = tl.program_id(0)
        page = tl.program_id(1)
        tokens = tl.arange(0, BLOCK_T)[:, None]
        dims = tl.arange(0, BLOCK_D)[None, :]
        mask = dims < head_size
        global_tokens = page * page_size + tokens
        flat = (global_tokens * heads + head) * head_size + dims

        key_positions = flat * key_bits
        key_bytes = key_positions // 8
        key_shifts = key_positions % 8
        key_low = tl.load(
            key_packed_ptr + key_bytes, mask=mask, other=0
        ).to(tl.int32)
        key_high = tl.load(
            key_packed_ptr + key_bytes + 1,
            mask=mask & (key_shifts + key_bits > 8), other=0,
        ).to(tl.int32)
        key_codes = (
            (key_low >> key_shifts) | (key_high << (8 - key_shifts))
        ) & ((1 << key_bits) - 1)
        metadata = (page * heads + head) * head_size + dims
        key_min = tl.load(key_min_ptr + metadata, mask=mask, other=0).to(tl.float32)
        key_scale = tl.load(
            key_scale_ptr + metadata, mask=mask, other=0
        ).to(tl.float32)
        keys = key_codes.to(tl.float32) * key_scale + key_min
        query = tl.load(
            query_ptr + head * head_size + dims, mask=dims < head_size, other=0
        ).to(tl.float32)
        scores = tl.sum(keys * query, axis=1) * scaling
        maximum = tl.max(scores, axis=0)
        weights = tl.exp(scores - maximum)
        denominator = tl.sum(weights, axis=0)

        value_positions = flat * value_bits
        value_bytes = value_positions // 8
        value_shifts = value_positions % 8
        value_low = tl.load(
            value_packed_ptr + value_bytes, mask=mask, other=0
        ).to(tl.int32)
        value_high = tl.load(
            value_packed_ptr + value_bytes + 1,
            mask=mask & (value_shifts + value_bits > 8), other=0,
        ).to(tl.int32)
        value_codes = (
            (value_low >> value_shifts) | (value_high << (8 - value_shifts))
        ) & ((1 << value_bits) - 1)
        value_qmax: tl.constexpr = (1 << (value_bits - 1)) - 1
        value_metadata = global_tokens * heads + head
        value_scale = tl.load(
            value_scale_ptr + value_metadata
        ).to(tl.float32)
        values = (value_codes.to(tl.float32) - value_qmax) * value_scale
        accumulator = tl.sum(weights[:, None] * values, axis=0)
        output_row = page * heads + head
        tl.store(max_ptr + output_row, maximum)
        tl.store(sum_ptr + output_row, denominator)
        tl.store(
            accumulator_ptr + output_row * head_size + dims,
            accumulator, mask=dims < head_size,
        )


def packed_dequantize(
    packed: torch.Tensor, norms: torch.Tensor, centroids: torch.Tensor,
    start_vector: int, count: int, dim: int, bits: int,
) -> torch.Tensor:
    """Fused unpack, codebook lookup, and norm scaling on CUDA."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not packed.is_cuda:
        raise ValueError("Triton packed dequantization requires CUDA tensors")
    output = torch.empty(
        count * dim, device=packed.device, dtype=torch.float32
    )
    elements = count * dim
    block = 256
    _packed_dequant_kernel[(triton.cdiv(elements, block),)](
        packed, norms.reshape(-1)[start_vector:], centroids, output,
        start_element=start_vector * dim,
        element_count=elements,
        dim=dim,
        bits=bits,
        BLOCK=block,
    )
    return output.reshape(count, dim)


def packed_chain_dequantize(
    anchor_packed: torch.Tensor, anchor_norms: torch.Tensor,
    anchor_centroids: torch.Tensor, delta_packed: torch.Tensor,
    delta_norms: torch.Tensor, delta_centroids: torch.Tensor,
    start_vector: int, count: int, dim: int, anchor_bits: int,
    delta_bits: int, target_layer: int,
) -> torch.Tensor:
    """Decode and sum an adjacent anchor/delta chain in one CUDA kernel."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not anchor_packed.is_cuda:
        raise ValueError("Triton packed chain decoding requires CUDA tensors")
    elements = count * dim
    output = torch.empty(elements, device=anchor_packed.device, dtype=torch.float32)
    block = 256
    _packed_chain_dequant_kernel[(triton.cdiv(elements, block),)](
        anchor_packed, anchor_norms.reshape(-1), anchor_centroids,
        delta_packed, delta_norms, delta_centroids, output,
        start_element=start_vector * dim,
        element_count=elements,
        vector_start=start_vector,
        packed_stride=delta_packed.stride(0),
        norm_stride=delta_norms.stride(0),
        dim=dim,
        anchor_bits=anchor_bits,
        delta_bits=delta_bits,
        target_layer=target_layer,
        BLOCK=block,
    )
    return output.reshape(count, dim)


def paged_kv_attention(
    query: torch.Tensor, key_packed: torch.Tensor, key_min: torch.Tensor,
    key_scale: torch.Tensor, value_packed: torch.Tensor,
    value_scale: torch.Tensor, cold_tokens: int, scaling: float,
    heads: int, head_size: int, page_size: int, key_bits: int,
    value_bits: int,
):
    """Return cold-page online-softmax max, denominator, and accumulator."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if cold_tokens % page_size:
        raise ValueError("direct packed attention requires complete cold pages")
    page_count = cold_tokens // page_size
    block_tokens = triton.next_power_of_2(page_size)
    block_dim = triton.next_power_of_2(head_size)
    maximum = torch.empty(
        page_count, heads, device=query.device, dtype=torch.float32
    )
    denominator = torch.empty_like(maximum)
    accumulator = torch.empty(
        page_count, heads, head_size, device=query.device, dtype=torch.float32
    )
    _paged_kv_attention_kernel[(heads, page_count)](
        query, key_packed, key_min, key_scale, value_packed, value_scale,
        maximum, denominator, accumulator,
        scaling=scaling,
        heads=heads, head_size=head_size, page_size=page_size,
        key_bits=key_bits, value_bits=value_bits,
        BLOCK_T=block_tokens, BLOCK_D=block_dim,
    )
    return maximum, denominator, accumulator
