"""Reversible hidden-width folding transforms for residual-stream codecs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FoldSpec:
    permutation: torch.Tensor
    alpha: torch.Tensor
    update: float = 0.0


def fit_fold(samples: torch.Tensor, mode: str, seed: int = 1234) -> FoldSpec:
    if samples.shape[-1] % 2:
        raise ValueError("folding requires an even hidden width")
    values = samples.float().reshape(-1, samples.shape[-1])
    left, right = values[:, 0::2], values[:, 1::2]
    count = left.shape[1]
    if mode == "adjacent_haar":
        return FoldSpec(torch.arange(count), torch.ones(count), update=0.5)
    if mode == "random_lifting":
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(count, generator=generator)
    elif mode == "correlation_lifting":
        left_centered = left - left.mean(0, keepdim=True)
        right_centered = right - right.mean(0, keepdim=True)
        left_unit = left_centered / left_centered.square().sum(0).sqrt().clamp_min(1e-12)
        right_unit = right_centered / right_centered.square().sum(0).sqrt().clamp_min(1e-12)
        correlation = (left_unit.T @ right_unit).abs()
        permutation = torch.empty(count, dtype=torch.long)
        available = torch.ones(count, dtype=torch.bool)
        # Greedy one-to-one matching is deterministic and cheap enough offline.
        order = correlation.max(1).values.argsort(descending=True)
        for row in order.tolist():
            scores = correlation[row].masked_fill(~available, -1)
            column = int(scores.argmax())
            permutation[row] = column
            available[column] = False
    else:
        raise ValueError(f"unknown fold mode: {mode}")
    paired = right[:, permutation]
    alpha = (left * paired).sum(0) / left.square().sum(0).clamp_min(1e-12)
    return FoldSpec(permutation, alpha, update=0.0)


def fold(values: torch.Tensor, spec: FoldSpec):
    left = values[..., 0::2].float()
    right = values[..., 1::2].float()[..., spec.permutation]
    detail = right - left * spec.alpha
    coarse = left + spec.update * detail
    return coarse, detail


def unfold(coarse: torch.Tensor, detail: torch.Tensor, spec: FoldSpec):
    left = coarse.float() - spec.update * detail.float()
    paired = detail.float() + left * spec.alpha
    inverse = torch.argsort(spec.permutation)
    right = paired[..., inverse]
    restored = torch.empty(
        (*left.shape[:-1], left.shape[-1] * 2), dtype=left.dtype,
        device=left.device,
    )
    restored[..., 0::2] = left
    restored[..., 1::2] = right
    return restored
