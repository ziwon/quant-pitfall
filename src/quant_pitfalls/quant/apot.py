"""Additive powers-of-two (APoT) levels, Li et al. (ICLR 2020).

Each level is a sum of n = bits / k terms, p_i in {0, 2^-i, 2^-(i+n), ...,
2^-(i+(2^k-2)n)}, scaled so the largest level equals alpha. bits=4, k=2 gives
{0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 33, 36, 48} / 48.

Pitfall 3: against a uniform grid with the same number of levels, each at its
MSE-optimal clip, APoT loses on a Gaussian (MSE ratio 1.63) and wins on a
Laplace (0.755).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class APoTSpec:
    bits: int  # magnitude bits; a sign is added on top when signed
    k: int = 2
    signed: bool = True


def apot_levels(bits: int, k: int = 2) -> torch.Tensor:
    """Unsigned levels normalized to [0, 1], sorted and deduplicated. bits must be a multiple of k."""
    if not isinstance(bits, int) or not isinstance(k, int) or k <= 0 or bits <= 0 or bits % k:
        raise ValueError("bits and k must be positive integers, with bits a multiple of k")
    n = bits // k
    levels = torch.zeros(1, dtype=torch.float64)
    for i in range(n):
        terms = torch.tensor([0.0] + [2.0 ** -(i + j * n) for j in range(2**k - 1)], dtype=torch.float64)
        levels = (levels[:, None] + terms[None, :]).flatten().unique(sorted=True)
    return levels / levels[-1]


def _clip(x: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError("APoT expects a floating-point tensor")
    clip = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    if not torch.isfinite(clip).all() or not (clip > 0).all():
        raise ValueError("alpha must be finite and positive")
    return clip


def apot_quantize(x: torch.Tensor, alpha: torch.Tensor | float, levels: torch.Tensor, *, signed: bool = True) -> torch.Tensor:
    """Clip and map to the nearest normalized level, preserving input dtype/device.

    Exact midpoint ties choose the larger magnitude (away from zero when signed).
    Unsigned quantization clips negative inputs to zero.
    """
    clip = _clip(x, alpha)
    grid = levels.to(dtype=x.dtype, device=x.device)
    if grid.ndim != 1 or grid.numel() < 2 or not torch.isfinite(grid).all():
        raise ValueError("levels must be a finite one-dimensional grid with at least two levels")
    if grid[0] != 0 or grid[-1] != 1 or not (grid[1:] > grid[:-1]).all():
        raise ValueError("levels must increase strictly from 0 to 1")
    magnitude = x.abs() if signed else x.clamp_min(0)
    normalized = (magnitude / clip).clamp(0, 1)
    midpoints = (grid[:-1] + grid[1:]) / 2
    indices = torch.bucketize(normalized.contiguous(), midpoints, right=True)
    result = grid[indices] * clip
    return result * x.sign() if signed else result


def apot_fake_quant(x: torch.Tensor, alpha: torch.Tensor | float, levels: torch.Tensor, *, signed: bool = True) -> torch.Tensor:
    """APoT forward values with gradient 1 inside the clip range and 0 outside."""
    clip = _clip(x, alpha)
    lower = -clip if signed else torch.zeros_like(clip)
    clipped = torch.where(x < lower, lower, torch.where(x > clip, clip, x))
    quantized = apot_quantize(x, clip, levels, signed=signed)
    return quantized.detach() + (clipped - clipped.detach())
