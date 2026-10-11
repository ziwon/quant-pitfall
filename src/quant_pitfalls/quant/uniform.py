"""Uniform quantizer with symmetric clipping and zero-point 0.

    s = alpha / q_max,  q = clip(round(x / s), q_min, q_max),  x_hat = s * q

Pitfall 1: with 4-bit narrow range (-7..7) on N(0, 1) the MSE-optimal clip is
alpha* = 2.47 sigma (MSE 0.0129 sigma^2); min/max clipping lands well above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rounding import Rounding, round_floor, round_nearest, round_ste


@dataclass(frozen=True)
class UniformSpec:
    bits: int
    signed: bool = True
    narrow_range: bool = True  # signed only: drop -2^(b-1) so the grid is symmetric
    rounding: Rounding = "nearest"

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int) or self.bits < (2 if self.signed else 1):
            raise ValueError("bits must be at least 2 when signed, or 1 when unsigned")
        if self.rounding not in ("nearest", "floor"):
            raise ValueError(f"Unknown rounding mode: {self.rounding}")

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.signed else 2**self.bits - 1

    @property
    def qmin(self) -> int:
        if not self.signed:
            return 0
        return -self.qmax if self.narrow_range else -(2 ** (self.bits - 1))


def _scale(x: torch.Tensor, alpha: torch.Tensor | float, spec: UniformSpec, axis: int | None) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError("The uniform quantizer expects a floating-point tensor")
    clip = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    if not torch.isfinite(clip).all() or not (clip > 0).all():
        raise ValueError("alpha must be finite and positive")
    if axis is not None:
        if not -x.ndim <= axis < x.ndim:
            raise ValueError("axis is outside the input dimensions")
        axis %= x.ndim
    if clip.ndim != 0:
        if axis is None or clip.ndim != 1 or clip.numel() != x.shape[axis]:
            raise ValueError("Per-channel alpha must be a vector matching the specified axis")
        shape = [1] * x.ndim
        shape[axis] = clip.numel()
        clip = clip.reshape(shape)
    return clip / spec.qmax


def quantize(x: torch.Tensor, alpha: torch.Tensor | float, spec: UniformSpec, *, axis: int | None = None) -> torch.Tensor:
    """Return integer codes q (in a float tensor). alpha is a scalar, or per-channel along axis."""
    scaled = x / _scale(x, alpha, spec, axis)
    rounded = round_nearest(scaled) if spec.rounding == "nearest" else round_floor(scaled)
    return rounded.clamp(spec.qmin, spec.qmax)


def dequantize(q: torch.Tensor, alpha: torch.Tensor | float, spec: UniformSpec, *, axis: int | None = None) -> torch.Tensor:
    """Map codes back to real values: s * q."""
    return q * _scale(q, alpha, spec, axis)


def fake_quant(x: torch.Tensor, alpha: torch.Tensor | float, spec: UniformSpec, *, axis: int | None = None) -> torch.Tensor:
    """Quantize and dequantize; the gradient is 1 inside the clip range and 0 outside (STE)."""
    scale = _scale(x, alpha, spec, axis)
    # Clamp before rounding so even a value just outside the range has zero gradient.
    scaled = x / scale
    scaled = torch.where(scaled < spec.qmin, spec.qmin, torch.where(scaled > spec.qmax, spec.qmax, scaled))
    return round_ste(scaled, spec.rounding) * scale
