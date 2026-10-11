"""Range observers and clip calibration.

Pitfall 5: an EMA observer m_t = (1 - c) m_{t-1} + c x_t keeps (1 - c)^t of its
initial value. With c = 0.01 the half-life is about 69 steps and covering 95%
of a jump takes about 298 steps (first integer step meeting the threshold: 299).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .uniform import UniformSpec, fake_quant


def _resize_loaded_buffers(module: nn.Module, state_dict: dict, prefix: str, names: tuple[str, ...]) -> None:
    """Allow fresh modules to restore the channel shape of calibrated buffers."""
    for name in names:
        value = state_dict.get(prefix + name)
        current = getattr(module, name)
        if value is not None and current.shape != value.shape:
            setattr(module, name, torch.empty_like(value, device=current.device))


def _minimum_alpha(dtype: torch.dtype, spec: UniformSpec | None = None) -> float:
    """Keep the scale normal and nonzero, including for all-zero channels."""
    return torch.finfo(dtype).tiny * (spec.qmax if spec is not None else 1)


class EMAMinMaxObserver(nn.Module):
    """Tracks an exponential moving average of the batch min and max.

    The first batch initializes the statistics directly; later batches are
    blended in with weight averaging_constant.
    """

    def __init__(self, averaging_constant: float = 0.01, *, symmetric: bool = True, axis: int | None = None) -> None:
        super().__init__()
        if not math.isfinite(averaging_constant) or not 0 < averaging_constant <= 1:
            raise ValueError("averaging_constant must be in (0, 1]")
        self.averaging_constant = averaging_constant
        self.symmetric = symmetric
        self.axis = axis
        self.register_buffer("min_val", torch.empty(0))
        self.register_buffer("max_val", torch.empty(0))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        _resize_loaded_buffers(self, state_dict, prefix, ("min_val", "max_val"))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update the running min/max from x and return x unchanged."""
        if not x.is_floating_point():
            raise TypeError("Observers expect a floating-point tensor")
        if x.numel() == 0:
            return x
        values = x.detach()
        if values.dtype in (torch.float16, torch.bfloat16):
            values = values.float()
        if not torch.isfinite(values).all():
            raise ValueError("Observer inputs must be finite")
        if self.axis is None:
            low, high = values.amin(), values.amax()
        else:
            if not -values.ndim <= self.axis < values.ndim:
                raise ValueError("axis is outside the input dimensions")
            axis = self.axis % values.ndim
            dims = tuple(i for i in range(values.ndim) if i != axis)
            low = values.amin(dims) if dims else values
            high = values.amax(dims) if dims else values
        if self.min_val.numel() == 0:
            self.min_val, self.max_val = low.clone(), high.clone()
        else:
            if self.min_val.shape != low.shape:
                raise ValueError("The observed channel count changed after initialization")
            self.min_val = self.min_val.to(low)
            self.max_val = self.max_val.to(high)
            self.min_val.lerp_(low, self.averaging_constant)
            self.max_val.lerp_(high, self.averaging_constant)
        return x

    def clip(self) -> torch.Tensor:
        """Symmetric max(|min|, |max|), or max(max, 0) for unsigned zero-point-0 data."""
        if self.min_val.numel() == 0:
            raise RuntimeError("Observe at least one nonempty batch before requesting a clip")
        return torch.maximum(self.min_val.abs(), self.max_val.abs()) if self.symmetric else self.max_val.clamp_min(0)


def mse_clip_search(x: torch.Tensor, spec: UniformSpec, *, num_candidates: int = 100, axis: int | None = None) -> torch.Tensor:
    """Search 0.05..1 times the observed range; ties select the smaller clip.

    Each channel is searched independently when axis is specified. All-zero
    ranges use a positive minimum scale. Calibration never records a gradient.
    """
    if not x.is_floating_point() or x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("Calibration needs a nonempty finite floating-point tensor")
    if not isinstance(num_candidates, int) or num_candidates < 2:
        raise ValueError("num_candidates must be an integer of at least 2")
    if axis is not None and not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is outside the input dimensions")
    with torch.no_grad():
        if axis is None:
            samples = x.detach().reshape(1, -1)
        else:
            moved = x.detach().movedim(axis, 0)
            samples = moved.reshape(moved.shape[0], -1)
        minimum = _minimum_alpha(x.dtype, spec)
        magnitudes = samples.abs() if spec.signed else samples.clamp_min(0)
        maximum = magnitudes.amax(dim=1).clamp_min(minimum)
        best_alpha = maximum.clone()
        best_error = torch.full_like(maximum, float("inf"), dtype=torch.float64)
        for ratio in torch.linspace(0.05, 1.0, num_candidates, dtype=torch.float64).tolist():
            alpha = (maximum * ratio).clamp_min(minimum)
            restored = fake_quant(samples, alpha, spec, axis=0)
            error = (restored - samples).double().square().mean(dim=1)
            improved = error < best_error
            best_alpha = torch.where(improved, alpha, best_alpha)
            best_error = torch.minimum(error, best_error)
    return best_alpha[0] if axis is None else best_alpha
