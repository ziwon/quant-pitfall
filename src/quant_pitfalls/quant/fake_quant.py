"""Observer plus quantizer with independent switches.

The method names follow torch.ao.quantization.FakeQuantize, so a staged
schedule reads the same as in PyTorch: observe first, then quantize, then
freeze the observer.
"""

from __future__ import annotations

import torch
from torch import nn

from .apot import APoTSpec, apot_fake_quant, apot_levels
from .observer import EMAMinMaxObserver, _minimum_alpha, _resize_loaded_buffers
from .uniform import UniformSpec, fake_quant


class FakeQuantize(nn.Module):
    def __init__(self, spec: UniformSpec | APoTSpec, observer: nn.Module | None = None, *, axis: int | None = None) -> None:
        super().__init__()
        if not isinstance(spec, (UniformSpec, APoTSpec)):
            raise TypeError("spec must be a UniformSpec or APoTSpec")
        self.spec = spec
        self.axis = axis
        self.observer = observer if observer is not None else EMAMinMaxObserver(symmetric=spec.signed, axis=axis)
        if not callable(getattr(self.observer, "clip", None)):
            raise TypeError("The observer must provide a clip() method")
        # A uniform quantizer starts at step 1 until observation supplies a range.
        self.register_buffer("alpha", torch.tensor(float(spec.qmax) if isinstance(spec, UniformSpec) else 1.0))
        self.register_buffer("observer_enabled", torch.tensor(True))
        self.register_buffer("fake_quant_enabled", torch.tensor(True))
        self.register_buffer("levels", apot_levels(spec.bits, spec.k) if isinstance(spec, APoTSpec) else torch.empty(0))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        _resize_loaded_buffers(self, state_dict, prefix, ("alpha",))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled.fill_(enabled)

    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled.fill_(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update the observer if enabled, then fake-quantize if enabled."""
        if bool(self.observer_enabled) and x.numel() != 0:
            with torch.no_grad():
                self.observer(x.detach())
                dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
                clip = torch.as_tensor(self.observer.clip(), device=x.device, dtype=dtype)
                if not torch.isfinite(clip).all() or not (clip >= 0).all():
                    raise ValueError("The observer must supply a finite nonnegative clip")
                minimum = _minimum_alpha(x.dtype, self.spec if isinstance(self.spec, UniformSpec) else None)
                self.alpha = clip.detach().clamp_min(minimum).clone()
        if not bool(self.fake_quant_enabled) or x.numel() == 0:
            return x
        if isinstance(self.spec, UniformSpec):
            return fake_quant(x, self.alpha, self.spec, axis=self.axis)
        alpha = self.alpha
        if alpha.ndim != 0:
            if self.axis is None or not -x.ndim <= self.axis < x.ndim:
                raise ValueError("Per-channel APoT needs a valid axis")
            axis = self.axis % x.ndim
            if alpha.ndim != 1 or alpha.numel() != x.shape[axis]:
                raise ValueError("alpha must match the channel count")
            shape = [1] * x.ndim
            shape[axis] = alpha.numel()
            alpha = alpha.reshape(shape)
        return apot_fake_quant(x, alpha, self.levels, signed=self.spec.signed)
