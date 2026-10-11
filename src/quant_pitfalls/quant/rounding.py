"""Rounding modes and the straight-through estimator (STE).

Pitfall 2: floor rounding has error U(-s, 0), so its MSE is s^2/3 against s^2/12
for round-to-nearest, and the -s/2 bias does not cancel when values are summed.
"""

from __future__ import annotations

from typing import Literal

import torch

Rounding = Literal["nearest", "floor"]


def round_nearest(x: torch.Tensor) -> torch.Tensor:
    """Round half up: floor(x + 0.5).

    torch.round rounds half to even. Integer kernels usually round by adding half
    an LSB and shifting, which rounds half up, so the fake-quant path and the
    integer simulator both use this rule to stay bit-exact on ties.
    """
    return torch.floor(x + 0.5)


def round_floor(x: torch.Tensor) -> torch.Tensor:
    """Round toward negative infinity, like an arithmetic right shift."""
    return torch.floor(x)


def round_ste(x: torch.Tensor, mode: Rounding = "nearest") -> torch.Tensor:
    """Round in the forward pass and pass the gradient through unchanged (Bengio et al., 2013)."""
    if mode == "nearest":
        rounded = round_nearest(x)
    elif mode == "floor":
        rounded = round_floor(x)
    else:
        raise ValueError(f"Unknown rounding mode: {mode}")
    return x + (rounded - x).detach()
