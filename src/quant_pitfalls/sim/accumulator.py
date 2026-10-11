"""Fixed-width integer accumulation.

The worst case for K products with |w| <= w_max and |a| <= a_max needs
ceil(log2(K * w_max * a_max + 1)) + 1 signed bits: 28 for 8-bit weights times
8-bit unsigned activations over a 3x3x256 window.

Pitfall 9: saturating addition is not associative. With an 8-bit accumulator,
[100, 100, -100] saturates to 27 while [100, -100, 100] gives 100. Two's
complement wrap-around returns 100 for both because the final sum fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class AccumulatorSpec:
    bits: int = 32
    overflow: Literal["saturate", "wrap"] = "saturate"

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int) or not 1 <= self.bits <= 62:
            raise ValueError("Accumulator bits must be in [1, 62] (int64 intermediates)")
        if self.overflow not in ("saturate", "wrap"):
            raise ValueError("Overflow must be saturate or wrap")

    @property
    def minimum(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def maximum(self) -> int:
        return (1 << (self.bits - 1)) - 1


def required_bits(w_max: int, a_max: int, fan_in: int) -> int:
    """Signed accumulator width that can never overflow."""
    if any(not isinstance(value, int) or value < 0 for value in (w_max, a_max, fan_in)):
        raise ValueError("Bounds and fan_in must be nonnegative integers")
    # bit_length is ceil(log2(bound + 1)) without floating-point rounding.
    return (w_max * a_max * fan_in).bit_length() + 1


def _integer_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Integer codes must have an integer dtype")
    return values.to(torch.int64)


def _check_sum_bound(values: torch.Tensor, terms: int) -> None:
    if values.numel():
        low, high = int(values.min()), int(values.max())
        if max(abs(low), abs(high)) * terms >= 1 << 63:
            raise OverflowError("The exact sum bound exceeds int64")


def apply_overflow(values: torch.Tensor, spec: AccumulatorSpec) -> torch.Tensor:
    """Clamp or wrap an int64 tensor into a signed two's-complement range."""
    if spec.overflow == "saturate":
        return values.clamp(spec.minimum, spec.maximum)
    unsigned = torch.bitwise_and(values, (1 << spec.bits) - 1)
    return torch.where(unsigned > spec.maximum, unsigned - (1 << spec.bits), unsigned)


def _ordered_saturation(values: torch.Tensor, spec: AccumulatorSpec) -> torch.Tensor:
    """Compose ordered clipping functions, preserving every sequential clamp.

    f(x)=clamp(x+d, l, u). Composing f2 after f1 yields
    d=d1+d2, l=clamp(l1+d2,l2,u2), u=clamp(u1+d2,l2,u2).
    Function composition is associative; saturating *addition* is not.
    Pairing adjacent functions preserves the original term order exactly.
    """
    delta = values
    lower, upper = torch.full_like(delta, spec.minimum), torch.full_like(delta, spec.maximum)
    while delta.shape[-1] > 1:
        if delta.shape[-1] % 2:
            delta = torch.cat((delta, torch.zeros_like(delta[..., :1])), -1)
            lower = torch.cat((lower, torch.full_like(lower[..., :1], spec.minimum)), -1)
            upper = torch.cat((upper, torch.full_like(upper[..., :1], spec.maximum)), -1)
        right_delta = delta[..., 1::2]
        right_lower, right_upper = lower[..., 1::2], upper[..., 1::2]
        lower = torch.minimum(torch.maximum(lower[..., ::2] + right_delta, right_lower), right_upper)
        upper = torch.minimum(torch.maximum(upper[..., ::2] + right_delta, right_lower), right_upper)
        delta = delta[..., ::2] + right_delta
    return torch.minimum(torch.maximum(delta[..., 0], lower[..., 0]), upper[..., 0])


def accumulate(products: torch.Tensor, spec: AccumulatorSpec, dim: int = -1, *, parallel: bool = False) -> torch.Tensor:
    """Sum in increasing dim-index order, starting at zero, with per-add overflow.

    Saturation applies clamp after *every* addition; it must not clamp only
    the final sum. Wrap is associative modulo 2**bits, so one final wrap is
    equivalent. Inputs and outputs use int64, with a checked exact-sum bound.
    Convolution supplies terms in input-channel, kernel-row, kernel-column
    order (the order of unfold); an optional bias is the final term.
    parallel=True composes ordered clipping functions in a tree, preserving
    exactly the same per-add saturation while reducing GPU kernel launches.
    """
    values = _integer_tensor(products).movedim(dim, -1)
    _check_sum_bound(values, values.shape[-1])
    if spec.overflow == "wrap":
        return apply_overflow(values.sum(-1), spec)
    if parallel and values.shape[-1] > 1:
        bound = max(abs(int(values.min())), abs(int(values.max()))) * values.shape[-1]
        if bound + (1 << (spec.bits - 1)) < 1 << 63:
            return _ordered_saturation(values, spec)
    result = torch.zeros(values.shape[:-1], dtype=torch.int64, device=values.device)
    for term in values.unbind(-1):
        result = (result + term).clamp(spec.minimum, spec.maximum)
    return result
