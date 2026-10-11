"""Numerical checks behind notebooks 1-3 and 6.

MSE values are for unit-variance inputs, so they read in units of sigma^2.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Literal

import torch

Distribution = Literal["gaussian", "laplace", "student_t"]
Quantizer = Callable[[torch.Tensor], torch.Tensor]


def pdf(dist: Distribution, x: torch.Tensor, *, nu: float = 3.0) -> torch.Tensor:
    """Unit-variance density. Laplace uses scale 1/sqrt(2); Student-t is rescaled by sqrt((nu - 2) / nu)."""
    if dist == "gaussian":
        return torch.exp(-x.square() / 2) / math.sqrt(2 * math.pi)
    if dist == "laplace":
        return torch.exp(-math.sqrt(2) * x.abs()) / math.sqrt(2)
    if dist == "student_t":
        if not math.isfinite(nu) or nu <= 2:
            raise ValueError("A unit-variance Student-t requires finite nu > 2")
        log_normalizer = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
        return torch.exp(log_normalizer - (nu + 1) / 2 * torch.log1p(x.square() / (nu - 2)))
    raise ValueError(f"Unknown distribution: {dist}")


def _simpson_weights(intervals: int, step: float) -> torch.Tensor:
    weights = torch.full((intervals + 1,), 2.0, dtype=torch.float64)
    weights[1:-1:2] = 4
    weights[0] = weights[-1] = 1
    return weights * (step / 3)


@lru_cache(maxsize=16)
def _integration_grid(dist: Distribution, limit: float, intervals: int) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...]]:
    x = torch.linspace(-limit, limit, intervals + 1, dtype=torch.float64)
    weights = _simpson_weights(intervals, 2 * limit / intervals) * pdf(dist, x)
    # Tail moments M_k = integral_limit^inf x^k p(x) dx, with x = limit/u.
    u = torch.linspace(0, 1, intervals + 1, dtype=torch.float64)
    tail_weights = _simpson_weights(intervals, 1 / intervals)
    moments = []
    if dist == "student_t":
        # nu=3: p(x)=2/[pi*(1+x^2)^2]. This form stays finite at u=0.
        factor = (2 / math.pi) * limit / (u.square() + limit**2).square()
        for power in range(3):
            moments.append(torch.dot(tail_weights, factor * limit**power * u ** (2 - power)).item())
    else:
        tail_x = limit / u[1:]
        density = pdf(dist, tail_x)
        for power in range(3):
            integrand = torch.zeros_like(u)
            integrand[1:] = tail_x**power * density * limit / u[1:].square()
            moments.append(torch.dot(tail_weights, integrand).item())
    return x, weights, tuple(moments)


def expected_mse(quantizer: Quantizer, dist: Distribution = "gaussian", *, limit: float = 14.0, intervals: int = 24_000) -> float:
    """E[(Q(x) - x)^2] by Simpson's rule on [-limit, limit] in float64, plus the tails.

    The quantizer is piecewise constant, so keep many points per step. Beyond the
    limit Q is saturated, so the tails add 2 * int_limit^inf (x - Q(limit))^2 p(x) dx
    (integrate with x = limit / u). Both endpoints must lie in the saturated region;
    asymmetric saturation is handled independently on the two sides. Student-t uses
    nu=3. The quantizer is called once on the entire grid. The tails are negligible for Gaussian and Laplace
    but hold about 9% of the variance of a unit-variance Student-t with nu = 3.
    """
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("limit must be finite and positive")
    if not isinstance(intervals, int) or intervals < 2 or intervals % 2:
        raise ValueError("intervals must be an even integer of at least 2")
    x, weights, (m0, m1, m2) = _integration_grid(dist, float(limit), intervals)
    with torch.no_grad():
        q = quantizer(x)
        if q.shape != x.shape or not torch.isfinite(q).all():
            raise ValueError("The quantizer must return finite values with the input shape")
        interior = torch.dot((q - x).square(), weights).item()
        negative, positive = q[0].item(), q[-1].item()
    tails = 2 * m2 + 2 * (negative - positive) * m1 + (negative**2 + positive**2) * m0
    return interior + tails


def optimal_clip(make_quantizer: Callable[[float], Quantizer], dist: Distribution, lo: float, hi: float) -> tuple[float, float]:
    """(alpha*, MSE*) over alpha in [lo, hi] by successive grid refinement."""
    if not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo < hi:
        raise ValueError("Clip bounds must be finite with 0 < lo < hi")
    # Keep the integration endpoints beyond every candidate's saturation point.
    limit = max(14.0, 2 * hi)
    lower, upper = lo, hi
    best_alpha, best_mse = lo, math.inf
    for _ in range(4):
        candidates = torch.linspace(lower, upper, 33, dtype=torch.float64).tolist()
        errors = [expected_mse(make_quantizer(alpha), dist, limit=limit) for alpha in candidates]
        index = min(range(len(errors)), key=errors.__getitem__)
        if errors[index] < best_mse:
            best_alpha, best_mse = candidates[index], errors[index]
        step = (upper - lower) / 32
        lower = max(lo, best_alpha - step)
        upper = min(hi, best_alpha + step)
    return best_alpha, best_mse


def ulp(w: torch.Tensor) -> torch.Tensor:
    """Spacing in w's dtype; zero/subnormal values use the smallest subnormal step."""
    if not w.is_floating_point():
        raise TypeError("ulp expects a floating-point tensor")
    info = torch.finfo(w.dtype)
    magnitude = w.to(torch.float64).abs().clamp_min(info.tiny)
    return (torch.exp2(torch.floor(torch.log2(magnitude))) * info.eps).to(w.dtype)


def vanishing_update_threshold(w: torch.Tensor, lr: float) -> torch.Tensor:
    """ulp(w)/(2*lr), calculated in float64 to avoid low-precision division error.

    This is a local half-spacing criterion; a step across a power-of-two boundary
    can have different spacing on its two sides. Optimizer arithmetic is separate.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    return ulp(w).to(torch.float64) / (2 * lr)
