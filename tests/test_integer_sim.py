import pytest
import torch
import torch.nn.functional as F

from quant_pitfalls.quant.uniform import UniformSpec
from quant_pitfalls.sim.accumulator import AccumulatorSpec
from quant_pitfalls.sim.integer import int_conv2d, quantize_multiplier, requantize


@pytest.mark.parametrize("m", [0.0007, 0.013, 0.24, 0.9])
def test_quantize_multiplier(m):
    m0, n = quantize_multiplier(m)
    assert 2**30 <= m0 < 2**31
    assert m0 * 2.0**-n == pytest.approx(m, rel=2**-30)


def test_int_conv_matches_float_conv_on_codes():
    # With a 32-bit accumulator nothing overflows, so the integer sum equals the exact sum.
    g = torch.Generator().manual_seed(0)
    x_q = torch.randint(0, 256, (2, 16, 8, 8), generator=g)
    w_q = torch.randint(-7, 8, (32, 16, 3, 3), generator=g)
    acc = int_conv2d(x_q, w_q, padding=1, acc=AccumulatorSpec(32))
    ref = F.conv2d(x_q.double(), w_q.double(), padding=1)
    assert torch.equal(acc.double(), ref)


def test_requantize_rounds_half_up():
    out = UniformSpec(8, signed=False)
    m0, n = 2**30, 31  # M = 0.5 exactly
    acc = torch.tensor([1, 3, 5, -1, 600])
    assert requantize(acc, m0, n, out).tolist() == [1, 2, 3, 0, 255]
