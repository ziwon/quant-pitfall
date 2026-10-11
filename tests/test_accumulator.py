import pytest
import torch

from quant_pitfalls.sim.accumulator import AccumulatorSpec, accumulate, required_bits


@pytest.mark.parametrize(
    "w_max, a_max, fan_in, bits",
    [(127, 255, 3 * 3 * 256, 28), (7, 255, 960, 22), (7, 255, 9, 15)],
)
def test_required_bits(w_max, a_max, fan_in, bits):
    assert required_bits(w_max, a_max, fan_in) == bits


def test_saturation_is_order_dependent():
    # Pitfall 9: saturating addition is not associative; wrap-around is exact when the final sum fits.
    sat = AccumulatorSpec(bits=8, overflow="saturate")
    wrap = AccumulatorSpec(bits=8, overflow="wrap")
    a = torch.tensor([100, 100, -100])
    b = torch.tensor([100, -100, 100])
    assert accumulate(a, sat).item() == 27
    assert accumulate(b, sat).item() == 100
    assert accumulate(a, wrap).item() == 100
    assert accumulate(b, wrap).item() == 100
