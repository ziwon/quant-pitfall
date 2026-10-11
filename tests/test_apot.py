import pytest
import torch

from quant_pitfalls.analysis import optimal_clip
from quant_pitfalls.quant.apot import apot_fake_quant, apot_levels
from quant_pitfalls.quant.uniform import UniformSpec, fake_quant


def test_apot_4bit_levels():
    expected = torch.tensor([0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 33, 36, 48], dtype=torch.float64) / 48
    assert torch.allclose(apot_levels(4, k=2).double(), expected)


@pytest.mark.parametrize(
    "dist, ratio, alpha_u, alpha_a",
    [("gaussian", 1.626, 2.90, 3.36), ("laplace", 0.755, 4.31, 5.73)],
)
def test_apot_vs_uniform_same_level_count(dist, ratio, alpha_u, alpha_a):
    # Pitfall 3: unsigned 4-bit magnitude plus sign (31 levels) for both quantizers.
    levels = apot_levels(4)
    mag = UniformSpec(4, signed=False)
    uniform = lambda a: (lambda x: torch.sign(x) * fake_quant(x.abs(), a, mag))
    apot = lambda a: (lambda x: apot_fake_quant(x, a, levels))
    au, mse_u = optimal_clip(uniform, dist, 1.5, 8.0)
    aa, mse_a = optimal_clip(apot, dist, 1.5, 9.0)
    assert au == pytest.approx(alpha_u, abs=0.03)
    assert aa == pytest.approx(alpha_a, abs=0.05)
    assert mse_a / mse_u == pytest.approx(ratio, abs=0.01)
