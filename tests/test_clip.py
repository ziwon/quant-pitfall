import pytest

from quant_pitfalls.analysis import expected_mse, optimal_clip
from quant_pitfalls.quant.uniform import UniformSpec, fake_quant


def uniform(spec):
    return lambda alpha: (lambda x: fake_quant(x, alpha, spec))


def test_4bit_gaussian_optimal_clip():
    # Pitfall 1: 4-bit narrow range (-7..7) on N(0, 1).
    alpha, mse = optimal_clip(uniform(UniformSpec(4)), "gaussian", 1.5, 4.0)
    assert alpha == pytest.approx(2.474, abs=0.01)
    assert mse == pytest.approx(0.01289, rel=0.01)


def test_4bit_gaussian_fixed_clips():
    q = uniform(UniformSpec(4))
    assert expected_mse(q(4.0), "gaussian") == pytest.approx(0.02722, rel=0.01)
    assert expected_mse(q(5.0), "gaussian") == pytest.approx(0.04252, rel=0.01)


def test_floor_vs_nearest_8bit():
    # Pitfall 2: 8-bit full range (-128..127), each at its own optimal clip.
    a_near, near = optimal_clip(uniform(UniformSpec(8, narrow_range=False)), "gaussian", 3.0, 6.0)
    a_floor, floor = optimal_clip(uniform(UniformSpec(8, narrow_range=False, rounding="floor")), "gaussian", 3.0, 6.0)
    assert a_near == pytest.approx(3.90, abs=0.03)
    assert a_floor == pytest.approx(3.60, abs=0.03)
    assert floor / near == pytest.approx(3.45, abs=0.05)
