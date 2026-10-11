import torch

from quant_pitfalls.quant.rounding import round_floor, round_nearest, round_ste


def test_rounding_error_moments():
    # Pitfall 2, in units of the step: nearest error ~ U(-1/2, 1/2), floor error ~ U(-1, 0).
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1_000_000, generator=g, dtype=torch.float64) * 100
    e_near = round_nearest(x) - x
    e_floor = round_floor(x) - x
    assert abs(e_near.mean().item()) < 2e-3
    assert abs(e_floor.mean().item() + 0.5) < 2e-3
    assert abs((e_near**2).mean().item() - 1 / 12) < 2e-3
    assert abs((e_floor**2).mean().item() - 1 / 3) < 2e-3


def test_round_nearest_ties_go_up():
    x = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5])
    assert round_nearest(x).tolist() == [-1.0, 0.0, 1.0, 2.0, 3.0]


def test_ste_passes_gradient():
    x = torch.linspace(-3, 3, 13, requires_grad=True)
    round_ste(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
