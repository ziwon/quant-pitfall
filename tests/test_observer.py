import pytest
import torch

from quant_pitfalls.quant.observer import EMAMinMaxObserver


@pytest.mark.parametrize("steps, expected", [(69, 1.5), (298, 1.95)])
def test_ema_lag(steps, expected):
    # Pitfall 5: c = 0.01 has a half-life of 69 steps and reaches 95% of a jump in 298.
    obs = EMAMinMaxObserver(averaging_constant=0.01)
    obs(torch.tensor([-1.0, 1.0]))
    for _ in range(steps):
        obs(torch.tensor([-2.0, 2.0]))
    assert float(obs.clip()) == pytest.approx(expected, abs=0.01)
