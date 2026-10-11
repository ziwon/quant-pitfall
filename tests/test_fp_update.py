import pytest
import torch

from quant_pitfalls.analysis import vanishing_update_threshold


@pytest.mark.parametrize(
    "dtype, w, threshold",
    [
        (torch.float32, 0.75, 0.0298),
        (torch.float16, 0.75, 244.1),
        (torch.bfloat16, 0.75, 1953.1),
        (torch.float16, 0.01, 3.815),
    ],
)
def test_vanishing_update_threshold(dtype, w, threshold):
    # Pitfall 6: an SGD step lr * g below half an ulp rounds back to w.
    lr = 1e-6
    wt = torch.tensor(w, dtype=dtype)
    t = float(vanishing_update_threshold(wt, lr))
    assert t == pytest.approx(threshold, rel=1e-3)
    assert torch.equal(wt + lr * t * 0.98, wt)
    assert not torch.equal(wt + lr * t * 1.02, wt)
