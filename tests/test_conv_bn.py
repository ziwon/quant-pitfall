import torch
from torch import nn

from quant_pitfalls.layers.conv_bn import QuantConvBn2d


def trained_pair():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    bn = nn.BatchNorm2d(16)
    with torch.no_grad():
        bn.running_mean.uniform_(-1, 1)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.uniform_(-1, 1)
    return conv, bn


def test_folding_matches_conv_bn_in_eval():
    # Pitfall 4: without a weight quantizer, the folded layer equals Conv + BN at inference.
    conv, bn = trained_pair()
    x = torch.randn(4, 8, 16, 16)
    ref = bn.eval()(conv(x))
    layer = QuantConvBn2d.from_float(conv, bn).eval()
    assert torch.allclose(layer(x), ref, atol=1e-5)


def test_frozen_bn_stats_survive_train_mode():
    # Pitfall 7: model.train() must not undo freeze_bn_stats().
    conv, bn = trained_pair()
    layer = QuantConvBn2d.from_float(conv, bn)
    layer.freeze_bn_stats()
    layer.train()
    mean, var = layer.bn.running_mean.clone(), layer.bn.running_var.clone()
    layer(torch.randn(8, 8, 16, 16))
    assert torch.equal(layer.bn.running_mean, mean)
    assert torch.equal(layer.bn.running_var, var)
