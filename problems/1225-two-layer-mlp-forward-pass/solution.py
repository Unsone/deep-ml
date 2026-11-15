import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output."""
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
    )

    # 只在写入参数时关闭梯度；用 copy_ 是原地拷贝，能保留正确的 dtype/device
    with torch.no_grad():
        model[0].weight.copy_(w1)
        model[0].bias.copy_(b1)
        model[2].weight.copy_(w2)
        model[2].bias.copy_(b2)

    out = model(x)          # x: (1, 2) -> out: (1, 1)
    return out.item()       # 标量 Python float