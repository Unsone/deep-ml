import torch
import torch.nn.functional as F
from typing import Tuple

def train_simple_cnn_with_backprop(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    learning_rate: float,
    kernel_size: int = 3,
    num_filters: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1
    flattened_size = out_h * out_w * num_filters

    W_conv = (torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.01).requires_grad_(True)
    b_conv = torch.zeros(num_filters, requires_grad=True)

    W_dense = (torch.randn(flattened_size, num_classes) * 0.01).requires_grad_(True)
    b_dense = torch.zeros(num_classes, requires_grad=True)

    optimizer = torch.optim.SGD([W_conv, b_conv, W_dense, b_dense], lr=learning_rate)

    for epoch in range(epochs):
        for i in range(n_samples):
            optimizer.zero_grad()

            x_i = X[i].unsqueeze(0).unsqueeze(0).float()   # (1,1,H,W)
            y_i = y[i].unsqueeze(0).float()

            # 用 unfold + matmul 代替 conv2d，避免触发 NNPACK
            patches = F.unfold(x_i, kernel_size)            # (1, k*k, out_h*out_w)
            Z_conv = W_conv.view(num_filters, -1) @ patches[0]   # (nf, out_h*out_w)
            Z_conv = Z_conv + b_conv.view(-1, 1)
            ReLU_flat = F.relu(Z_conv).reshape(1, -1)       # (1, nf*out_h*out_w)

            Z_dense = ReLU_flat @ W_dense + b_dense
            loss = F.cross_entropy(Z_dense, y_i.argmax(dim=1))
            loss.backward()
            optimizer.step()

    return (
        W_conv.detach(),
        b_conv.detach(),
        W_dense.detach(),
        b_dense.detach()
    )