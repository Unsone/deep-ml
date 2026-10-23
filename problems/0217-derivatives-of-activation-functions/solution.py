import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    """
    
    # 1. 计算 Sigmoid 的导数
    # 创建叶子张量，requires_grad=True 表示需要计算梯度
    x_sigmoid = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_sigmoid = torch.sigmoid(x_sigmoid)
    y_sigmoid.backward()  # 反向传播计算梯度
    sigmoid_deriv = x_sigmoid.grad.item() # 提取标量值
    
    # 2. 计算 Tanh 的导数
    x_tanh = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_tanh = torch.tanh(x_tanh)
    y_tanh.backward()
    tanh_deriv = x_tanh.grad.item()
    
    # 3. 计算 ReLU 的导数
    x_relu = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_relu = torch.relu(x_relu)
    y_relu.backward()
    relu_deriv = x_relu.grad.item()
    
    # 构建并返回字典
    return {
        'sigmoid': sigmoid_deriv,
        'tanh': tanh_deriv,
        'relu': relu_deriv
    }