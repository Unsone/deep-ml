import torch

def adam_optimizer(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    t: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    :param parameter: Current parameter value (scalar or tensor)
    :param grad: Current gradient (scalar or tensor)
    :param m: First moment estimate (scalar or tensor)
    :param v: Second moment estimate (scalar or tensor)
    :param t: Current timestep
    :param learning_rate: Learning rate (default=0.001)
    :param beta1: First moment decay rate (default=0.9)
    :param beta2: Second moment decay rate (default=0.999)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_m, updated_v) as torch.Tensors
    """
    parameter = torch.as_tensor(parameter, dtype=torch.float64)
    grad = torch.as_tensor(grad, dtype=torch.float64)
    m = torch.as_tensor(m, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64)
    # Your code here
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    M = m / (1 - beta1 ** t)
    V = v / (1 - beta2 ** t)

    parameter = parameter - learning_rate * (M / (torch.sqrt(V) + epsilon))

    return torch.round(parameter, decimals=5), torch.round(m, decimals=5), torch.round(v, decimals=5)