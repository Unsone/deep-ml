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

    ============================================================
    为什么返回值(m, v)和实际用于更新参数的值(m_hat, v_hat)不一样？

    - m, v 是 Adam 的"持久状态"(persistent state), 需要在
      timestep 之间保存并递推:
          m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
          v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
      它们保存的是"未修正的原始矩估计"(biased moments), 与
      PyTorch Adam 的 state['exp_avg'] / state['exp_avg_sq'] 一致。

    - 偏差修正(bias correction)只是因为 m, v 初始化为 0, 在早期
      偏向 0, 所以临时算修正值用于本步更新:
          m_hat = m_t / (1 - beta1^t)
          v_hat = v_t / (1 - beta2^t)
          theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    - 如果把修正后的 m_hat, v_hat 写回状态并返回, 下一步递推就会在
      "已放大"的值上继续累积(因为 1-beta^t < 1, 修正后更大),
      导致后续更新偏离标准 Adam —— 相当于修正被重复累积进状态。
    ============================================================
    """
    parameter = torch.as_tensor(parameter, dtype=torch.float64)
    grad = torch.as_tensor(grad, dtype=torch.float64)
    m = torch.as_tensor(m, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64)

    # 1) 更新原始矩估计 (biased, 作为持久状态返回)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # 2) 偏差修正 —— 仅用于本步计算, 不写回状态
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # 3) 用修正值更新参数
    parameter = parameter - learning_rate * (m_hat / (torch.sqrt(v_hat) + epsilon))

    # 返回: 更新后的参数 + 未修正的 m, v (供下一步递推)
    return torch.round(parameter, decimals=5), torch.round(m, decimals=5), torch.round(v, decimals=5)