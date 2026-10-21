import torch

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> torch.Tensor:
    """
    Compute the GRPO objective function.
    """
    
    # ==========================================
    # 模块 0: 数据类型转换 (确保可以进行向量化计算)
    # ==========================================
    # 将输入统一转换为 float 类型的 Tensor
    rhos = torch.as_tensor(rhos, dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.float32)
    pi_theta_old = torch.as_tensor(pi_theta_old, dtype=torch.float32)
    pi_theta_ref = torch.as_tensor(pi_theta_ref, dtype=torch.float32)

    # ==========================================
    # 模块 1: 计算 PPO 风格的截断代理目标 (Clipped Surrogate Objective)
    # ==========================================
    # 1.1 未截断的目标: rho * A
    surr1 = rhos * A
    
    # 1.2 截断的目标: clamp(rho, 1-eps, 1+eps) * A
    rhos_clipped = torch.clamp(rhos, 1.0 - epsilon, 1.0 + epsilon)
    surr2 = rhos_clipped * A
    
    # 1.3 取两者中的最小值
    surrogate_objective = torch.min(surr1, surr2)

    # ==========================================
    # 模块 2: 计算重要性加权的 KL 散度惩罚
    # ==========================================
    # 题目公式: D_KL = rho_i * (pi_ref / pi_theta - log(pi_ref / pi_theta) - 1)
    # 且题目已知: pi_theta = rho * pi_theta_old
    
    # 2.1 计算当前策略概率 pi_theta
    pi_theta = rhos * pi_theta_old
    
    # 2.2 计算参考策略与当前策略的比值 (添加 1e-8 防止除以 0 导致 NaN)
    ref_ratio = pi_theta_ref / (pi_theta + 1e-8)
    
    # 2.3 计算 KL 散度项 (添加 1e-8 防止 log(0))
    kl_divergence = rhos * (ref_ratio - torch.log(ref_ratio + 1e-8) - 1)

    # ==========================================
    # 模块 3: 组合最终目标函数并求均值
    # ==========================================
    # 最终目标 = 截断代理目标 - beta * KL惩罚
    final_objective = surrogate_objective - beta * kl_divergence
    
    # 题目要求返回平均值 (average)
    return final_objective.mean()