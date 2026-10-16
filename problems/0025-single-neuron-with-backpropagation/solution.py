import torch

def train_neuron(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    initial_weights: torch.Tensor, 
    initial_bias: float, 
    learning_rate: float, 
    epochs: int
    ) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.
    """

    # ============================================================
    # 【坑 1】为什么 weights 要 .clone().detach()，而 bias 不用？
    # ------------------------------------------------------------
    # - weights 是 torch.Tensor（可变对象），多个变量名可指向同一块内存。
    #   不 clone 的话，若后续发生原地修改（如 weights -= ...），
    #   外部传入的 initial_weights 会被污染。
    # - .detach() 是为了切断计算图：
    #   即便本题是"手动反向传播"、没调用 loss.backward()，
    #   但如果外部传入的 initial_weights 带 requires_grad=True，
    #   PyTorch 仍会自动构建计算图，在 epochs 很大时可能内存泄漏。
    #   .detach() 是防御性编程，让 weights 变成纯数据张量。
    # - bias 是 Python float（不可变对象），
    #   `bias = bias - lr * dl_db` 会创建新对象，不会污染原始值；
    #   且 float 本身没有计算图概念，所以不需要 clone/detach。
    #   （如果题目把 initial_bias 也定义成 tensor，就必须同样处理！）
    # ============================================================
    weights = initial_weights.clone().detach()
    bias = initial_bias
    mse_values = []
    N = features.shape[0]

    for epoch in range(epochs):
        # ---------- 前向传播 ----------
        # z = XW + b, a = sigmoid(z)
        predictions = torch.sigmoid(features @ weights + bias)
        error = predictions - labels

        # ---------- 计算 MSE 损失 ----------
        # L = mean((a - y)^2)
        mse = torch.mean(error ** 2)

        # ============================================================
        # 【坑 2】mse 是 torch.Tensor，不是 Python float！
        # 题目要求返回 list[float]，所以要用 .item() 取出 Python 标量。
        # 虽然 round(tensor) 也能工作，但返回的 list 里装的是 tensor，
        # 不符合类型注解 list[float]，测试可能失败。
        # ============================================================
        mse_values.append(mse.item())

        # ---------- 反向传播（手动链式法则） ----------
        # 链式法则：dL/dw = dL/da * da/dz * dz/dw
        #          dL/db = dL/da * da/dz * dz/db

        # 第一步：dL/da
        # L = mean((a - y)^2)  =>  dL/da = 2(a - y) / N
        dl_dp = 2 * error / N

        # 第二步：da/dz
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)
        dp_dz = predictions * (1 - predictions)

        # 第三步：合并 dL/dz
        dl_dz = dl_dp * dp_dz

        # ============================================================
        # 【坑 3】为什么是 features.T @ dl_dz，而不是反过来？
        # ------------------------------------------------------------
        # 维度匹配决定顺序：
        #   features: (N, D),  dl_dz: (N,)
        #   目标 dL_dw 形状必须和 weights 一致，即 (D,)
        #   features.T (D, N) @ dl_dz (N,) -> (D,) ✅
        #
        # 数学依据：dL/dW_j = Σ_i (dL/dz_i) * X_ij
        # 即对第 j 个权重的梯度，要把所有样本 i 的贡献按 X_ij 加权求和。
        #
        # 一句话记忆：谁对谁求导，结果长得像谁。
        # W 的梯度要长得像 W，所以用 X^T 去乘 dl_dz，把 N 维消掉。
        # 反过来写 (dl_dz @ features.T) 会直接维度报错。
        # ============================================================
        dl_dw = features.T @ dl_dz

        # ============================================================
        # 【坑 4】为什么 bias 的梯度要 .sum()？
        # ------------------------------------------------------------
        # z_i = Σ_j X_ij W_j + b，b 对所有样本 i 共享同一个值。
        # dL/db = Σ_i (dL/dz_i) * (dz_i/db) = Σ_i (dL/dz_i) * 1
        # 所以要把所有样本的 dl_dz 求和。
        #
        # 直观理解：b 是共享的"基准线"，每个样本都在对它提意见，
        # 要听取所有样本的意见，加起来才决定 b 该怎么调。
        #
        # 对比 weights：每个 W_j 只被一个特征维度影响，
        # 矩阵乘法 X^T @ dl_dz 已经自动在样本维度上做了加权求和，
        # 所以 weights 不需要额外 .sum()。
        # ============================================================
        dl_db = dl_dz.sum()

        # ---------- 梯度下降更新参数 ----------
        weights = weights - learning_rate * dl_dw
        bias = bias - learning_rate * dl_db

    # ============================================================
    # 【坑 5】返回值必须做类型转换 + 四舍五入！
    # ------------------------------------------------------------
    # 题目原文："each rounded to four decimal places"
    # - weights: tensor -> list[float]，每个元素 round 4 位
    # - bias: 更新后是 tensor（因为 dl_db 是 tensor），要 float() 再 round
    # - mse_values: 已经是 list[float]（因为用了 .item()），只需 round
    # ============================================================
    final_weights = [round(w, 4) for w in weights.tolist()]
    final_bias = round(float(bias), 4)
    final_mse = [round(m, 4) for m in mse_values]

    return final_weights, final_bias, final_mse