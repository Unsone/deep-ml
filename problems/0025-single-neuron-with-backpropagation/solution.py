import torch

def train_neuron(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    initial_weights: torch.Tensor, 
    initial_bias: float, 
    learning_rate: float, 
    epochs: int
    ) -> tuple[list[float], float, list[float]]:
    
    weights = initial_weights.clone().detach()
    bias = initial_bias
    mse_values = []
    N = features.shape[0]

    for epoch in range(epochs):
        # 前向传播
        predictions = torch.sigmoid(features @ weights + bias)
        error = predictions - labels
        mse = torch.mean(error ** 2)
        mse_values.append(mse.item())   # ← 转成 Python float

        # 反向传播
        dl_dp = 2 * error / N
        dp_dz = predictions * (1 - predictions)
        dl_dz = dl_dp * dp_dz
        dl_dw = features.transpose(0, 1) @ dl_dz   # 或 features.T
        dl_db = dl_dz.sum()

        # 更新参数
        weights = weights - learning_rate * dl_dw
        bias = bias - learning_rate * dl_db

    # 格式化输出
    final_weights = [round(w, 4) for w in weights.tolist()]
    final_bias = round(float(bias), 4)
    final_mse = [round(m, 4) for m in mse_values]

    return final_weights, final_bias, final_mse