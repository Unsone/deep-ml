import torch
import math

def flash_attention_forward(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    block_size: int = 2) -> torch.Tensor:
    
    seq_len, d_model = Q.shape
    
    # 初始化输出 O，分母 l，以及全局最大值 Z_max_pre
    attention = torch.zeros((seq_len, d_model), device=Q.device)
    l = torch.zeros((seq_len, 1), device=Q.device)
    Z_max_pre = torch.full((seq_len, 1), float('-inf'), device=Q.device)

    for j in range(0, seq_len, block_size):
        # 1. 提取 K 和 V 的当前块
        K_j = K[j : j + block_size, :]
        V_j = V[j : j + block_size, :]
        
        # 2. 计算当前 Q 与 K_j 的局部注意力分数
        # 注意：这里除法写在外面等同于 Q @ (K_j.T / sqrt(d))
        Z = Q @ K_j.T / math.sqrt(d_model)
        
        # 3. 找出当前块的最大值
        # 【修复点】Z.max() 返回 (values, indices)，必须加 [0] 取 values
        Z_max_j = Z.max(dim=1, keepdim=True)[0] 
        
        # 4. 更新全局最大值
        Z_max = torch.max(Z_max_pre, Z_max_j)
        
        # 5. 计算局部指数（减去最新的全局最大值以防溢出）
        Z_exp = torch.exp(Z - Z_max)
        
        # 6. 计算缩放因子：因为全局最大值可能变大了，之前的累积结果需要按比例缩小
        scale_factor = torch.exp(Z_max_pre - Z_max)
        
        # 7. 更新分母（过去的分母 * 缩放因子 + 当前块的指数和）
        l = l * scale_factor + Z_exp.sum(dim=1, keepdim=True)
        
        # 8. 更新分子输出（过去的输出 * 缩放因子 + 当前块的 P @ V_j）
        attention = attention * scale_factor + Z_exp @ V_j
        
        # 9. 更新历史最大值
        Z_max_pre = Z_max

    # 【修复点】循环结束后，必须除以分母 l 进行最终的 Softmax 归一化
    return attention / l 