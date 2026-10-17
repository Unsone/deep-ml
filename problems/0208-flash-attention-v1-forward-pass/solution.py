import torch
import math

def flash_attention_forward(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    block_size: int = 2) -> torch.Tensor:
    """
    Flash Attention v1 前向传播实现
    """
    seq_len, d_model = Q.shape
    
    # ==========================================
    # 1. 初始化变量
    # ==========================================
    # softmax_numerator (分子): 累加过程中的 未归一化的 P * V
    softmax_numerator = torch.zeros((seq_len, d_model), device=Q.device)
    
    # softmax_denominator (分母): 累加过程中的 指数和 sum(exp)
    softmax_denominator = torch.zeros((seq_len, 1), device=Q.device)
    
    # global_max (全局最大值): 用于防止 exp() 溢出
    global_max = torch.full((seq_len, 1), float('-inf'), device=Q.device)

    # ==========================================
    # 2. 遍历 K 和 V 的块
    # ==========================================
    for j in range(0, seq_len, block_size):
        K_j = K[j : j + block_size, :]
        V_j = V[j : j + block_size, :]
        
        # --- 计算局部 Attention 分数 ---
        # scores 相当于公式里的 S (或 Z)
        scores = Q @ K_j.T / math.sqrt(d_model)
        
        # 提取当前块的局部最大值（注意加上 [0] 解包）
        block_max = scores.max(dim=1, keepdim=True)[0] 
        
        # 更新全局最大值
        new_global_max = torch.max(global_max, block_max)
        
        # 计算局部指数（减去最新的全局最大值防溢出）
        # block_exp 相当于局部块的 P
        block_exp = torch.exp(scores - new_global_max)
        
        # 计算历史结果的缩放因子 (如果最大值变了，老结果要按比例缩水)
        rescale_factor = torch.exp(global_max - new_global_max)
        
        # ==========================================
        # 3. 更新分子和分母
        # ==========================================
        # 更新分母：历史分母 * 缩放因子 + 当前块的指数和
        softmax_denominator = (softmax_denominator * rescale_factor) + block_exp.sum(dim=1, keepdim=True)
        
        # 更新分子：历史分子 * 缩放因子 + 当前块的 (P @ V)
        softmax_numerator = (softmax_numerator * rescale_factor) + (block_exp @ V_j)
        
        # 更新历史最大值
        global_max = new_global_max

    # ==========================================
    # 4. 最终归一化
    # ==========================================
    # 最后，把累计的分子除以累计的分母，得到真正的 Softmax 结果
    return softmax_numerator / softmax_denominator