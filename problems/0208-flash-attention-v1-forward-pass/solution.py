import torch
import math

def flash_attention_forward(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    block_size: int = 2) -> torch.Tensor:
    """
    Flash Attention v1 前向传播实现 (复习版)
    
    【核心思想】：
    传统的 Attention 需要生成完整的 (N, N) 注意力矩阵，显存开销为 O(N^2)。
    Flash Attention 将 K 和 V 按行切块 (Tiling)，每次只算一小块分数，
    并通过“在线 Softmax (Online Softmax)”动态更新结果，显存开销降为 O(N)。
    """
    seq_len, d_model = Q.shape
    
    # ==========================================
    # 1. 初始化变量 (使用直观的命名法)
    # ==========================================
    # softmax_numerator (分子): 累加过程中未归一化的 P * V 结果
    # 为什么 V 在这里？因为数学上 Attention = (e^S / 分母) @ V = (e^S @ V) / 分母。
    # 为了避免过早除法导致分块无法拼合，我们只累加乘法部分。
    softmax_numerator = torch.zeros((seq_len, d_model), device=Q.device)
    
    # softmax_denominator (分母): 累加过程中的指数和 sum(exp)
    softmax_denominator = torch.zeros((seq_len, 1), device=Q.device)
    
    # global_max (全局最大值): 用于防止 exp() 数值溢出
    # 初始化为负无穷，保证第一块数据进来时，rescale_factor = exp(-inf - new_max) = 0
    global_max = torch.full((seq_len, 1), float('-inf'), device=Q.device)

    # ==========================================
    # 2. 遍历 K 和 V 的块 (Tiling)
    # ==========================================
    for j in range(0, seq_len, block_size):
        # 每次取出 block_size 行 K 和 V，形状为 (block_size, d_model)
        K_j = K[j : j + block_size, :]
        V_j = V[j : j + block_size, :]
        
        # ==========================================
        # 3. 计算局部 Attention 分数
        # ==========================================
        # Q: (seq_len, d_model), K_j.T: (d_model, block_size)
        # 结果 scores: (seq_len, block_size)，只生成了个小矩阵，不爆显存！
        scores = Q @ K_j.T / math.sqrt(d_model)
        
        # 【易错点回顾】：.max() 返回的是一个 tuple (values, indices)，
        # 必须加上 [0] 提取 values，否则 torch.maximum 会报错类型不匹配！
        block_max = scores.max(dim=1, keepdim=True)[0] 
        
        # 比较历史最大值和当前块的最大值，取较大者作为新的全局最大值
        new_global_max = torch.maximum(global_max, block_max)
        
        # 计算局部指数（减去最新的全局最大值 global_max，防止 exp() 溢出）
        # block_exp 相当于局部块的 P，形状 (seq_len, block_size)
        block_exp = torch.exp(scores - new_global_max)
        
        # 【核心魔法 - 缩放因子】：
        # 如果全局最大值变大了（从 global_max 变成 new_global_max），
        # 之前基于老最大值算出来的累加结果就“偏大”了，必须乘上这个因子缩回去，保持数学等价。
        rescale_factor = torch.exp(global_max - new_global_max)
        
        # ==========================================
        # 4. 更新分子和分母 (坚决不在此处做除法！)
        # ==========================================
        # 更新分母：历史分母 * 缩放因子 + 当前块的指数和
        softmax_denominator = (softmax_denominator * rescale_factor) + block_exp.sum(dim=1, keepdim=True)
        
        # 更新分子：历史分子 * 缩放因子 + 当前块的 (P @ V)
        softmax_numerator = (softmax_numerator * rescale_factor) + (block_exp @ V_j)
        
        # 更新历史最大值
        global_max = new_global_max

    # ==========================================
    # 5. 最终归一化 (最后的一步除法)
    # ==========================================
    # 循环结束后，分子和分母都累加完毕。
    # 此时执行一次除法，完美等价于标准的 Softmax(S) @ V
    return softmax_numerator / softmax_denominator