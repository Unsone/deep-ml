import torch
import math

def flash_attention_forward(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    block_size: int = 2) -> torch.Tensor:
    """
    Flash Attention v1 前向传播实现
    
    核心思想：
    传统的 Attention 需要生成完整的 (N, N) 注意力矩阵，显存开销为 O(N^2)。
    Flash Attention 将 K 和 V 切块，每次只算一小块 (N, block_size) 的分数，
    并通过“在线 Softmax (Online Softmax)”动态更新结果，显存开销降为 O(N)。
    """
    seq_len, d_model = Q.shape
    
    # ==========================================
    # 1. 初始化全局变量 (Running Statistics)
    # ==========================================
    # attention / O: 最终输出（分子部分，尚未除以分母），初始化为全 0
    attention = torch.zeros((seq_len, d_model), device=Q.device)
    
    # l: Softmax 的分母（指数和），初始化为全 0
    l = torch.zeros((seq_len, 1), device=Q.device)
    
    # Z_max_pre: 全局最大值（用于数值稳定性），初始化为负无穷。
    # 为什么是负无穷？保证第一块数据进来时，scale_factor = exp(-inf - Z_max) = 0，不影响初始状态。
    Z_max_pre = torch.full((seq_len, 1), float('-inf'), device=Q.device)

    # ==========================================
    # 2. 外层循环：遍历 K 和 V 的块 (Tiling)
    # ==========================================
    for j in range(0, seq_len, block_size):
        # 取出当前的 K 块和 V 块，形状均为 (block_size, d_model)
        K_j = K[j : j + block_size, :]
        V_j = V[j : j + block_size, :]
        
        # ==========================================
        # 3. 计算局部打分 S
        # ==========================================
        # Q: (seq_len, d_model), K_j.T: (d_model, block_size)
        # 结果 Z: (seq_len, block_size)，只生成了个小矩阵，不爆显存！
        Z = Q @ K_j.T / math.sqrt(d_model)
        
        # ==========================================
        # 4. 在线 Softmax 更新逻辑
        # ==========================================
        # 【易错点】：Z.max() 返回的是一个 tuple (values, indices)，
        # 必须加上 [0] 提取 values，否则 torch.maximum 会报错类型不匹配！
        Z_max_j = Z.max(dim=1, keepdim=True)[0] 
        
        # 比较历史最大值和当前块的最大值，取较大者作为新的全局最大值
        Z_max = torch.maximum(Z_max_pre, Z_max_j)
        
        # 计算局部指数（减去最新的全局最大值 Z_max，防止 exp() 溢出）
        Z_exp = torch.exp(Z - Z_max)
        
        # 计算缩放因子 scale_factor：
        # 因为全局最大值可能变大了（从 Z_max_pre 变成 Z_max），
        # 之前基于老最大值算出来的结果“偏大”了，必须乘上这个因子缩回去，保证数学等价。
        scale_factor = torch.exp(Z_max_pre - Z_max)
        
        # 更新分母 l: l_new = scale_factor * l_old + sum(Z_exp)
        l = l * scale_factor + Z_exp.sum(dim=1, keepdim=True)
        
        # 更新输出分子 attention: O_new = scale_factor * O_old + (P @ V_j)
        attention = attention * scale_factor + Z_exp @ V_j
        
        # 更新历史最大值，供下一次循环使用
        Z_max_pre = Z_max

    # ==========================================
    # 5. 最终归一化
    # ==========================================
    # 【易错点】：循环过程中只累加了分子。分母 l 是动态更新的，
    # 必须在所有块遍历完毕后，统一做一次除法归一化。
    return attention / l