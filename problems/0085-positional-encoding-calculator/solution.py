import torch
import math

def pos_encoding(position: int, d_model: int):
    """
    计算 Transformer 模型的位置编码。

    参数:
        position: 序列长度（位置数量）
        d_model: 模型维度

    返回:
        形状为 (position, d_model) 的 torch.Tensor，数据类型为 float16，
        如果 position == 0 或 d_model <= 0，则返回 -1。
    """
    # 1. 边界条件检查
    if position <= 0 or d_model <= 0:
        return -1

    # 2. 初始化位置编码矩阵 (position, d_model)
    pe = torch.zeros(position, d_model, dtype=torch.float16)

    # 3. 创建位置列向量 (position, 1)
    position_vec = torch.arange(0, position, dtype=torch.float16).unsqueeze(1)

    # 4. 计算分母项 div_term
    # 公式中的 10000^(2i/d_model) 取对数后变为 exp(2i * -log(10000) / d_model)
    # 我们只需要计算一半的维度 (d_model/2)，因为偶数索引用 sin，奇数索引用 cos
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float16) * (-math.log(10000.0) / d_model))

    # 5. 计算 sin 和 cos
    # 偶数维度 (0, 2, 4...) 使用 sin
    pe[:, 0::2] = torch.sin(position_vec * div_term)
    
    # 奇数维度 (1, 3, 5...) 使用 cos
    # 注意：如果 d_model 是奇数，切片长度可能不一致，需要处理
    if d_model % 2 == 0:
        pe[:, 1::2] = torch.cos(position_vec * div_term)
    else:
        # 如果 d_model 是奇数，最后一个维度只能由 sin 填充
        pe[:, 1::2] = torch.cos(position_vec * div_term[:-1])

    return pe