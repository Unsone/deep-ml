import torch
import math

def pos_encoding(position: int, d_model: int):
    """
    计算 Transformer 模型的位置编码。
    
    返回:
        形状为 (position, d_model) 的 torch.Tensor，数据类型为 float16，
        如果 position == 0 或 d_model <= 0，则返回 -1。
    """
    # 1. 边界条件检查
    if position <= 0 or d_model <= 0:
        return -1

    # 2. 初始化位置编码矩阵 (position, d_model)
    # pe[:, 0::2] 和 pe[:, 1::2] 是 Python 切片语法：
    #   - 逗号前 ':' 表示取所有行
    #   - 逗号后 '0::2' 表示从第 0 列开始，步长为 2（即提取所有偶数列：0, 2, 4...）
    #   - 逗号后 '1::2' 表示从第 1 列开始，步长为 2（即提取所有奇数列：1, 3, 5...）
    pe = torch.zeros(position, d_model, dtype=torch.float16)

    # 3. 创建位置列向量 (position, 1)
    # torch.arange(0, position) 类似于 Python 的 range()，生成一维连续序列 [0, 1, 2, ..., position-1]
    # .unsqueeze(1) 的作用是"增加维度"：将形状从 (position,) 变为 (position, 1) 的列向量。
    # 【为什么要列向量？】
    # 目的是为了利用 PyTorch 的"广播机制(Broadcasting)"。
    # 当 (position, 1) 和 (1, d_model/2) 相乘时，会自动扩展成 (position, d_model/2) 的矩阵，
    # 从而避免写双重 for 循环，实现向量化高速计算。
    position_vec = torch.arange(0, position, dtype=torch.float16).unsqueeze(1)

    # 4. 计算分母项 div_term
    # 对应公式 10000^(2i/d_model) 的对数形式，使用 exp 避免数值溢出
    # torch.arange(0, d_model, 2) 生成偶数索引 [0, 2, 4...]，
    # 注意：这里的 div_term 长度 = ceil(d_model / 2) (向上取整)。
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float16) * 
                         (-math.log(10000.0) / d_model))

    # 5. 计算 sin 和 cos
    # 偶数维度 (0, 2, 4...) 使用 sin 填充，长度刚好等于 div_term 的长度
    pe[:, 0::2] = torch.sin(position_vec * div_term)
    
    # 【为什么这里要分类讨论？】
    # 因为当 d_model 为奇数时，偶数列和奇数列的数量是不相等的！
    # 假设 d_model = 5：
    #   - 偶数列 pe[:, 0::2] 提取到 3 列 (索引 0, 2, 4)
    #   - 奇数列 pe[:, 1::2] 提取到 2 列 (索引 1, 3)
    # 但是 div_term 的长度是根据偶数索引生成的，所以始终为 3。
    # 如果直接用 pe[:, 1::2] = torch.cos(...)，会把长度为 3 的数据塞进只有 2 列的格子里，导致 Shape 不匹配报错。
    if d_model % 2 == 0:
        # 偶数长度：完美对半，直接赋值
        pe[:, 1::2] = torch.cos(position_vec * div_term)
    else:
        # 奇数长度：奇数列比偶数列少 1 列。
        # 所以我们需要截断 div_term，去掉最后一个元素 (div_term[:-1])，使其长度变为 2，才能完美匹配奇数列的长度。
        pe[:, 1::2] = torch.cos(position_vec * div_term[:-1])

    return pe