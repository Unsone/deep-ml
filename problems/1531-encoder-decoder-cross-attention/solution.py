import numpy as np


def cross_attention(decoder_x, encoder_out, W_q, W_k, W_v, src_pad_mask=None):
    """
    Encoder-decoder attention: queries from decoder states, keys and values from encoder output.

    decoder_x:    (tgt_len, d_model)
    encoder_out:  (src_len, d_model)
    W_q, W_k:     (d_model, d_k)
    W_v:          (d_model, d_v)
    src_pad_mask: optional (src_len,) boolean / 0-1 array, True or 1 at padded source positions

    Returns (output, weights) with shapes (tgt_len, d_v) and (tgt_len, src_len).
    Masked source positions must have weight exactly 0.
    """

    # =====================================================================
    # 一、Cross-Attention 是什么？和一般（自）注意力的区别
    # ---------------------------------------------------------------------
    # 普通自注意力(self-attention)：Q、K、V 都来自同一个序列，
    #   Q@K.T 得到的是"方阵" (seq_len, seq_len)，即"这句话里每个词看这句话的其他词"。
    #
    # 交叉注意力(cross-attention)：Q 来自"解码器状态"，K、V 来自"编码器输出"。
    #   - Q 来自 decoder_x：长度为 tgt_len（目标端已生成的部分）
    #   - K、V 来自 encoder_out：长度为 src_len（整句源文）
    #   所以分数矩阵 Q@K.T 是"长方形" (tgt_len, src_len)：
    #   每个目标词(query) 都要看一眼 整句源文(key)，
    #   语义是"把源句子的信息翻译/传递给目标端"，这是编码器和解码器之间的桥。
    #
    # 两个关键不同点：
    #   1) 分数矩阵是长方形，不是方阵 —— 因为 Q 和 K 来自不同长度的序列。
    #   2) 只需要 padding mask，不需要"因果 mask"。
    #      - 解码器自注意力要因果 mask：不能偷看还没生成的目标词。
    #      - cross-attention 不用因果 mask：源句子是编码器一次性看完的完整输入，
    #        每个目标词本来就该看整句源文，不存在"未来"。
    # =====================================================================

    # ---------------------------------------------------------------------
    # 二、投影得到 Q、K、V
    #   Q = decoder_x @ W_q -> (tgt_len, d_k)
    #   K = encoder_out @ W_k -> (src_len, d_k)
    #   V = encoder_out @ W_v -> (src_len, d_v)
    # 注意 K、V 都来自 encoder_out（同源），Q 来自 decoder_x（目标端）。
    # ---------------------------------------------------------------------
    Q = decoder_x @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v

    # d_k 这里存的是"缩放因子" sqrt(d_k)，用于 scaled dot-product，
    # 防止点积随维度变大而过大，导致 softmax 过于尖锐、梯度消失。
    d_k = np.sqrt(Q.shape[1])

    # ---------------------------------------------------------------------
    # 三、缩放点积分数矩阵
    #   scores = Q @ K.T / sqrt(d_k)  -> (tgt_len, src_len)
    # 必须显式转置 K：因为 Q 是 (tgt_len, d_k)，K 是 (src_len, d_k)，
    # 直接 Q@K 形状不合法；K.T 是 (d_k, src_len)，乘完得到长方形矩阵。
    # 这个矩阵第 i 行第 j 列 = 第 i 个目标词(query) 对第 j 个源词(key) 的原始打分。
    # ---------------------------------------------------------------------
    scores = Q @ K.T / d_k
    tgt_len, src_len = scores.shape

    # ---------------------------------------------------------------------
    # 四、Padding Mask（交叉注意力为什么必须做这一步）
    # ---------------------------------------------------------------------
    # 背景：一批句子长短不一，短的会在末尾补 <pad> 占位符凑齐长度。
    #       这些 <pad> 不是真词、没有语义。
    # 不屏蔽会怎样：softmax 会给"所有"源位置分权重，包括 padding；
    #       于是 softmax @ V 时 padding 的 value 也被加权混进输出，
    #       等于模型在"关注空位"，输出被污染，训练学歪。
    # 做法：softmax 之前，把 padding 位置的分数设为 -inf。
    #       因为 exp(-inf) 精确等于 0.0，softmax 后这些位置权重"恰好为 0"，
    #       其余真实词权重重新归一化、和仍为 1，padding 对输出零贡献。
    # 为什么必须 -inf 而不是 -1e9：题目要求权重"恰好为 0"；
    #       -1e9 经 exp 是个极小正数，不是精确 0，会漏一点点 padding 信息。
    #
    # 下面用最朴素的双层 for 循环：
    #   外层 i 遍历每个目标位置(query 行)，内层 j 遍历每个源位置(key 列)，
    #   若第 j 个源位置是 padding(src_pad_mask[j] 为真)，就把 scores[i,j] 打成 -inf。
    # （等价的 NumPy 写法是：scores = np.where(mask[None, :], -np.inf, scores)，
    #   其中 mask[None, :] 把一维 (src_len,) 变成 (1, src_len) 广播到每一行，
    #   意思就是"同一份源 mask 作用到每个 query 行"——因为 padding 是源句子的属性，
    #   与是哪个目标词来看无关。）
    #
    # 说明：cross-attention 的 mask 只针对"源序列的 padding"，没有因果 mask。
    #       题目保证至少有一个源位置未被 mask，所以不会出现整行全 -inf 的 nan 情况。
    # ---------------------------------------------------------------------
    if src_pad_mask is not None:
        for i in range(tgt_len):
            for j in range(src_len):
                if src_pad_mask[j]:
                    scores[i, j] = -np.inf

    # ---------------------------------------------------------------------
    # 五、沿源序列方向做数值稳定的 softmax（axis=-1 对二维即 axis=1）
    #   softmax 在"源序列"这一维做：每个目标词(query) 对所有源词(key) 的分数
    #   归一化成一个概率分布（这一行和为 1）。
    # 步骤：
    #   1) 每行减最大值：softmax 对整体平移不变，减最大值数学上等价，
    #      但能防止 exp 溢出成 inf / inf/inf=nan（数值稳定）。
    #   2) 逐元素 exp：padding 位置的 -inf 在这里变成精确的 0.0。
    #   3) 除以每行的和：归一化，每行权重和为 1。
    # 结果 softmax（即 weights）形状 (tgt_len, src_len)，
    # padding 列权重精确为 0，其余列和为 1。
    # ---------------------------------------------------------------------
    scores_max = scores.max(axis=-1, keepdims=True)   # 每行最大值 (tgt_len, 1)
    scores_new = scores - scores_max                  # 减最大值，数值稳定
    exp_scores = np.exp(scores_new)                   # exp；-inf -> 0.0
    softmax = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # 每行归一化

    # ---------------------------------------------------------------------
    # 六、加权求和得到输出
    #   attention = softmax @ V -> (tgt_len, d_v)
    # 注意加权对象是投影后的 V（不是 W_v！W_v 只用来算 V）。
    # 每个目标位置的输出 = 它对所有源位置的注意力权重 × 对应源位置的 value 之和，
    # 即"把整句源文按相关性加权融合"后的表示。
    # ---------------------------------------------------------------------
    attention = softmax @ V

    # 返回：
    #   attention: (tgt_len, d_v)   —— 交叉注意力的输出
    #   softmax:   (tgt_len, src_len)—— 注意力权重矩阵（题目要求的 weights）
    return attention, softmax