import numpy as np


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """标准缩放点积注意力。Q:(H,D), K/V:(L,H,D), 返回 (H,D)。"""
    num_heads, head_dim = Q.shape

    # 把 head 维提到前面:(L, H, D) -> (H, L, D)
    K_ = K.transpose(1, 0, 2)
    V_ = V.transpose(1, 0, 2)

    # 缩放点积分数 (H, L)
    scores = np.einsum('hd,hld->hl', Q, K_) / np.sqrt(head_dim)

    # 数值稳定 softmax
    scores_new = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores_new)
    softmax = exp_scores / exp_scores.sum(axis=-1, keepdims=True)   # (H, L)

    # 加权 V (H, D)
    attention = np.einsum('hl,hld->hd', softmax, V_)
    return attention


def paged_attention(query, key_cache, value_cache, block_table, context_len):
    num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]

    if context_len == 0:
        return np.zeros((num_heads, head_dim))

    # 按页表重建 K, V
    key_list, value_list = [], []
    remaining = context_len
    for phys_idx in block_table:
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        key_list.append(key_cache[phys_idx, :take])
        value_list.append(value_cache[phys_idx, :take])
        remaining -= take

    K = np.concatenate(key_list, axis=0)     # (L, H, D)
    V = np.concatenate(value_list, axis=0)

    return self_attention(query, K, V)