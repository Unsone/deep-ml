import torch

def multi_head_latent_attention(
    X: torch.Tensor,
    W_dkv: torch.Tensor,
    W_uk: torch.Tensor,
    W_uv: torch.Tensor,
    W_dq: torch.Tensor,
    W_uq: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).

    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads

    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV latent of shape (seq_len, d_c_kv)
    """
    seq_len, d_model = X.shape
    d_c_kv = W_dkv.shape[1]
    d_c_q = W_dq.shape[1]
    d_heads = d_model // n_heads

    C_KV = X @ W_dkv
    K = C_KV @ W_uk
    V = C_KV @ W_uv
    C_Q = X @ W_dq
    Q = C_Q @ W_uq

    output = multi_head_attention(Q, K, V, n_heads) @ W_o
    return output, C_KV
    pass

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / (d_k)**(1/2)
    scores_max = scores.max(dim = -1, keepdim = True).values
    scores_new = scores - scores_max
    exp_scores = torch.exp(scores_new)
    softmax = exp_scores / exp_scores.sum(dim = -1, keepdim = True)
    attention = softmax @ V
    return attention

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int):
    seq_len, d_model = Q.shape
    d_heads = d_model // n_heads
    Q_ = Q.view(seq_len, n_heads, d_heads).transpose(0, 1)
    K_ = K.view(seq_len, n_heads, d_heads).transpose(0, 1)
    V_ = V.view(seq_len, n_heads, d_heads).transpose(0, 1)
    attention = self_attention(Q_, K_, V_)
    return attention.transpose(0,1).contiguous().view(seq_len, d_model)

