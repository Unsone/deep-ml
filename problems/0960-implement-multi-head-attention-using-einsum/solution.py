import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    # (1) one contraction per matrix: (B, S, d_in) x (d_in, d_out) -> (B, S, d_out)
    Q = np.einsum('bsi,io->bso', X, W_q)
    K = np.einsum('bsi,io->bso', X, W_k)
    V = np.einsum('bsi,io->bso', X, W_v)
    return Q, K, V

def self_attention(Q, K, V, causal: bool = True):
    # Q, K, V: (B, H, S, d_head); heads are treated as an extra batch axis
    d_head = Q.shape[-1]
    S = Q.shape[-2]

    # (3) scaled scores: (B, H, S_query, S_key)
    scores = np.einsum('bhsd,bhtd->bhst', Q, K) / np.sqrt(d_head)

    # (4) causal mask: block key positions t > query position s
    if causal:
        future = np.triu(np.ones((S, S), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)

    # (5) stable softmax over the key axis
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    softmax = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # (6) aggregate values: (B, H, S, S) x (B, H, S, d_head) -> (B, H, S, d_head)
    attention = np.einsum('bhst,bhtd->bhsd', softmax, V)
    return attention

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    X = np.asarray(X, dtype=float)
    W_q, W_k, W_v, W_o = (np.asarray(w, dtype=float) for w in (W_q, W_k, W_v, W_o))

    B, S, _ = X.shape
    d_out = W_q.shape[1]
    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
    d_head = d_out // num_heads

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)

    # (2) split into heads: (B, S, d_out) -> (B, S, H, d_head) -> (B, H, S, d_head)
    Q_ = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K_ = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V_ = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)

    attentions = self_attention(Q_, K_, V_, causal)

    # merge heads: (B, H, S, d_head) -> (B, S, H, d_head) -> (B, S, d_out)
    attentions = attentions.transpose(0, 2, 1, 3).reshape(B, S, d_out)

    # (7) output projection
    out = np.einsum('bso,op->bsp', attentions, W_o)
    return out.tolist()