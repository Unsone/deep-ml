import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    attention = multi_head_attention(X, weights, num_heads)
    h = layer_norm(X + attention, weights['gamma1'], weights['beta1'], eps)

    ffn_out = FFN(h, weights)
    out = layer_norm(h + ffn_out, weights['gamma2'], weights['beta2'], eps)
    return out
    pass
    
def multi_head_attention(X, weights, num_heads):
    batch, seq_len, d_model = X.shape
    d_k = d_model // num_heads
    Q, K, V = compute_QKV(X, weights)
    
    def split_heads(tensor):
        return tensor.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    Q_ = split_heads(Q)
    K_ = split_heads(K)
    V_ = split_heads(V)
    
    attention = self_attention(Q_, K_, V_)
    full_attention = attention.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return full_attention @ weights['W_o']
    
def self_attention(Q, K, V):
    sqrt_d_k = np.sqrt(Q.shape[-1])
    scores = Q @ np.swapaxes(K, -1, -2) / sqrt_d_k
    scores_max = scores.max(axis = -1, keepdims = True)
    scores_new = scores - scores_max
    exp_scores = np.exp(scores_new)
    softmax = exp_scores / exp_scores.sum(axis = -1, keepdims = True)
    attention = softmax @ V
    return attention

def compute_QKV(X, weights):
    Q = X @ weights['W_q']
    K = X @ weights['W_k']
    V = X @ weights['W_v']
    return Q, K, V
    
def layer_norm(X, gamma, beta, eps):
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    normalized = (X - mean) / np.sqrt(variance + eps)
    return gamma * normalized + beta
    
def FFN(X, weights):
    hidden = np.maximum(0, X @ weights['W1'] + weights['b1'])
    return hidden @ weights['W2'] + weights['b2']