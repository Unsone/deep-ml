import numpy as np


def _sigmoid(z):
    """数值稳定的 sigmoid（避免 exp 溢出）。"""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """Simplified Gated DeltaNet linear attention forward pass."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64)
    w = np.asarray(rms_weight, dtype=np.float64).reshape(-1)

    T, d = k.shape
    if T == 0:
        return []

    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)
    k_norm = k / (np.linalg.norm(k, axis=-1, keepdims=True) + eps)

    alpha = np.exp(-np.logaddexp(0.0, a)) * np.exp(float(A_log))
    beta = _sigmoid(b)
    g_silu = g * _sigmoid(g) 

    S = np.zeros((d, d), dtype=np.float64)
    y = np.empty((T, d), dtype=np.float64)

    for t in range(T):
        S = alpha[t] * S
        delta = (v[t] - S.T @ k_norm[t]) * beta[t]
        S = S + np.outer(k_norm[t], delta)
        o = S.T @ q_norm[t]
        o_norm = o / np.sqrt(np.mean(o * o) + eps) * w
        y[t] = o_norm * g_silu[t]

    return np.round(y, 4).tolist()