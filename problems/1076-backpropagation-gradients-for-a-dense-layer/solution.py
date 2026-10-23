import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.
    """

    z = np.dot(W, a_prev) + b 
    a = 1.0 / (1.0 + np.exp(-z))
    dC_da = 2 * (a - y)
    da_dz = a * (1 - a)
    delta = dC_da * da_dz
    dW = np.outer(delta, a_prev)
    db = delta
    da_prev = np.dot(W.T, delta)
    return {
        'dW': dW.tolist(),
        'db': db.tolist(),
        'da_prev': da_prev.tolist()
    }