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
    Q = decoder_x @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v
    d_k = np.sqrt(Q.shape[1])

    scores = Q @ K.T / d_k

    if src_pad_mask is not None:
        mask = np.asarray(src_pad_mask, dtype=bool)
        scores = np.where(mask[None, :], -np.inf, scores)

    scores_max = scores.max(axis=-1, keepdims=True)
    scores_new = scores - scores_max
    exp_scores = np.exp(scores_new)
    softmax = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    attention = softmax @ V
    return attention, softmax