import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    X = np.asarray(X, dtype=float)
    W_query = np.asarray(W_query, dtype=float)
    W_key = np.asarray(W_key, dtype=float)
    W_value = np.asarray(W_value, dtype=float)

    B, T, _ = X.shape
    d_out = W_query.shape[-1]

    Q = np.dot(X, W_query)
    K = np.dot(X, W_key)
    V = np.dot(X, W_value)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_out)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return (attn @ V).tolist()




