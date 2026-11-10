import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        N = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return N 

    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    A = softmax(S)
    output = A @ V
    return output


