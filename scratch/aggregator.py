import torch

def aggregate_unbiased(grads, n_k, p_k, n_total):
    # Eq (9) and (29): Unbiased estimate [cite: 209, 380, 424]
    M = len(grads)
    g_sum = torch.zeros_like(grads[0])
    for i in range(M):
        weight = n_k[i] / (n_total * p_k[i])
        g_sum += weight * grads[i]
    return g_sum / M