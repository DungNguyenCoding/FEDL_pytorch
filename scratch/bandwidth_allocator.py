import numpy as np

def allocate_bandwidth(B, snrs):
    # Theorem 3: More bandwidth to worse channels [cite: 446, 461, 462]
    R_k = np.log2(1 + snrs)
    inv_R_sum = np.sum(1 / R_k)
    return (B / R_k) * (1 / inv_R_sum)