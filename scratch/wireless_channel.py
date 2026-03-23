import numpy as np

class WirelessChannel:
    def __init__(self, K=30, B=1e6, P_device=24, noise_density=-174):
        self.B = B
        self.P_W = 10**((P_device - 30) / 10) # 24 dBm to Watts [cite: 471]
        self.N_W = 10**((noise_density - 30 + 10 * np.log10(B)) / 10) # [cite: 471]

    def get_snr(self, distances):
        # LTE path loss model [cite: 470]
        pl_db = 128.1 + 37.6 * np.log10(distances)
        pl_linear = 10**(pl_db / 10)
        return (self.P_W / pl_linear) / self.N_W

    def get_latency(self, snr, q=16, S=1e5):
        # T_k^U = qS / (B * log2(1 + snr)) [cite: 273]
        return (q * S) / (self.B * np.log2(1 + snr))