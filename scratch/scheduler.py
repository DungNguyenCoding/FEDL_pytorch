import torch
from scipy.optimize import bisect

class FEELScheduler:
    def __init__(self, rho, n_total):
        self.rho = rho
        self.n_total = n_total

    def get_pk(self, n_k, g_norms, T_u):
        def objective(lmbda):
            # Eq (25): Optimal p_k formula [cite: 322]
            denom = torch.clamp((1 - self.rho) * T_u + lmbda, min=1e-9)
            p_k = (n_k / self.n_total) * g_norms * torch.sqrt(self.rho / denom)
            return p_k.sum().item() - 1
        
        lmbda_star = bisect(objective, -1e10, 1e10)
        denom = torch.clamp((1 - self.rho) * T_u + lmbda_star, min=1e-9)
        p_k = (n_k / self.n_total) * g_norms * torch.sqrt(self.rho / denom)
        return p_k / p_k.sum() # Normalized [cite: 322, 325]