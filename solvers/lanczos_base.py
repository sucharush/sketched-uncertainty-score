import numpy as np
from solvers.krylov_base import KrylovSolverBase

class LanczosBase(KrylovSolverBase):
    def __init__(self, G_matvec, p, verbose=False):
        super().__init__(G_matvec, p, verbose)
        self.alphas = []
        self.betas = []

    def _build_Hessenberg(self):
        alpha = np.array(self.alphas)
        # Use beta[:-1] to exclude the last element for correct dimensions
        beta = np.array(self.betas[:-1]) if len(self.betas) > 0 else np.array([])
        T = np.diag(alpha)
        if len(beta) > 0:
            T += np.diag(beta, 1) + np.diag(beta, -1)
        self.H = T