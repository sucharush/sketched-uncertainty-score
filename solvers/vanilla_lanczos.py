import numpy as np
from solvers.lanczos_base import LanczosBase

class VanillaLanczos(LanczosBase):
    def __init__(self, G_matvec, p, reorth=False, verbose=False):
        super().__init__(G_matvec, p, verbose)
        self.reorth = reorth

    def run(self, num_steps):
        alphas = []
        betas = []
        V = []
        q_prev = np.zeros(self.p)
        q = self._initialize_vector()

        if self.reorth:
            V = [q]

        for i in range(num_steps):
            w = self.G_matvec(q)
            alpha_i = np.dot(q, w)
            w -= alpha_i * q
            # - beta_{j-1} * v_{j-1}
            if i > 0:
                w -= betas[-1] * q_prev

            if self.reorth:
                for _ in range(1):
                    w = self._reorthogonalize(np.column_stack(V), w)

            beta_i = np.linalg.norm(w)
            if beta_i < 1e-12:
                self.custom_print(f"   High-memory Lanczos: Early termination after {i} iterations.")
                break

            alphas.append(alpha_i)
            betas.append(beta_i)

            q_prev, q = q, w / beta_i
            if self.reorth:
                V.append(q)

        if self.reorth:
            self.V = np.column_stack(V)[:, : len(alphas)]
        else:
            self.V = None  # no full basis stored
        self.alphas = alphas
        self.betas = betas
        
    def get_basis(self, ortho = True):
        "U for G~U \Gamma Ut"
        U, _ = self.get_top_ritzpairs(return_vectors=True)
        if U is None:
            raise ValueError("No basis stored.")
        if ortho:
            U, _ = np.linalg.qr(U, mode='reduced')
        return U