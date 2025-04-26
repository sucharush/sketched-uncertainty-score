import numpy as np
from solvers.lanczos_base import LanczosBase


class VanillaLanczos(LanczosBase):
    def __init__(self, G_matvec, p, reorth=True, verbose=False, store_full_basis=False):
        super().__init__(G_matvec, p, verbose)
        self.reorth = reorth
        self.store_full_basis = store_full_basis

    def run(self, num_steps):
        alphas = []
        betas = []
        V = []
        q_prev = np.zeros(self.p)
        q = self._initialize_vector()

        if self.reorth or self.store_full_basis:
            V = [q]
        # V = [q]

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
                    # print("orth check:", np.linalg.norm(np.column_stack(V).T @ w))

            beta_i = np.linalg.norm(w)

            if beta_i < 1e-12:
                self.custom_print(
                    f"   High-memory Lanczos: Early termination after {i} iterations."
                )
                break
            alphas.append(alpha_i)
            betas.append(beta_i)

            q_prev, q = q, w / beta_i
            if self.reorth or self.store_full_basis:
                V.append(q)
            # V.append(q)

        if self.reorth or self.store_full_basis:
            self.V = np.column_stack(V)[:, : len(alphas)]
        else:
            self.V = None  # no full basis stored
        # self.V = np.column_stack(V)[:, : len(alphas)]
        self.alphas = alphas
        self.betas = betas

    def get_basis(self, k=None, ortho=True):
        "U for G~U \Gamma Ut"
        U, _ = self.get_top_ritzpairs(k=k, return_vectors=True)
        if U is None:
            raise ValueError("No basis stored.")
        if ortho:
            U, _ = np.linalg.qr(U, mode="reduced")
        return U
