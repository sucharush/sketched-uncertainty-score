from krylov_base import KrylovBase
import numpy as np

class ClassicalLanczos(KrylovBase):
    def __init__(self, G_matvec, p, reorth=False, verbose=False):
        super().__init__(G_matvec, p, verbose)
        self.reorth = reorth

    def run(self, num_steps):
        q_prev = np.zeros(self.p)
        q = self._initialize_vector()
        V = [q]
        alpha = []
        beta = []

        for i in range(num_steps):
            z = self.G_matvec(q)
            a = np.dot(q, z)
            alpha.append(a)

            z -= a * q
            if i > 0:
                z -= beta[-1] * q_prev

            if self.reorth:
                z = self._reorthogonalize(np.column_stack(V), z)

            b = np.linalg.norm(z)
            if b < 1e-12:
                break

            beta.append(b)
            q_prev, q = q, z / b
            V.append(q)

        self.V = np.column_stack(V)
        self.tridiagonal = (np.array(alpha), np.array(beta))

    def build_reduced_matrix(self):
        alpha, beta = self.tridiagonal
        T = np.diag(alpha)
        if len(beta) > 0:
            T += np.diag(beta, 1) + np.diag(beta, -1)
        self.H = T
