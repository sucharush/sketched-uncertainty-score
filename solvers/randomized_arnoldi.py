import numpy as np
from scipy.linalg import solve, lstsq, qr, norm
from solvers.krylov_base import KrylovSolverBase
from sketch.sketch_base import Sketcher

class RGSArnoldi(KrylovSolverBase):
    def __init__(self, G_matvec, p, sketch: Sketcher, verbose=False):
        super().__init__(G_matvec, p, verbose)
        if sketch.p != p:
            raise ValueError("Sketcher's input dimension does not match p.")
        self.sketcher = sketch
        self.s = sketch.s
        self._H_cols = None

    def _solve_least_squares(self, G, p):
        """Solve (G^T G) y = G^T p robustly."""
        Gt = G.T.conj()  
        GtG = Gt @ G
        Gtp = Gt @ p
        try:
            return solve(GtG, Gtp, assume_a='sym')  # assume_a helps performance if symmetric
        except np.linalg.LinAlgError:
            return lstsq(GtG, Gtp, cond=None)[0]

    def run(self, num_steps):
        Q = []
        P = []
        H = []

        q0 = self._initialize_vector()
        Q.append(q0)
        P.append(self.sketcher.apply_sketch(q0))

        for i in range(num_steps):
            w = self.G_matvec(Q[-1])
            p = self.sketcher.apply_sketch(w)

            G = np.column_stack(P)
            y = self._solve_least_squares(G, p)
            q_proj = w - np.column_stack(Q) @ y
            p_proj = self.sketcher.apply_sketch(q_proj)

            beta = norm(p_proj)
            if beta < 1e-12:
                self.custom_print(f"RGSArnoldi: Early termination after {i+1} steps.")
                break

            q_next = q_proj / beta
            p_next = p_proj / beta

            Q.append(q_next)
            P.append(p_next)

            col = np.zeros(i + 2, dtype=np.complex128)
            col[:i + 1] = y
            col[i + 1] = beta
            H.append(col)

        self.V = np.column_stack(P)[:, :len(H)]
        self._H_cols = H

    def _build_Hessenberg(self):
        if self._H_cols is None:
            raise ValueError("No Hessenberg columns stored. Run the algorithm first.")
        maxlen = max(len(col) for col in self._H_cols)
        self.H = np.column_stack([np.pad(col, (0, maxlen - len(col))) for col in self._H_cols])[:maxlen - 1, :]

    def get_basis(self, k=None, ortho=True):
        U, _ = self.get_top_ritzpairs(k=k,return_vectors=True)
        if U is None:
            raise ValueError("No basis stored.")
        if ortho:
            U, _ = qr(U, mode='economic')  # = numpy 'reduced'
        return U
