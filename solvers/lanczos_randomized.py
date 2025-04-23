from .lanczos_base import LanczosBase
from sketch.sketch_base import Sketcher
import numpy as np

class RandomizedLanczos(LanczosBase):
    def __init__(self, G_matvec, p, sketch: Sketcher, verbose=False):
        super().__init__(G_matvec, p, verbose)
        self.sketcher = sketch
        self.richardson_iters = 100
        self.V_full = None
        self.V = None

    def _solve_system(self, G, p):
        ##### Richardson iteration: doesn't work, \rho(GtG) is too large
        
        # print(f"cond(GtG): {np.linalg.cond(G.T@G)}") 
        # """Solve G^T G y = G^T p using Richardson iterations."""
        # n = G.shape[1]
        # y = np.zeros(n, dtype=np.complex128)
        # for _ in range(self.richardson_iters):
        #     residual = p - G @ y
        #     y += G.T @ residual   
        # return y
        Gt = G.conj().T  
        GtG = Gt @ G
        Gtp = Gt @ p
        try:
            y = np.linalg.solve(GtG, Gtp)
        except ValueError:
            y = np.linalg.lstsq(GtG, Gtp, rcond=None)[0]
        return y


    def run(self, num_steps, richardson_iters=50):
        # print("Running Randomized Lanczos: ")
        self.richardson_iters = richardson_iters
        # initialize
        q = self._initialize_vector().astype(np.complex128)
        V = np.zeros(
            (self.p, num_steps + 1), dtype=np.complex128
        )  # Columns 0 to num_steps
        V[:, 0] = q
        ps = [self.sketcher.apply_sketch(q)]
        alphas = []
        betas = []

        for i in range(num_steps):
            current_q = V[:, i]
            w = self.G_matvec(current_q)

            if i > 0:
                w -= betas[i - 1] * V[:, i - 1]

            current_p = self.sketcher.apply_sketch(w)
            alpha = current_p @ ps[i]
            alphas.append(alpha)

            w -= alpha * current_q
            current_p -= alpha * ps[i]

            # reorthogonalize
            if i > 0:
                G_prev = np.column_stack(ps[:i])
                Q_prev = V[:, :i]
                y = self._solve_system(G_prev, current_p)
                w -= Q_prev @ y
                current_p -= G_prev @ y

            beta = np.linalg.norm(w)
            betas.append(beta)

            if beta < 1e-12:
                self.custom_print(f"   Randomized Lanczos: Early termination after {i+1} iterations.")
                # print("-------------------------------------------------")
                break

            q_next = w / beta
            p_next = self.sketcher.apply_sketch(q_next)

            V[:, i + 1] = q_next
            ps.append(p_next)

        if len(alphas) > 0:
            alphas = alphas[:i+1]
            betas = betas[:i+1]  
            # print(f"len(alphas) = {len(alphas)}, len(betas)={len(betas[:-1])}")
            self.tridiagonal = (alphas, betas[:-1]) 
            self.V_full = V[:, : len(alphas)]
            # self.V = np.array(ps[:-1]).T
            ps = ps[:i+1] 
            self.V = np.array(ps).T
        # if i == num_steps - 1:
            # print("   Finished without early termination.")
            # print("-------------------------------------------------")

