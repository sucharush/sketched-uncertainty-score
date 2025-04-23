import numpy as np
from solvers.lanczos_base import LanczosBase
from sketch.sketch_base import Sketcher
from solvers.vanilla_lanczos import VanillaLanczos


class SketchedLanczos(LanczosBase):
    def __init__(self, G_matvec, p, sketch: Sketcher, verbose=False):
        super().__init__(G_matvec, p, verbose)
        if sketch.p != p:
            raise ValueError(
                f"The dimension of the sketch matrix (s, p) = ({sketch.s, sketch.p}) \
                must match the dimension p = {p} of the vectors."
            )
        self.s = sketch.s
        self.sketcher = sketch
        self.SV = []
        self.sketched_hm_basis = None

    def run(self, num_steps, pre_steps=0):
        """
        Runs the (Preconditioned) Sketched Lanczos algorithm.

        Parameters:
        -----------
        num_steps : int
            Total iterations to perform.
        pre_steps : int, optional
            High-memory Lanczos steps before switching to sketched mode.

        """
        sketched_hm_basis = None
        current_matvec = self.G_matvec
        if pre_steps > 0:
            sketched_steps = num_steps - pre_steps
            if sketched_steps <= 0:
                raise ValueError(
                    f"Total steps ({num_steps}) must be greater than preconditioning steps ({pre_steps})."
                )

            # run high-memory lanczos
            self.hm_lanczos = VanillaLanczos(self.G_matvec, self.p, reorth=True)
            self.hm_lanczos.run(num_steps=pre_steps)

            # eigenvalue deflation
            U0, lambdas0 = self.hm_lanczos.get_top_ritzpairs(
                pre_steps, return_vectors=True
            )
            Lambda0 = np.diag(lambdas0)
            original_matvec = self.G_matvec

            def deflated_matvec(v):
                # print("shape:", v.shape)
                return original_matvec(v) - U0 @ (Lambda0 @ (U0.T @ v))

            current_matvec = deflated_matvec
            sketched_hm_basis = self.sketcher.apply_sketch(U0)

        alphas = []
        betas = []
        # V_sketch = np.zeros((self.s, num_steps), dtype=np.complex128)
        q_prev = np.zeros(self.p)
        q = self._initialize_vector()

        V_sketch = [self.sketcher.apply_sketch(q)]

        sketched_steps = num_steps if pre_steps is None else num_steps - pre_steps
        for i in range(sketched_steps):
            w = current_matvec(q)
            alpha_i = np.dot(q, w)
            w -= alpha_i * q
            # - beta_{j-1} * v_{j-1}
            if i > 0:
                w -= betas[-1] * q_prev

            beta_i = np.linalg.norm(w)
            if beta_i < 1e-12:
                self.custom_print(
                    f"   Sketched Lanczos: Early termination after {pre_steps + i + 1} iterations (including {pre_steps} for eigenvalue deflation)."
                )
                break

            alphas.append(alpha_i)
            betas.append(beta_i)

            q_prev, q = q, w / beta_i

            V_sketch.append(self.sketcher.apply_sketch(q))

        self.V = np.column_stack(V_sketch)[:, : len(alphas)]  # align with H dim
        self.alphas = alphas
        self.betas = betas
        # if pre_steps > 0:
        #     self.G_matvec = original_matvec
        self.sketched_hm_basis = sketched_hm_basis

    def get_basis(self, ortho=True):
        """
        Returns the full basis for preconditioned version.

        Parameters:
        -----------
        ortho : bool, optional
            Orthonormalize the basis if True (default).

        Returns:
        --------
        np.ndarray
            Full basis U (concatenated if preconditioning used).
        """
        Us, _ = self.get_top_ritzpairs(return_vectors=True)
        if self.sketched_hm_basis is not None:
            # raise ValueError("Please first run method with eigenvalue deflation steps.")
            SU0 = self.sketched_hm_basis
            U = np.hstack((SU0, Us))
        else:
            U = Us

        if ortho:
            U, _ = np.linalg.qr(U, mode="reduced")

        return U
