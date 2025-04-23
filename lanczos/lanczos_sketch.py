import numpy as np
from .lanczos_base import LanczosBase
from .lanczos_hm import HighMemoryLanczos
# from ..sketch.sketch_base import Sketcher
from sketch.sketch_base import Sketcher

class SketchedLanczos(LanczosBase):
    def __init__(self, G_matvec, p, sketch: Sketcher, verbose=False):
        """
        Parameters
        ----------
        G_matvec : callable
            A function or lambda that, given a vector of shape (p,),
            returns the matrix-vector product G @ v.
        p : int
            Dimension of the vectors to be multiplied.
        sketcher : a sketching method from ABC Sketcher.
        """
        super().__init__(G_matvec, p, verbose)
        if sketch.p != p:
            raise ValueError(
                f"The dimension of the sketch matrix (s, p) = ({sketch.s, sketch.p}) \
                must match the dimension p = {p} of the vectors."
            )
        self.s = sketch.s
        self.sketcher = sketch

        # Create an instance of the full-memory Lanczos:
        self.full_lanczos = HighMemoryLanczos(G_matvec, p)

    def run_full_then_sketch(self, num_steps_full):
        """
        (1) Run a full-memory Lanczos to get basis V (p x num_steps_full)
        (2) Sketch the basis using SRFT
        (3) Orthonormalize in the sketch space, if desired
        """
        # print("Running Full Lanczos + Sketch: ")

        # --- A) Use the HighMemoryLanczos object to get a full basis ---
        self.full_lanczos.run(num_steps_full)
        U0, _ = self.full_lanczos.get_top_eigenpairs(num_steps_full)

        # --- B) Apply SRFT to V => shape (s, num_steps_full) ---
        V_sketched = self.sketcher.apply_sketch(U0)

        # --- C) Orthonormalize in the sketch space ---
        Q_sketched, _ = np.linalg.qr(
            V_sketched, mode="reduced"
        )  # shape (s, num_steps_full)

        # Return the sketched orthonormal basis
        return Q_sketched

    # Sketched-only Lanczos, if you still want that method:
    def sketched_lanczos(self, k, v0=None):
        """
        Lanczos procedure that tracks the usual alpha/beta scalars
        but only stores the (s x k) sketched vectors, not the full V (p x k).

        Parameters
        ----------
        k : int
            Number of Lanczos iterations.
        v0 : ndarray, optional
            Initial vector of shape (p,). If None, use random.

        Returns
        -------
        Q : ndarray, shape (s, m)
            Orthonormal basis (in the sketched space) for the Lanczos vectors,
            where m is <= k if early breakdown occurs.
        """
        # print("Running Sketched Lanczos: ")
        V_sketch = np.zeros((self.s, k), dtype=np.complex128)
        alphas = np.zeros(k)
        betas = np.zeros(k - 1)

        if v0 is None:
            v = np.random.randn(self.p)
        else:
            v = v0.copy()
        v /= np.linalg.norm(v)

        beta_prev = 0.0
        v_prev_prev = None
        v_prev = v.copy()

        for i in range(k):
            w = self.G_matvec(v_prev)
            alpha = np.dot(v_prev, w)
            alphas[i] = alpha
            w -= alpha * v_prev

            if i > 0:
                w -= beta_prev * v_prev_prev

            beta = np.linalg.norm(w)
            if i < k - 1:
                betas[i] = beta

            # Check for breakdown
            if beta < 1e-12:
                alphas = alphas[: i + 1]
                betas = betas[:i]
                self.custom_print(f"   Sketched Lanczos: Early termination after {i+1} iterations.")
                # print("-------------------------------------------------")
                break

            v_current = w / beta
            # Compute the SRFT of the new Lanczos vector
            V_sketch[:, i] = self.sketcher.apply_sketch(v_current)

            # Update for next iteration
            v_prev_prev = v_prev.copy()
            v_prev = v_current
            beta_prev = beta

        # Store the final tridiagonal
        self.tridiagonal = (alphas, betas)

        # Orthonormalize the sketched vectors in (s x number_of_kept_vectors)-space
        m = len(alphas)
        # Q, _ = np.linalg.qr(V_sketch[:, :m], mode="reduced")
        # return Q
        self.V = V_sketch[:, :m]
        # if i == k-1:
            # print("Sketched Lanczos: Finished without early termination.")
            # print("-------------------------------------------------")


    def preconditioned_version(self, k, k0=5, verbose=False):
        """
        "Preconditioned" (deflation + sketched Lanczos) approach:
        1) Use a high-memory Lanczos to get top-k0 approximate eigenpairs (U0, Lambda0).
        2) Define a deflated operator: A_defl(v) = A*v - U0 * Lambda0 * (U0^T v).
        3) Run sketched Lanczos on the deflated operator to get an (s x k1) basis U_S.
        4) Combine srft(U0) and U_S, and do a final QR in sketched space,
            returning an (s x (k0 + k1)) orthonormal basis.
        """
        # print(f"Running Preconditioned Sketched Lanczos with k0={k0}: ")
        k1 = k - k0
        # Step 1: top-k0 approximate eigenpairs
        self.full_lanczos.run(num_steps=k0)
        U0, lambdas = self.full_lanczos.get_top_eigenpairs(k0)
        Lambda0 = np.diag(lambdas)
        if verbose:
            print(f"top-{k0} approx. eigenvalues: {lambdas}")
        original_matvec = self.G_matvec

        # Step 2: eigenvalue deflation
        def deflated_matvec(v):
            # print("shape:", v.shape)
            return original_matvec(v) - U0 @ (Lambda0 @ (U0.T @ v))

        # Step 3: run sketched Lanczos on the deflated operator
        self.G_matvec = deflated_matvec
        self.sketched_lanczos(k1)
        # print(self.V.shape)
        U_S, _ = self.get_top_eigenpairs(k1)
        self.G_matvec = original_matvec

        # Step 4: combine srft(U0) with U_S in the sketched domain, then QR
        SU0 = self.sketcher.apply_sketch(U0)  # shape (s, k0)
        combined = np.hstack((SU0, U_S))  # shape (s, k0 + k1)
        Q, _ = np.linalg.qr(combined, mode="reduced")
        return Q