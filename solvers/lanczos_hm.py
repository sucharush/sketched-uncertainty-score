import numpy as np
import os
from .lanczos_base import LanczosBase

class HighMemoryLanczos(LanczosBase):
    def run(self, num_steps, ortho = True):
        """
        Run the standard Lanczos algorithm with full reorthogonalization.

        Parameters:
        num_steps : int
            Number of Lanczos steps to perform.
        """
        # print("Running High-memory Lanczos: ")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        # Initialize the starting vector
        v = self._initialize_vector()
        V = np.zeros((self.p, num_steps + 1))  # Columns 0 to num_steps
        V[:, 0] = v

        alphas = []
        betas = []
        beta_prev = 0.0  # Initialize beta_{-1} as 0

        for j in range(num_steps):
            # new direction
            w = self.G_matvec(V[:, j])
            # - beta_{j-1} * v_{j-1}
            if j > 0:
                w -= beta_prev * V[:, j - 1]

            # - alpha_j * v_{j}
            alpha = np.dot(V[:, j], w)
            alphas.append(alpha)
            w -= alpha * V[:, j]

            if ortho:
                for _ in range(1):
                    w = self._reorthogonalize(V[:, : j + 1], w)

            beta = np.linalg.norm(w)
            betas.append(beta)

            # if beta is zero, stop the iteration early
            if beta < 1e-12:
                self.custom_print(f"   High-memory Lanczos: Early termination after {j+1} iterations.")
                # print("-------------------------------------------------")
                break

            # normalize to get v_{j+1}
            if j + 1 <= num_steps:  # Ensure we don't exceed allocated space
                V[:, j + 1] = w / beta

            beta_prev = beta  # Update beta

        # len(alphas) diagonal, len(betas)-1 off-diagonal
        self.tridiagonal = (alphas, betas[:-1])
        self.V = V[:, : len(alphas)]
        if j == num_steps-1:
            # print("High-memory Lanczos: Finished without early termination.")
            # print("-------------------------------------------------")
            pass