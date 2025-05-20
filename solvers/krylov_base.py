from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eig, eigh

class KrylovSolverBase(ABC): 
    def __init__(self, G_matvec, p, verbose=False):
        self.G_matvec = G_matvec
        self.p = p
        self.verbose = verbose
        self.V = None
        self.H = None
        self.custom_print = self._create_verbose_function(verbose)

    def _create_verbose_function(self, verbose):
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        return verbose_print

    def _initialize_vector(self):
        v = np.random.randn(self.p)
        return v / np.linalg.norm(v)

    def _reorthogonalize(self, V, w):
        for i in range(V.shape[1]):
            proj = np.dot(V[:, i], w)
            w -= proj * V[:, i]
        return w

    @abstractmethod
    def _build_Hessenberg(self):
        pass

    @abstractmethod
    def run(self, num_steps):
        pass

    @abstractmethod
    def get_basis(self, k = None, ortho=True):
        "Basis U/U_S in the paper — essentially from the Ritz vectors"
        pass

    # def get_top_ritzpairs(self, k=None, return_vectors=False):
    #     # always build the Hessenberg matrix first
    #     self._build_Hessenberg()
    #     evals, evecs = np.linalg.eigh(self.H)
    #     idx = np.argsort(evals)[::-1]
    #     evals = evals[idx]
    #     evecs = evecs[:, idx]

    #     if k is not None:
    #         evals = evals[:k]
    #         evecs = evecs[:, :k]

    #     if return_vectors:
    #         if self.V is None:
    #             raise ValueError("Cannot return eigenvectors: basis V is not stored.")
    #         return self.V @ evecs, evals
    #     else:
    #         return None, evals
    def get_top_ritzpairs(self, k=None, return_vectors=False):
        
        self._build_Hessenberg()

        # Check symmetry / Hermitian
        H = self.H
        if np.allclose(H, H.T.conj(), atol=1e-10):
            evals, evecs = eigh(H, lower=True, check_finite=False)
        else:
            evals, evecs = eig(H, check_finite=False)

        idx = np.argsort(evals.real)[::-1]  # sort by real part -- TODO: check correctness
        evals = evals[idx]
        evecs = evecs[:, idx]

        if k is not None:
            evals = evals[:k]
            evecs = evecs[:, :k]

        if return_vectors:
            if self.V is None:
                raise ValueError("Cannot return eigenvectors: basis V is not stored.")
            return self.V @ evecs, evals
        else:
            return None, evals
