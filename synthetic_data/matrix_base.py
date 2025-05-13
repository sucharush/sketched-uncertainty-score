from abc import ABC, abstractmethod
import numpy as np

class BaseMatrixGenerator(ABC):
    def __init__(self, n, seed=42):
        self.n = n
        self.seed = seed
        self.Q = None  # optional orthogonal transform
        # self.eigenvalues = None

    @abstractmethod
    def _generate_diag(self):
        """Return an array of length n representing the spectrum."""
        pass

    def _build(self):
        np.random.seed(self.seed)
        self.diag_vals = self._generate_diag()
        self.eigenvalues = np.sort(self.diag_vals)[::-1]
        # self.Q, _ = np.linalg.qr(np.random.randn(self.n, self.n))
        self.Q = np.eye(self.n)

    def matvec(self, x):
        return self.Q @ (self.diag_vals * (self.Q.T @ x))

    def get_matvec(self):
        Q = self.Q
        QT = Q.T 
        diag = self.diag_vals
        return lambda x: Q @ (diag * (QT @ x))


    def get_true_eigenvalues(self, top_k=None):
        return self.eigenvalues[:top_k] if top_k else self.eigenvalues
