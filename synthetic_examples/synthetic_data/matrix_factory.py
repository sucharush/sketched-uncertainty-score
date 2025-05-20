import numpy as np
from synthetic_examples.synthetic_data.matrix_base import BaseMatrixGenerator
# synthetic_data.matrix_base import BaseMatrixGenerator
class PolyDecayMatrix(BaseMatrixGenerator):
    """
    Diagonal spectrum:
        λ_1, ..., λ_R = R, R-1, ..., 1
        λ_{R+1:} = (i - R + 2)^(-d)

    Parameters
    ----------
    n : int
        Matrix size.
    R : int
        Number of leading eigenvalues with flat (descending) profile.
    d : float
        Polynomial decay exponent for the tail.
    seed : int
        Random seed.
    """
    def __init__(self, n, R=20, d=1.0, seed=44):
        self.R = R
        self.d = d
        super().__init__(n, seed)
        self._build()

    # def _generate_diag(self):
    #     diag = np.ones(self.n)
    #     diag[:self.R] = np.arange(1, self.R + 1)[::-1]
    #     diag[self.R:] = [float(i - self.R + 2) ** (-self.d) for i in range(self.R, self.n)]
    #     return np.array(diag)
    def _generate_diag(self):
        return self.R / (np.arange(1, self.n + 1) ** self.d)

class ExpDecayMatrix(BaseMatrixGenerator):
    """
    Diagonal spectrum:
        λ_i = R * exp(-d * i)

    Parameters
    ----------
    n : int
        Matrix size.
    R : int
        Scale factor. Controls the largest eigenvalue.
    d : float
        Exponential decay rate (larger = faster decay).
    seed : int
        Random seed.
    """
    def __init__(self, n, R=100, d=0.2, seed=42):  
        self.d = d
        self.R = R
        super().__init__(n, seed)
        self._build()

    def _generate_diag(self):
        return self.R * np.exp(-self.d * np.arange(self.n))





