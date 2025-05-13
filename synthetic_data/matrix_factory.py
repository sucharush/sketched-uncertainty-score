import numpy as np
from synthetic_data.matrix_base import BaseMatrixGenerator
class PolyDecayMatrix(BaseMatrixGenerator):
    def __init__(self, n, R=10, d=1.0, seed=44):
        self.R = R
        self.d = d
        super().__init__(n, seed)
        self._build()

    def _generate_diag(self):
        diag = np.ones(self.n)
        diag[:self.R] = np.arange(1, self.R + 1)[::-1]
        diag[self.R:] = [float(i - self.R + 2) ** (-self.d) for i in range(self.R, self.n)]
        return np.array(diag)

class ExpDecayMatrix(BaseMatrixGenerator):
    def __init__(self, n, rate=0.2, seed=42):
        self.rate = rate
        super().__init__(n, seed)
        self._build()

    def _generate_diag(self):
        scale = 100  # or choose based on desired max eigenvalue
        return scale * np.exp(-self.rate * np.arange(self.n))
        # return np.exp(-self.rate * np.arange(self.n))

class LowRankPlusNoise(BaseMatrixGenerator):
    def __init__(self, n, rank=10, noise_level=0.1, seed=42):
        self.rank = rank
        self.noise_level = noise_level
        super().__init__(n, seed)
        self._build()

    def _generate_diag(self):
        signal = np.ones(self.rank)
        noise = self.noise_level * np.random.rand(self.n - self.rank)
        return np.concatenate([signal, noise])
