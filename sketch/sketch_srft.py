from sketch.sketch_base import Sketcher
import numpy as np

class SRFTSketcher(Sketcher):
    def __init__(self, p, s=256, rfft=True):
        self.p = p
        self.s = s
        self.rfft = rfft
        self._D = None
        self._subset = None
        self._initialize_srft()

    def _initialize_srft(self):
        self._D = np.sign(np.random.randn(self.p))
        if self.rfft:
            fft_size = self.p // 2 + 1
        else:
            fft_size = self.p
        self._subset = np.sort(np.random.choice(fft_size, self.s, replace=False))
    
    def set_p(self, p):
        super().set_p(p)
        self._initialize_srft()
    
    def set_s(self, s):
        super().set_s(s)
        self._initialize_srft()

    def apply_sketch(self, x):

        if self._D is None or self._subset is None:
            self._initialize_srft()

        is_vector = (x.ndim == 1)
        if is_vector:
            x = x[:, np.newaxis]  # make it (p, 1)

        x_d = self._D[:, None] * x

        if self.rfft:
            x_f = np.fft.rfft(x_d, axis=0, norm="ortho")
        else:
            x_f = np.fft.fft(x_d, axis=0, norm="ortho")

        x_sub = x_f[self._subset, :]

        x_sketch = np.sqrt(self.p / self.s) * x_sub

        return x_sketch[:, 0] if is_vector else x_sketch

