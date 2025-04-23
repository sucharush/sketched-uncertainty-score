from sketch.sketch_base import Sketcher
import numpy as np

class GaussianSketcher(Sketcher):
    def __init__(self, p, s=256):
        self.p = p
        self.s = s
        self.S = None
        self._initialize_gaussian()

    def _initialize_gaussian(self):
        self.S = np.random.normal(size=(self.s, self.p)) / np.sqrt(self.s)
    
    def set_p(self, p):
        super().set_p(p)
        self._initialize_gaussian()
        
    def set_s(self, s):
        super().set_s(s)
        self._initialize_gaussian()

    def apply_sketch(self, x):

        if self.S is None:
            self._initialize_gaussian()

        x_sketched = self.S @ x  # Works for both vector and matrix x

        return x_sketched