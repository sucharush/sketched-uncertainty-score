from abc import ABC, abstractmethod

class Sketcher(ABC):
    @abstractmethod
    def apply_sketch(self, x):
        """
        Apply the sketching technique to vector or matrix x.
        """
        pass
    