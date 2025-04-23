import numpy as np

class LanczosBase:
    def __init__(self, G_matvec, p, verbose=False):
        """
        Initialize the Lanczos algorithm.

        Parameters:
        G_matvec : callable
            Function to perform the matrix-vector multiplication with matrix G.
        p : int
            Dimension of the vector space.
        """
        self.G_matvec = G_matvec
        self.p = p
        self.tridiagonal = None  # Ensure this attribute is always available
        self.T = None
        self.V = None  
        self.verbose = verbose
        self.custom_print = self._create_verbose_function(self.verbose)
    
    def _create_verbose_function(self, verbose):
        """Return a print function that only outputs if verbose is True."""
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        return verbose_print
    
    def build_tridiagonal(self):
        if self.tridiagonal is None:
            raise ValueError("Tridiagonal matrix not set. Run the Lanczos process first.")
        alphas, betas = self.tridiagonal
        T = np.diag(alphas)
        if len(betas) > 0:
            T += np.diag(betas, 1) + np.diag(betas, -1)
        self.T = T
        
    def get_top_eigenpairs(self, k=None, evec=True):
        """
        Compute the eigenpairs of the stored tridiagonal matrix T.
        Optionally return only the top k eigenpairs.
        
        returns:
            U0 (p, k): approx. top-k eigenvectors
            evals (0, k): approx. top-k eigenvalues
        """
        # TODO: rename
        # always rebuild the Hessenberg!!!!
        self.build_tridiagonal()
        evals, evecs = np.linalg.eigh(self.T)
        if self.V is None:
            raise ValueError("Basis not set. Run the method first to compute the basis.")
        
        idx = np.argsort(evals)[::-1]  # sort
        evals = evals[idx]
        evecs = evecs[:, idx]

        if k is not None:
            if k > len(evals):
                k = len(evals)
            evals = evals[:k]
            evecs = evecs[:, :k]

        if evec:
            U0 = self.V @ evecs
            # if not np.allclose(U0.T@U0, np.identity(len(evals)),atol=1e-8):
            #     print("Warning: Ritz vectors are not orthogonal.")
            return U0, evals
        else:
            return None, evals
        
    
    def _initialize_vector(self):
        """Generate a random normalized vector."""
        v = np.random.randn(self.p)
        return v / np.linalg.norm(v)

    def _reorthogonalize(self, V, w):
        """Perform reorthogonalization to ensure numerical stability."""
        for i in range(V.shape[1]):
            proj = np.dot(V[:, i], w)
            w -= proj * V[:, i]
        return w

    def run(self, num_steps):
        """Run the Lanczos algorithm, to be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")
