import numpy as np
import pytest
from solvers.vanilla_lanczos import VanillaLanczos
from solvers.sketched_lanczos import SketchedLanczos
from solvers.randomized_arnoldi import RGSArnoldi

# pytest tests/test_krylov_methods.py -v

class IdentitySketch:
    def __init__(self, p):
        self.p = self.s = p
    def apply_sketch(self, x):
        return x
    
def test_identity_sketch():
    p = 5
    S = IdentitySketch(p)
    X = np.random.randn(p, 3)
    Sx = S.apply_sketch(X)
    assert np.allclose(Sx, X)
    
SOLVER_CONFIGS = [
    ("HighMemoLanczos", VanillaLanczos, {"reorth": True}),
    ("LowMemoLanczos", VanillaLanczos, {"reorth": False}),
    ("SketchedLanczos", SketchedLanczos, {"sketch": IdentitySketch(100)}),
    # TODO: test preconditioned sketched lanczos
    ("RGSArnoldi", RGSArnoldi, {"sketch": IdentitySketch(100)}),
]

@pytest.fixture
def diag_problem():
    p = 100
    k = 15
    sig_ev = 10
    high_values = np.linspace(10, 10 + 5 * (sig_ev - 1), sig_ev)[::-1]
    low_values = np.linspace(1, 0.1, p - sig_ev)
    diag = np.concatenate([high_values, low_values])
    A = np.diag(diag)
    def G_matvec(v): return diag * v
    return A, G_matvec, p, k, diag

@pytest.mark.parametrize("name, SolverClass, kwargs", SOLVER_CONFIGS)
def test_ritz_value_accuracy(name, SolverClass, kwargs, diag_problem):
    A, G_matvec, p, k, true_evals = diag_problem
    solver = SolverClass(G_matvec=G_matvec, p=p, **kwargs)
    if isinstance(solver, SketchedLanczos):
        solver.run(num_steps=k, pre_steps=0)
    else:
        solver.run(num_steps=k)
    _, ritz_vals = solver.get_top_ritzpairs(return_vectors=False)
    err = np.linalg.norm(ritz_vals[:10] - true_evals[:10])
    assert err < 1e-2
    
@pytest.mark.parametrize("name, SolverClass, kwargs", SOLVER_CONFIGS[:2])  # Only vanilla supports dimension check
def test_dimension_consistency(name, SolverClass, kwargs, diag_problem):
    _, G_matvec, p, k, _ = diag_problem
    solver = SolverClass(G_matvec=G_matvec, p=p, **kwargs)
    solver.run(num_steps=k)
    solver._build_Hessenberg()
    alphas = solver.alphas
    betas = solver.betas
    H = solver.H
    V = solver.V
    assert len(betas) == len(alphas)
    assert H.shape == (len(alphas), len(alphas))
    if V is not None:
        assert V.shape[1] == len(alphas)
        if hasattr(solver, 's'):
            assert V.shape[0] == solver.s
        else:
            assert V.shape[0] == solver.p
            