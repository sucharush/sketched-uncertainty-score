import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt
from typing import Tuple
from sketch.sketch_base import Sketcher
from solvers.vanilla_lanczos import VanillaLanczos
from solvers.sketched_lanczos import SketchedLanczos
from solvers.randomized_arnoldi import RGSArnoldi
from synthetic_examples.synthetic_data.matrix_factory import PolyDecayMatrix
import os


def create_poly_decay_matvec(n, R=10, d=1, seed=44):
    """
    Create a symmetric matrix-vector product function A(x) = Q Λ Qᵗ x
    where Λ is a polynomial decay diagonal, and Q is orthogonal.

    Parameters
    ----------
    n : int
        Matrix size.
    R : int
        Number of leading large diagonal entries.
    d : float
        Polynomial decay exponent.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    function
        A function matvec(x) that applies A = Q Λ Qᵗ to a vector x.
    """

    if seed is not None:
        np.random.seed(seed)

    # 1. Construct polynomial decay spectrum
    diag_vals = np.ones(n)
    diag_vals[:R] = np.arange(1, R + 1)[::-1]
    diag_vals[R:] = np.array([float(i - R + 2) ** (-d) for i in range(R, n)])

    # 2. Generate orthogonal matrix Q via QR decomposition
    Q, _ = qr(np.random.randn(n, n))  # Q @ Q.T = I

    # 3. Define the matvec: A(x) = Q @ Λ @ Q.T @ x
    def matvec(x):
        if len(x) != n:
            raise ValueError(f"Vector x must have length {n}.")
        return Q @ (diag_vals * (Q.T @ x))

    return matvec, np.sort(diag_vals)[::-1]


def eigen_diag_desc(real_eigens, top_k):
    eigens = real_eigens[:top_k]
    return eigens


def top_percent(eigs, percent=0.9):
    """Return top `percent` of eigenvalues by descending |value|"""
    k = int(len(eigs) * percent)
    idx = np.argsort(-np.abs(eigs))[:k]
    return eigs[idx]


def collect_ritz_values(p, k, G_matvec, real_eigens, sketch, verbose):
    # (A) HighMemoryLanczos
    hlz = VanillaLanczos(G_matvec=G_matvec, p=p, verbose=verbose)
    hlz.run(num_steps=k)
    _, L0 = hlz.get_top_ritzpairs(return_vectors=True)  # store for repeated use
    # print(np.diag(_))

    # (B) SketchedLanczos
    slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
    # Q = slz.sketched_lanczos(k=k)
    slz.run(num_steps=k, pre_steps=0)
    _, L_sketched = slz.get_top_ritzpairs(return_vectors=True)
    # print("Q", type(Q))

    # (C) Randomized Lanczos
    rlz = RGSArnoldi(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
    rlz.run(num_steps=k)
    _, L_randomized = rlz.get_top_ritzpairs()  # store for repeated use

    real_eigens = eigen_diag_desc(real_eigens, top_k=k)
    # print(len(real_eigens))
    return L0, L_sketched, L_randomized, real_eigens


def plot_eigenvalues(
    L0, L_sketched, L_randomized, real_eigens, ylog=False, show_sub=False
):
    """
    Plot computed eigenvalues (Ritz values) and true eigenvalues on a base-10 log scale.

    Each input is sorted in descending order and plotted against its index.

    Parameters
    ----------
    L0 : array-like
        Eigenvalues from High Memory Lanczos.
    L_sketched : array-like
        Eigenvalues from Sketched Lanczos.
    L_randomized : array-like
        Eigenvalues from Randomized Lanczos.
    real_eigens : array-like
        True eigenvalues.
    """
    # Sort each set in descending order
    L0_sorted = np.sort(L0)[::-1]
    L_sketched_sorted = np.sort(L_sketched)[::-1]
    L_randomized_sorted = np.sort(L_randomized)[::-1]
    real_eigens_sorted = np.sort(real_eigens)[::-1]

    # Create index arrays for each
    indices_L0 = np.arange(len(L0_sorted))
    indices_L_sketched = np.arange(len(L_sketched_sorted))
    indices_L_randomized = np.arange(len(L_randomized_sorted))
    indices_real = np.arange(len(real_eigens_sorted))

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot each set with distinct markers and colors
    plt.scatter(
        indices_L0, L0_sorted, marker="x", color="green", label="High Memory Lanczos"
    )
    plt.scatter(
        indices_L_sketched,
        L_sketched_sorted,
        marker="x",
        color="red",
        label="Sketched Lanczos",
    )
    plt.scatter(
        indices_L_randomized,
        L_randomized_sorted,
        marker="x",
        color="blue",
        label="Randomized Arnoldi",
    )
    plt.scatter(
        indices_real,
        real_eigens_sorted,
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="True Eigenvalues",
    )
    plt.xticks(np.arange(len(real_eigens_sorted)))  # force integer ticks

    plt.xlabel("Index (sorted in descending order)")
    plt.ylabel("Eigenvalue (Ritz value)")
    plt.title("Comparison of Computed and True Eigenvalues")
    plt.legend()
    plt.grid(True)

    if ylog:
        # Set the y-axis to a base-10 logarithmic scale
        plt.yscale("log", base=10)
    if show_sub:
        plt.show()


def plot_eigenvalue_plane(
    L0, L_sketched, L_randomized, real_eigens, x_log=False, show_sub=False
):
    """
    Plot eigenvalues in the complex plane (Re vs Im) with distinct markers.

    Parameters
    ----------
    L0, L_sketched, L_randomized, real_eigens : array-like
        Eigenvalue arrays (can be complex).
    x_log : bool
        Whether to apply log-scale on the real axis (log(|Re(λ)|)).
    show_sub : bool
        Whether to call plt.show() inside the function.
    """
    plt.figure(figsize=(8, 6))

    def transform(x):
        return np.log10(np.abs(x.real)) * np.sign(x.real) if x_log else x.real

    # alpha = 0.6
    # def transform(x):
    #     # alpha = alpha  # between 0 (heavily compressed) and 1 (no compression)
    #     return np.sign(x.real) * (np.abs(x.real) ** alpha)

    plt.scatter(
        transform(top_percent(L0)),
        top_percent(L0).imag,
        marker="x",
        color="green",
        label="High Memory Lanczos",
    )
    plt.scatter(
        transform(top_percent(L_sketched)),
        top_percent(L_sketched).imag,
        marker="+",
        color="red",
        label="Sketched Lanczos",
    )
    plt.scatter(
        transform(top_percent(L_randomized)),
        top_percent(L_randomized).imag,
        marker="1",
        color="blue",
        label="Randomized Arnoldi",
    )

    plt.scatter(
        transform(real_eigens),
        real_eigens.imag,
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="True Eigenvalues",
    )
    all_real = np.concatenate(
        [L0.real, L_sketched.real, L_randomized.real, real_eigens.real]
    )
    min_pos = np.min(all_real[all_real > 0])
    max_val = np.max(all_real)
    plt.xlim(min_pos - 1e-1, max_val * 1.2)

    plt.ylim(-0.15, 0.15)

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)

    # plt.xlabel(f"$\mathrm{{sign}}(\mathrm{{Re}}(\lambda)) \cdot |\mathrm{{Re}}(\lambda)|^{{{alpha}}}$" if x_log else "Real Part")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.xscale("symlog")
    plt.title("Eigenvalues in Complex Plane" + (" (Log-Re)" if x_log else ""))
    plt.legend()

    if show_sub:
        plt.show()


# def poly_decay_matvec(x, n, R=10, d=1):
#     """
#     Mat-vec product for the polynomial decay matrix (size n x n) with vector x.

#     Parameters
#     ----------
#     x : np.ndarray of shape (p,)
#         The input vector to be multiplied by the poly-decay matrix.
#     n : int
#         Size of the matrix (and vector).
#     R : int
#         Number of leading 1s on the diagonal.
#     d : float
#         Decay parameter.

#     Returns
#     -------
#     y : np.ndarray of shape (n,)
#         The resulting product of poly_decay_matrix(n, R, d) @ x
#     """
#     if len(x) != n:
#         raise ValueError(f"Vector x must have length {n}.")

#     diag_vals = np.ones(n)
#     diag_vals[:R] = np.arange(1, R+1)
#     for i in range(R, n):
#         diag_vals[i] = float(i - R + 2) ** (-d)

#     return diag_vals * x


def plot_ritz_sweep(
    k_list,
    s_list,
    p=2000,
    R=20,
    d=1.0,
    sketch_class=None,
    matrix_class=PolyDecayMatrix,
    save_dir=None,
    verbose=False,
    show=True,
    x_log=False,
):
    """
    For each (k, s) pair, plot and optionally save eigenvalue comparisons.

    Parameters
    ----------
    k_list : list of int
        Krylov iteration counts to try.
    s_list : list of int
        Sketch sizes to try.
    p : int
        Matrix size.
    d : float
        Polynomial decay exponent.
    sketch_class : callable
        Sketch class constructor with signature sketch_class(p, s).
    save_dir : str or None
        If provided, saves figures under this directory.
    verbose : bool
        If True, print progress.
    show : bool
        Whether to call plt.show(). If False, only saves.
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    # G_matvec, true_eigs = create_poly_decay_matvec(n=p, R=10, d=d)
    mat = matrix_class(n=p, R=R, d=d)
    G_matvec = mat.get_matvec()
    true_eigs = mat.get_true_eigenvalues()
    # G_matvec, true_eigs = create_poly_decay_matvec(n=p, R=10, d=1)

    for k in k_list:
        for s in s_list:
            if s <= k:
                continue
            if verbose:
                print(f"[Plotting] k={k}, s={s}")
            sketch = sketch_class(p=p, s=s)
            L0, Ls, Lr, eigs_k = collect_ritz_values(
                p=p,
                k=k,
                G_matvec=G_matvec,
                real_eigens=true_eigs,
                sketch=sketch,
                verbose=verbose,
            )
            # L0, Ls, Lr, eigs_k = collect_ritz_values(p, k, G_matvec, true_eigs, sketch, verbose=False)
            plot_eigenvalue_plane(L0, Ls, Lr, eigs_k, x_log=x_log)
            plt.title(f"Ritz Values: $k={k}$, $s={s}$")
            if save_dir:
                fname = os.path.join(save_dir, f"ritz_k{k}_s{s}.png")
                plt.savefig(fname, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()
