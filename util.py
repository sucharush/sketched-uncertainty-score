import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sketch.sketch_base import Sketcher
from solvers.vanilla_lanczos import VanillaLanczos
from solvers.sketched_lanczos import SketchedLanczos
from solvers.randomized_arnoldi import RGSArnoldi

import numpy as np
from scipy.linalg import qr

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
    diag_vals[:R] = np.arange(1, R+1)[::-1]
    diag_vals[R:] = np.array([float(i - R + 2) ** (-d) for i in range(R, n)])

    # 2. Generate orthogonal matrix Q via QR decomposition
    Q, _ = qr(np.random.randn(n, n))  # Q @ Q.T = I

    # 3. Define the matvec: A(x) = Q @ Λ @ Q.T @ x
    def matvec(x):
        if len(x) != n:
            raise ValueError(f"Vector x must have length {n}.")
        return Q @ (diag_vals * (Q.T @ x))

    return matvec, np.sort(diag_vals)[::-1]


def create_G_matvec(n, sig_ev = 20):
    """
    Creates a matrix-vector multiplication function for a diagonal matrix.

    Parameters:
        p (int): The dimension of the matrix and vector.
        sig_ev (int): The number of significant eigenvalues.

    Returns:
        function: A function that takes a vector `v` and returns the product of the diagonal matrix and `v`.
    """

    # Create the diagonal values: higher values for significant eigenvalues, decaying for the rest
    high_values = np.linspace(10, 10 + 5 * (sig_ev - 1), sig_ev)[::-1]
    low_values = np.linspace(1, 0.1, n - sig_ev)
    diag = np.concatenate([high_values, low_values])

    # Define the matrix-vector multiplication function
    def G_matvec(v):
        return diag * v

    return G_matvec


def plot_results(G_matvec, p, k, sketch: Sketcher, k0=5, outer_runs=10, num_samples=20, with_plot=False, verbose=True):
    """

    Args:
        G_matvec (function): Matrix-vector product function that defines the matrix G.
        p (int): Dimension of the vector space for G.
        k (int): Rank (num. of iterations in Lanczos).
        sketch (Sketcher): Sketching method to be used.
        k0 (int, optional): Number of iterations for the full Lanczos process. Defaults to 5.
        outer_runs (int, optional): Number of times the process is repeated on each set of samples. Defaults to 10.
        num_samples (int, optional): Number of samples tested. Defaults to 20.
        with_plot (bool, optional): Defaults to False.


    Returns:
        Dataframe: summary of results.
    """
    def create_verbose_function(verbose):
        """Return a print function that only outputs if verbose is True."""
        def verbose_print(*args, **kwargs):
            """Print arguments if verbose is enabled."""
            if verbose:
                print(*args, **kwargs)
        return verbose_print
    custom_print = create_verbose_function(verbose)

    # We'll do 5 "outer" runs, each with `num_samples` random vectors
    outer_runs = 5

    high_all = []
    sketch_all = []
    precond_all = []
    post_sketch_all = []
    rd_lcz_all = []

    for i in range(outer_runs):
        custom_print(f"Running {i+1} ...")
        # (A) HighMemoryLanczos
        hlz = VanillaLanczos(G_matvec=G_matvec, p=p, reorth=True, verbose=verbose)
        custom_print("   ------------Running High-memory Lanczos--------------")
        hlz.run(num_steps=k)
        U0 = hlz.get_basis()  # store for repeated use
        # print(np.diag(_))

        # (B) SketchedLanczos
        slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch,verbose=verbose)
        # Q = slz.sketched_lanczos(k=k)
        custom_print("   --------------Running Sketched Lanczos---------------")
        slz.run(num_steps=k)
        Q = slz.get_basis()
        # print("Q", type(Q))
        custom_print("   -------Running Preconditioned Sketched Lanczos-------")
        slz.run(num_steps=k, pre_steps=k0)
        Q_precond = slz.get_basis()
        # custom_print("   --------------Running Lanczos + Sketch---------------")
        # Q_postsketch = slz.run_full_then_sketch(num_steps_full=k)
        
        # (C) Randomized Lanczos
        rlz = RGSArnoldi(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
        custom_print("   -------------Running Randomized Arnoldi--------------")
        rlz.run(num_steps=k)
        U_small = rlz.get_basis()  # store for repeated use

        high_vals = []
        sketch_vals = []
        precond_vals = []
        post_sketch_vals = []
        rd_lcz_vals = []

        for _ in range(num_samples):
            v = np.random.normal(size=(p,)).astype(np.complex128)
            v = v / np.linalg.norm(v)

            # 1) High memory measure
            high_vals.append(np.linalg.norm(U0.T @ v))

            # 2) Sketched
            v_sketched = sketch.apply_sketch(v)
            # print(Q.shape, Q_precond.shape, v_sketched.shape)
            sketch_vals.append(np.linalg.norm(Q.T @ v_sketched))

            # 3) Preconditioned
            precond_vals.append(np.linalg.norm(Q_precond.T @ v_sketched))

            # # 4) Post-sketch measure
            # post_sketch_vals.append(np.linalg.norm(Q_postsketch.T @ v_sketched))
            
            # 5) Randomized
            rd_lcz_vals.append(np.linalg.norm(U_small.T @ v_sketched))            
            

        high_all.append(high_vals)
        sketch_all.append(sketch_vals)
        precond_all.append(precond_vals)
        post_sketch_all.append(post_sketch_vals)
        rd_lcz_all.append(rd_lcz_vals)

    high = np.array(high_all)
    sketched = np.array(sketch_all)
    precond = np.array(precond_all)
    post_sketch = np.array(post_sketch_all)
    rd_lcz = np.array(rd_lcz_all)

    data = []
    for _, (method_name, arr) in enumerate(
        # zip(
        #     ["High Memory", "Sketched", "Preconditioned", "Post Sketched", "Randomized Lanczos"],
        #     [high, sketched, precond, post_sketch, rd_lcz],
        # )
        zip(
            ["High Memory", "Sketched", "Preconditioned", "Randomized Arnoldi"],
            [high, sketched, precond, rd_lcz],
        )
    ):
        for outer_run in range(outer_runs):
            for sample_idx in range(num_samples):
                data.append(
                    {
                        "Method": method_name,
                        "Sample": sample_idx,
                        "Value": arr[outer_run, sample_idx],
                    }
                )

    df = pd.DataFrame(data)
    if with_plot:
        plt.figure()
        sns.lineplot(
            data=df,
            x="Sample",
            y="Value",
            hue="Method",
            style="Method",
            errorbar="sd",
            err_style="band",
            markers=["x", "x", "x", "x"],
            dashes=False,
            err_kws={"alpha": 0.2},
        )

        plt.title("Results Over Multiple Samples (Mean ± Std Over 5 Outer Runs)")
        plt.xlabel("Sample Index")
        plt.ylabel("Norm Value")
        plt.legend()
        plt.show()
    return df

def aggregate_plot_results(df, num_samples):
    """
    Collapse the original output of `plot_results()` by averaging over samples
    for each method and outer run.

    Returns a DataFrame with one row per (Method, Run).
    """
    df = df.copy()
    df['Run'] = df.groupby('Method').cumcount() // num_samples
    summary_df = df.groupby(['Method', 'Run']).agg(
        Value=('Value', 'mean')
    ).reset_index()
    return summary_df

def run_experiment_grid(
    p_list,
    k_list,
    s_list=None,
    d_list=[1.0],
    outer_runs=5,
    num_samples=20,
    sketch_class=None,  # a constructor like: lambda p, s → sketcher instance
    verbose=False
):
    """
    Run projection experiments across (p, k, d) grid and return summarized results.

    Returns:
        all_df : pd.DataFrame
            Columns: ['Method', 'Run', 'Value', 'p', 'k', 'd']
    """
    results = []
    s_list = [2 * x for x in k_list] if s_list is None else s_list
    # print(f"Sketch size: {s_list}")
    for d in d_list:
        for p in p_list:
            for k,s in zip(k_list, s_list):
                # s=100
                # s = s
                run_id = f"p={p}_k={k}_s={s}_d={d}"
                print(f"Running: {run_id}")

                # 1. Matrix-vector product
                G_matvec, eigvals = create_poly_decay_matvec(n=p, d=d, seed=42)

                # 2. Sketcher
                sketch = sketch_class(p=p, s=s, rfft = False) if sketch_class else None

                # 3. Run projection experiment
                df = plot_results(
                    G_matvec=G_matvec,
                    p=p,
                    k=k,
                    sketch=sketch,
                    k0=5,
                    outer_runs=outer_runs,
                    num_samples=num_samples,
                    with_plot=False,
                    verbose=verbose
                )

                # 4. Aggregate over samples
                df_summary = aggregate_plot_results(df, num_samples=num_samples)

                # 5. Attach metadata
                df_summary["p"] = p
                df_summary["k"] = k
                df_summary["s"] = s
                df_summary["d"] = d
                df_summary["run_id"] = run_id

                results.append(df_summary)

    all_df = pd.concat(results, ignore_index=True)
    return all_df


def calculate_gaps(df):
    """
    Calculate the average gaps between the 'High Memory' method and other methods,
    and format the results as 'mean ± std'.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'Method', 'Sample', and 'Value'.

    Returns:
        dict: A dictionary with keys as methods and values as formatted strings showing mean and std.
    """

    # Compute means for each sample and method
    mean_df = df.groupby(['Method', 'Sample']).mean().reset_index()

    # Pivot the table so each method becomes a column
    pivot_df = mean_df.pivot(index='Sample', columns='Method', values='Value')

    # Calculate the differences from 'High Memory'
    pivot_df['Error Sketched'] = np.abs(pivot_df['High Memory'] - pivot_df['Sketched'])
    pivot_df['Error Preconditioned'] = np.abs(pivot_df['High Memory'] - pivot_df['Preconditioned'])
    # pivot_df['Error Post Sketched'] = np.abs(pivot_df['High Memory'] - pivot_df['Post Sketched'])
    pivot_df['Error Randomized Arnoldi'] = np.abs(pivot_df['High Memory'] - pivot_df['Randomized Arnoldi'])

    # Initialize result dictionary
    results = {}

    # Calculate mean and std of these gaps and format results
    # for method in ['Sketched', 'Preconditioned', 'Post Sketched', 'Randomized Lanczos']:
    for method in ['Sketched', 'Preconditioned', 'Randomized Arnoldi']:
        gap_key = f"Error {method}"
        mean_gap = pivot_df[gap_key].mean()
        # print(mean_gap)
        std_gap = pivot_df[gap_key].std()
        # print(std_gap)
        results[method] = f"{mean_gap:.3f} ± {std_gap:.3f}"
    # Also calculate and format results for 'High Memory' itself
    mean_high = pivot_df['High Memory'].mean()
    std_high = pivot_df['High Memory'].std()
    results['High Memory'] = f"{mean_high:.3f} ± {std_high:.3f}"
    # results['High Memory'] = f"{mean_high:.3f}"

    return results

def eigen_diag_desc(real_eigens, top_k):
    eigens = real_eigens[:top_k]
    return eigens

def collect_ritz_values(p, k, G_matvec, real_eigens, sketch, verbose):
    # (A) HighMemoryLanczos
    hlz = VanillaLanczos(G_matvec=G_matvec, p=p, verbose=verbose)
    hlz.run(num_steps=k)
    _, L0 = hlz.get_top_ritzpairs(return_vectors=True)  # store for repeated use
    # print(np.diag(_))

    # (B) SketchedLanczos
    slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch,verbose=verbose)
    # Q = slz.sketched_lanczos(k=k)
    slz.run(num_steps=k, pre_steps=0)
    _, L_sketched = slz.get_top_ritzpairs(return_vectors=True)
    # print("Q", type(Q))

    # (C) Randomized Lanczos
    rlz = RGSArnoldi(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
    rlz.run(num_steps=k)
    _, L_randomized = rlz.get_top_ritzpairs()  # store for repeated use
    
    real_eigens = eigen_diag_desc(real_eigens, top_k=k)
    # print(real_eigens)
    return L0, L_sketched, L_randomized, real_eigens


def plot_eigenvalues(L0, L_sketched, L_randomized, real_eigens, ylog = False):
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
    plt.scatter(indices_L0, L0_sorted, marker='x', color="green", label='High Memory Lanczos')
    plt.scatter(indices_L_sketched, L_sketched_sorted, marker='x', color='red', label='Sketched Lanczos')
    plt.scatter(indices_L_randomized, L_randomized_sorted, marker='x', color='blue', label='Randomized Arnoldi')
    plt.scatter(indices_real, real_eigens_sorted, marker='o', facecolors='none', edgecolors='black', label='True Eigenvalues')
    plt.xticks(np.arange(len(real_eigens_sorted)))  # force integer ticks

    plt.xlabel('Index (sorted in descending order)')
    plt.ylabel('Eigenvalue (Ritz value)')
    plt.title('Comparison of Computed and True Eigenvalues')
    plt.legend()
    plt.grid(True)
    
    if ylog:
    # Set the y-axis to a base-10 logarithmic scale
        plt.yscale('log', base=10)
    
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