import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lanczos.lanczos_hm import HighMemoryLanczos
from lanczos.lanczos_sketch import SketchedLanczos
from lanczos.lanczos_randomized import RandomizedLanczos

def create_verbose_function(verbose):
    """Return a print function that only outputs if verbose is True."""
    def verbose_print(*args, **kwargs):
        """Print arguments if verbose is enabled."""
        if verbose:
            print(*args, **kwargs)
    return verbose_print

def create_poly_decay_matvec(n, R=10, d=1):
    """
    Create a matrix-vector product function for a polynomial decay matrix of size n x n.

    Parameters
    ----------
    n : int
        Size of the matrix (and vector).
    R : int, optional
        Number of leading 1s on the diagonal. Defaults to 10.
    d : float, optional
        Decay parameter. Defaults to 1.

    Returns
    -------
    function
        A function that takes a vector x of shape (n,) and returns the product of
        the poly decay matrix and x.
    """
    # Precompute the diagonal values once
    diag_vals = np.ones(n)
    diag_vals[:R] = np.arange(1, R+1)[::-1]
    for i in range(R, n):
        diag_vals[i] = float(i - R + 2) ** (-d)

    # Define the matrix-vector product function using the precomputed diagonal
    def poly_decay_matvec(x):
        """
        Matrix-vector product with a precomputed diagonal of a poly decay matrix.

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            The input vector to be multiplied by the poly-decay matrix.

        Returns
        -------
        y : np.ndarray of shape (n,)
            The resulting product of poly_decay_matrix @ x
        """
        if len(x) != n:
            raise ValueError(f"Vector x must have length {n}.")
        return diag_vals * x
    
    return poly_decay_matvec

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


def plot_results(G_matvec, p, k, sketch, k0=5, outer_runs=10, num_samples=20, with_plot=False, verbose=True):
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
        hlz = HighMemoryLanczos(G_matvec=G_matvec, p=p, verbose=verbose)
        custom_print("   ------------Running High-memory Lanczos--------------")
        hlz.run(num_steps=k)
        U0, _ = hlz.get_top_eigenpairs()  # store for repeated use
        # print(np.diag(_))

        # (B) SketchedLanczos
        slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch,verbose=verbose)
        # Q = slz.sketched_lanczos(k=k)
        custom_print("   --------------Running Sketched Lanczos---------------")
        slz.sketched_lanczos(k=k)
        Q, _ = slz.get_top_eigenpairs()
        # print("Q", type(Q))
        custom_print("   -------Running Preconditioned Sketched Lanczos-------")
        Q_precond = slz.preconditioned_version(k=k, k0=k0)
        custom_print("   --------------Running Lanczos + Sketch---------------")
        Q_postsketch = slz.run_full_then_sketch(num_steps_full=k)
        
        # (C) Randomized Lanczos
        rlz = RandomizedLanczos(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
        custom_print("   -------------Running Randomized Lanczos--------------")
        rlz.run(num_steps=k)
        U_small, _ = rlz.get_top_eigenpairs()  # store for repeated use

        high_vals = []
        sketch_vals = []
        precond_vals = []
        post_sketch_vals = []
        rd_lcz_vals = []

        for _ in range(num_samples):
            v = np.random.normal(size=(p,))
            v = v / np.linalg.norm(v)

            # 1) High memory measure
            high_vals.append(np.linalg.norm(U0.T @ v))

            # 2) Sketched
            v_sketched = sketch.apply_sketch(v)
            # print(type(Q), type(v_sketched))
            sketch_vals.append(np.linalg.norm(Q.T @ v_sketched))

            # 3) Preconditioned
            precond_vals.append(np.linalg.norm(Q_precond.T @ v_sketched))

            # 4) Post-sketch measure
            post_sketch_vals.append(np.linalg.norm(Q_postsketch.T @ v_sketched))
            
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
            ["High Memory", "Sketched", "Preconditioned", "Randomized Lanczos"],
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
    pivot_df['Error Post Sketched'] = np.abs(pivot_df['High Memory'] - pivot_df['Post Sketched'])
    pivot_df['Error Randomized Lanczos'] = np.abs(pivot_df['High Memory'] - pivot_df['Randomized Lanczos'])

    # Initialize result dictionary
    results = {}

    # Calculate mean and std of these gaps and format results
    # for method in ['Sketched', 'Preconditioned', 'Post Sketched', 'Randomized Lanczos']:
    for method in ['Sketched', 'Preconditioned', 'Randomized Lanczos']:
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

def eigen_diag_desc(matvec, p, top_k):
    eigens = matvec(np.ones(p))[:top_k]
    return eigens

def collect_ritz_values(p, k, G_matvec, sketch, verbose):
    # (A) HighMemoryLanczos
    hlz = HighMemoryLanczos(G_matvec=G_matvec, p=p, verbose=verbose)
    hlz.run(num_steps=k)
    U0, L0 = hlz.get_top_eigenpairs()  # store for repeated use
    # print(np.diag(_))

    # (B) SketchedLanczos
    slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch,verbose=verbose)
    # Q = slz.sketched_lanczos(k=k)
    slz.sketched_lanczos(k=k)
    Q, L_sketched = slz.get_top_eigenpairs()
    # print("Q", type(Q))

    # (C) Randomized Lanczos
    rlz = RandomizedLanczos(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
    rlz.run(num_steps=k)
    U_small, L_randomized = rlz.get_top_eigenpairs()  # store for repeated use
    
    real_eigens = eigen_diag_desc(G_matvec, p=p, top_k=k)
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
    plt.scatter(indices_L0, L0_sorted, marker='x', color="green", label='High Memory Ritz')
    plt.scatter(indices_L_sketched, L_sketched_sorted, marker='x', color='red', label='Sketched Ritz')
    plt.scatter(indices_L_randomized, L_randomized_sorted, marker='x', color='blue', label='Randomized Ritz')
    plt.scatter(indices_real, real_eigens_sorted, marker='o', facecolors='none', edgecolors='black', label='True Eigenvalues')
    
    plt.xlabel('Index (sorted in descending order)')
    plt.ylabel('Eigenvalue')
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