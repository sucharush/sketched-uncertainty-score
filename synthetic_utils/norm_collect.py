import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sketch.sketch_base import Sketcher
from solvers.vanilla_lanczos import VanillaLanczos
from solvers.sketched_lanczos import SketchedLanczos
from solvers.randomized_arnoldi import RGSArnoldi

from synthetic_data.matrix_factory import PolyDecayMatrix 

# def plot_results(
#     G_matvec,
#     p,
#     k,
#     sketch: Sketcher,
#     k0=5,
#     outer_runs=10,
#     num_samples=10,
#     methods=("high", "sketched", "precond", "randomized"),
#     v = None,
#     with_plot=False,
#     verbose=True
# ):
#     def create_verbose_function(verbose):
#         def verbose_print(*args, **kwargs):
#             if verbose:
#                 print(*args, **kwargs)
#         return verbose_print

#     custom_print = create_verbose_function(verbose)

#     data = []

#     for i in range(outer_runs):
#         custom_print(f"Running outer iteration {i+1}/{outer_runs}...")

#         if "high" in methods:
#             hlz = VanillaLanczos(G_matvec=G_matvec, p=p, reorth=True, verbose=verbose)
#             hlz.run(num_steps=k)
#             U0 = hlz.get_basis()
#         if "sketched" in methods or "precond" in methods:
#             slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
#             slz.run(num_steps=k)
#             Q = slz.get_basis()
#         if "precond" in methods:
#             slz.run(num_steps=k, pre_steps=k0)
#             Q_precond = slz.get_basis()
#         if "randomized" in methods:
#             rlz = RGSArnoldi(G_matvec=G_matvec, p=p, sketch=sketch, verbose=verbose)
#             rlz.run(num_steps=k)
#             U_small = rlz.get_basis()

#         for sample_id in range(num_samples):
#             if not v:
#                 v = np.random.normal(size=(p,)).astype(np.complex128)
#             v = v.astype(np.complex128)
#             v = v / np.linalg.norm(v)
#             v_sketched = sketch.apply_sketch(v)

#             if "high" in methods:
#                 data.append({
#                     "Method": "High Memory",
#                     "Sample": sample_id,
#                     "Value": np.linalg.norm(U0.T @ v)
#                 })
#             if "sketched" in methods:
#                 data.append({
#                     "Method": "Sketched",
#                     "Sample": sample_id,
#                     "Value": np.linalg.norm(Q.T @ v_sketched)
#                 })
#             if "precond" in methods:
#                 data.append({
#                     "Method": "Preconditioned",
#                     "Sample": sample_id,
#                     "Value": np.linalg.norm(Q_precond.T @ v_sketched)
#                 })
#             if "randomized" in methods:
#                 data.append({
#                     "Method": "Randomized Arnoldi",
#                     "Sample": sample_id,
#                     "Value": np.linalg.norm(U_small.T @ v_sketched)
#                 })

#     df = pd.DataFrame(data)

#     if with_plot:
#         plt.figure()
#         sns.lineplot(
#             data=df, x="Sample", y="Value", hue="Method", style="Method",
#             errorbar="sd", err_style="band",
#             markers=["x"]*len(methods), dashes=False, err_kws={"alpha": 0.2}
#         )
#         plt.title("Projection Norm vs Sample")
#         plt.xlabel("Sample")
#         plt.ylabel("Norm Value")
#         plt.legend()
#         plt.grid()
#         plt.show()

#     return df

def plot_results(
    G_matvec,
    p,
    k,
    sketch: Sketcher,
    v_list=None,
    k0=5,
    outer_runs=10,
    methods=("high", "sketched", "precond", "randomized"),
    with_plot=False,
    verbose=True
):
    """
    Run projection experiment for a set of fixed vectors v (one per outer run).

    Parameters
    ----------
    v_list : list[np.ndarray] or None
        Optional list of vectors to use, length = outer_runs.
    """
    def create_verbose_function(verbose):
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        return verbose_print

    custom_print = create_verbose_function(verbose)
    data = []

    # If not given, generate outer_runs vectors
    if v_list is None:
        v_list = [np.random.normal(size=(p,)).astype(np.complex128) for _ in range(outer_runs)]
        v_list = [v / np.linalg.norm(v) for v in v_list]

    for run_id in range(outer_runs):
        v = v_list[run_id]
        v_sketched = sketch.apply_sketch(v)

        custom_print(f"Run {run_id+1}/{outer_runs}")

        if "high" in methods:
            hlz = VanillaLanczos(G_matvec=G_matvec, p=p, reorth=True, verbose=False)
            hlz.run(num_steps=k)
            U0 = hlz.get_basis()
            val = np.linalg.norm(U0.T @ v)
            data.append({"Method": "High Memory", "Value": val, "Run": run_id})

        if "sketched" in methods or "precond" in methods:
            slz = SketchedLanczos(G_matvec=G_matvec, p=p, sketch=sketch, verbose=False)
            slz.run(num_steps=k)
            Q = slz.get_basis()
            if "sketched" in methods:
                val = np.linalg.norm(Q.T @ v_sketched)
                data.append({"Method": "Sketched", "Value": val, "Run": run_id})
            if "precond" in methods:
                slz.run(num_steps=k, pre_steps=k0)
                Q_precond = slz.get_basis()
                val = np.linalg.norm(Q_precond.T @ v_sketched)
                data.append({"Method": "Preconditioned", "Value": val, "Run": run_id})

        if "randomized" in methods:
            rlz = RGSArnoldi(G_matvec=G_matvec, p=p, sketch=sketch, verbose=False)
            rlz.run(num_steps=k)
            U_small = rlz.get_basis()
            val = np.linalg.norm(U_small.T @ v_sketched)
            data.append({"Method": "Randomized Arnoldi", "Value": val, "Run": run_id})

    df = pd.DataFrame(data)

    if with_plot:
        plt.figure()
        sns.pointplot(
            data=df, x="Run", y="Value", hue="Method", dodge=0.2,
            markers="o", linestyles="-", errorbar="sd"
        )
        plt.title("Projection Norm over Outer Runs")
        plt.xlabel("Run ID")
        plt.ylabel("Norm Value")
        plt.grid(True)
        plt.legend()
        plt.show()

    return df



def aggregate_plot_results(df, num_samples):
    """
    Collapse raw projection results to (Method, Run)-level by averaging over samples.

    Parameters
    ----------
    df : pd.DataFrame
        Raw result dataframe from `plot_results`.
    num_samples : int
        Number of samples per outer run.

    Returns
    -------
    pd.DataFrame
        Columns: Method, Run, Value
    """
    df = df.copy()
    df["Run"] = df.groupby("Method").cumcount() // num_samples
    df_summary = df.groupby(["Method", "Run"]).agg(Value=("Value", "mean")).reset_index()
    return df_summary


def run_experiment_grid(
    p,
    d,
    k_list,
    s_list=None,
    outer_runs=5,
    num_samples=20,
    sketch_class=None,
    verbose=False
):
    """
    Run a grid of projection experiments on a PolyDecayMatrix with fixed (p, d),
    and varying (k, s).

    Parameters
    ----------
    p : int
        Matrix dimension.
    d : float
        Polynomial decay exponent.
    k_list : list[int]
        Krylov steps to test.
    s_list : list[int], optional
        Sketch sizes. Defaults to [2 * k for k in k_list].
    outer_runs : int
        Repetitions per experiment setting.
    num_samples : int
        Number of random vectors per run.
    sketch_class : callable
        Constructor: lambda p, s → sketch instance.
    verbose : bool
        If True, print debug messages.

    Returns
    -------
    pd.DataFrame
        Summary of projection values with metadata.
    """
    results = []
    s_list = [2 * k for k in k_list] if s_list is None else s_list
    # 1. Generate synthetic matrix and matvec
    mat = PolyDecayMatrix(n=p, d=d, seed=42)
    G_matvec = mat.get_matvec()

    for k, s in zip(k_list, s_list):
        print(f"Running: p={p}, k={k}, s={s}, d={d}")

        # 2. Initialize sketcher (if any)
        sketch = sketch_class(p=p, s=s) if sketch_class else None

        # 3. Run projection experiment
        df_raw = plot_results(
            G_matvec=G_matvec,
            p=p,
            k=k,
            sketch=sketch,
            k0=5,
            outer_runs=outer_runs,
            # num_samples=num_samples,
            with_plot=False,
            verbose=verbose
        )

        # 4. Aggregate over samples
        df_summary = aggregate_plot_results(df_raw, num_samples=num_samples)

        # 5. Attach metadata
        df_summary["p"] = p
        df_summary["k"] = k
        df_summary["s"] = s
        df_summary["d"] = d
        df_summary["run_id"] = f"p={p}_k={k}_s={s}_d={d}"

        results.append(df_summary)

    all_df = pd.concat(results, ignore_index=True)
    return all_df

import pandas as pd
import numpy as np

def run_projection_sweep_with_fixed_v(
    p,
    d,
    k_list,
    s_list,
    sketch_class,
    outer_runs=10,
    methods=("high", "sketched", "randomized"),
    verbose=False
):
    """
    Run projection norm experiments for all (k, s) pairs using the same v's.

    Parameters
    ----------
    p : int
        Matrix dimension.
    d : float
        Decay exponent for PolyDecayMatrix.
    k_list : list[int]
        Krylov steps to test.
    s_list : list[int]
        Sketch sizes to test.
    sketch_class : callable
        Constructor: sketch_class(p, s)
    outer_runs : int
        Number of test vectors (each reused across settings)
    methods : tuple[str]
        Subset of methods to run.
    verbose : bool
        If True, print progress.

    Returns
    -------
    pd.DataFrame
        Columns: ['Method', 'Run', 'Value', 'k', 's']
    """
    # Create matrix and test vectors
    mat = PolyDecayMatrix(n=p, d=d)
    G_matvec = mat.get_matvec()
    v_list = [np.random.randn(p).astype(np.complex128) for _ in range(outer_runs)]
    v_list = [v / np.linalg.norm(v) for v in v_list]

    all_df = []

    for s in s_list:
        sketch = sketch_class(p=p, s=s)
        for k in k_list:
            if s <= k:
                continue
            if verbose:
                print(f"Running: k={k}, s={s}")
            df = plot_results(
                G_matvec=G_matvec,
                p=p,
                k=k,
                sketch=sketch,
                v_list=v_list,
                outer_runs=outer_runs,
                methods=methods,
                with_plot=False,
                verbose=verbose
            )
            df["k"] = k
            df["s"] = s
            all_df.append(df)

    return pd.concat(all_df, ignore_index=True)
