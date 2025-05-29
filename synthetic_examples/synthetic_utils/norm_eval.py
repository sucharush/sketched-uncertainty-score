import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from synthetic_examples.synthetic_utils.norm_collect import run_experiment_grid


def make_type_map(k_fixed, s_fixed):
    return {
        "sweep_s": (f"Varying $s$, fixed $k={k_fixed}$", "$s$"),
        "fixed_s": (f"Fixed $s={s_fixed}$, varying $k$", "$k$"),
        "s_eq_2k": ("$s=2k$, varying $k$", "$k$"),
    }


def build_comparison_data(cfg, sketch_class):
    p = cfg["p"]
    d = cfg["d"]
    R = cfg["R"]
    outer_runs = cfg["outer_runs"]
    num_samples = cfg["num_samples"]
    k_list2 = cfg["k_list2"]
    s_list1 = cfg["s_list1"]
    s_fixed = cfg["s_fixed"]
    k_fixed = cfg["k_fixed"]
    sketch_class = cfg["sketch_class"]
    matrix_class = cfg["matrix_class"]

    cfg["type_map"] = make_type_map(k_fixed, s_fixed)

    # (1) Varying s, fixed k
    k_list1 = [k_fixed] * len(s_list1)
    df1 = run_experiment_grid(
        p=p,
        d=d,
        k_list=k_list1,
        s_list=s_list1,
        outer_runs=outer_runs,
        num_samples=num_samples,
        sketch_class=sketch_class,
        matrix_class=matrix_class,
        R=R,
        verbose=False,
    )
    df1["xval"] = df1["s"]
    df1["type"] = "sweep_s"

    # # (2) Fixed s, varying k
    # s_list2 = [s_fixed] * len(k_list2)
    # df2 = run_experiment_grid(
    #     p=p,
    #     d=d,
    #     k_list=k_list2,
    #     s_list=s_list2,
    #     outer_runs=outer_runs,
    #     num_samples=num_samples,
    #     sketch_class=sketch_class,
    #     matrix_class=matrix_class,
    #     R=R,
    #     verbose=False,
    # )
    # df2["xval"] = df2["k"]
    # df2["type"] = "fixed_s"

    # (3) s = 2k
    s_list3 = [2 * k for k in k_list2]
    df3 = run_experiment_grid(
        p=p,
        d=d,
        k_list=k_list2,
        s_list=s_list3,
        outer_runs=outer_runs,
        num_samples=num_samples,
        sketch_class=sketch_class,
        matrix_class=matrix_class,
        R=R,
        verbose=False,
    )
    df3["xval"] = df3["k"]
    df3["type"] = "s_eq_2k"

    # all_df = pd.concat([df1, df2, df3], ignore_index=True)
    all_df = pd.concat([df1, df3], ignore_index=True)

    base_df = all_df[all_df["Method"] == "Lanczos (w/o reorth.)"]
    diff_df = all_df[all_df["Method"] != "Lanczos (w/o reorth.)"].copy()
    merged_df = diff_df.merge(
        base_df[["xval", "type", "Run", "Value"]],
        on=["xval", "type", "Run"],
        suffixes=("", "_base"),
    )
    merged_df["AbsDiff"] = np.abs(merged_df["Value"] - merged_df["Value_base"])
    return all_df, merged_df


def plot_comparison_grid(cfg, all_df, merged_df, **kwargs):
    type_map = cfg["type_map"]
    figsize = kwargs.get("figsize", (10, 10))
    linewidth = kwargs.get("linewidth", 1.5)
    palette = kwargs.get("palette", "deep")
    title_prefix = kwargs.get("title_prefix", "")
    suptitle = kwargs.get("suptitle", None)

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey="row")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # for i, t in enumerate(["sweep_s", "fixed_s", "s_eq_2k"]):
    for i, t in enumerate(["sweep_s", "s_eq_2k"]):
        sns.pointplot(
            data=all_df[all_df["type"] == t],
            x="xval",
            y="Value",
            hue="Method",
            dodge=0.1,
            errorbar="sd",
            join=True,
            markers="o",
            linestyles="-",
            linewidth=linewidth,
            capsize=0.1,
            palette=palette,
            ax=axes[0, i],
        )
        axes[0, i].set_title(title_prefix + type_map[t][0])
        axes[0, i].set_xlabel(type_map[t][1])
        axes[0, i].set_ylabel("$\\|U^T v\\|_2$ or $\\|U_S^T (Sv)\\|_2$")
        axes[0, i].grid(True)

        sns.pointplot(
            data=merged_df[merged_df["type"] == t],
            x="xval",
            y="AbsDiff",
            hue="Method",
            dodge=0.1,
            errorbar="sd",
            join=True,
            markers="o",
            linestyles="-",
            linewidth=linewidth,
            capsize=0.1,
            palette=palette,
            ax=axes[1, i],
        )
        axes[1, i].set_xlabel(type_map[t][1])
        axes[1, i].set_ylabel("$|\\|U^T v\\|_2 - \\|U_S^T (Sv)\\|_2|$")
        axes[1, i].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_comparison_grid(cfg, merged_df, **kwargs):
    k_fixed = cfg["k_fixed"]
    s_fixed = cfg["s_fixed"]
    type_map = make_type_map(k_fixed, s_fixed)
    figsize = kwargs.get("figsize", (18, 8))
    linewidth = kwargs.get("linewidth", 1.5)
    palette = kwargs.get("palette", "deep")
    title_prefix = kwargs.get("title_prefix", "")
    suptitle = kwargs.get("suptitle", None)

    # fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=20, y=1.05)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # for i, t in enumerate(["sweep_s", "fixed_s", "s_eq_2k"]):
    for i, t in enumerate(["sweep_s", "s_eq_2k"]):
        sns.pointplot(
            data=merged_df[merged_df["type"] == t],
            x="xval",
            y="AbsDiff",
            hue="Method",
            dodge=0.1,
            errorbar="sd",
            join=True,
            markers="o",
            linestyles="-",
            linewidth=linewidth,
            capsize=0.1,
            palette=palette,
            ax=axes[i],
        )
        axes[i].set_title(title_prefix + type_map[t][0], fontsize=20)
        axes[i].set_xlabel(type_map[t][1], fontsize=15)
        axes[i].set_ylabel(r"Error $|\|U^\top v\|_2 - \|U_S^\top (Sv)\|_2|$", fontsize=15)
        axes[i].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
def plot_comparison_individual(cfg, merged_df, save_dir=None, **kwargs):
    k_fixed = cfg["k_fixed"]
    s_fixed = cfg["s_fixed"]
    type_map = make_type_map(k_fixed, s_fixed)
    plot_types = ["sweep_s", "s_eq_2k"]

    figsize = kwargs.get("figsize", (8, 6))
    linewidth = kwargs.get("linewidth", 1.5)
    palette = kwargs.get("palette", "deep")
    title_prefix = kwargs.get("title_prefix", "")
    suptitle = kwargs.get("suptitle", None)

    # Compute global y-axis limits across all subplots
    y_min = merged_df["AbsDiff"].min()
    y_max = merged_df["AbsDiff"].max()
    y_margin = 0.05 * (y_max - y_min)
    y_bounds = (y_min - y_margin, y_max + y_margin)

    for t in plot_types:
        fig, ax = plt.subplots(figsize=figsize)

        sns.pointplot(
            data=merged_df[merged_df["type"] == t],
            x="xval",
            y="AbsDiff",
            hue="Method",
            dodge=0.1,
            errorbar="sd",
            join=True,
            markers="o",
            linestyles="-",
            linewidth=linewidth,
            capsize=0.1,
            palette=palette,
            ax=ax,
        )

        if suptitle:
            fig.suptitle(suptitle, fontsize=16, y=1.02)

        ax.set_title(title_prefix + type_map[t][0], fontsize=16)
        ax.set_xlabel(type_map[t][1], fontsize=14)
        ax.set_ylabel(r"Error $|\|U^\top v\|_2 - \|U_S^\top (Sv)\|_2|$", fontsize=14)
        ax.set_ylim(*y_bounds)
        ax.grid(True)

        plt.tight_layout()
        # NEW: save if save_dir is provided
        if save_dir is not None:
            # os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"norm_{t}.pdf")
            print(fname)
            plt.savefig(fname, bbox_inches="tight")
        plt.show()

