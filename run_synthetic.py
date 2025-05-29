import warnings
warnings.filterwarnings("ignore")

from synthetic_examples.synthetic_utils.norm_eval import (
    build_comparison_data, 
    plot_comparison_individual
)
from synthetic_examples.synthetic_utils.eigen_eval import plot_ritz_sweep
from sketch.sketch_srft import SRFTSketcher
from synthetic_examples.synthetic_data.matrix_factory import PolyDecayMatrix


def main():
    config = {
        "p": 2000,
        "R": 1000,
        "d": 2,
        "k_fixed": 20,
        "s_fixed": 100,
        "k_list2": [10, 20, 30, 40, 50, 60],
        "s_list1": [20, 40, 60, 80, 100],
        "outer_runs": 5,
        "num_samples": 1,
        "sketch_class": SRFTSketcher,
        "matrix_class": PolyDecayMatrix
    }
    # Plot Ritz values
    plot_ritz_sweep(
        k_list=[20, 40],
        s_list=[50, 100],
        p=config["p"],
        R=config["R"],
        d=config["d"],
        sketch_class=config["sketch_class"],
        matrix_class=config["matrix_class"],
        verbose=True,
        x_log=True,
    )
    
    # Run norm comparison
    all_df, merged_df = build_comparison_data(cfg=config, sketch_class=SRFTSketcher)
    plot_comparison_individual(cfg=config, all_df=all_df, merged_df=merged_df)


if __name__ == "__main__":
    main()
