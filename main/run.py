import numpy as np
import argparse
import torch
from main.networks import SmallNet, LeNet
from main.run_util import run_experiment, SOLVER_MAP
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_and_report(model_key, solver, ood_name, n_runs=5):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if model_key == "mlp":
        model_fn = lambda: SmallNet()
        model_ckpt = "models/mlp.pt"
        id_dataset = "mnist"
        flatten = True
        steps = 50
    elif model_key == "lenet":
        model_fn = lambda: LeNet()
        model_ckpt = "models/lenet.pt"
        id_dataset = "fashion"
        flatten = False
        steps = 100
    else:
        raise ValueError(f"Unsupported model: {model_key}")
    if solver not in SOLVER_MAP:
        raise ValueError(f"Unsupported solver key: {solver}")
    solver_name = SOLVER_MAP[solver]

    config = {
        "model_fn": model_fn,
        "model_ckpt": model_ckpt,
        "id_dataset": id_dataset,
        "ood_dataset": ood_name.lower(),
        "flatten": flatten,
        "batch_size": 2000,
        "steps": steps,
        "ood_size": 500,
        "device": device,
        "method": solver,  
        "name": f"{model_key.upper()} on {id_dataset.upper()} vs {ood_name.upper()} using basis from {solver_name}"
    }


    scores = []
    for i in range(n_runs):
        print(f"\n==== Run {i+1} ====")
        score = run_experiment(config)
        scores.append(score)
                                                         
    mean, std = np.mean(scores), np.std(scores)
    print(f"\n>>> {config['name']} (AUROC over {n_runs} runs): {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "lenet"])
    parser.add_argument("--solver", type=str, choices=list(SOLVER_MAP.keys()), default="sl")
    parser.add_argument("--ood", type=str, required=True, choices=["fashion", "mnist", "kmnist", "rotate"])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    run_and_report(args.model, args.solver, args.ood, args.runs)
