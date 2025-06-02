from torchvision import datasets, transforms
import torch
from uncertainty.ggn import GGNMatVecOperator
from uncertainty.evaluation_slu import SLUEvaluator
from sketch.sketch_srft import SRFTSketcher
from solvers.sketched_lanczos import SketchedLanczos
from solvers.vanilla_lanczos import VanillaLanczos
from solvers.randomized_arnoldi import RGSArnoldi
import os
import numpy as np

def load_dataset(name, flatten=True, train=True, batch_size=2000, device="cpu", rotate_angle=None):

    transform_list = []
    if rotate_angle is not None:
        transform_list.append(transforms.RandomRotation((rotate_angle, rotate_angle)))
    transform_list.append(transforms.ToTensor())
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)

    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion": datasets.FashionMNIST,
        "kmnist": datasets.KMNIST,
    }

    dataset_cls = dataset_map[name]
    dataset = dataset_cls(root="./data", train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X, Y = next(iter(loader))
    return X.to(device), Y.to(device)


def load_id_train_subset(name, flatten=True, device="cpu"):
    path = f"data/id_train_{name.lower()}.pt"
    data = torch.load(path)
    X, Y = data["X"], data["Y"]
    
    if flatten:
        X = X.view(X.size(0), -1)

    return X.to(device), Y.to(device)

def build_solver(method, ggn: GGNMatVecOperator, num_params, steps):
    s = 2 * steps
    if method == "sl":
        sketch = SRFTSketcher(p=num_params, s=s)
        return SketchedLanczos(ggn.numpy_interface, p=num_params, sketch=sketch), sketch
    elif method == "ra":
        sketch = SRFTSketcher(p=num_params, s=s)
        return RGSArnoldi(ggn.numpy_interface, p=num_params, sketch=sketch), sketch
    elif method == "ll":
        return VanillaLanczos(ggn.numpy_interface, p=num_params, reorth=False, store_full_basis=True), None
    elif method == "hl":
        return VanillaLanczos(ggn.numpy_interface, p=num_params, reorth=True), None
    else:
        raise ValueError(f"Unknown method: {method}")
# solver_config.py
SOLVER_MAP = {
    "sl": "sketched Lanczos",
    "ra": "randomized Arnoldi",
    "ll": "low-memory Lanczos",
    "hl": "high-memory Lanczos"
}


def run_experiment(config):
    # === Setup model ===
    device = torch.device(config.get("device", "cpu"))
    model = config["model_fn"]().to(device)

    ckpt = config.get("model_ckpt")
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded model from {ckpt}")
    model.eval()

    # === Load train and Id datasets ===
    train_X, train_Y = load_id_train_subset(config["id_dataset"], flatten=config["flatten"], device=config["device"])
    id_X, _ = load_dataset(
        config["id_dataset"], flatten=config["flatten"], train=True,
        batch_size=config["ood_size"], device=device
    )
    
    # === GGN + Sketch + Lanczos ===
    ggn = GGNMatVecOperator(model, train_X, train_Y, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    solver, sketch = build_solver(config["method"], ggn, num_params, config["steps"])
    solver.run(num_steps=config["steps"])
    Us = solver.get_basis(k=int(0.9*config["steps"]))
    # Us = solver.get_basis()
    evaluator = SLUEvaluator(model, Us, sketch, device=config["device"], flatten=config["flatten"])
    
    # === Load OoD data and AUROC evaluation ===
    if config["ood_dataset"] == "rotate":
        angles = [90, 180, 270]
        aucs = []
        for angle in angles:
            ood_X_angle, _ = load_dataset(
                config["id_dataset"], flatten=config["flatten"], train=True,
                batch_size=config["ood_size"], rotate_angle=angle, device=device
            )
            # === AUROC evaluation ===
            auc = evaluator.compute_auroc(id_X, ood_X_angle)
            aucs.append(auc)
            print(f"[{config['name']}] Rotation ({angle}°) AUROC = {auc:.4f}")
        auroc = float(np.mean(aucs))
        print(f"[{config['name']}] Rotation (avg) AUROC = {auroc:.4f}")
    else:
        ood_X, _ = load_dataset(
            config["ood_dataset"], flatten=config["flatten"], train=False,
            batch_size=config["ood_size"], device=device
        )

        # === AUROC evaluation ===
        auroc = evaluator.compute_auroc(id_X, ood_X)

        print(f"[{config['name']}] AUROC = {auroc:.4f}")
        # evaluator.plot_histogram()
    return auroc
