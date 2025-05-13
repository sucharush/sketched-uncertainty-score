import torch
from main.networks import SmallNet, LeNet

device = "mps" if torch.backends.mps.is_available() else "cpu"

def auto_complete_config(base_config):
    config = base_config.copy()
    id_name = config["id_dataset"].capitalize()
    ood_name = config["ood_dataset"].capitalize()
    model_name = config["model_fn"]().__class__.__name__
    config["name"] = f"{model_name} on {id_name} vs {ood_name}"
    return config


CONFIGS = {
    "mlp": {
        "name": "MLP on MNIST vs FashionMNIST",
        "model_fn": lambda: SmallNet(),
        "model_ckpt": "models/mlp.pt",
        "id_dataset": "mnist",
        "ood_dataset": "kmnist",
        "flatten": True,
        "batch_size": 2000,
        "steps": 50,
        "ood_size": 500,
        "device": device
    },
    "lenet": {
        "name": "LeNet on FashionMNIST vs Rotated",
        "model_fn": lambda: LeNet(),
        "model_ckpt": "models/lenet.pt",
        "id_dataset": "fashion",
        "ood_dataset": "rotate",
        "flatten": False,
        "batch_size": 2000,
        "steps": 100,
        "ood_size": 500,
        "device": device
    }
}
