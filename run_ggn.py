import numpy as np
import matplotlib.pyplot as plt
from solvers.vanilla_lanczos import VanillaLanczos
from uncertainty.ggn import GGNMatVecOperator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
# 1. Small model
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
    def forward(self, x):
        return self.net(x)
# 2. Train 
def train(model, X, Y, epochs=300, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = F.cross_entropy(out, Y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 and verbose:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        if epoch + 1 >= epochs and verbose:
            print(f'Epoch {epoch}, Final Loss: {loss.item()}')
            
EPOCHS_BASED_ON_SIZE = {10: 50, 100: 100, 1000: 100}
sample_sizes = [10, 100, 1000]
lanczos_steps = [10, 100, 1000]
sample_colors = {10: "tab:blue", 100: "tab:orange", 1000: "tab:green"}
step_styles = {10: "-", 100: "--", 1000: ":"}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

# ------------------------------
# Run experiment
# ------------------------------
model = SmallNet().to(device)
def compute_lanczos_spectra(X_batch, Y_batch):
    results = {}
    for n_samples in sample_sizes:
        X_sub = X_batch[:n_samples]
        Y_sub = Y_batch[:n_samples]

        model = SmallNet().to(device)
        train(model, X_sub, Y_sub, epochs=EPOCHS_BASED_ON_SIZE[n_samples], verbose=False)

        ggn_op = GGNMatVecOperator(model, X_sub, Y_sub, device=device)
        num_params = sum(p.numel() for p in model.parameters())

        for steps in lanczos_steps:
            print(f"Running: samples={n_samples}, steps={steps}")
            solver = VanillaLanczos(G_matvec=ggn_op.numpy_interface, p=num_params)
            solver.run(num_steps=steps)
            _, evals = solver.get_top_ritzpairs()
            results[(n_samples, steps)] = evals
    return results


# ------------------------------
# Plotting
# ------------------------------
def plot_all(results):
    plt.figure(figsize=(10, 6))
    for (samples, steps), evals in results.items():
        clean_evals = evals[evals > 0]
        x_vals = np.arange(1, len(clean_evals) + 1)
        plt.plot(x_vals, clean_evals, color=sample_colors[samples], linestyle=step_styles[steps],
                 label=f"samples={samples}, steps={steps}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title("Top GGN Eigenvalues across Sample Sizes & Lanczos Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_by_sample(results, samples_to_plot=[100, 1000]):
    for sample in samples_to_plot:
        plt.figure(figsize=(8, 6))
        for (samples, steps), evals in results.items():
            if samples == sample:
                clean_evals = evals[evals > 0]
                x_vals = np.arange(1, len(clean_evals) + 1)
                plt.plot(x_vals, clean_evals, color=sample_colors[samples], linestyle=step_styles[steps],
                         label=f"steps={steps}")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue (log scale)")
        plt.title(f"Top GGN Eigenvalues (samples={sample})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ------------------------------
# Main entry point
# ------------------------------
def main():
    transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=2000, shuffle=True)
    X_batch, Y_batch = next(iter(loader))
    X_batch = X_batch.float().to(device)
    Y_batch = Y_batch.long().to(device)

    results = compute_lanczos_spectra(X_batch, Y_batch)
    plot_all(results)
    plot_by_sample(results)


if __name__ == "__main__":
    main()
