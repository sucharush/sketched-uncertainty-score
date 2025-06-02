# Sketch-based Uncertainty Score for Neural Networks

Exploratory repo for sketched Krylov methods applied to uncertainty estimation in neural networks, based on [Miani et al., 2024](https://arxiv.org/pdf/2409.15008). 

Includes synthetic demos and **in-progress** uncertainty score experiments.

## How to 
To generate the plots for synthetic data:
```bash
python -m run_synthetic
```
To generate the plots for ggn spectrum:
```bash
python -m run_ggn
```
To evaluate uncertainty-score-based AUROC performance using different basis methods:

```bash
python -m main.run --model mlp --solver sl --steps 50 --ood kmnist --runs 5
```
Parameters:
```
--model     Pretrained model to use (`mlp` or `lenet`)
--solver    Basis construction method (`hl` = High-memory Lanczos,
           `ll` = Low-memory Lanczos, `sl` = Sketched Lanczos [default],
           `ra` = Randomized Arnoldi)
--steps     Number of iterations in Krylov methods
--ood       Out-of-distribution dataset (`fashion`, `mnist`, `kmnist`, or `rotate`)
--runs      Number of independent runs to average over (default: 5)
```

OR generate the full table by 
<!-- (~30 mins on `mps`) -->
```bash
./run_auroc.sh
```
## Results

|                     |                          |       MLP (MNIST) vs |                     |       |          LeNet (FashionMNIST) vs          |                     |
|----------------------|--------------------------|---------------------|---------------------|-------------------------------|---------------------|---------------------|
|        Solver                | FashionMNIST             | KMNIST              | Rotation (avg)      | MNIST                         | KMNIST              | Rotation (avg)      |
| Lanczos (reorth.)    | 0.61 ± 0.01              | 0.28 ± 0.01         | 0.37 ± 0.02         | 0.81 ± 0.01                   | 0.68 ± 0.03         | 0.45 ± 0.02         |
| Lanczos (w/o reorth.)| 0.59 ± 0.02              | 0.28 ± 0.01         | 0.38 ± 0.01         | 0.78 ± 0.01                   | 0.65 ± 0.01         | 0.42 ± 0.01         |
| Sketched Lanczos     | 0.59 ± 0.01              | 0.19 ± 0.01         | 0.28 ± 0.01         | 0.71 ± 0.02                   | 0.50 ± 0.01         | 0.39 ± 0.00         |
| Randomized Arnoldi   | 0.59 ± 0.02              | 0.18 ± 0.01         | 0.27 ± 0.01         | 0.69 ± 0.01                   | 0.48 ± 0.01         | 0.40 ± 0.02         |


