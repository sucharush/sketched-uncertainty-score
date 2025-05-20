# Sketch-based Uncertainty Score for Neural Networks

Exploratory repo for sketched Krylov methods applied to uncertainty estimation in neural networks, based on [Miani et al., 2024](https://arxiv.org/pdf/2409.15008). 

Includes synthetic demos and **in-progress** uncertainty score experiments.

## How to 
To evaluate uncertainty-score-based AUROC performance using different basis methods:

```bash
python -m main.run --model mlp --solver sl --ood kmnist --runs 5
```

- `--model`  Pretrained model to use (`mlp` or `lenet`)
- `--solver` Basis construction method (`hl` = High-memory Lanczos,`ll` = Low-memory Lanczos, `sl` = Sketched Lanczos (default), `ra` = Randomized Arnoldi)
- `--ood`   Out-of-distribution dataset (`fashion`, `mnist`, `kmnist`, or `rotate`)
- `--runs`   Number of independent runs to average over (default: 5)

OR generate the full table by 
<!-- (~30 mins on `mps`) -->
```bash
./run_all.sh
```
## Results
| Solver     |            |    MLP (MNIST) vs      |                  |          |     LeNet (FashionMNIST) vs    |                  |
|------------|:-------------------------:|:--------:|:----------------:|:-------------------------------:|:--------:|:----------------:|
|            | FashionMNIST              | KMNIST   | Rotation (avg)   | MNIST                          | KMNIST   | Rotation (avg)   |
| High-Mem Lanczos  | 0.63 ± 0.03               | 0.30 ± 0.02 | 0.39 ± 0.01   | 0.81 ± 0.01                    | 0.67 ± 0.01 | 0.44 ± 0.00    |
| Low-Mem  Lanczos  | 0.60 ± 0.01               | 0.28 ± 0.01 | 0.37 ± 0.02   | 0.79 ± 0.01                    | 0.66 ± 0.01 | 0.43 ± 0.01    |
| Sketched Lanczos  | 0.59 ± 0.02               | 0.21 ± 0.02 | 0.28 ± 0.03   | 0.70 ± 0.01                    | 0.53 ± 0.03 | 0.40 ± 0.01    |
| Randomized Arnoldi    | 0.62 ± 0.00               | 0.20 ± 0.01 | 0.27 ± 0.01   | 0.70 ± 0.03                    | 0.49 ± 0.03 | 0.41 ± 0.01    |

