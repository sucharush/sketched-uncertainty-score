## Sketched Krylov Approximations for Uncertainty in Neural Networks

Exploratory repo for sketched Krylov methods applied to uncertainty estimation in neural networks, based on [Miani et al., 2024](https://arxiv.org/pdf/2409.15008). 

Includes synthetic demos and **in-progress** SLU experiments.

### How to 
To evaluate SLU-based AUROC performance:

```bash
python -m main.run --model mlp --ood kmnist --runs 10
```

- `--model`  Pretrained model to use (`mlp` or `lenet`)
- `--ood`   Out-of-distribution dataset (`fashion`, `mnist`, `kmnist`, or `rotate`)
- `--runs`   Number of independent runs to average over (default: 5)

