from uncertainty.evaluation_base import BaseEvaluator
import torch
import torch.nn.functional as F
from functorch import make_functional, jacrev
import numpy as np
from sketch.sketch_base import Sketcher

class SLUEvaluator(BaseEvaluator):
    def __init__(self, model, Us, sketch: Sketcher, device=None, flatten = True):
        super().__init__(model, device, flatten)
        # if isinstance(Us, np.ndarray):
        #     Us = torch.from_numpy(Us).float()
        if np.iscomplexobj(Us):
            Us = torch.from_numpy(Us.astype(np.complex64))  # or complex128 if needed
        else:
            Us = torch.from_numpy(Us).float()
        self.Us = Us.to(self.device)
        self.sketch = sketch
        
        self.fmodel, self.params = make_functional(self.model)
        self.params_flat = torch.cat([p.view(-1) for p in self.params]).to(self.device)
        
    def _compute_outputs(self, p, x):
        params_unflat = []
        idx = 0
        for param in self.params:
            numel = param.numel()
            params_unflat.append(p[idx:idx+numel].view_as(param))
            idx += numel
        return self.fmodel(tuple(params_unflat), x.unsqueeze(0)).squeeze(0)


    def compute_score(self, x):
        Us_np = self.Us.detach().cpu().numpy()

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        # Full Jacobian
        J = jacrev(lambda p: self._compute_outputs(p, x))(self.params_flat)
        J_np = J.detach().cpu().numpy()
        # print(f"J shape: {J_np.shape}")

        SJt_np = self.sketch.apply_sketch(J_np.T)  # (s, n_params)
        proj_np = Us_np.T @ SJt_np            # (s, s)
        # print(f"proj shape: {proj_np.shape}")

        # Frobenius norms and SLU score
        full_norm_sq = np.linalg.norm(J_np, ord='fro') ** 2
        sketch_norm_sq = np.linalg.norm(proj_np, ord='fro') ** 2
        # print(f"Full norm: {full_norm_sq}, Sketch norm: {sketch_norm_sq}")

        return float(full_norm_sq - sketch_norm_sq)
