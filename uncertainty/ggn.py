import torch
import torch.nn.functional as F
# from functorch import make_functional, jvp, vjp, vmap
from functorch import  make_functional
from torch.func import vmap, grad, jvp, vjp
import numpy as np


class GGNMatVecOperator:
    def __init__(self, model, X, Y, loss_type="cross_entropy", device=None):
        self.device = device if device is not None else X.device
        self.X = X.to(self.device)
        self.Y = Y.to(self.device)
        self.loss_type = loss_type
        self._make_operator(model)
        
    def _make_operator(self, model):
        self.fmodel, self.params = make_functional(model)
        self.params_flat = torch.cat([p.view(-1) for p in self.params]).to(self.device)

        def compute_outputs(p, x):
            # Unflatten parameters
            params_unflat = []
            idx = 0
            for param in self.params:
                numel = param.numel()
                params_unflat.append(p[idx:idx+numel].view_as(param).to(self.device))
                idx += numel
            return self.fmodel(tuple(params_unflat), x.unsqueeze(0)).squeeze(0)

        def make_loss_fn(y):
            if self.loss_type == "cross_entropy":
                return lambda o: F.cross_entropy(o.unsqueeze(0), y.unsqueeze(0), reduction='mean')
            elif self.loss_type == "mse":
                return lambda o: F.mse_loss(o.unsqueeze(0), y.unsqueeze(0).float(), reduction='mean')
            elif self.loss_type == "binary_cross_entropy":
                return lambda o: F.binary_cross_entropy_with_logits(o.unsqueeze(0), y.unsqueeze(0).float())
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        def ggn_matvec(v_flat):
            v_flat = v_flat.to(self.device)

            def per_sample_ggn(x, y):
                loss_fn = make_loss_fn(y)

                # 1. Compute Jv
                outputs, jvp_out = jvp(
                    lambda p: compute_outputs(p, x),
                    (self.params_flat,),
                    (v_flat,)
                )
                # 2. Compute H Jv
                grad_fn = grad(loss_fn)
                H_jvp = jvp(grad_fn, (outputs,), (jvp_out,))[1]
                
                # 3. Compute J^T (H Jv)
                _, vjp_fn = vjp(lambda p: compute_outputs(p, x), self.params_flat)
                return vjp_fn(H_jvp)[0]

            return vmap(per_sample_ggn)(self.X, self.Y).sum(0)

        # Save it once
        self._ggn_matvec_fn = ggn_matvec

    
    # def numpy_interface(self, v_np):
    #     v_torch = torch.from_numpy(v_np).float().to(self.device)
    #     Gv = self._ggn_matvec_fn(v_torch)
    #     return Gv.detach().cpu().numpy()
    def numpy_interface(self, v_np):
        if np.iscomplexobj(v_np):
            
            v_real = torch.from_numpy(v_np.real.astype(np.float32)).to(self.device)
            v_imag = torch.from_numpy(v_np.imag.astype(np.float32)).to(self.device)

            Gv_real = self._ggn_matvec_fn(v_real).detach()
            Gv_imag = self._ggn_matvec_fn(v_imag).detach()
            # Gv linear operator
            return Gv_real.cpu().numpy() + 1j * Gv_imag.cpu().numpy()
        else:
            v_torch = torch.from_numpy(v_np.astype(np.float32)).to(self.device)
            Gv = self._ggn_matvec_fn(v_torch).detach()
            return Gv.cpu().numpy()



