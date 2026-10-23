from typing import Dict, Tuple

import torch
from dhg import Hypergraph


class InnerLoopAdaptation:
    def _inner_loop_eval(
        self,
        support_batch: Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        X_s, hg_s, index_s, index_label = support_batch
        X_s = X_s.to(device)
        y_s = torch.tensor(index_label, device=device, dtype=torch.float32).view(-1)
        if hasattr(hg_s, "to"):
            hg_s = hg_s.to(device)

        fast_params: Dict[str, torch.Tensor] = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        for _ in range(self.inner_steps):
            logits = self.model.functional_forward(X_s, hg_s, index_s, params=fast_params).view(-1)
            loss = self._loss(logits, y_s)
            gradients = torch.autograd.grad(
                loss,
                list(fast_params.values()),
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            fast_params = {
                name: param - self.inner_lr * (gradient if gradient is not None else torch.zeros_like(param))
                if name.startswith("classifier.")
                else param
                for (name, param), gradient in zip(fast_params.items(), gradients)
            }
        return fast_params
