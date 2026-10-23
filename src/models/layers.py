"""Hypergraph encoder and pair-concentration layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph


class HGNNConv(nn.Module):
    """A lightweight HGNN convolution used by the published model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.0,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU(inplace=False)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        return X if self.is_last else self.act(X)

    def functional_forward(
        self, X: torch.Tensor, hg: Hypergraph, params: dict
    ) -> torch.Tensor:
        X = F.linear(X, params["theta.weight"], params.get("theta.bias"))
        X = hg.smoothing_with_HGNN(X)
        return X if self.is_last else F.relu(X, inplace=False)


class Concentration(nn.Module):
    """Construct pair features from node representations.

    For a pair ``(i, j)``, the representation is
    ``[z_i, z_j, |z_i-z_j|, z_i*z_j]``.
    """

    def __init__(self, hidden_channels: int, hidden_size: int):
        super().__init__()

    @staticmethod
    def _pair_tensor(X: torch.Tensor, GP_info) -> torch.Tensor:
        pairs = torch.as_tensor(GP_info, device=X.device, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                f"Expected pair indices with shape [n, 2], got {tuple(pairs.shape)}."
            )
        if pairs.numel() and (
            int(pairs.min()) < 0 or int(pairs.max()) >= X.shape[0]
        ):
            raise IndexError("Pair index is outside the feature matrix.")

        z1 = X.index_select(0, pairs[:, 0])
        z2 = X.index_select(0, pairs[:, 1])
        return torch.cat((z1, z2, torch.abs(z1 - z2), z1 * z2), dim=1)

    def forward(self, X: torch.Tensor, GP_info) -> torch.Tensor:
        return self._pair_tensor(X, GP_info)

    def functional_forward(self, X: torch.Tensor, GP_info) -> torch.Tensor:
        return self._pair_tensor(X, GP_info)
