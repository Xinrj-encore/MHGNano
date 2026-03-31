"""
HyperGraph Neural Network layers for Protein-Protein Interaction prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph


class HGNNConv(nn.Module):
    """
    Hypergraph Neural Network Convolution Layer.

    Implements hypergraph convolution without BatchNorm and Dropout
    for stable meta-learning.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        bias: Whether to use bias in linear transformation
        use_bn: Kept for API compatibility (unused)
        drop_rate: Kept for API compatibility (unused)
        is_last: Whether this is the last layer (no activation)
    """

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
        self.act = nn.ReLU(inplace=True)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Node features [N, in_channels]
            hg: Hypergraph structure

        Returns:
            Output features [N, out_channels]
        """
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
        return X

    def functional_forward(
        self, X: torch.Tensor, hg: Hypergraph, params: dict
    ) -> torch.Tensor:
        """
        Functional forward for MAML (supports custom parameters).

        Args:
            X: Node features
            hg: Hypergraph structure
            params: Dictionary of parameters {'theta.weight', 'theta.bias'}

        Returns:
            Output features
        """
        W = params["theta.weight"]
        b = params.get("theta.bias", None)
        X = F.linear(X, W, b)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = F.relu(X, inplace=True)
        return X


class Concentration(nn.Module):
    """
    Concentration layer for aggregating hyperedge information.

    Aggregates node embeddings into hyperedge representations and
    then concentrates them for final prediction.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Node features [N, in_channels]
            hg: Hypergraph structure

        Returns:
            Concentrated features
        """
        # Aggregate node features to hyperedges
        edge_features = self.theta(X)
        # Apply hypergraph smoothing
        X = hg.smoothing_with_HGNN(edge_features)
        return X

    def functional_forward(
        self, X: torch.Tensor, hg: Hypergraph, params: dict
    ) -> torch.Tensor:
        """Functional forward for MAML."""
        W = params["theta.weight"]
        b = params.get("theta.bias", None)
        edge_features = F.linear(X, W, b)
        X = hg.smoothing_with_HGNN(edge_features)
        return X
