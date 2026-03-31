"""
HyperGraph Meta-Learning Network for Protein-Protein Interaction Prediction.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph

from .layers import HGNNConv, Concentration


class HyperGraphMetaNetwork(nn.Module):
    """
    Hypergraph Neural Network for cross-species PPI prediction.

    This model combines hypergraph convolutional layers with a
    classifier for predicting protein-protein interactions.

    Architecture:
        1. Two-layer HGNN encoder (320 -> 256 -> 128)
        2. Concentration layer for hyperedge aggregation
        3. Three-layer MLP classifier (512 -> 1024 -> 256 -> 1)

    Args:
        in_channels: Input feature dimension (default: 320 for ESM-8M)
        hid_channels: Hidden dimension (default: 256)
        num_classes: Embedding dimension (default: 128)
        num_hyperedges: Number of hyperedges (for compatibility)
        use_bn: Kept for CLI compatibility (unused)
        drop_rate: Kept for CLI compatibility (unused)
    """

    def __init__(
        self,
        in_channels: int = 320,
        hid_channels: int = 256,
        num_classes: int = 128,
        num_hyperedges: int = 0,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        # HGNN encoder layers
        self.layer1 = HGNNConv(
            in_channels, hid_channels, use_bn=False, drop_rate=0.0
        )
        self.layer2 = HGNNConv(
            hid_channels, num_classes, use_bn=False, drop_rate=0.0, is_last=True
        )

        # Concentration layer
        self.concentration = Concentration(hid_channels, hid_channels)

        # Classifier (no BN/Dropout for stable meta-learning)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def encode(self, x: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """
        Encode protein features using HGNN.

        Args:
            x: Input features [N, in_channels]
            hg: Hypergraph structure

        Returns:
            Encoded features [N, num_classes]
        """
        hidden1 = self.layer1(x, hg)
        return self.layer2(hidden1, hg)

    def forward(
        self,
        X: torch.Tensor,
        hg: Hypergraph,
        GP_info: torch.Tensor,
        phase: str = "train",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Node features [N, in_channels]
            hg: Hypergraph structure
            GP_info: Hyperedge indices for concentration
            phase: 'train' or 'eval'

        Returns:
            Predictions [N, 1]
        """
        X = self.encode(X, hg)
        concen_Z = self.concentration(X, GP_info)
        out = self.classifier(concen_Z)
        return out

    def functional_forward(
        self,
        X: torch.Tensor,
        hg: Hypergraph,
        hg_index: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Functional forward for MAML (supports custom parameters).

        Args:
            X: Node features
            hg: Hypergraph structure
            hg_index: Hyperedge indices
            params: Dictionary of model parameters

        Returns:
            Predictions
        """
        if params is None:
            return self.forward(X, hg, hg_index)

        def p(name: str) -> torch.Tensor:
            return params[name]

        # Layer 1
        if hasattr(self.layer1, "functional_forward"):
            x = self.layer1.functional_forward(
                X,
                hg,
                {"theta.weight": p("layer1.theta.weight"), "theta.bias": p("layer1.theta.bias")},
            )
        else:
            x = self.layer1(X, hg)

        # Layer 2
        if hasattr(self.layer2, "functional_forward"):
            x = self.layer2.functional_forward(
                x,
                hg,
                {"theta.weight": p("layer2.theta.weight"), "theta.bias": p("layer2.theta.bias")},
            )
        else:
            x = self.layer2(x, hg)

        # Concentration
        if hasattr(self.concentration, "functional_forward"):
            z = self.concentration.functional_forward(x, hg_index)
        else:
            z = self.concentration(x, hg_index)

        # Classifier (functional)
        h = z
        h = F.linear(h, p("classifier.0.weight"), p("classifier.0.bias"))
        h = F.relu(h)
        h = F.linear(h, p("classifier.2.weight"), p("classifier.2.bias"))
        h = F.relu(h)
        out = F.linear(h, p("classifier.4.weight"), p("classifier.4.bias"))

        return out
