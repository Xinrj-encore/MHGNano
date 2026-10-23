"""
Test basic model functionality.
"""
import torch
import pytest
from dhg import Hypergraph

from models import HyperGraphMetaNetwork, MetaLearner, HGNNConv


def test_hgnn_conv():
    """Test HGNN convolution layer."""
    conv = HGNNConv(320, 256)

    # Create dummy data
    X = torch.randn(100, 320)
    hg = Hypergraph(100, [[0, 1, 2], [3, 4, 5]])

    out = conv(X, hg)
    assert out.shape == (100, 256)


def test_hypergraph_meta_network():
    """Test HyperGraphMetaNetwork forward pass."""
    model = HyperGraphMetaNetwork(
        in_channels=320,
        hid_channels=256,
        num_classes=128,
        num_hyperedges=10,
    )

    X = torch.randn(100, 320)
    hg = Hypergraph(100, [[0, 1, 2], [3, 4, 5]])
    GP_info = torch.tensor([[0, 1], [3, 4], [1, 2]])

    out = model(X, hg, GP_info)
    assert out.shape == (3, 1)


def test_meta_learner():
    """Test MetaLearner initialization."""
    base = HyperGraphMetaNetwork(
        in_channels=320,
        hid_channels=256,
        num_classes=128,
        num_hyperedges=10,
    )

    meta = MetaLearner(base, inner_lr=1e-3, inner_steps=5, meta_lr=1e-4)
    assert meta.inner_lr == 1e-3
    assert meta.inner_steps == 5


if __name__ == "__main__":
    test_hgnn_conv()
    test_hypergraph_meta_network()
    test_meta_learner()
    print("All tests passed!")
