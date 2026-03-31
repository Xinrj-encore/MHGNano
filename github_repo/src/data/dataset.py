"""
Data loading and processing utilities.
"""
import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from dhg import Hypergraph


def build_hypergraph_from_pickle(
    pkl_path: str, num_nodes: Optional[int] = None
) -> Hypergraph:
    """
    Build hypergraph from pickle file.

    Args:
        pkl_path: Path to pickle file containing hyperedge list
        num_nodes: Number of nodes (optional)

    Returns:
        Hypergraph object
    """
    with open(pkl_path, "rb") as f:
        edge_list_data = pickle.load(f)
    hg = Hypergraph(
        edge_list_data["num_vertices"], edge_list_data["PPI_cliques_list"]
    )
    return hg


def load_tensor_if_exists(
    path: str, map_location=None
) -> Optional[torch.Tensor]:
    """Load tensor if file exists."""
    if os.path.exists(path):
        return torch.load(path, map_location=map_location)
    return None


def load_species_data(
    species: str,
    data_root: str = "./data/yuanhe",
    model_size_tag: str = "8M",
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Hypergraph, List]:
    """
    Load data for a specific species.

    Args:
        species: Species identifier
        data_root: Root directory for data
        model_size_tag: ESM model size (8M, 35M, 150M)
        device: Device to load tensors to

    Returns:
        (X, hg, y): Features, hypergraph, and labels
    """
    sp_dir = os.path.join(data_root, species)

    # Load hypergraph
    hyper_pkl = os.path.join(
        sp_dir, f"PPI_cliques_Hyperedge_{species}_{model_size_tag}.pkl"
    )
    hg = build_hypergraph_from_pickle(hyper_pkl)

    # Load features
    X_path = os.path.join(
        sp_dir, f"protein_feature_CT_{species}_{model_size_tag}.pt"
    )
    X = load_tensor_if_exists(X_path, map_location=device)
    if X is None:
        raise FileNotFoundError(f"Missing feature tensor: {X_path}")

    # Load train/val/test indices
    with open(f"{data_root}/{species}/train_idx_{species}.pkl", "rb") as f:
        train_idx = pickle.load(f)
    with open(f"{data_root}/{species}/val_idx_{species}.pkl", "rb") as f:
        val_idx = pickle.load(f)
    with open(f"{data_root}/{species}/test_idx_{species}.pkl", "rb") as f:
        test_idx = pickle.load(f)

    # Combine support and query sets
    support_idx = train_idx[0]
    val_idx[0][0].extend(test_idx[0][0])
    val_idx[0][1].extend(test_idx[0][1])
    query_idx = val_idx[0]

    y = [support_idx, query_idx]
    return X, hg, y


def make_task(
    X: torch.Tensor,
    hg: Hypergraph,
    GP_info: torch.Tensor,
    y: torch.Tensor,
    n_support: int,
    n_query: int,
    shuffle: bool = True,
) -> Dict[str, Tuple]:
    """
    Create a meta-learning task with support and query sets.

    Args:
        X: Node features
        hg: Hypergraph
        GP_info: Hyperedge indices
        y: Labels
        n_support: Number of support samples
        n_query: Number of query samples
        shuffle: Whether to shuffle data

    Returns:
        Task dictionary with 'support' and 'query' keys
    """
    N = X.size(0)
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    assert n_support + n_query <= N

    s_idx = idx[:n_support]
    q_idx = idx[n_support : n_support + n_query]

    support = (X[s_idx], hg, GP_info[s_idx], y[s_idx])
    query = (X[q_idx], hg, GP_info[q_idx], y[q_idx])

    return {"support": support, "query": query}


def sample_meta_batch(
    tasks: List[Dict[str, Tuple]], meta_batch_size: int
) -> List[Dict[str, Tuple]]:
    """Sample a batch of tasks for meta-training."""
    if meta_batch_size <= len(tasks):
        import random
        return random.sample(tasks, meta_batch_size)
    return tasks
