"""Data loading utilities for the reviewer-facing reproduction code."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from dhg import Hypergraph

PairSplit = Tuple[List[List[int]], List[float]]
SpeciesSplits = Dict[str, PairSplit]


def build_hypergraph_from_pickle(
    pkl_path: str, num_nodes: Optional[int] = None
) -> Hypergraph:
    """Load a serialized clique-derived hypergraph."""
    with open(pkl_path, "rb") as handle:
        payload = pickle.load(handle)
    n_vertices = int(payload["num_vertices"])
    if num_nodes is not None and n_vertices != int(num_nodes):
        raise ValueError(
            f"Hypergraph vertex count ({n_vertices}) does not match "
            f"the requested count ({num_nodes})."
        )
    return Hypergraph(n_vertices, payload["PPI_cliques_list"])


def _load_tensor(path: str, device: Optional[torch.device]) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        tensor = torch.load(path, map_location=device)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, found {type(tensor)!r}.")
    return tensor.float()


def _canonical_split(raw: object, fold: int, path: str) -> PairSplit:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} does not contain any folds.")
    if fold < 0 or fold >= len(raw):
        raise IndexError(f"Fold {fold} is unavailable in {path} (n={len(raw)}).")

    selected = raw[fold]
    if not isinstance(selected, (list, tuple)) or len(selected) != 2:
        raise ValueError(f"Fold {fold} in {path} must be [pairs, labels].")
    pairs, labels = selected
    if len(pairs) != len(labels):
        raise ValueError(
            f"Pair/label length mismatch in {path}, fold {fold}: "
            f"{len(pairs)} != {len(labels)}."
        )

    normalized_pairs = [[int(pair[0]), int(pair[1])] for pair in pairs]
    normalized_labels = [float(label) for label in labels]
    return normalized_pairs, normalized_labels


def load_species_data(
    species: str,
    data_root: str = "./data/yuanhe",
    model_size_tag: str = "8M",
    device: Optional[torch.device] = None,
    fold: int = 0,
) -> Tuple[torch.Tensor, Hypergraph, SpeciesSplits]:
    """Load features, a hypergraph, and independent train/validation/test splits.

    The split files contain five folds. Only the requested fold is loaded.
    Validation and test pairs are deliberately kept separate: validation is
    used for model selection and test is reserved for final evaluation.
    """
    species_dir = os.path.join(data_root, species)
    hyper_path = os.path.join(
        species_dir,
        f"PPI_cliques_Hyperedge_{species}_{model_size_tag}.pkl",
    )
    feature_path = os.path.join(
        species_dir,
        f"protein_feature_CT_{species}_{model_size_tag}.pt",
    )

    if not os.path.exists(hyper_path):
        raise FileNotFoundError(hyper_path)
    if not os.path.exists(feature_path):
        raise FileNotFoundError(feature_path)

    features = _load_tensor(feature_path, device)
    hypergraph = build_hypergraph_from_pickle(hyper_path, features.shape[0])

    splits: SpeciesSplits = {}
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(
            species_dir, f"{split_name}_idx_{species}.pkl"
        )
        if not os.path.exists(split_path):
            raise FileNotFoundError(split_path)
        with open(split_path, "rb") as handle:
            raw = pickle.load(handle)
        splits[split_name] = _canonical_split(raw, fold, split_path)

    return features, hypergraph, splits


def sample_meta_batch(tasks: List[Dict[str, Tuple]], meta_batch_size: int):
    """Sample species tasks without changing their pair split contents."""
    import random

    if meta_batch_size <= 0 or meta_batch_size >= len(tasks):
        return tasks
    return random.sample(tasks, meta_batch_size)
