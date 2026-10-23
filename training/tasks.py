import random
from typing import Dict, List, Tuple

import torch
from dhg import Hypergraph

from .data import load_species_data
from .sampling import positive_pairs_from_splits, resample_split_negatives


def make_meta_task(X, hypergraph, support_split, query_split):
    support_pairs, support_labels = support_split
    query_pairs, query_labels = query_split
    support = (
        X,
        hypergraph,
        torch.as_tensor(support_pairs, dtype=torch.long),
        torch.as_tensor(support_labels, dtype=torch.float32),
    )
    query = (
        X,
        hypergraph,
        torch.as_tensor(query_pairs, dtype=torch.long),
        torch.as_tensor(query_labels, dtype=torch.float32),
    )
    return {"support": support, "query": query}


def build_species_tasks(
    species_data: Dict[str, Tuple],
    species_names: List[str],
    mode: str,
    seed: int,
    negative_sampling: str = "original",
    negative_ratio: float = 1.0,
) -> List[Dict]:
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported task mode: {mode}")

    tasks = []
    for task_index, species in enumerate(species_names):
        X, hypergraph, splits = species_data[species]
        support_split = splits["train"]
        query_name = "test" if mode == "test" else "val"
        query_split = splits[query_name]

        if mode == "train" and negative_sampling != "original":
            rng = random.Random(seed + task_index * 1009)
            positive_pairs = positive_pairs_from_splits(splits)
            options = {
                "mode": negative_sampling,
                "neg_ratio": negative_ratio,
                "rng": rng,
                "positive_pairs": positive_pairs,
                "num_nodes": X.size(0),
            }
            support_split = resample_split_negatives(support_split, **options)
            query_split = resample_split_negatives(query_split, **options)

        tasks.append(make_meta_task(X, hypergraph, support_split, query_split))
    return tasks


def load_species_group(species_names, data_root, model_size, kingdom=None, species_dirs=None):
    species_dirs = species_dirs or {}
    return {
        species: load_species_data(
            species,
            species_dirs.get(species, data_root),
            model_size,
            kingdom=kingdom if species not in species_dirs else None,
        )
        for species in species_names
    }
