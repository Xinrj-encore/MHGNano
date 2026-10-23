import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from dhg import Hypergraph
def build_hypergraph_from_pickle(pkl_path: str, num_nodes: Optional[int] = None) -> Hypergraph:
    with open(pkl_path, "rb") as f:
        edge_list_data = pickle.load(f)
    hg = Hypergraph(edge_list_data["num_vertices"], edge_list_data["PPI_cliques_list"])
    return hg

def load_tensor_if_exists(path: str, map_location=None) -> Optional[torch.Tensor]:
    if os.path.exists(path):
        return torch.load(path, map_location=map_location)
    return None

def make_task(
    X: torch.Tensor,
    hg: Hypergraph,
    GP_info: torch.Tensor,
    y: torch.Tensor,
    n_support: int,
    n_query: int,
    shuffle: bool = True,
) -> Dict[str, Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor]]:
    N = X.size(0)
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    assert n_support + n_query <= N
    s_idx = idx[:n_support]
    q_idx = idx[n_support : n_support + n_query]
    support = (X[s_idx], hg, GP_info[s_idx], y[s_idx])
    query = (X[q_idx], hg, GP_info[q_idx], y[q_idx])
    return {"support": support, "query": query}

def sample_meta_batch(tasks: List[Dict[str, Tuple]], meta_batch_size: int) -> List[Dict[str, Tuple]]:
    return random.sample(tasks, meta_batch_size) if meta_batch_size <= len(tasks) else tasks


def resolve_species_dir(data_root: str, species: str, kingdom: Optional[str] = None) -> Path:
    root = Path(data_root)
    if kingdom not in {None, "eukaryotes", "prokaryotes"}:
        raise ValueError("kingdom must be eukaryotes or prokaryotes")
    if kingdom:
        candidates = [root / kingdom / species, root / species, root]
    else:
        candidates = [
            root / species,
            root / "eukaryotes" / species,
            root / "prokaryotes" / species,
            root,
        ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find data for {species!r}. Searched: {searched}")


def load_species_data(
    species: str,
    data_root: str,
    model_size_tag: str = "8M",
    device=None,
    kingdom: Optional[str] = None,
):
    species_dir = resolve_species_dir(data_root, species, kingdom)
    hypergraph_path = species_dir / f"PPI_cliques_Hyperedge_{species}_{model_size_tag}.pkl"
    if not hypergraph_path.is_file():
        hypergraph_path = species_dir / f"PPI_cliques_Hyperedge_{species}_8M.pkl"
    hypergraph = build_hypergraph_from_pickle(str(hypergraph_path))
    feature_path = species_dir / f"protein_feature_CT_{species}_{model_size_tag}.pt"
    features = load_tensor_if_exists(str(feature_path), map_location=device)
    if features is None:
        raise FileNotFoundError(f"Missing feature tensor: {feature_path}")
    splits = {}
    for split_name in ("train", "val", "test"):
        split_path = species_dir / f"{split_name}_idx_{species}.pkl"
        with split_path.open("rb") as handle:
            records = pickle.load(handle)
        splits[split_name] = records[0]
    return features, hypergraph, splits
