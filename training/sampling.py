import random
from typing import Dict, List, Optional, Tuple
import numpy as np
def normalize_pair(pair) -> Tuple[int, int]:
    u, v = int(pair[0]), int(pair[1])
    return (u, v) if u <= v else (v, u)


def split_pairs_by_label(idx_label_split: Tuple[List[int], List[int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    pairs, labels = idx_label_split
    pos_pairs, neg_pairs = [], []
    for pair, label in zip(pairs, labels):
        norm_pair = normalize_pair(pair)
        if float(label) > 0.5:
            pos_pairs.append(norm_pair)
        else:
            neg_pairs.append(norm_pair)
    return pos_pairs, neg_pairs


def pack_pair_split(
    pos_pairs: List[Tuple[int, int]],
    neg_pairs: List[Tuple[int, int]],
    rng: random.Random,
) -> Tuple[List[List[int]], List[float]]:
    items = [(list(pair), 1.0) for pair in pos_pairs] + [(list(pair), 0.0) for pair in neg_pairs]
    rng.shuffle(items)
    if not items:
        return [], []
    pairs, labels = zip(*items)
    return list(pairs), list(labels)


def positive_pairs_from_splits(splits: Dict[str, Tuple[List[int], List[int]]]) -> set:
    positive_pairs = set()
    for split_name in ("train", "val", "test"):
        pos_pairs, _ = split_pairs_by_label(splits[split_name])
        positive_pairs.update(pos_pairs)
    return positive_pairs


def sample_existing_negatives(
    idx_label_split: Tuple[List[int], List[int]],
    neg_ratio: float,
    rng: random.Random,
) -> Tuple[List[List[int]], List[float]]:
    pos_pairs, neg_pairs = split_pairs_by_label(idx_label_split)
    target_neg = min(len(neg_pairs), int(round(len(pos_pairs) * neg_ratio)))
    sampled_neg = rng.sample(neg_pairs, target_neg) if target_neg < len(neg_pairs) else list(neg_pairs)
    return pack_pair_split(pos_pairs, sampled_neg, rng)


def generate_degree_matched_negatives(
    pos_pairs: List[Tuple[int, int]],
    positive_pairs: set,
    num_nodes: int,
    target_neg: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    if target_neg <= 0 or num_nodes <= 1:
        return []

    degree = np.ones(num_nodes, dtype=np.float64)
    for u, v in pos_pairs:
        if 0 <= u < num_nodes:
            degree[u] += 1.0
        if 0 <= v < num_nodes:
            degree[v] += 1.0
    prob = degree / degree.sum()
    nodes = np.arange(num_nodes)

    seed = rng.randrange(0, 2**32 - 1)
    np_rng = np.random.default_rng(seed)
    negatives = set()
    max_attempts = max(target_neg * 100, 1000)
    attempts = 0
    while len(negatives) < target_neg and attempts < max_attempts:
        attempts += 1
        u, v = np_rng.choice(nodes, size=2, replace=False, p=prob)
        pair = normalize_pair((u, v))
        if pair in positive_pairs or pair in negatives:
            continue
        negatives.add(pair)

    if len(negatives) < target_neg:
        print(f"[NegSampling] Only generated {len(negatives)}/{target_neg} degree-matched negatives.")
    return list(negatives)


def resample_split_negatives(
    idx_label_split: Tuple[List[int], List[int]],
    mode: str,
    neg_ratio: float,
    rng: random.Random,
    positive_pairs: Optional[set] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[List[List[int]], List[float]]:
    if mode == "original":
        return idx_label_split
    if neg_ratio <= 0:
        raise ValueError("--neg_ratio must be positive when negative sampling is enabled.")
    if mode == "existing_random":
        return sample_existing_negatives(idx_label_split, neg_ratio, rng)
    if mode == "degree_matched":
        pos_pairs, neg_pairs = split_pairs_by_label(idx_label_split)
        if positive_pairs is None or num_nodes is None:
            raise ValueError("degree_matched negative sampling requires positive_pairs and num_nodes.")
        target_neg = int(round(len(pos_pairs) * neg_ratio))
        generated_neg = generate_degree_matched_negatives(
            pos_pairs=pos_pairs,
            positive_pairs=positive_pairs,
            num_nodes=num_nodes,
            target_neg=target_neg,
            rng=rng,
        )
        if len(generated_neg) < target_neg and neg_pairs:
            needed = min(target_neg - len(generated_neg), len(neg_pairs))
            generated_set = set(generated_neg)
            fallback = [pair for pair in neg_pairs if pair not in generated_set]
            generated_neg.extend(rng.sample(fallback, min(needed, len(fallback))))
        return pack_pair_split(pos_pairs, generated_neg, rng)
    raise ValueError(f"Unknown negative sampling mode: {mode}")

