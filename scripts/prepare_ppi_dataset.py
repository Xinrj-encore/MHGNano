#!/usr/bin/env python3
"""Prepare deterministic PPI graph, hyperedge, and train/validation/test files.

The input is a STRING-style whitespace-delimited links file and a FASTA file.
Generated files are compatible with ``src/data/dataset.py``. Raw biological
data are not downloaded or redistributed by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path, id_field: str = "first-token") -> dict[str, str]:
    """Read FASTA records, retaining sequences of length 51 through 800."""
    if id_field not in {"first-token", "description"}:
        raise ValueError("id_field must be 'first-token' or 'description'")
    sequences: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []

    def save_record() -> None:
        if header is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        identifier = header.split(maxsplit=1)[0] if id_field == "first-token" else header
        if 51 <= len(sequence) <= 800:
            if identifier in sequences and sequences[identifier] != sequence:
                raise ValueError(f"Conflicting FASTA sequences for {identifier!r}")
            sequences[identifier] = sequence

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                save_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at {path}:{line_number}")
                chunks = []
            elif header is None:
                raise ValueError(f"Sequence data before first FASTA header at {path}:{line_number}")
            else:
                chunks.append(line)
    save_record()
    return sequences


def iter_string_links(path: Path) -> Iterable[tuple[str, str, int]]:
    """Stream STRING links rows (protein1 protein2 combined_score)."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"Expected 3 columns at {path}:{line_number}")
            try:
                score = int(fields[2])
            except ValueError:
                if line_number == 1 or fields[2].lower() in {"combined_score", "score"}:
                    continue
                raise ValueError(f"Non-integer confidence score at {path}:{line_number}")
            if fields[0] != fields[1]:
                yield fields[0], fields[1], score


def stratified_folds(labels: list[int], n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Return deterministic stratified (train-validation, test) index folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    if set(by_class) != {0, 1} or any(len(rows) < n_splits for rows in by_class.values()):
        raise ValueError("Each class must contain at least n_splits records")

    test_folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(by_class):
        rows = by_class[label]
        random.Random(f"{seed}:{label}:folds").shuffle(rows)
        for fold_index, row in enumerate(rows):
            test_folds[fold_index % n_splits].append(row)
    all_indices = set(range(len(labels)))
    return [
        (sorted(all_indices - set(test_indices)), sorted(test_indices))
        for test_indices in test_folds
    ]


def make_splits(
    pairs: list[list[int]], labels: list[int], n_splits: int, seed: int, validation_fraction: float
) -> dict[str, list[list[object]]]:
    """Create stratified folds with a validation subset drawn from trainval."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    split_records: dict[str, list[list[object]]] = {name: [] for name in ("train", "val", "test")}
    for fold_index, (trainval_indices, test_indices) in enumerate(stratified_folds(labels, n_splits, seed)):
        train_indices: list[int] = []
        val_indices: list[int] = []
        for label in (0, 1):
            rows = [index for index in trainval_indices if labels[index] == label]
            random.Random(f"{seed}:{fold_index}:{label}:validation").shuffle(rows)
            n_val = max(1, math.ceil(len(rows) * validation_fraction))
            if n_val >= len(rows):
                raise ValueError("Not enough examples per class for train and validation")
            val_indices.extend(rows[:n_val])
            train_indices.extend(rows[n_val:])

        for split_name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ):
            random.Random(f"{seed}:{fold_index}:{split_name}:order").shuffle(indices)
            split_records[split_name].append(
                [[pairs[index] for index in indices], [float(labels[index]) for index in indices]]
            )
    return split_records


def _write_tsv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def prepare_dataset(
    links_path: Path,
    fasta_path: Path,
    output_root: Path,
    species: str,
    positive_score_cutoff: int = 500,
    positive_cap: int = 10_000,
    negative_ratio: int = 9,
    n_splits: int = 5,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    fasta_id_field: str = "first-token",
) -> Path:
    """Build files consumed by the released training/evaluation code."""
    if positive_cap <= 0 or negative_ratio <= 0:
        raise ValueError("positive_cap and negative_ratio must be positive")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / species
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    sequences = read_fasta(fasta_path, fasta_id_field)
    positives: list[tuple[str, str, int]] = []
    seen_positive_keys: set[tuple[str, str]] = set()
    for a, b, score in iter_string_links(links_path):
        if score <= positive_score_cutoff or a not in sequences or b not in sequences:
            continue
        pair = (a, b) if a < b else (b, a)
        if pair in seen_positive_keys:
            continue
        if len(positives) < positive_cap:
            seen_positive_keys.add(pair)
            positives.append((pair[0], pair[1], score))
    if len(positives) < n_splits:
        raise ValueError(f"Only {len(positives)} positive pairs remain; need at least {n_splits}")

    positive_keys = {(a, b) for a, b, _ in positives}
    protein_ids = sorted({protein for a, b, _ in positives for protein in (a, b)})
    protein_set = set(protein_ids)
    negatives: list[tuple[str, str, int]] = []
    seen_negative_keys: set[tuple[str, str]] = set()
    negative_limit = len(positives) * negative_ratio
    for a, b, score in iter_string_links(links_path):
        if score > positive_score_cutoff or a not in protein_set or b not in protein_set:
            continue
        pair = (a, b) if a < b else (b, a)
        if pair in positive_keys or pair in seen_negative_keys:
            continue
        seen_negative_keys.add(pair)
        negatives.append((pair[0], pair[1], score))
        if len(negatives) >= negative_limit:
            break
    if len(negatives) < n_splits:
        raise ValueError(f"Only {len(negatives)} candidate negatives remain; need at least {n_splits}")

    node_id = {protein: index for index, protein in enumerate(protein_ids)}
    positive_pairs = [[node_id[a], node_id[b]] for a, b, _ in positives]
    negative_pairs = [[node_id[a], node_id[b]] for a, b, _ in negatives]
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    splits = make_splits(all_pairs, all_labels, n_splits, split_seed, validation_fraction)

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Install networkx (see requirements.txt) to build cliques") from exc
    graph = nx.Graph()
    graph.add_nodes_from(range(len(protein_ids)))
    graph.add_edges_from(positive_pairs)
    cliques = [sorted(clique) for clique in nx.find_cliques(graph)]
    cliques.sort(key=lambda clique: (len(clique), clique))

    temp_path = Path(tempfile.mkdtemp(prefix=f".{species}.", dir=output_root))
    try:
        with (temp_path / f"Protein_list_{species}.pkl").open("xb") as handle:
            pickle.dump(protein_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _write_tsv(temp_path / "Protein_list.csv", ("Entry", "ID"), node_id.items())
        _write_tsv(
            temp_path / "PPI.csv",
            ("protein1", "protein2", "combined_score"),
            ((node_id[a], node_id[b], score) for a, b, score in positives),
        )
        _write_tsv(
            temp_path / f"pos_network_{species}.tsv",
            ("protein1", "protein2", "combined_score"),
            positives,
        )
        _write_tsv(
            temp_path / f"neg_network_{species}.tsv",
            ("protein1", "protein2", "combined_score"),
            negatives,
        )
        with (temp_path / f"PPI_cliques_Hyperedge_{species}_8M.pkl").open("xb") as handle:
            pickle.dump(
                {
                    "num_vertices": len(protein_ids),
                    "PPI_edge_list": positive_pairs,
                    "PPI_cliques_list": cliques,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        split_counts: dict[str, list[dict[str, int]]] = {}
        for split_name, fold_rows in splits.items():
            with (temp_path / f"{split_name}_idx_{species}.pkl").open("xb") as handle:
                pickle.dump(fold_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
            split_counts[split_name] = []
            for pairs, labels in fold_rows:
                positive_count = sum(label > 0.5 for label in labels)
                split_counts[split_name].append(
                    {"pairs": len(pairs), "positive": positive_count, "negative": len(pairs) - positive_count}
                )
        manifest = {
            "species": species,
            "inputs": {
                "links_file": links_path.name,
                "links_sha256": sha256_file(links_path),
                "fasta_file": fasta_path.name,
                "fasta_sha256": sha256_file(fasta_path),
                "fasta_id_field": fasta_id_field,
            },
            "preprocessing": {
                "sequence_length_inclusive": [51, 800],
                "positive_rule": f"STRING combined_score > {positive_score_cutoff}",
                "negative_rule": f"STRING combined_score <= {positive_score_cutoff}",
                "positive_induced_negative_universe": True,
                "selection_order": "input order after canonical undirected-pair deduplication",
                "positive_cap": positive_cap,
                "negative_to_positive_ratio": negative_ratio,
                "actual_negative_to_positive_ratio": len(negatives) / len(positives),
                "actual_positive_pairs": len(positives),
                "actual_negative_pairs": len(negatives),
            },
            "split": {
                "method": "deterministic stratified K-fold; validation sampled from each fold's trainval portion",
                "n_splits": n_splits,
                "validation_fraction_of_trainval": validation_fraction,
                "seed": split_seed,
                "counts_by_fold": split_counts,
                "pair_overlap_between_splits": False,
            },
            "hypergraph": {
                "construction": "NetworkX maximal cliques of the undirected positive PPI graph",
                "hyperedge_count": len(cliques),
                "weighting": "uniform unit weight per hyperedge (DHG e_weight default None -> 1.0)",
                "vertex_weight": "uniform unit weight (DHG v_weight default None -> 1.0)",
            },
        }
        with (temp_path / "preprocessing_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.rename(temp_path, destination)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", required=True, type=Path, help="STRING protein.links file")
    parser.add_argument("--fasta", required=True, type=Path, help="Matching STRING protein FASTA")
    parser.add_argument("--species", required=True, help="Species label used in output filenames")
    parser.add_argument("--output-root", type=Path, default=Path("./data/yuanhe"))
    parser.add_argument("--positive-score-cutoff", type=int, default=500)
    parser.add_argument("--positive-cap", type=int, default=10_000)
    parser.add_argument("--negative-ratio", type=int, default=9)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fasta-id-field", choices=("first-token", "description"), default="first-token")
    args = parser.parse_args()
    output = prepare_dataset(
        args.links,
        args.fasta,
        args.output_root,
        args.species,
        positive_score_cutoff=args.positive_score_cutoff,
        positive_cap=args.positive_cap,
        negative_ratio=args.negative_ratio,
        n_splits=args.folds,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        fasta_id_field=args.fasta_id_field,
    )
    print(f"Wrote dataset artifacts to {output}")


if __name__ == "__main__":
    main()
