#!/usr/bin/env python3
"""Hash pair split contents and report pair/node overlap.

This script intentionally uses the same pickle layout as the MHGNano,
TRGH-PPI runner, and HIGH-PPI runner: a list of folds, where each fold is
``(pairs, labels)``. Run it in an environment containing numpy because numpy
objects may be referenced by the pickle files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_records(payload: Any, fold: int) -> list[tuple[int, int, int]]:
    folds = payload
    pairs, labels = folds[fold]
    records = []
    for pair, label in zip(pairs, labels):
        u, v = int(pair[0]), int(pair[1])
        if u > v:
            u, v = v, u
        records.append((u, v, int(float(label) > 0.5)))
    return sorted(records)


def content_hash(records: list[tuple[int, int, int]]) -> str:
    data = "".join(f"{u}\t{v}\t{label}\n" for u, v, label in records)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def pair_set(records: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(u, v) for u, v, _ in records}


def node_set(records: list[tuple[int, int, int]]) -> set[int]:
    return {node for u, v, _ in records for node in (u, v)}


def unique_pair_labels(
    records: list[tuple[int, int, int]],
) -> tuple[dict[tuple[int, int], int], int]:
    pair_labels: dict[tuple[int, int], int] = {}
    conflicts = 0
    for u, v, label in records:
        pair = (u, v)
        if pair in pair_labels and pair_labels[pair] != label:
            conflicts += 1
        pair_labels[pair] = label
    return pair_labels, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    species_dir = args.data_root / args.species
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = species_dir / f"{split}_idx_{args.species}.pkl"
        with path.open("rb") as handle:
            records = canonical_records(pickle.load(handle), args.fold)
        pair_labels, within_conflicts = unique_pair_labels(records)
        unique_records = [
            (u, v, label) for (u, v), label in sorted(pair_labels.items())
        ]
        splits[split] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "n_records": len(records),
            "n_positive_records": sum(label == 1 for _, _, label in records),
            "n_negative_records": sum(label == 0 for _, _, label in records),
            "n_unique_pairs": len(pair_labels),
            "n_unique_positive": sum(label == 1 for label in pair_labels.values()),
            "n_unique_negative": sum(label == 0 for label in pair_labels.values()),
            "within_split_duplicate_records": len(records) - len(pair_labels),
            "within_split_conflicting_labels": within_conflicts,
            "pair_sha256": content_hash(unique_records),
            "pairs": set(pair_labels),
            "pair_labels": pair_labels,
            "positive_nodes": node_set(
                [record for record in unique_records if record[2] == 1]
            ),
        }

    rows = []
    for split, item in splits.items():
        other_pair_overlaps = {
            other: len(item["pairs"] & splits[other]["pairs"])
            for other in splits
            if other != split
        }
        same_label_overlaps = {}
        cross_label_overlaps = {}
        for other in splits:
            if other == split:
                continue
            shared = item["pairs"] & splits[other]["pairs"]
            same_label_overlaps[other] = sum(
                item["pair_labels"][pair] == splits[other]["pair_labels"][pair]
                for pair in shared
            )
            cross_label_overlaps[other] = sum(
                item["pair_labels"][pair] != splits[other]["pair_labels"][pair]
                for pair in shared
            )
        negative_records = []
        with (species_dir / f"{split}_idx_{args.species}.pkl").open("rb") as handle:
            negative_records = [
                record
                for record in canonical_records(pickle.load(handle), args.fold)
                if record[2] == 0
            ]
        negative_nodes = node_set(negative_records)
        rows.append(
            {
                "data_root": str(args.data_root),
                "species": args.species,
                "fold": args.fold,
                "split": split,
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "n_records": item["n_records"],
                "n_positive_records": item["n_positive_records"],
                "n_negative_records": item["n_negative_records"],
                "n_unique_pairs": item["n_unique_pairs"],
                "n_unique_positive": item["n_unique_positive"],
                "n_unique_negative": item["n_unique_negative"],
                "within_split_duplicate_records": item[
                    "within_split_duplicate_records"
                ],
                "within_split_conflicting_labels": item[
                    "within_split_conflicting_labels"
                ],
                "canonical_pair_label_sha256": item["pair_sha256"],
                "pair_overlap_train": other_pair_overlaps.get("train", 0),
                "pair_overlap_val": other_pair_overlaps.get("val", 0),
                "pair_overlap_test": other_pair_overlaps.get("test", 0),
                "same_label_pair_overlap_train": same_label_overlaps.get("train", 0),
                "same_label_pair_overlap_val": same_label_overlaps.get("val", 0),
                "same_label_pair_overlap_test": same_label_overlaps.get("test", 0),
                "cross_label_pair_overlap_train": cross_label_overlaps.get("train", 0),
                "cross_label_pair_overlap_val": cross_label_overlaps.get("val", 0),
                "cross_label_pair_overlap_test": cross_label_overlaps.get("test", 0),
                "negative_nodes_in_positive_node_universe": len(
                    negative_nodes & item["positive_nodes"]
                ),
            }
        )

    fieldnames = list(rows[0])
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
