#!/usr/bin/env python3
"""Summarize maximal-clique hyperedge sizes from prepared species pickles."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path


SOURCES = (
    ("Eukaryotic", "H. sapiens", "zhenhe/human/PPI_cliques_Hyperedge_human_8M.pkl"),
    ("Eukaryotic", "C. elegans", "zhenhe/elegans/PPI_cliques_Hyperedge_elegans_8M.pkl"),
    ("Eukaryotic", "S. cerevisiae", "zhenhe/cerevisiae/PPI_cliques_Hyperedge_cerevisiae_8M.pkl"),
    ("Eukaryotic", "D. melanogaster", "zhenhe/melanogaster/PPI_cliques_Hyperedge_melanogaster_8M.pkl"),
    ("Eukaryotic", "D. rerio", "zhenhe/rerio/PPI_cliques_Hyperedge_rerio_8M.pkl"),
    ("Eukaryotic", "M. musculus", "zhenhe/mouse/PPI_cliques_Hyperedge_mouse_8M.pkl"),
    ("Prokaryotic", "Streptococcus", "yuanhe/1326/PPI_cliques_Hyperedge_1326_8M.pkl"),
    ("Prokaryotic", "Salmonella enterica", "yuanhe/41514/PPI_cliques_Hyperedge_41514_8M.pkl"),
    ("Prokaryotic", "Bifidobacterium", "yuanhe/77635/PPI_cliques_Hyperedge_77635_8M.pkl"),
    ("Prokaryotic", "Pseudomonas aeruginosa", "yuanhe/208964/PPI_cliques_Hyperedge_208964_8M.pkl"),
    ("Prokaryotic", "Actinobacteria", "yuanhe/Actinobacteria/PPI_cliques_Hyperedge_Actinobacteria_8M.pkl"),
    ("Prokaryotic", "E. coli", "demo/ratio-1-9/coli/PPI_cliques_Hyperedge_coli.pkl"),
)

FIELDS = (
    "group",
    "species",
    "source",
    "protein_nodes",
    "ppi_edges",
    "hyperedges",
    "size_1",
    "size_2",
    "size_3",
    "size_4",
    "size_ge5",
    "size_ge3",
    "pct_size_ge3",
    "max_size",
    "mean_size",
)


def summarize(data_root: Path) -> list[dict[str, object]]:
    rows = []
    for group, species, relative_path in SOURCES:
        source = data_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        # These pickle files must come from a trusted preprocessing pipeline.
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        cliques = payload["PPI_cliques_list"]
        if not cliques:
            raise ValueError(f"No cliques found in {source}")
        sizes = Counter(map(len, cliques))
        total = sum(sizes.values())
        size_ge3 = sum(count for size, count in sizes.items() if size >= 3)
        row = {
            "group": group,
            "species": species,
            "source": relative_path,
            "protein_nodes": payload["num_vertices"],
            "ppi_edges": len(payload["PPI_edge_list"]),
            "hyperedges": total,
            "size_1": sizes[1],
            "size_2": sizes[2],
            "size_3": sizes[3],
            "size_4": sizes[4],
            "size_ge5": sum(count for size, count in sizes.items() if size >= 5),
            "size_ge3": size_ge3,
            "pct_size_ge3": 100 * size_ge3 / total,
            "max_size": max(sizes),
            "mean_size": sum(size * count for size, count in sizes.items()) / total,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Root containing zhenhe/, yuanhe/, and demo/ratio-1-9/")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV path")
    args = parser.parse_args()

    rows = summarize(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} species summaries to {args.output}")


if __name__ == "__main__":
    main()
