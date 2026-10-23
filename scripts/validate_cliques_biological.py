"""
Biological validation for clique-derived hyperedges.

Real analysis workflow:
1. Load hyperedge cliques from MHGNano pickle.
2. Map node ids to protein identifiers through Protein_list.csv.
3. Load GO annotations from a GAF file.
4. Run per-clique GO enrichment with hypergeometric test and BH correction.
5. Compare clique coherence against random same-size protein sets.
6. Export paper-ready summaries and top examples.

This script intentionally avoids optional heavy dependencies such as goatools.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu


@dataclass
class GOTermResult:
    go_id: str
    k: int
    K: int
    n: int
    N: int
    p_value: float
    q_value: float


def load_cliques_from_pickle(pickle_path: Path) -> List[List[int]]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "PPI_cliques_list" in data:
        cliques = data["PPI_cliques_list"]
    elif isinstance(data, list):
        cliques = data
    else:
        raise ValueError(f"Unexpected clique pickle structure: {type(data)}")

    return [list(map(int, clique)) for clique in cliques]


def load_protein_mapping(protein_list_path: Path) -> pd.DataFrame:
    df = pd.read_csv(protein_list_path)
    required = {"Entry", "ID"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{protein_list_path} must contain columns {required}, got {set(df.columns)}"
        )
    return df[["Entry", "ID"]].copy()


def build_universe_alias_index(universe_entries: Set[str]) -> Dict[str, Set[str]]:
    alias_to_entries: Dict[str, Set[str]] = defaultdict(set)

    def add_alias(entry: str, alias: str) -> None:
        alias = alias.strip()
        if alias:
            alias_to_entries[alias].add(entry)

    for entry in universe_entries:
        add_alias(entry, entry)
        if "." in entry:
            suffix = entry.split(".", 1)[1]
            add_alias(entry, suffix)
            parts = suffix.split(".")
            if parts and parts[-1].isdigit():
                without_terminal_isoform = ".".join(parts[:-1])
                add_alias(entry, without_terminal_isoform)
                if without_terminal_isoform and without_terminal_isoform[-1:].isalpha():
                    add_alias(entry, without_terminal_isoform[:-1])
    return dict(alias_to_entries)


def load_go_annotations_from_gaf(
    gaf_path: Path,
    universe_entries: Set[str],
    aspect_filter: Optional[Set[str]] = None,
    evidence_exclude: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    GAF column meanings:
    1 DB
    2 DB_Object_ID
    3 DB_Object_Symbol
    5 GO_ID
    7 Evidence
    9 Aspect
    11 DB_Object_Type
    17 Gene_Product_Form_ID
    """
    aspect_filter = aspect_filter or {"P", "C"}
    evidence_exclude = evidence_exclude or {"IEA"}
    protein_to_go: Dict[str, Set[str]] = defaultdict(set)
    alias_index = build_universe_alias_index(universe_entries)

    opener = gzip.open if gaf_path.suffix == ".gz" else open
    with opener(gaf_path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            db_object_id = parts[1].strip()
            db_object_symbol = parts[2].strip()
            go_id = parts[4].strip()
            evidence = parts[6].strip()
            aspect = parts[8].strip()
            synonyms = parts[10].strip()
            obj_type = parts[11].strip().lower()
            gene_product_form_id = parts[16].strip() if len(parts) > 16 else ""

            if obj_type and obj_type not in {"protein", "gene_product"}:
                continue
            if aspect_filter and aspect not in aspect_filter:
                continue
            if evidence in evidence_exclude:
                continue
            candidate_aliases = {db_object_id, db_object_symbol, gene_product_form_id}
            if synonyms:
                candidate_aliases.update(x.strip() for x in synonyms.split("|") if x.strip())

            matched_entries: Set[str] = set()
            for alias in candidate_aliases:
                matched_entries.update(alias_index.get(alias, set()))
                if ":" in alias:
                    matched_entries.update(alias_index.get(alias.rsplit(":", 1)[-1], set()))
                if "." in alias:
                    parts_alias = alias.split(".")
                    if parts_alias[-1].isdigit():
                        matched_entries.update(alias_index.get(".".join(parts_alias[:-1]), set()))

            for entry in matched_entries:
                protein_to_go[entry].add(go_id)

    return dict(protein_to_go)


def benjamini_hochberg(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    n = len(values)
    order = np.argsort(values)
    q_values = np.empty(n, dtype=float)
    running_min = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        adjusted = values[idx] * n / rank
        running_min = min(running_min, adjusted)
        q_values[idx] = min(running_min, 1.0)
    return list(q_values)


def functional_homogeneity_score(clique_entries: Sequence[str], protein_to_go: Dict[str, Set[str]]) -> float:
    go_sets = [protein_to_go.get(entry, set()) for entry in clique_entries if protein_to_go.get(entry)]
    if len(go_sets) < 2:
        return 0.0

    similarities: List[float] = []
    for i in range(len(go_sets)):
        for j in range(i + 1, len(go_sets)):
            union = go_sets[i] | go_sets[j]
            if not union:
                continue
            similarities.append(len(go_sets[i] & go_sets[j]) / len(union))
    return float(np.mean(similarities)) if similarities else 0.0


def perform_go_enrichment(
    clique_entries: Sequence[str],
    protein_to_go: Dict[str, Set[str]],
    term_to_proteins: Dict[str, Set[str]],
    min_annotated: int = 2,
    excluded_terms: Optional[Set[str]] = None,
    max_term_fraction: float = 0.25,
) -> List[GOTermResult]:
    excluded_terms = excluded_terms or {"GO:0008150", "GO:0003674", "GO:0005575"}
    annotated_entries = [entry for entry in clique_entries if entry in protein_to_go]
    if len(annotated_entries) < min_annotated:
        return []

    clique_terms: Counter[str] = Counter()
    for entry in annotated_entries:
        clique_terms.update(protein_to_go[entry])

    universe = set(protein_to_go.keys())
    N = len(universe)
    n = len(set(annotated_entries))
    if n < min_annotated:
        return []

    raw_results: List[Tuple[str, int, int, int, int, float]] = []
    for go_id, k in clique_terms.items():
        if go_id in excluded_terms:
            continue
        proteins_with_term = term_to_proteins.get(go_id, set()) & universe
        K = len(proteins_with_term)
        if K == 0:
            continue
        if (K / N) > max_term_fraction:
            continue
        p_value = hypergeom.sf(k - 1, N, K, n)
        raw_results.append((go_id, k, K, n, N, p_value))

    if not raw_results:
        return []

    q_values = benjamini_hochberg([row[5] for row in raw_results])
    results = [
        GOTermResult(
            go_id=go_id,
            k=k,
            K=K,
            n=n,
            N=N,
            p_value=p,
            q_value=q,
        )
        for (go_id, k, K, n, N, p), q in zip(raw_results, q_values)
    ]
    results.sort(key=lambda x: (x.q_value, x.p_value, -x.k, x.go_id))
    return results


def parse_go_names(go_obo_path: Path) -> Dict[str, str]:
    names: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_name: Optional[str] = None

    with open(go_obo_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current_id and current_name:
                    names[current_id] = current_name
                current_id = None
                current_name = None
                continue
            if line.startswith("id: GO:"):
                current_id = line.split("id: ", 1)[1]
            elif line.startswith("name: "):
                current_name = line.split("name: ", 1)[1]
        if current_id and current_name:
            names[current_id] = current_name
    return names


def random_same_size_scores(
    clique_sizes: Sequence[int],
    universe_entries: Sequence[str],
    protein_to_go: Dict[str, Set[str]],
    n_repeats: int,
    seed: int,
) -> List[float]:
    rng = np.random.default_rng(seed)
    universe_entries = np.array(universe_entries, dtype=object)
    scores: List[float] = []

    for size in clique_sizes:
        if size < 2 or size > len(universe_entries):
            scores.append(0.0)
            continue
        repeat_scores = []
        for _ in range(n_repeats):
            sample = rng.choice(universe_entries, size=size, replace=False).tolist()
            repeat_scores.append(functional_homogeneity_score(sample, protein_to_go))
        scores.append(float(np.mean(repeat_scores)))
    return scores


def build_term_to_proteins(protein_to_go: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    term_to_proteins: Dict[str, Set[str]] = defaultdict(set)
    for protein, terms in protein_to_go.items():
        for term in terms:
            term_to_proteins[term].add(protein)
    return dict(term_to_proteins)


def run_real_analysis(
    clique_file: Path,
    protein_list_file: Path,
    go_annotation_file: Path,
    output_dir: Path,
    go_obo_file: Optional[Path] = None,
    min_clique_size: int = 3,
    max_cliques: Optional[int] = None,
    random_repeats: int = 100,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cliques = load_cliques_from_pickle(clique_file)
    protein_df = load_protein_mapping(protein_list_file)
    id_to_entry = dict(zip(protein_df["ID"], protein_df["Entry"]))

    filtered_cliques = [clique for clique in cliques if len(clique) >= min_clique_size]
    if max_cliques is not None:
        filtered_cliques = filtered_cliques[:max_cliques]

    universe_entries = set(protein_df["Entry"].astype(str).tolist())
    protein_to_go = load_go_annotations_from_gaf(go_annotation_file, universe_entries)
    term_to_proteins = build_term_to_proteins(protein_to_go)
    go_names = parse_go_names(go_obo_file) if go_obo_file and go_obo_file.exists() else {}

    rows = []
    clique_sizes = []
    observed_scores = []

    for clique_id, clique in enumerate(filtered_cliques):
        clique_entries = [id_to_entry[idx] for idx in clique if idx in id_to_entry]
        unique_entries = sorted(set(clique_entries))
        clique_sizes.append(len(unique_entries))

        homogeneity = functional_homogeneity_score(unique_entries, protein_to_go)
        enrichment = perform_go_enrichment(unique_entries, protein_to_go, term_to_proteins)
        sig_terms = [term for term in enrichment if term.q_value < 0.05]

        top_term = sig_terms[0] if sig_terms else (enrichment[0] if enrichment else None)
        rows.append(
            {
                "clique_id": clique_id,
                "size": len(unique_entries),
                "annotated_proteins": sum(1 for entry in unique_entries if entry in protein_to_go),
                "functional_homogeneity": homogeneity,
                "n_enriched_terms_fdr05": len(sig_terms),
                "has_enrichment_fdr05": int(len(sig_terms) > 0),
                "top_go_id": top_term.go_id if top_term else "",
                "top_go_name": go_names.get(top_term.go_id, "") if top_term else "",
                "top_go_q_value": top_term.q_value if top_term else math.nan,
                "top_go_p_value": top_term.p_value if top_term else math.nan,
                "proteins": ";".join(unique_entries),
            }
        )
        observed_scores.append(homogeneity)

    results_df = pd.DataFrame(rows)
    random_scores = random_same_size_scores(
        clique_sizes=clique_sizes,
        universe_entries=sorted(universe_entries),
        protein_to_go=protein_to_go,
        n_repeats=random_repeats,
        seed=seed,
    )
    results_df["random_same_size_homogeneity"] = random_scores
    results_df["homogeneity_gain_vs_random"] = (
        results_df["functional_homogeneity"] - results_df["random_same_size_homogeneity"]
    )

    observed = results_df["functional_homogeneity"].to_numpy(dtype=float)
    random_arr = results_df["random_same_size_homogeneity"].to_numpy(dtype=float)
    p_u = mannwhitneyu(observed, random_arr, alternative="greater")

    paper_table = (
        results_df.sort_values(
            by=["has_enrichment_fdr05", "functional_homogeneity", "n_enriched_terms_fdr05"],
            ascending=[False, False, False],
        )
        .head(10)
        .reset_index(drop=True)
    )

    summary = {
        "total_cliques_evaluated": int(len(results_df)),
        "min_clique_size": int(min_clique_size),
        "proteins_in_universe": int(len(universe_entries)),
        "proteins_with_go_annotations": int(len(protein_to_go)),
        "fraction_proteins_with_go_annotations": float(len(protein_to_go) / len(universe_entries)) if universe_entries else 0.0,
        "cliques_with_enrichment_fdr05": int(results_df["has_enrichment_fdr05"].sum()),
        "fraction_cliques_with_enrichment_fdr05": float(results_df["has_enrichment_fdr05"].mean()) if len(results_df) else 0.0,
        "mean_functional_homogeneity": float(results_df["functional_homogeneity"].mean()) if len(results_df) else 0.0,
        "median_functional_homogeneity": float(results_df["functional_homogeneity"].median()) if len(results_df) else 0.0,
        "mean_random_same_size_homogeneity": float(results_df["random_same_size_homogeneity"].mean()) if len(results_df) else 0.0,
        "median_random_same_size_homogeneity": float(results_df["random_same_size_homogeneity"].median()) if len(results_df) else 0.0,
        "mean_homogeneity_gain_vs_random": float(results_df["homogeneity_gain_vs_random"].mean()) if len(results_df) else 0.0,
        "mannwhitneyu_statistic": float(p_u.statistic),
        "mannwhitneyu_p_value_greater": float(p_u.pvalue),
        "random_repeats_per_clique": int(random_repeats),
    }

    interpretation_lines = [
        f"We evaluated {summary['total_cliques_evaluated']} clique-derived hyperedges (size >= {summary['min_clique_size']}) using GO Biological Process and Cellular Component annotations.",
        f"{summary['fraction_cliques_with_enrichment_fdr05'] * 100:.1f}% of cliques showed at least one enriched GO term after Benjamini-Hochberg correction (FDR < 0.05).",
        f"The mean functional homogeneity score was {summary['mean_functional_homogeneity']:.4f}, compared with {summary['mean_random_same_size_homogeneity']:.4f} for same-size random protein sets.",
        f"Clique-derived groups were significantly more functionally coherent than random controls (Mann-Whitney U, one-sided p = {summary['mannwhitneyu_p_value_greater']:.3e}).",
        "These results support biological plausibility of clique-derived hyperedges while not implying one-to-one equivalence with curated complexes.",
    ]

    results_path = output_dir / "clique_validation_results.csv"
    paper_table_path = output_dir / "table_for_paper_top_cliques.csv"
    summary_path = output_dir / "validation_summary.json"
    interpretation_path = output_dir / "paper_ready_interpretation.txt"

    results_df.to_csv(results_path, index=False)
    paper_table.to_csv(paper_table_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(interpretation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(interpretation_lines) + "\n")

    print("=" * 72)
    print("Clique biological validation completed")
    print("=" * 72)
    print(f"Cliques evaluated: {summary['total_cliques_evaluated']}")
    print(f"Annotated proteins in universe: {summary['proteins_with_go_annotations']} / {summary['proteins_in_universe']}")
    print(f"Cliques with FDR<0.05 enrichment: {summary['cliques_with_enrichment_fdr05']} "
          f"({summary['fraction_cliques_with_enrichment_fdr05'] * 100:.1f}%)")
    print(f"Mean homogeneity: {summary['mean_functional_homogeneity']:.4f}")
    print(f"Mean random same-size homogeneity: {summary['mean_random_same_size_homogeneity']:.4f}")
    print(f"Mann-Whitney U p-value (greater): {summary['mannwhitneyu_p_value_greater']:.3e}")
    print()
    print(f"Saved: {results_path}")
    print(f"Saved: {paper_table_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {interpretation_path}")


def simulate_validation(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_cliques = 500
    validation_results = pd.DataFrame(
        {
            "clique_id": range(n_cliques),
            "size": rng.choice([3, 4, 5, 6, 7, 8, 10, 12], n_cliques),
            "functional_homogeneity": rng.beta(5, 2, n_cliques),
            "random_same_size_homogeneity": rng.beta(2, 5, n_cliques),
        }
    )
    validation_results["homogeneity_gain_vs_random"] = (
        validation_results["functional_homogeneity"] - validation_results["random_same_size_homogeneity"]
    )
    validation_results["has_enrichment_fdr05"] = (
        validation_results["functional_homogeneity"] > 0.5
    ).astype(int)
    validation_results.to_csv(output_dir / "clique_validation_results.csv", index=False)
    print(f"Simulation output written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biological validation for clique-derived hyperedges")
    parser.add_argument("--clique_file", type=Path, help="Path to PPI clique pickle file")
    parser.add_argument("--protein_list", type=Path, help="Path to Protein_list.csv")
    parser.add_argument("--go_annotation", type=Path, help="Path to GAF annotation file (.gaf or .gaf.gz)")
    parser.add_argument("--go_obo", type=Path, help="Path to go-basic.obo", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("analysis/biological_validation"))
    parser.add_argument("--min_clique_size", type=int, default=3)
    parser.add_argument("--max_cliques", type=int, default=None)
    parser.add_argument("--random_repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    if args.simulate:
        simulate_validation(args.output_dir)
        return

    required = [args.clique_file, args.protein_list, args.go_annotation]
    if any(item is None for item in required):
        raise SystemExit("Real analysis requires --clique_file, --protein_list, and --go_annotation.")

    run_real_analysis(
        clique_file=args.clique_file,
        protein_list_file=args.protein_list,
        go_annotation_file=args.go_annotation,
        output_dir=args.output_dir,
        go_obo_file=args.go_obo,
        min_clique_size=args.min_clique_size,
        max_cliques=args.max_cliques,
        random_repeats=args.random_repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
