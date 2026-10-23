"""Recompute matched-seed ESM-2 scale summaries and paired tests."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parent
SIZES = ("8M", "35M", "150M", "650M")
SOURCE_NAMES = {
    "8 million": "8M",
    "35 million": "35M",
    "150 million": "150M",
    "650 million": "650M",
}
METRICS = ("Precision", "Recall", "Acc", "F1", "AUROC", "AUPRC")


def bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    result = np.empty_like(values)
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, values[index] * len(values) / rank)
        result[index] = min(running, 1.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "PR-D-26-01971_table_8_30seed_wide.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "recomputed")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data["size"] = data["method"].map(SOURCE_NAMES)
    if data["size"].isna().any():
        raise ValueError("Unexpected ESM-2 model labels.")

    summary_rows = []
    friedman_rows = []
    pairwise_rows = []
    for metric in METRICS:
        wide = data.pivot(index="seed", columns="size", values=metric).reindex(
            columns=SIZES
        )
        if wide.isna().any().any() or len(wide) != 30:
            raise ValueError(f"{metric} does not have 30 matched values per size.")
        for size in SIZES:
            summary_rows.append(
                {
                    "metric": metric,
                    "size": size,
                    "n": len(wide),
                    "mean": wide[size].mean(),
                    "sd": wide[size].std(ddof=1),
                }
            )
        test = friedmanchisquare(*(wide[size] for size in SIZES))
        friedman_rows.append(
            {"metric": metric, "friedman_p": test.pvalue, "best_size": wide.mean().idxmax()}
        )
        for left, right in combinations(SIZES, 2):
            test = wilcoxon(wide[left], wide[right], alternative="two-sided")
            pairwise_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{left} - {right}",
                    "mean_diff": (wide[left] - wide[right]).mean(),
                    "p_wilcoxon": test.pvalue,
                }
            )

    friedman = pd.DataFrame(friedman_rows)
    friedman["q_friedman"] = bh_adjust(friedman["friedman_p"].tolist())
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["q_wilcoxon"] = bh_adjust(pairwise["p_wilcoxon"].tolist())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "mean_sd.csv", index=False)
    friedman.to_csv(args.out_dir / "friedman.csv", index=False)
    pairwise.to_csv(args.out_dir / "pairwise_wilcoxon.csv", index=False)


if __name__ == "__main__":
    main()
