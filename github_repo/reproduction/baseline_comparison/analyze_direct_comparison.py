"""Recompute independent TRGH-PPI/HIGH-PPI versus MHGNano-few tests."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


ROOT = Path(__file__).resolve().parent
METRICS = {
    "Precision": "precision",
    "Recall": "recall",
    "Acc": "acc",
    "F1": "f1",
    "AUROC": "auroc",
    "AUPRC": "aucpr",
}
TABLE_PREFIX = {
    "prokaryotic_to_prokaryotic": "Table2_",
    "eukaryotic_to_eukaryotic": "Table3_",
    "eukaryotic_to_prokaryotic": "Table4_",
    "prokaryotic_to_eukaryotic": "Table5_",
}
DIRECTION = {
    ("TRGH-PPI", "prokaryotic_to_prokaryotic"): "prok_to_prok_208964",
    ("TRGH-PPI", "eukaryotic_to_eukaryotic"): "euk_to_euk_elegans",
    ("TRGH-PPI", "eukaryotic_to_prokaryotic"): "euk_to_prok_208964",
    ("TRGH-PPI", "prokaryotic_to_eukaryotic"): "prok_to_euk_mouse",
    ("HIGH-PPI", "prokaryotic_to_prokaryotic"): "prok_to_prok_208964",
    ("HIGH-PPI", "eukaryotic_to_eukaryotic"): "euk_to_euk_elegans",
    ("HIGH-PPI", "eukaryotic_to_prokaryotic"): "euk_to_prok_208964",
    ("HIGH-PPI", "prokaryotic_to_eukaryotic"): "prok_to_euk_elegans",
}


def bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    q_values = np.empty_like(values)
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, values[index] * len(values) / rank)
        q_values[index] = min(running, 1.0)
    return q_values


def load_reference(path: Path) -> dict[tuple[str, str], np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["method"] == "MHGNano-few"].copy()
    output = {}
    for setting, prefix in TABLE_PREFIX.items():
        subset = frame.loc[frame["table_id"].str.startswith(prefix)]
        for metric in METRICS:
            output[setting, metric] = subset.loc[
                subset["metric"] == metric, "value"
            ].to_numpy(float)

    for metric in METRICS:
        per_seed = (
            frame.loc[frame["table_id"].str.startswith(tuple(TABLE_PREFIX.values()))]
            .query("metric == @metric")
            .groupby("seed", sort=True)["value"]
            .mean()
        )
        output["overall_ppi", metric] = per_seed.to_numpy(float)
    return output


def current_values(
    frame: pd.DataFrame, model: str, setting: str, metric_column: str
) -> np.ndarray:
    if setting == "overall_ppi":
        return (
            frame.loc[frame["model"] == model]
            .groupby("seed", sort=True)[metric_column]
            .mean()
            .to_numpy(float)
        )
    direction = DIRECTION[model, setting]
    return frame.loc[
        (frame["model"] == model) & (frame["direction"] == direction),
        metric_column,
    ].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct",
        type=Path,
        default=ROOT / "trgh_high_30seed_direct_raw.csv",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "mhgnano_main_30seed_long.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "recomputed_independent_tests.csv",
    )
    args = parser.parse_args()

    direct = pd.read_csv(args.direct)
    reference = load_reference(args.reference)
    rows = []
    settings = [*TABLE_PREFIX, "overall_ppi"]
    for model, setting, (metric, column) in (
        (model, setting, item)
        for model in ("TRGH-PPI", "HIGH-PPI")
        for setting in settings
        for item in METRICS.items()
    ):
        current = current_values(direct, model, setting, column)
        baseline = reference[setting, metric]
        mw = mannwhitneyu(current, baseline, alternative="two-sided")
        welch = ttest_ind(current, baseline, equal_var=False)
        rows.append(
            {
                "model": model,
                "setting": setting,
                "metric": metric,
                "n_new": len(current),
                "n_reference": len(baseline),
                "new_mean": current.mean(),
                "reference_mean": baseline.mean(),
                "mean_diff_new_minus_reference": current.mean() - baseline.mean(),
                "p_mannwhitney": mw.pvalue,
                "p_welch": welch.pvalue,
            }
        )
    results = pd.DataFrame(rows)
    results["q_mannwhitney"] = bh_adjust(results["p_mannwhitney"].tolist())
    results["q_welch"] = bh_adjust(results["p_welch"].tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
