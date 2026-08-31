"""Regenerate class-ratio summaries from the archived seed-level CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS = ("auroc", "auprc", "f1", "precision", "recall", "acc")
TASK_ORDER = ("yuanhe", "zhenhe", "yz", "zy")
TASK_LABELS = {
    "yuanhe": "Prok. to Prok.",
    "zhenhe": "Euk. to Euk.",
    "yz": "Prok. to Euk.",
    "zy": "Euk. to Prok.",
}
SOURCES = {
    1: "ratio_1_all_maml_eval_metrics.csv",
    3: "ratio_3_all_maml_eval_metrics.csv",
    5: "ratio_5_all_maml_eval_metrics.csv",
    9: "ratio_9_paper_original_all_maml_eval_metrics.csv",
}


def task_from_run(run: str) -> str:
    for task in TASK_ORDER:
        if str(run).startswith(task):
            return task
    raise ValueError(f"Cannot identify transfer task from run name: {run}")


def load() -> pd.DataFrame:
    frames = []
    for ratio, filename in SOURCES.items():
        frame = pd.read_csv(ROOT / filename)
        required = {"run", "stage", "seed", *METRICS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing {sorted(missing)}")
        frame = frame.loc[frame["stage"] == "after_eval"].copy()
        frame["ratio"] = ratio
        frame["task"] = frame["run"].map(task_from_run)
        frame["data_protocol"] = (
            "original_paper_split" if ratio == 9 else "pair_disjoint"
        )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    counts = data.groupby(["ratio", "task"]).size()
    if not (counts == 5).all():
        raise ValueError("Each ratio/task group must contain exactly five seeds.")
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    long = data.melt(
        id_vars=("ratio", "task", "data_protocol", "seed"),
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    summary = (
        long.groupby(["ratio", "task", "data_protocol", "metric"], sort=False)
        .value.agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sd"})
    )
    return summary


def make_wide(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(
        index=["ratio", "task", "data_protocol"],
        columns="metric",
        values=["mean", "sd"],
    )
    wide.columns = [f"{metric}_{statistic}" for statistic, metric in wide.columns]
    return wide.reset_index().sort_values(["ratio", "task"])


def plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 8.0), sharex=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        subset = summary.loc[summary["metric"] == metric]
        for task in TASK_ORDER:
            series = subset.loc[subset["task"] == task].sort_values("ratio")
            axis.errorbar(
                series["ratio"],
                series["mean"],
                yerr=series["sd"],
                marker="o",
                linewidth=1.2,
                capsize=2.5,
                label=TASK_LABELS[task],
            )
        axis.set_title(metric.upper())
        axis.set_xticks([1, 3, 5, 9])
        axis.set_xticklabels(["1:1", "1:3", "1:5", "1:9"])
        axis.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data = load()
    summary = summarize(data)
    wide = make_wide(summary)
    data.to_csv(ROOT / "after_eval_raw_metrics.csv", index=False)
    summary.to_csv(ROOT / "after_eval_summary_mean_sd_long.csv", index=False)
    wide.to_csv(ROOT / "after_eval_summary_mean_sd_wide.csv", index=False)
    if args.plot:
        plot(summary, ROOT / "fig_class_imbalance_robustness.png")

    quality = {
        "after_eval_rows": int(len(data)),
        "groups": int(data.groupby(["ratio", "task"]).ngroups),
        "seeds_per_group": 5,
        "seed_set": [42, 123, 456, 789, 2023],
        "protocol_by_ratio": {
            f"1:{ratio}": "original_paper_split" if ratio == 9 else "pair_disjoint"
            for ratio in SOURCES
        },
    }
    (ROOT / "data_quality_checks.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
