import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


METRIC_COLUMNS = ["acc", "precision", "recall", "f1", "auroc", "auprc", "threshold", "val_loss"]
BASE_COLUMNS = [
    "script",
    "run",
    "setting",
    "stage",
    "seed",
    "loss_type",
    "model_size",
    "checkpoint_path",
    "unseen_species",
    "inner_steps",
    "inner_lr",
    "meta_lr",
    "neg_sampling",
    "neg_ratio",
    "use_val_threshold",
]


def add_statistical_args(parser):
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Label used to group this run in multiseed statistical reports.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="",
        help="CSV file to append final before/after evaluation metrics. Default: <out_dir>/maml_eval_metrics.csv.",
    )
    parser.add_argument(
        "--stats-out-dir",
        type=str,
        default="",
        help="Directory for summary_mean_sd_ci.csv, paired_tests.csv, and statistical_report.md.",
    )
    parser.add_argument(
        "--stats-baseline-run",
        type=str,
        default="",
        help="Optional run label for paired between-run tests at --stats_compare_stage.",
    )
    parser.add_argument(
        "--stats-compare-stage",
        type=str,
        default="after_eval",
        choices=["before_eval", "after_eval"],
        help="Stage used when --stats_baseline_run is set.",
    )
    parser.add_argument(
        "--no-stats-report",
        action="store_true",
        help="Append the per-run CSV rows but skip regenerating the statistical report.",
    )


def write_maml_eval_statistics(
    metrics: Mapping[str, float],
    args,
    script_name: str,
    extra: Optional[Mapping[str, object]] = None,
):
    results_csv = Path(args.results_csv) if args.results_csv else Path(args.out_dir) / "maml_eval_metrics.csv"
    run_name = args.run_name or default_run_name(args, script_name)
    common = _common_fields(args, script_name, run_name, extra)
    rows = [
        _stage_row(common, metrics, "before_eval"),
        _stage_row(common, metrics, "after_eval"),
    ]

    _append_rows(results_csv, rows)
    print(f"[Stats] Appended final evaluation rows to {results_csv}")

    if args.no_stats_report:
        return

    stats_out_dir = Path(args.stats_out_dir) if args.stats_out_dir else results_csv.parent / "statistics"
    write_statistics_report(
        input_csv=results_csv,
        out_dir=stats_out_dir,
        baseline_run=args.stats_baseline_run,
        compare_stage=args.stats_compare_stage,
    )


def write_statistics_report(
    input_csv: Path,
    out_dir: Path,
    baseline_run: str = "",
    compare_stage: str = "after_eval",
):
    stats_module_dir = Path(__file__).resolve().parent
    if str(stats_module_dir) not in sys.path:
        sys.path.insert(0, str(stats_module_dir))

    try:
        from statistical_report_from_csv import (
            add_corrections,
            paired_group_tests,
            paired_stage_tests,
            read_rows,
            select_metrics,
            summarize,
            write_csv,
            write_markdown,
        )
    except Exception as exc:
        print(f"[Stats] Could not import statistical_report_from_csv.py: {exc}")
        return

    rows = read_rows(input_csv)
    if not rows:
        print(f"[Stats] No rows found in {input_csv}; skipped report generation.")
        return

    metrics = select_metrics(rows, ["acc", "precision", "recall", "f1", "auroc", "auprc"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = summarize(rows, metrics, group_col="run", stage_col="setting")
    write_csv(
        out_dir / "summary_mean_sd_ci.csv",
        summary_rows,
        ["group", "stage", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high", "mean_sd", "mean_ci95"],
    )

    test_rows = paired_stage_tests(
        rows,
        metrics,
        group_col="run",
        stage_col="setting",
        pair_col="seed",
        stage_a="before_eval",
        stage_b="after_eval",
    )
    if baseline_run:
        test_rows.extend(
            paired_group_tests(
                rows,
                metrics,
                group_col="run",
                stage_col="setting",
                pair_col="seed",
                baseline_group=baseline_run,
                compare_stage=compare_stage,
            )
        )
        test_rows = add_corrections(test_rows)

    if test_rows:
        write_csv(
            out_dir / "paired_tests.csv",
            test_rows,
            [
                "group",
                "comparison",
                "metric",
                "n",
                "mean_a",
                "mean_b",
                "mean_diff",
                "cohen_dz",
                "p_ttest",
                "p_wilcoxon",
                "p_ttest_bonferroni",
                "p_ttest_bh",
                "p_wilcoxon_bonferroni",
                "p_wilcoxon_bh",
            ],
        )

    write_markdown(out_dir / "statistical_report.md", summary_rows, test_rows)
    print(f"[Stats] Wrote {out_dir / 'summary_mean_sd_ci.csv'}")
    if test_rows:
        print(f"[Stats] Wrote {out_dir / 'paired_tests.csv'}")
    print(f"[Stats] Wrote {out_dir / 'statistical_report.md'}")


def default_run_name(args, script_name: str) -> str:
    out_tag = Path(getattr(args, "out_dir", ".")).name or "output"
    loss_type = getattr(args, "loss_type", "loss")
    threshold_tag = "selected_threshold" if getattr(args, "use_val_threshold", False) else "fixed_threshold"
    return f"{script_name}:{out_tag}:{loss_type}:{threshold_tag}"


def _common_fields(args, script_name: str, run_name: str, extra: Optional[Mapping[str, object]]):
    common = {
        "script": script_name,
        "run": run_name,
        "seed": getattr(args, "seed", ""),
        "loss_type": getattr(args, "loss_type", ""),
        "model_size": getattr(args, "model_size", ""),
        "checkpoint_path": getattr(args, "checkpoint_path", ""),
        "inner_steps": getattr(args, "inner_steps", ""),
        "inner_lr": getattr(args, "inner_lr", ""),
        "meta_lr": getattr(args, "meta_lr", ""),
        "neg_sampling": getattr(args, "neg_sampling", ""),
        "neg_ratio": getattr(args, "neg_ratio", ""),
        "use_val_threshold": getattr(args, "use_val_threshold", False),
    }
    if extra:
        common.update({key: _stringify(value) for key, value in extra.items()})
    return common


def _stage_row(common: Mapping[str, object], metrics: Mapping[str, float], setting: str):
    if setting == "before_eval":
        values = {
            "acc": metrics.get("before_acc", float("nan")),
            "precision": metrics.get("before_precision", float("nan")),
            "recall": metrics.get("before_recall", float("nan")),
            "f1": metrics.get("before_f1", float("nan")),
            "auroc": metrics.get("before_auroc", float("nan")),
            "auprc": metrics.get("before_aucpr", float("nan")),
            "threshold": metrics.get("before_threshold", float("nan")),
            "val_loss": "",
        }
    else:
        values = {
            "acc": metrics.get("acc", float("nan")),
            "precision": metrics.get("precision", float("nan")),
            "recall": metrics.get("recall", float("nan")),
            "f1": metrics.get("f1", float("nan")),
            "auroc": metrics.get("val_auroc", float("nan")),
            "auprc": metrics.get("val_aucpr", float("nan")),
            "threshold": metrics.get("threshold", float("nan")),
            "val_loss": metrics.get("val_loss", float("nan")),
        }

    row = {key: _stringify(value) for key, value in common.items()}
    row["setting"] = setting
    row["stage"] = setting
    for key, value in values.items():
        row[key] = _format_number(value)
    return row


def _append_rows(path: Path, rows: List[Mapping[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[Dict[str, str]] = []
    existing_fields: List[str] = []

    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    replacement_keys = {_row_key(row) for row in rows}
    existing_rows = [row for row in existing_rows if _row_key(row) not in replacement_keys]

    preferred_fields = BASE_COLUMNS + [key for key in ["setting", "stage"] if key not in BASE_COLUMNS] + METRIC_COLUMNS
    new_fields = []
    for row in rows:
        for key in row.keys():
            if key not in preferred_fields and key not in new_fields:
                new_fields.append(key)

    fieldnames = _merge_fields(existing_fields, preferred_fields + new_fields)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _row_key(row: Mapping[str, object]):
    return (
        str(row.get("script", "")),
        str(row.get("run", "")),
        str(row.get("seed", "")),
        str(row.get("setting", "")),
    )


def _merge_fields(existing: Iterable[str], preferred: Iterable[str]) -> List[str]:
    fields: List[str] = []
    for key in list(existing) + list(preferred):
        if key and key not in fields:
            fields.append(key)
    return fields


def _format_number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _stringify(value)
    if math.isnan(out):
        return ""
    return f"{out:.8g}"


def _stringify(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return "" if value is None else str(value)
