"""Repeat held-out evaluation and retain seed-level metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from data import load_species_data
from models import HyperGraphMetaNetwork, MetaLearner
from train import build_evaluation_tasks, set_seed


METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")


def evaluate_one(
    checkpoint: str,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(seed)
    species_data = {
        species_id: load_species_data(
            species_id,
            args.data_root,
            args.model_size,
            fold=args.fold,
        )
        for species_id in args.species
    }
    feature_widths = {features.shape[1] for features, _, _ in species_data.values()}
    if len(feature_widths) != 1:
        raise ValueError(
            f"Evaluation species have inconsistent feature widths: {sorted(feature_widths)}."
        )
    inferred_in_channels = feature_widths.pop()
    if args.in_channels is not None and args.in_channels != inferred_in_channels:
        raise ValueError(
            f"--in_channels={args.in_channels} does not match the loaded feature "
            f"width ({inferred_in_channels})."
        )
    checkpoint_payload = torch.load(checkpoint, map_location=device)
    model = HyperGraphMetaNetwork(in_channels=inferred_in_channels).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    meta = MetaLearner(
        model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        loss_type=args.loss_type,
    )
    tasks = build_evaluation_tasks(
        species_data, args.species, adapt=not args.no_adapt
    )
    result = meta.meta_evaluate(
        tasks,
        device,
        threshold=args.threshold,
        adapt=not args.no_adapt,
    )
    return {metric: float(result[metric]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoint_template",
        help="Template containing {seed}, used for one checkpoint per seed.",
    )
    parser.add_argument("--data_root", default="./data/yuanhe")
    parser.add_argument(
        "--model_size", default="8M", choices=["8M", "35M", "150M", "650M"]
    )
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_lr", type=float, default=1e-4)
    parser.add_argument(
        "--loss_type", choices=["bce", "weighted_bce"], default="bce"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_adapt", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2023],
    )
    parser.add_argument("--output", default="./results/evaluation_seed_level.json")
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_template:
        parser.error("Provide --checkpoint or --checkpoint_template.")
    if args.checkpoint and args.checkpoint_template:
        parser.error("Use only one checkpoint source.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        checkpoint = (
            args.checkpoint_template.format(seed=seed)
            if args.checkpoint_template
            else args.checkpoint
        )
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        metrics = evaluate_one(checkpoint, args, seed, device)
        rows.append({"seed": seed, "checkpoint": str(checkpoint), **metrics})

    summary = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[metric] = {
            "mean": float(np.nanmean(values)),
            "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    payload = {
        "protocol": {
            "threshold": args.threshold,
            "adapt": not args.no_adapt,
            "fold": args.fold,
            "seeds": args.seeds,
            "test_labels_used_for_threshold": False,
        },
        "rows": rows,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
