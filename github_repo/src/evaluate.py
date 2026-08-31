"""Evaluate a checkpoint on held-out species using the fixed cutoff protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import load_species_data
from models import HyperGraphMetaNetwork, MetaLearner
from train import build_evaluation_tasks, set_seed
from utils import print_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_adapt", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = HyperGraphMetaNetwork(in_channels=inferred_in_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    metrics = meta.meta_evaluate(
        tasks,
        device,
        threshold=args.threshold,
        adapt=not args.no_adapt,
    )
    print_metrics(metrics, prefix="Evaluation")
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "species": args.species,
        "fold": args.fold,
        "seed": args.seed,
        "adapt": not args.no_adapt,
        "threshold": args.threshold,
        "metrics": metrics,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
