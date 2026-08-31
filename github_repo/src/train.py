"""Train MHGNano with species-level meta-learning tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from data import load_species_data, sample_meta_batch
from models import HyperGraphMetaNetwork, MetaLearner
from utils import Checkpointer, EarlyStopping


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_meta_tasks(
    species_data: Dict[str, Tuple], species: List[str]
) -> List[Dict[str, Tuple]]:
    """Use train as support and validation as query for meta-training."""
    tasks = []
    for species_id in species:
        features, hypergraph, splits = species_data[species_id]
        tasks.append(
            {
                "support": (features, hypergraph, *splits["train"]),
                "query": (features, hypergraph, *splits["val"]),
            }
        )
    return tasks


def build_evaluation_tasks(
    species_data: Dict[str, Tuple],
    species: List[str],
    adapt: bool = True,
) -> List[Dict[str, Tuple]]:
    """Use target train as support and target test as query."""
    tasks = []
    for species_id in species:
        features, hypergraph, splits = species_data[species_id]
        support_split = splits["train"] if adapt else ([], [])
        tasks.append(
            {
                "support": (features, hypergraph, *support_split),
                "query": (features, hypergraph, *splits["test"]),
            }
        )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="./data/yuanhe")
    parser.add_argument(
        "--model_size",
        default="8M",
        choices=["8M", "35M", "150M", "650M"],
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help="Input feature dimension. Defaults to the loaded feature width.",
    )
    parser.add_argument("--hid_channels", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=128)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_lr", type=float, default=1e-4)
    parser.add_argument(
        "--loss_type", choices=["bce", "weighted_bce"], default="bce"
    )
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--meta_batch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--species",
        nargs="+",
        default=["1326", "41514", "77635", "208964"],
    )
    parser.add_argument("--out_dir", default="./output/yuanhe")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(args.species) < 2:
        raise ValueError(
            "Provide at least one training species and one held-out species."
        )

    train_species = args.species[:-1]
    held_out_species = args.species[-1]
    species_data = {
        species_id: load_species_data(
            species_id,
            args.data_root,
            args.model_size,
            fold=args.fold,
        )
        for species_id in train_species
    }
    tasks = build_meta_tasks(species_data, train_species)
    feature_widths = {features.shape[1] for features, _, _ in species_data.values()}
    if len(feature_widths) != 1:
        raise ValueError(
            f"Training species have inconsistent feature widths: {sorted(feature_widths)}."
        )
    inferred_in_channels = feature_widths.pop()
    if args.in_channels is not None and args.in_channels != inferred_in_channels:
        raise ValueError(
            f"--in_channels={args.in_channels} does not match the loaded feature "
            f"width ({inferred_in_channels})."
        )
    args.in_channels = inferred_in_channels

    model = HyperGraphMetaNetwork(
        in_channels=args.in_channels,
        hid_channels=args.hid_channels,
        num_classes=args.num_classes,
    ).to(device)
    meta = MetaLearner(
        model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        loss_type=args.loss_type,
    )
    checkpointer = Checkpointer(args.out_dir, args.save_every)
    early_stopping = EarlyStopping(args.patience, delta=1e-4)
    start_epoch = checkpointer.try_resume(meta, args.resume)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "device": str(device),
                "train_species": train_species,
                "held_out_species": held_out_species,
                "adaptable_parameters": "classifier.*",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="meta-training"):
        batch = sample_meta_batch(tasks, args.meta_batch_size)
        metrics = meta.meta_train_step(batch, device, args.threshold)
        checkpointer.save_last(epoch, meta, metrics)
        checkpointer.save_periodic(epoch, meta, metrics)
        is_best = checkpointer.save_if_best(epoch, meta, metrics)

        if epoch % 10 == 0 or is_best:
            print(
                f"[epoch {epoch:05d}] loss={metrics['meta_loss']:.4f} "
                f"val_auprc={metrics['meta_aucpr']:.4f} "
                f"val_f1={metrics['meta_f1']:.4f}"
                f"{' *best*' if is_best else ''}"
            )
        if epoch % 20 == 0 and early_stopping(metrics["meta_aucpr"]):
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"Best checkpoint: {checkpointer.best_path_pt}")


if __name__ == "__main__":
    main()
