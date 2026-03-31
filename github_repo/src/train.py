"""
Training script for HyperGraph Meta-Learning.
"""
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from dhg import Hypergraph
from tqdm import tqdm

from models import HyperGraphMetaNetwork, MetaLearner
from data import load_species_data, sample_meta_batch
from utils import Checkpointer, EarlyStopping


def build_tasks_from_species_list(
    species_data: Dict[str, Tuple], splist: List[str]
) -> List[Dict[str, Tuple]]:
    """Build meta-learning tasks from species list."""
    tasks = []
    for sp in splist:
        X, hg, y = species_data[sp]
        s_idx = y[0][0]
        s_label = y[0][1]
        q_idx = y[1][0]
        q_label = y[1][1]
        support = (X, hg, s_idx, s_label)
        query = (X, hg, q_idx, q_label)
        tasks.append({"support": support, "query": query})
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Train HyperGraph Meta-Learning model"
    )
    parser.add_argument(
        "--data_root", type=str, default="./data/yuanhe", help="Data root directory"
    )
    parser.add_argument(
        "--model_size", type=str, default="8M", choices=["8M", "35M", "150M"]
    )
    parser.add_argument("--in_channels", type=int, default=320)
    parser.add_argument("--hid_channels", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=128)

    # MAML hyperparameters
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--meta_batch_size", type=int, default=3)
    parser.add_argument("--n_support", type=int, default=64)
    parser.add_argument("--n_query", type=int, default=64)

    # Training settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)

    # Species list
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=["1326", "41514", "77635", "208964"],
        help="List of species IDs",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split species
    if len(args.species) > 1:
        unseen_species = args.species[-1]
        train_species = args.species[:-1]
    else:
        unseen_species = args.species[0]
        train_species = args.species

    print(f"Training species: {train_species}")
    print(f"Unseen species: {unseen_species}")

    # Load species data
    num_hyperedges = 0
    species_data = {}
    for sp in train_species:
        print(f"Loading {sp}...")
        X, hg, y = load_species_data(sp, args.data_root, args.model_size, device=None)
        species_data[sp] = (X, hg, y)
        num_hyperedges = max(num_hyperedges, len(hg.e))

    # Create model
    base = HyperGraphMetaNetwork(
        in_channels=args.in_channels,
        hid_channels=args.hid_channels,
        num_classes=args.num_classes,
        num_hyperedges=num_hyperedges,
    ).to(device)

    meta = MetaLearner(
        base_model=base,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
    )

    print(f"Model parameters: {sum(p.numel() for p in base.parameters()):,}")

    # Checkpointer
    checkpointer = Checkpointer(args.out_dir, save_every=args.save_every)
    start_epoch = checkpointer.try_resume(meta, args.resume)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, delta=0.001)

    # Training loop
    print("\n===== Meta-Training =====")
    loss_list = []

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Training"):
        tasks = build_tasks_from_species_list(species_data, train_species)

        if len(tasks) == 0:
            raise RuntimeError("No valid tasks constructed.")

        batch = sample_meta_batch(tasks, args.meta_batch_size)
        metrics, before_loss = meta.meta_train_step(batch, device)
        loss_list.append(before_loss)

        # Save checkpoints
        checkpointer.save_last(epoch, meta, metrics)
        checkpointer.save_periodic(epoch, meta, metrics)
        is_best = checkpointer.save_if_best(epoch, meta, metrics, monitor="meta_aucpr")

        # Early stopping check
        if epoch % 20 == 0 and epoch > 1:
            improved = early_stopping(
                metrics["meta_aucpr"], loss_list[-2], before_loss
            )
            if not improved:
                print(f"[Epoch {epoch}] No improvement (best={early_stopping.best_auc_pr:.4f})")
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Print progress
        if epoch % 10 == 0:
            print(
                f"[Epoch {epoch:04d}] "
                f"loss={metrics['meta_loss']:.4f} | "
                f"acc={metrics['meta_acc']:.4f} | "
                f"auroc={metrics['meta_auroc']:.4f} | "
                f"aucpr={metrics['meta_aucpr']:.4f}"
                f"{' *BEST*' if is_best else ''}"
            )

    print(f"\nTraining complete! Best model saved to {checkpointer.best_path_pt}")


if __name__ == "__main__":
    main()
