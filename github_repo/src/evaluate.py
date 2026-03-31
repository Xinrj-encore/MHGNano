"""
Evaluation script for HyperGraph Meta-Learning.
"""
import argparse

import torch

from models import HyperGraphMetaNetwork, MetaLearner
from data import load_species_data, sample_meta_batch
from utils import print_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate HyperGraph Meta-Learning model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data/yuanhe")
    parser.add_argument("--model_size", type=str, default="8M", choices=["8M", "35M", "150M"])
    parser.add_argument("--species", type=str, nargs="+", default=["208964"])
    parser.add_argument("--in_channels", type=int, default=320)
    parser.add_argument("--hid_channels", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=128)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_lr", type=float, default=1e-4)
    parser.add_argument("--meta_batch_size", type=int, default=3)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
    num_hyperedges = 10000  # Will be updated
    base = HyperGraphMetaNetwork(
        in_channels=args.in_channels,
        hid_channels=args.hid_channels,
        num_classes=args.num_classes,
        num_hyperedges=num_hyperedges,
    ).to(device)

    base.load_state_dict(checkpoint["model_state_dict"])

    meta = MetaLearner(
        base_model=base,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
    )

    # Load test species data
    species_data = {}
    for sp in args.species:
        print(f"Loading {sp}...")
        X, hg, y = load_species_data(sp, args.data_root, args.model_size, device=None)
        species_data[sp] = (X, hg, y)

    # Build tasks
    from train import build_tasks_from_species_list
    tasks = build_tasks_from_species_list(species_data, args.species)
    batch = sample_meta_batch(tasks, args.meta_batch_size)

    # Evaluate
    print("\n===== Evaluation =====")
    metrics = meta.meta_evaluate(batch, device)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
