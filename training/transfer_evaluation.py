"""Evaluate a trained source-domain MHGNano checkpoint on target species."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from maml_statistics import add_statistical_args, write_maml_eval_statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--kingdom", choices=("eukaryotes", "prokaryotes"))
    parser.add_argument(
        "--species-dir",
        action="append",
        default=[],
        metavar="SPECIES=PATH",
        help="Exact data directory for a species outside the common kingdom root; repeatable.",
    )
    parser.add_argument("--train-species", nargs="+", required=True)
    parser.add_argument("--validation-species", required=True)
    parser.add_argument("--test-species", nargs="+", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-size", default="8M", choices=("8M", "35M", "150M", "650M"))
    parser.add_argument("--in-channels", type=int, default=320)
    parser.add_argument("--hid-channels", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=128)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument("--inner-steps", type=int)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type",
        choices=("bce", "weighted_bce", "focal", "weighted_focal", "bce_pairwise", "pairwise_auc"),
    )
    parser.add_argument("--focal-gamma", type=float)
    parser.add_argument("--pairwise-lambda", type=float)
    parser.add_argument("--pairwise-margin", type=float)
    parser.add_argument("--pairwise-max-pairs", type=int)
    parser.add_argument(
        "--support-size",
        type=int,
        default=64,
        help="Number of labeled target-train pairs used for few-shot adaptation.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./output/transfer_eval")
    add_statistical_args(parser)
    return parser


def _parse_species_dirs(entries: list[str]) -> dict[str, str]:
    result = {}
    for entry in entries:
        species, separator, path = entry.partition("=")
        if not separator or not species or not path:
            raise ValueError(f"Expected SPECIES=PATH, got {entry!r}")
        if species in result:
            raise ValueError(f"Duplicate --species-dir for {species!r}")
        result[species] = path
    return result


def _sample_support_split(split, support_size: int, seed: int):
    pairs, labels = split
    if support_size < 2:
        raise ValueError("--support-size must be at least 2 to include both classes")
    if len(pairs) != len(labels):
        raise ValueError("Support pair and label counts do not match")

    positive = [index for index, label in enumerate(labels) if float(label) > 0.5]
    negative = [index for index, label in enumerate(labels) if float(label) <= 0.5]
    if not positive or not negative:
        raise ValueError("Target training split must contain positive and negative pairs")

    count = min(support_size, len(pairs))
    if count == len(pairs):
        return pairs, labels

    positive_count = min(len(positive), max(1, round(count * len(positive) / len(pairs))))
    negative_count = count - positive_count
    if negative_count > len(negative):
        negative_count = len(negative)
        positive_count = count - negative_count
    if positive_count > len(positive):
        positive_count = len(positive)
        negative_count = count - positive_count

    rng = random.Random(seed)
    selected = rng.sample(positive, positive_count) + rng.sample(negative, negative_count)
    rng.shuffle(selected)
    return [pairs[index] for index in selected], [labels[index] for index in selected]


def evaluate_checkpoint(args) -> dict[str, float]:
    import torch

    from .meta_learner import MetaLearner
    from .model import MHGNano
    from .tasks import build_species_tasks, load_species_group

    all_species = args.train_species + [args.validation_species] + args.test_species
    if len(set(all_species)) != len(all_species):
        raise ValueError("Training, validation, and test species must be disjoint")
    if args.support_size < 2:
        raise ValueError("--support-size must be at least 2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_dirs = _parse_species_dirs(args.species_dir)

    test_data = load_species_group(
        args.test_species, args.data_root, args.model_size, args.kingdom, species_dirs
    )
    model = MHGNano(
        in_channels=args.in_channels,
        hid_channels=args.hid_channels,
        num_classes=args.num_classes,
        # Kept for constructor compatibility; the current model does not use it.
        num_hyperedges=0,
        use_bn=False,
        drop_rate=0.0,
    ).to(device)

    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_model

        load_model(model, str(checkpoint_path))
        checkpoint = {}
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state)

    saved_hyperparams = checkpoint.get("hyperparams", {})
    args.inner_lr = args.inner_lr if args.inner_lr is not None else saved_hyperparams.get("inner_lr", 1e-2)
    args.inner_steps = (
        args.inner_steps if args.inner_steps is not None else saved_hyperparams.get("inner_steps", 5)
    )
    args.loss_type = args.loss_type or saved_hyperparams.get("loss_type", "bce")
    args.focal_gamma = (
        args.focal_gamma
        if args.focal_gamma is not None
        else saved_hyperparams.get("focal_gamma", 2.0)
    )
    args.pairwise_lambda = (
        args.pairwise_lambda
        if args.pairwise_lambda is not None
        else saved_hyperparams.get("pairwise_lambda", 0.2)
    )
    args.pairwise_margin = (
        args.pairwise_margin
        if args.pairwise_margin is not None
        else saved_hyperparams.get("pairwise_margin", 0.0)
    )
    args.pairwise_max_pairs = (
        args.pairwise_max_pairs
        if args.pairwise_max_pairs is not None
        else saved_hyperparams.get("pairwise_max_pairs", 4096)
    )
    learner = MetaLearner(
        base_model=model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        pairwise_lambda=args.pairwise_lambda,
        pairwise_margin=args.pairwise_margin,
        pairwise_max_pairs=args.pairwise_max_pairs,
    )

    for task_index, species in enumerate(args.test_species):
        features, hypergraph, splits = test_data[species]
        splits["train"] = _sample_support_split(
            splits["train"], args.support_size, args.seed + task_index
        )
        test_data[species] = (features, hypergraph, splits)

    test_tasks = build_species_tasks(
        test_data, args.test_species, "test", seed=args.seed
    )
    metrics = learner.meta_evaluate(
        test_tasks, device, threshold=args.threshold, select_threshold=False
    )
    print(metrics)
    write_maml_eval_statistics(
        metrics,
        args,
        script_name="cross_domain_checkpoint_eval",
        extra={
            "train_species": args.train_species,
            "val_species": args.validation_species,
            "unseen_species": args.test_species,
            "support_size": args.support_size,
        },
    )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
