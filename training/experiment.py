import random

import numpy as np
import torch
from tqdm import tqdm

from maml_statistics import write_maml_eval_statistics
from .checkpoint import Checkpointer, EarlyStopping
from .data import sample_meta_batch
from .meta_learner import MetaLearner
from .model import MHGNano
from .tasks import build_species_tasks, load_species_group


def run_experiment(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_species = args.train_species
    validation_species = args.validation_species
    test_species = args.test_species
    all_species = train_species + [validation_species] + test_species
    if len(set(all_species)) != len(all_species):
        raise ValueError("Training, validation, and test species must be disjoint")
    if args.epochs < 2:
        raise ValueError("epochs must be at least 2 because validation runs every two epochs")

    print(
        f"Training species: {train_species}; "
        f"validation species: {validation_species}; test species: {test_species}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_dirs = dict(item.split("=", 1) for item in args.species_dir)
    train_data = load_species_group(
        train_species, args.data_root, args.model_size, args.kingdom, species_dirs
    )
    validation_data = load_species_group(
        [validation_species], args.data_root, args.model_size, args.kingdom, species_dirs
    )
    num_hyperedges = max(len(data[1].e) for data in train_data.values())

    base_model = MHGNano(
        in_channels=args.in_channels,
        hid_channels=args.hid_channels,
        num_classes=args.num_classes,
        num_hyperedges=num_hyperedges,
        use_bn=False,
        drop_rate=0.0,
    ).to(device)
    learner = MetaLearner(
        base_model=base_model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        pairwise_lambda=args.pairwise_lambda,
        pairwise_margin=args.pairwise_margin,
        pairwise_max_pairs=args.pairwise_max_pairs,
    )

    task_options = {
        "seed": args.seed,
        "negative_sampling": args.neg_sampling,
        "negative_ratio": args.neg_ratio,
    }
    train_tasks = build_species_tasks(train_data, train_species, "train", **task_options)
    validation_tasks = build_species_tasks(
        validation_data, [validation_species], "val", **task_options
    )
    checkpointer = Checkpointer(args.out_dir, save_every=args.save_every)
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    print("\n===== Meta-Training =====")
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training epochs"):
        train_batch = sample_meta_batch(train_tasks, args.meta_batch_size)
        train_metrics = learner.meta_train_step(train_batch, device)
        if epoch % 2:
            print(
                f"[Epoch {epoch:03d}] TRAIN loss={train_metrics['meta_loss']:.4f}, "
                f"acc={train_metrics['meta_acc']:.4f}, aucpr={train_metrics['meta_aucpr']:.4f}"
            )
            continue

        validation_metrics = learner.meta_evaluate(
            validation_tasks,
            device,
            threshold=None if args.use_val_threshold else 0.5,
            select_threshold=args.use_val_threshold,
        )
        print(
            f"[Epoch {epoch:03d}] TRAIN loss={train_metrics['meta_loss']:.4f}, "
            f"aucpr={train_metrics['meta_aucpr']:.4f} | "
            f"VAL loss={validation_metrics['val_loss']:.4f}, "
            f"aucpr={validation_metrics['val_aucpr']:.4f}, "
            f"auroc={validation_metrics['val_auroc']:.4f}, "
            f"threshold={validation_metrics['threshold']:.4f}"
        )
        metrics = {**train_metrics, **validation_metrics}
        checkpointer.save_last(epoch, learner, metrics)
        checkpointer.save_periodic(epoch, learner, metrics)
        if early_stopping(validation_metrics["val_aucpr"]):
            checkpointer.save_if_best(epoch, learner, metrics, monitor="val_aucpr")
        if early_stopping.early_stop:
            print(f"Early stopping; best validation AUPRC={early_stopping.best:.6f}")
            break

    if not checkpointer.best_path_pt.is_file():
        raise RuntimeError("Training ended without a best checkpoint")
    checkpoint = torch.load(checkpointer.best_path_pt, map_location=device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    test_data = load_species_group(
        test_species, args.data_root, args.model_size, args.kingdom, species_dirs
    )
    test_tasks = build_species_tasks(test_data, test_species, "test", **task_options)
    learner = MetaLearner(
        base_model=base_model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        pairwise_lambda=args.pairwise_lambda,
        pairwise_margin=args.pairwise_margin,
        pairwise_max_pairs=args.pairwise_max_pairs,
    )
    eval_batch = sample_meta_batch(test_tasks, args.meta_batch_size)
    threshold = checkpoint.get("metrics", {}).get("threshold", 0.5)
    if not args.use_val_threshold:
        threshold = 0.5
    metrics = learner.meta_evaluate(
        eval_batch, device, threshold=float(threshold), select_threshold=False
    )
    print(metrics)
    write_maml_eval_statistics(
        metrics,
        args,
        script_name="train_meta",
        extra={
            "train_species": train_species,
            "val_species": validation_species,
            "unseen_species": test_species,
        },
    )
