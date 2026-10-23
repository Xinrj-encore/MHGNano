import math
import random

from .data_io import stratified_folds


def make_splits(
    pairs: list[list[int]],
    labels: list[int],
    n_splits: int,
    seed: int,
    validation_fraction: float,
):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    split_records = {name: [] for name in ("train", "val", "test")}
    for fold_index, (trainval, test) in enumerate(stratified_folds(labels, n_splits, seed)):
        train_indices = []
        val_indices = []
        for label in (0, 1):
            rows = [index for index in trainval if labels[index] == label]
            random.Random(f"{seed}:{fold_index}:{label}:validation").shuffle(rows)
            n_val = max(1, math.ceil(len(rows) * validation_fraction))
            if n_val >= len(rows):
                raise ValueError("Not enough examples per class for train and validation")
            val_indices.extend(rows[:n_val])
            train_indices.extend(rows[n_val:])

        for name, indices in (("train", train_indices), ("val", val_indices), ("test", test)):
            random.Random(f"{seed}:{fold_index}:{name}:order").shuffle(indices)
            split_records[name].append(
                [[pairs[index] for index in indices], [float(labels[index]) for index in indices]]
            )
    return split_records
