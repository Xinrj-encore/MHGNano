from collections import defaultdict
from pathlib import Path
from typing import Iterable
import hashlib
import random


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path, id_field: str = "first-token") -> dict[str, str]:
    if id_field not in {"first-token", "description"}:
        raise ValueError("id_field must be 'first-token' or 'description'")
    sequences = {}
    header = None
    chunks = []

    def save_record():
        if header is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        identifier = header.split(maxsplit=1)[0] if id_field == "first-token" else header
        if 51 <= len(sequence) <= 800:
            if identifier in sequences and sequences[identifier] != sequence:
                raise ValueError(f"Conflicting FASTA sequences for {identifier!r}")
            sequences[identifier] = sequence

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                save_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at {path}:{line_number}")
                chunks = []
            elif header is None:
                raise ValueError(f"Sequence data before first FASTA header at {path}:{line_number}")
            else:
                chunks.append(line)
    save_record()
    return sequences


def iter_string_links(path: Path) -> Iterable[tuple[str, str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"Expected 3 columns at {path}:{line_number}")
            try:
                score = int(fields[2])
            except ValueError:
                if line_number == 1 or fields[2].lower() in {"combined_score", "score"}:
                    continue
                raise ValueError(f"Non-integer confidence score at {path}:{line_number}")
            if fields[0] != fields[1]:
                yield fields[0], fields[1], score


def stratified_folds(labels: list[int], n_splits: int, seed: int):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_class = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    if set(by_class) != {0, 1} or any(len(rows) < n_splits for rows in by_class.values()):
        raise ValueError("Each class must contain at least n_splits records")

    test_folds = [[] for _ in range(n_splits)]
    for label, rows in sorted(by_class.items()):
        random.Random(f"{seed}:{label}:folds").shuffle(rows)
        for fold_index, row in enumerate(rows):
            test_folds[fold_index % n_splits].append(row)
    all_indices = set(range(len(labels)))
    return [
        (sorted(all_indices - set(indices)), sorted(indices))
        for indices in test_folds
    ]
