#!/usr/bin/env python3
"""Build deterministic nanobody-antigen pair files from FASTA and labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sequences(fasta_path: Path, nanobody_marker: str):
    records: dict[str, tuple[str, str]] = {}
    sequences = {"nb": set(), "at": set()}
    identifiers = {"nb": defaultdict(set), "at": defaultdict(set)}
    marker = nanobody_marker.casefold()

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        description = record.description.strip()
        sequence = str(record.seq).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"Empty sequence for FASTA record {description!r}")
        kind = "nb" if marker in description.casefold() else "at"
        sequences[kind].add(sequence)
        identifiers[kind][sequence].add(record.id.strip())

        for alias in {record.id.strip(), description}:
            previous = records.get(alias)
            current = (sequence, kind)
            if previous is not None and previous != current:
                raise ValueError(f"Conflicting FASTA records share identifier {alias!r}")
            records[alias] = current

    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    if not sequences["nb"] or not sequences["at"]:
        raise ValueError(
            f"FASTA must contain both nanobody and antigen records (marker={nanobody_marker!r})"
        )

    sequence_ids = {
        kind: {sequence: min(identifiers[kind][sequence]) for sequence in sequences[kind]}
        for kind in ("nb", "at")
    }
    id_kinds = {
        identifier: kind
        for kind in ("nb", "at")
        for identifier in sequence_ids[kind].values()
    }
    return records, sequence_ids, id_kinds


def read_pairs(pair_path: Path, records, sequence_ids):
    mapped_pairs: list[tuple[str, str, int]] = []
    with pair_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 3:
                raise ValueError(f"Expected two protein IDs and a label at {pair_path}:{line_number}")
            first, second, raw_label = (field.strip() for field in row[:3])
            if raw_label.casefold() in {"label", "class", "interaction"}:
                continue
            try:
                label_value = float(raw_label)
            except ValueError as exc:
                raise ValueError(f"Invalid label at {pair_path}:{line_number}: {raw_label!r}") from exc
            if label_value not in {0.0, 1.0}:
                raise ValueError(f"Expected binary label 0 or 1 at {pair_path}:{line_number}")
            if first not in records or second not in records:
                missing = first if first not in records else second
                raise ValueError(f"FASTA has no sequence for {missing!r} ({pair_path}:{line_number})")

            first_sequence, first_kind = records[first]
            second_sequence, second_kind = records[second]
            if (first_kind, second_kind) == ("at", "nb"):
                first_sequence, second_sequence = second_sequence, first_sequence
                first_kind, second_kind = second_kind, first_kind
            if (first_kind, second_kind) != ("nb", "at"):
                raise ValueError(
                    f"Expected one nanobody and one antigen at {pair_path}:{line_number}"
                )
            mapped_pairs.append(
                (
                    sequence_ids["nb"][first_sequence],
                    sequence_ids["at"][second_sequence],
                    int(label_value),
                )
            )

    if not mapped_pairs:
        raise ValueError(f"No labeled pairs found in {pair_path}")
    if not any(label == 1 for _, _, label in mapped_pairs):
        raise ValueError("The pair file contains no positive interactions")
    return mapped_pairs


def write_delimited(path: Path, header, rows, delimiter=",") -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def prepare_nai_dataset(
    pair_path: Path,
    fasta_path: Path,
    output_dir: Path,
    nanobody_marker: str = "nb",
) -> Path:
    pair_path = pair_path.resolve()
    fasta_path = fasta_path.resolve()
    output_dir = output_dir.resolve()
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(fasta_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    records, sequence_ids, id_kinds = read_sequences(fasta_path, nanobody_marker)
    pairs = read_pairs(pair_path, records, sequence_ids)
    positives = [(nb, at, label) for nb, at, label in pairs if label == 1]
    positive_proteins = {protein for nb, at, _ in positives for protein in (nb, at)}
    negatives = [
        (nb, at)
        for nb, at, label in pairs
        if label == 0 and nb in positive_proteins and at in positive_proteins
    ]
    protein_list = sorted(positive_proteins)
    protein_id = {protein: index for index, protein in enumerate(protein_list)}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        write_delimited(
            stage_dir / "Protein_list.csv",
            ("Entry", "ID"),
            ((protein, protein_id[protein]) for protein in protein_list),
        )
        write_delimited(
            stage_dir / "pos_network_nano.tsv",
            ("protein1", "protein2", "label"),
            positives,
            delimiter="\t",
        )
        write_delimited(
            stage_dir / "neg_network_nano.tsv",
            ("protein1", "protein2"),
            negatives,
            delimiter="\t",
        )
        with (stage_dir / "Protein_list_nano.pkl").open("xb") as handle:
            pickle.dump(protein_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        manifest = {
            "task": "nanobody-antigen interaction prediction",
            "inputs": {
                "pairs_file": pair_path.name,
                "pairs_sha256": sha256_file(pair_path),
                "fasta_file": fasta_path.name,
                "fasta_sha256": sha256_file(fasta_path),
            },
            "mapping": {
                "nanobody_marker": nanobody_marker,
                "sequence_deduplication": "uppercase sequence; identical sequences use the lexicographically smallest original FASTA ID within entity type",
                "pair_orientation": "nanobody first, antigen second",
                "protein_list_scope": "proteins appearing in at least one positive pair",
            },
            "counts": {
                "input_pairs": len(pairs),
                "unique_nanobody_sequences": len(sequence_ids["nb"]),
                "unique_antigen_sequences": len(sequence_ids["at"]),
                "positive_pairs": len(positives),
                "input_negative_pairs": sum(label == 0 for _, _, label in pairs),
                "retained_negative_pairs": len(negatives),
                "positive_proteins": len(protein_list),
                "positive_nanobodies": sum(id_kinds[protein] == "nb" for protein in protein_list),
                "positive_antigens": sum(id_kinds[protein] == "at" for protein in protein_list),
                "negative_to_positive_ratio": len(negatives) / len(positives),
            },
            "negative_filter": "retain label-0 pairs only when both proteins occur in a positive pair",
        }
        with (stage_dir / "preprocessing_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.rename(stage_dir, output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(f"Wrote {len(positives)} positive and {len(negatives)} retained negative pairs to {output_dir}")
    return output_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = project_root / "data" / "nano"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=default_data_dir / "all_pair_data.pair.tsv")
    parser.add_argument("--fasta", type=Path, default=default_data_dir / "all_pair_data.seqs.fasta")
    parser.add_argument("--output-dir", type=Path, default=default_data_dir / "generated")
    parser.add_argument("--nanobody-marker", default="nb")
    args = parser.parse_args()
    prepare_nai_dataset(args.pairs, args.fasta, args.output_dir, args.nanobody_marker)


if __name__ == "__main__":
    main()
