#!/usr/bin/env python3
"""Extract mean-pooled frozen ESM-2 protein features for a prepared species."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import torch

from scripts.prepare_ppi_dataset import read_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--species-dir", required=True, type=Path)
    parser.add_argument("--species", required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--model-tag", default="8M")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string")
    parser.add_argument("--fasta-id-field", choices=("first-token", "description"), default="first-token")
    args = parser.parse_args()

    species_dir = args.species_dir.resolve()
    mapping_path = species_dir / "Protein_list.csv"
    output_path = species_dir / f"protein_feature_CT_{args.species}_{args.model_tag}.pt"
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing feature tensor: {output_path}")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["ID"]))
    expected_ids = [row["Entry"] for row in rows]
    if [int(row["ID"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Protein_list.csv IDs must be contiguous and start at zero")
    sequence_by_id = read_fasta(args.fasta, args.fasta_id_field)
    missing = [protein for protein in expected_ids if protein not in sequence_by_id]
    if missing:
        raise ValueError(f"FASTA is missing {len(missing)} proteins, e.g. {missing[:5]}")
    sequences = [sequence_by_id[protein] for protein in expected_ids]

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to extract ESM-2 features") from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    feature_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch_sequences = sequences[start : start + args.batch_size]
            encoded = tokenizer(
                batch_sequences,
                add_special_tokens=False,
                truncation=True,
                max_length=800,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            feature_batches.append(pooled.cpu().float())
    features = torch.cat(feature_batches, dim=0)
    if features.shape[0] != len(expected_ids):
        raise AssertionError("Feature count does not match Protein_list.csv")

    with tempfile.NamedTemporaryFile(dir=species_dir, suffix=".pt", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(features, temp_path)
        # A hard link gives exclusive creation semantics and will fail if a
        # concurrent process has already created the requested artifact.
        os.link(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Saved {tuple(features.shape)} features to {output_path}")


if __name__ == "__main__":
    main()
