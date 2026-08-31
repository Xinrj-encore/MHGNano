# MHGNano

MHGNano is a hypergraph meta-learning implementation for cross-species
protein-protein interaction (PPI) prediction and few-shot transfer to
nanobody-antigen interaction (NAI) prediction.

This revision accompanies the August 2026 reviewer-response package. It
includes the corrected train/validation/test protocol, a fixed-cutoff
evaluation implementation, and the compact seed-level evidence needed to
inspect the added analyses.

## What Is Implemented

- Frozen ESM-2 protein features.
- Maximal-clique-derived hyperedges from positive PPI graphs.
- A two-layer HGNN encoder and pairwise classifier.
- MAML-style classifier adaptation from a meta-learned initialization.
- Held-out evaluation with target `train` support and target `test` query.
- Fixed `0.5` probability cutoff for Accuracy, Precision, Recall, and F1.
  AUROC and AUPRC use continuous prediction probabilities.

## Installation

```bash
git clone https://github.com/Xinrj-encore/MHGNano.git
cd MHGNano
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Choose a PyTorch build appropriate for the local CUDA driver before installing
the remaining requirements when GPU execution is required.

## Processed Data Layout

Raw STRING, sequence, and NAI data are not redistributed. The training scripts
expect precomputed artifacts in this layout:

```text
data/
└── <collection>/
    └── <species_id>/
        ├── PPI_cliques_Hyperedge_<species_id>_<model_size>.pkl
        ├── protein_feature_CT_<species_id>_<model_size>.pt
        ├── train_idx_<species_id>.pkl
        ├── val_idx_<species_id>.pkl
        └── test_idx_<species_id>.pkl
```

Each index file is a list of folds. A fold stores `[pairs, labels]`, where a
pair is `[protein_i, protein_j]`. `--fold 0` selects the first fold.

For the manuscript preprocessing, sequences were restricted to 51--800 amino
acids; STRING confidence greater than 500 defined positive PPI edges; and
candidate negatives were restricted to the positive-induced protein universe.
Maximal cliques were enumerated from the undirected positive PPI graph with
NetworkX 3.6.1.

## Train

The final species passed to `--species` is held out. During meta-training,
source-species training pairs are support and source-species validation pairs
are query. Test pairs are never appended to validation pairs.

```bash
python src/train.py \
  --data_root ./data/yuanhe \
  --species 1326 41514 77635 208964 \
  --model_size 8M \
  --fold 0 \
  --seed 42 \
  --out_dir ./output/yuanhe_seed42
```

The default model size is `8M`; `35M`, `150M`, and `650M` are also accepted
when matching feature and hypergraph artifacts are available. The input width
is inferred from the feature tensor, so it is not fixed to the 320 dimensions
used by ESM-2 8M.

## Evaluate

Few-shot evaluation adapts only the classifier on the held-out species training
split, then evaluates once on its test split:

```bash
python src/evaluate.py \
  --checkpoint ./output/yuanhe_seed42/best_model.pt \
  --data_root ./data/yuanhe \
  --species 208964 \
  --fold 0 \
  --threshold 0.5 \
  --output ./output/yuanhe_seed42/test_metrics.json
```

Use `--no_adapt` for an explicit no-target-adaptation evaluation. The command
always reports Accuracy, Precision, Recall, F1, AUROC, and AUPRC.

For one checkpoint per seed, preserve seed-level results rather than reporting
only an aggregate:

```bash
python src/evaluate_with_stats.py \
  --checkpoint_template './output/yuanhe_seed{seed}/best_model.pt' \
  --data_root ./data/yuanhe \
  --species 208964 \
  --seeds 42 123 456 789 2023 \
  --threshold 0.5 \
  --output ./results/yuanhe_seed_level.json
```

## Reviewer-Relevant Evidence

The `reproduction/` directory contains compact, versioned evidence used by the
revision:

- `protocol_audit/`: cutoff protocol, split-audit method, and documented
  historical versus strict pair-disjoint controls.
- `class_imbalance/`: five-seed 1:1, 1:3, 1:5, and reference 1:9 results.
- `baseline_comparison/`: 30-seed direct TRGH-PPI/HIGH-PPI results and the
  independent-comparison statistics against MHGNano-few.
- `esm2_scale/`: 30 matched-seed ESM-2 scale results and corrected tests.

Read [REPRODUCTION.md](REPRODUCTION.md) before interpreting a comparison.
It records which analyses use matched paired seeds and which use independent
tests, and it distinguishes the historical split files from the strict
pair-disjoint control.

The GO-enrichment and same-size random-control workflow used to assess the
biological coherence of clique-derived hyperedges is implemented in
`scripts/validate_cliques_biological.py`. It accepts a clique pickle, protein
identifier mapping, and GO annotation file; those source databases are not
redistributed.

## Scope

The numerical evidence is specific to the listed datasets, splits, feature
artifacts, and evaluation protocols. It is not a claim of universal superiority
over all PPI methods or all biological settings.

## License

This repository is released under the MIT License. Please cite the MHGNano
manuscript and the original data and baseline sources when using this code.
