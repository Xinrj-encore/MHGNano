# MHGNano

MHGNano is a hypergraph meta-learning framework for cross-species
protein-protein interaction (PPI) prediction and transfer to
nanobody-antigen interaction (NAI) prediction.

The maintained implementation is in [`github_repo/`](github_repo/). The
commands below are written for the repository root; enter `github_repo/`
before running Python scripts.

## Installation

The released environment uses Python 3.10 or newer, PyTorch, DHG, NumPy,
SciPy, pandas, scikit-learn, NetworkX, Transformers, and Biopython. A
conda-based installation is:

```bash
git clone https://github.com/Xinrj-encore/MHGNano.git
cd MHGNano/github_repo
conda env create -f environment.yml
conda activate hypergraph-meta-ppi
```

Alternatively, install the dependencies with pip after selecting a PyTorch
build compatible with the local CUDA driver:

```bash
cd MHGNano/github_repo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Preprocessing

Raw STRING networks, protein sequences, and the NAI source files are not
redistributed. The released training and evaluation code consumes the
preprocessed artifacts listed below. The preprocessing protocol used for the
reported experiments is:

1. Read the STRING protein-link file and the corresponding protein FASTA
   file for each species.
2. Keep protein sequences with length greater than 50 and no more than 800
   amino acids.
3. Define positive PPI edges using STRING `combined_score > 500`. Retain
   the selected positive edges and construct candidate negative edges from
   pairs whose two proteins both belong to the positive-induced protein
   universe. Negative proteins may therefore also occur in positive pairs;
   isolated proteins are not introduced as trivial negatives.
4. Map the retained protein identifiers to contiguous integer node IDs.
   Build an undirected graph from the positive PPI edges and enumerate its
   maximal cliques with NetworkX. Each clique is stored as one
   clique-derived hyperedge. Negative pairs are used for pairwise
   supervision and are not used to construct hyperedges.
5. Generate frozen protein features with
   `facebook/esm2_t6_8M_UR50D`. Mean-pool the residue representations to
   obtain one feature vector per protein. The `35M`, `150M`, and `650M`
   ESM-2 variants can be used when matching feature files are available.
6. Create deterministic stratified five-fold splits, with validation sampled
   from each fold's trainval portion. Canonical pairs are exclusive to one of
   `train`, `val`, or `test` within a fold. The source-task protocol uses
   `train` as support and `val` as query; held-out evaluation uses target
   `train` as support and target `test` as query.

Each species directory must contain:

```text
data/<collection>/<species>/
├── PPI_cliques_Hyperedge_<species>_<model_size>.pkl
├── protein_feature_CT_<species>_<model_size>.pt
├── train_idx_<species>.pkl
├── val_idx_<species>.pkl
└── test_idx_<species>.pkl
```

The clique pickle contains `num_vertices`, `PPI_edge_list`, and
`PPI_cliques_list`. Each split pickle is a list of folds; a fold is
`[pairs, labels]`, where each pair is `[protein_i, protein_j]`. NAI files
use the same pair/label convention after mapping nanobody and antigen
entities to integer IDs and retaining entities represented in positive
pairs.

The implementation used to inspect the clique construction is
[`scripts/validate_cliques_biological.py`](github_repo/scripts/validate_cliques_biological.py).
It accepts a clique pickle, a protein mapping, and a GO annotation file for
the optional biological-coherence analysis.

## Training

Run from `MHGNano/github_repo/`. The last species passed to `--species` is
held out; all preceding species are used as meta-training tasks.

```bash
python src/train.py \
  --data_root ./data/yuanhe \
  --species 1326 41514 77635 208964 \
  --model_size 8M \
  --fold 0 \
  --seed 42 \
  --out_dir ./output/yuanhe_seed42
```

The default training configuration uses frozen ESM-2 features, a two-layer
hypergraph encoder, five classifier adaptation steps, inner-loop learning
rate `1e-3`, meta-learning rate `1e-4`, meta-batch size `3`, and early
stopping based on validation AUPRC. The input feature width is inferred from
the loaded feature tensor.

## Held-Out Evaluation

Few-shot evaluation adapts the pair classifier on the held-out species'
training split and evaluates on its test split:

```bash
python src/evaluate.py \
  --checkpoint ./output/yuanhe_seed42/best_model.pt \
  --data_root ./data/yuanhe \
  --species 208964 \
  --model_size 8M \
  --fold 0 \
  --threshold 0.5 \
  --output ./output/yuanhe_seed42/test_metrics.json
```

For an explicit no-target-adaptation evaluation, add `--no_adapt`. For NAI
transfer, point `--data_root` to the prepared NAI target directory and use
the corresponding target species/entity identifier.

Accuracy, Precision, Recall, and F1 use the fixed probability cutoff `0.5`
with `p >= 0.5`. AUROC and AUPRC use continuous prediction probabilities.
Test/query labels are not used to select the cutoff.

To retain seed-level results for repeated evaluation:

```bash
python src/evaluate_with_stats.py \
  --checkpoint_template './output/yuanhe_seed{seed}/best_model.pt' \
  --data_root ./data/yuanhe \
  --species 208964 \
  --model_size 8M \
  --seeds 42 123 456 789 2023 \
  --threshold 0.5 \
  --output ./results/yuanhe_seed_level.json
```

## Reproduction Materials

Reviewer-facing protocols, archived seed-level results, and scripts are in
[`github_repo/reproduction/`](github_repo/reproduction/):

- `protocol_audit/`: cutoff and pair-split audits;
- `class_imbalance/`: 1:1, 1:3, 1:5, and reference 1:9 results;
- `baseline_comparison/`: 30-seed TRGH-PPI and HIGH-PPI comparison results;
- `esm2_scale/`: matched-seed ESM-2 scale results and corrected tests.
- `hyperedge_size_distribution/`: per-species clique-size counts and the appendix table;
- `clique_biological_validation/`: aggregate GO-enrichment and matched-size control results.

The detailed evidence map and rerun boundaries are documented in
[`github_repo/REPRODUCTION.md`](github_repo/REPRODUCTION.md). Generated
checkpoints and runtime logs are written to the user-selected `output/`,
`results/`, and `logs/` directories.

All reported threshold-dependent results follow the common `0.5` protocol.
The numerical conclusions are specific to the stated datasets, species
splits, feature artifacts, and evaluation settings.

## License

This project is released under the MIT License. Please cite the MHGNano
manuscript and the original data and baseline sources when using the code.
