# MHGNano

MHGNano is a hypergraph meta-learning framework for cross-species protein--protein interaction (PPI) prediction and few-shot transfer to nanobody--antigen interaction (NAI) prediction. This repository contains the model, data-preparation entry points, training/evaluation commands, and reviewer-requested reproducibility evidence.

The repository includes the prepared PPI and NAI datasets used by the
experiments in [`data/`](data/), including class-ratio sensitivity splits and
pair-disjoint and randomized-hyperedge controls. Raw full STRING downloads,
per-species PPI FASTA source files, and trained checkpoints are not included.
Binary feature tensors and split pickle files use Git LFS; install Git LFS
before cloning. The dataset inventory and source attribution are in
[`data/README.md`](data/README.md).

## Install

```bash
git clone https://github.com/Xinrj-encore/MHGNano.git
cd MHGNano
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run with Python 3.10 or newer. Choose a PyTorch build compatible with the local CUDA driver when using a GPU. The code was developed with DHG 0.9.5 and NetworkX 3.6.1.

## Prepare PPI Data

To regenerate a new PPI dataset, download the STRING protein links file and matching protein FASTA for each organism. The script applies the manuscript sequence-length filter (51--800 aa), treats scores `>500` as positive and scores `<=500` as candidate negatives, retains up to 10,000 positives and up to nine candidate negatives per positive, and restricts candidate negatives to proteins in the positive-pair universe. It canonicalizes undirected pairs, removes duplicate pairs, and records input checksums and split counts in a manifest. The exact prepared inputs and supplied historical split files used in the manuscript are already included under `data/zhenhe/` and `data/yuanhe/`; the generator below is for creating a separate new dataset.

```bash
python -m scripts.prepare_ppi_dataset \
  --links ./raw/9606.protein.links.v12.0.txt \
  --fasta ./raw/9606.protein.sequences.v12.0.fa \
  --species 9606 \
  --output-root ./data/yuanhe \
  --positive-cap 10000 \
  --negative-ratio 9 \
  --folds 5 \
  --validation-fraction 0.2 \
  --split-seed 42
```

This creates `data/yuanhe/9606/` containing the positive graph, maximal-clique hyperedges, five train/validation/test folds, protein ID mapping, and `preprocessing_manifest.json`. Output creation is exclusive: rerunning to an existing species directory stops instead of overwriting it.

Within each fold, the splitter keeps each canonical undirected pair in exactly one of train, validation, or test, and approximately stratifies labels. It uses a deterministic stratified five-fold assignment, then samples validation from the remaining training portion. This is a clean regeneration workflow, not a replacement for the fixed split artifacts used to produce the manuscript tables.

## Extract Frozen ESM-2 Features

Feature extraction reads `Protein_list.csv` and emits rows in its integer ID order. The default uses `facebook/esm2_t6_8M_UR50D`, no special tokens, truncation at 800 residues, and attention-mask mean pooling.

```bash
python -m scripts.extract_esm2_features \
  --fasta ./raw/9606.protein.sequences.v12.0.fa \
  --species-dir ./data/yuanhe/9606 \
  --species 9606 \
  --model-name facebook/esm2_t6_8M_UR50D \
  --model-tag 8M \
  --batch-size 4
```

The first run downloads the model from Hugging Face unless it is already cached. The feature file is written as `protein_feature_CT_9606_8M.pt`; existing outputs are not overwritten.

## Train and Evaluate

The last species listed is held out. Source `train` pairs are support and source `val` pairs are query during meta-training. Few-shot evaluation adapts the classifier on held-out `train` pairs and reports metrics on held-out `test` pairs. Use `--no_adapt` for zero-shot evaluation.

```bash
python src/train.py \
  --data_root ./data/yuanhe \
  --species 1326 41514 77635 208964 \
  --model_size 8M --fold 0 --seed 42 \
  --out_dir ./output/yuanhe_seed42

python src/evaluate.py \
  --checkpoint ./output/yuanhe_seed42/best_model.pt \
  --data_root ./data/yuanhe --species 208964 --fold 0 \
  --threshold 0.5 --output ./output/yuanhe_seed42/test_metrics.json
```

Main model defaults are listed in [`configs/yuanhe.yaml`](configs/yuanhe.yaml). `src/evaluate_with_stats.py` preserves seed-level results for existing checkpoints. The fixed threshold is 0.5 for Accuracy, Precision, Recall, and F1; AUROC and AUPRC use continuous scores.

## Hyperedge Weights

Each hyperedge is one maximal clique from the undirected positive PPI graph. The released model constructs `dhg.Hypergraph(num_vertices, cliques)` without custom edge- or vertex-weight arrays. Under DHG 0.9.5, omitted `e_weight` and `v_weight` default to unit weights (1.0); no learned, degree-based, or size-based reweighting is applied. The library's incidence-based HGNN normalization is then used by the convolution layer. See the [DHG Hypergraph API](https://deephypergraph.readthedocs.io/en/latest/api/dhg.html).

## Reproducibility Materials

- [`REPRODUCTION.md`](REPRODUCTION.md): split semantics, model settings, seed schedules, and statistical-testing families.
- [`reproduction/protocol_audit/`](reproduction/protocol_audit/): fixed-cutoff and pair-overlap audits.
- [`reproduction/class_imbalance/`](reproduction/class_imbalance/): five-seed 1:1, 1:3, 1:5, and reference 1:9 results.
- [`reproduction/baseline_comparison/`](reproduction/baseline_comparison/): direct HIGH-PPI/TRGH-PPI runs and comparisons.
- [`reproduction/esm2_scale/`](reproduction/esm2_scale/): 30-seed ESM-2 scale summaries and corrected tests.
- [`reproduction/hyperedge_size_distribution/`](reproduction/hyperedge_size_distribution/): per-species hyperedge counts and the appendix table.
- [`reproduction/clique_biological_validation/`](reproduction/clique_biological_validation/): multi-species validation summary and representative enriched cliques.
- `tests/test_ppi_preprocessing.py`: deterministic split, disjointness, and no-overwrite checks.

Run the preprocessing tests with `python -m unittest discover -s tests -p 'test_ppi_preprocessing.py'`.

The repository includes the processed model inputs, fixed split files,
sensitivity/control split indices, seed-level evidence, and analysis code.
Reproducing the published results does not require regenerating these inputs.
The full STRING link tables and PPI FASTA source files are omitted; they can
be obtained from STRING under its terms. The uploaded files retain the pair
labels and feature tensors used by the model. See [`data/README.md`](data/README.md)
for the collection-to-experiment mapping and notes on which split files share
the primary networks and features.

## License

Released under the MIT License. Please cite the MHGNano manuscript and original data and baseline sources when using this code.
