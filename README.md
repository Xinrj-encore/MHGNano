# MHGNano

MHGNano predicts protein interactions with ESM-2 sequence representations, clique-derived hypergraphs, and MAML-based adaptation. This repository contains command-line workflows for preparing PPI data, extracting protein features, meta-training MHGNano, and evaluating a trained checkpoint on a target species or the NAI task.

## Repository layout

- `training/`: MHGNano, MAML training, evaluation, data loading, and checkpoint code.
- `scripts/prepare_ppi_dataset.py`: generic STRING PPI preprocessing and split generation.
- `scripts/prepare_nai_dataset.py`: deterministic nanobody-antigen pair mapping and network export.
- `scripts/extract_esm2_features.py`: frozen ESM-2 feature extraction.
- `train_meta.py`: cross-species meta-training entry point.
- `evaluate_cross_domain_checkpoint.py`: zero-shot and few-shot target evaluation entry point.
- `meta_maml_nano.py`: compatibility entry point for checkpoint-only NAI evaluation.
- `maml_statistics.py`: per-run evaluation CSV output and an optional statistical-report hook.
- `data/`: example input and prepared data files. The preprocessing commands can also use data stored elsewhere.
- `WORKFLOW.md`: a compact command reference for the main workflow.

## Setup

Use Python 3.10 or newer. Install the listed dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU use, install a PyTorch build compatible with the installed CUDA version. ESM-2 weights are downloaded from Hugging Face the first time feature extraction runs. `requirements.txt` lists package names without pinned versions, so record the package and CUDA versions used for a particular result.

## Data preparation

### 1. Prepare a species dataset

Provide a STRING links file and its matching protein FASTA. FASTA identifiers use the first header token by default, and sequences outside 51–800 residues are excluded. The default positive rule is `combined_score > 500`; candidate operational negatives have scores at or below 500 and are restricted to proteins present in the selected positive pairs. Up to 10,000 positive pairs are taken in input-file order, followed by up to nine negatives per positive in input-file order. These negatives are sampling labels, not experimentally confirmed non-interactions. The actual counts and ratio are written to `preprocessing_manifest.json`.

```bash
python prepare_dataset.py \
  --links /path/to/9606.protein.links.txt \
  --fasta /path/to/9606.protein.sequences.fa \
  --species human \
  --kingdom eukaryotes \
  --data-root data/generated
```

By default, the script creates five deterministic, pair-stratified folds using split seed 42. Within each fold, validation pairs are sampled from the training/validation portion. The output directory is not overwritten if it already exists. Maximal cliques are built from the positive interaction graph and saved as hyperedges with uniform weights.

The generated directory contains `Protein_list.csv` and `Protein_list_<species>.pkl`, `PPI.csv`, positive and negative network TSV files, `PPI_cliques_Hyperedge_<species>_8M.pkl`, train/validation/test split pickles, and a preprocessing manifest.

For the full preparation-to-training workflow, each species directory needs `Protein_list.csv` (used by feature extraction), `PPI_cliques_Hyperedge_<species>_<model-size>.pkl` (the loader falls back to the shared 8M hypergraph), `protein_feature_CT_<species>_<model-size>.pt`, and `train_idx_<species>.pkl`, `val_idx_<species>.pkl`, and `test_idx_<species>.pkl`. 

### 2. Extract frozen ESM-2 features

The default and example configuration uses the Hugging Face checkpoint `facebook/esm2_t6_8M_UR50D` (ESM-2, 8M variant; 320-dimensional embeddings). The encoder is frozen: the script mean-pools its token representations for each protein and truncates sequences at 800 tokens. Protein IDs are read in the order in `Protein_list.csv`. The feature tensor is saved in the species directory and existing tensors are not overwritten.

The released CLI supports these ESM-2 sizes:

| Size | Hugging Face checkpoint (`--model-name`) | Feature width (`--in-channels`) |
| --- | --- | ---: |
| 8M | `facebook/esm2_t6_8M_UR50D` | 320 |
| 35M | `facebook/esm2_t12_35M_UR50D` | 480 |
| 150M | `facebook/esm2_t30_150M_UR50D` | 640 |
| 650M | `facebook/esm2_t33_650M_UR50D` | 1280 |

```bash
python scripts/extract_esm2_features.py \
  --fasta /path/to/9606.protein.sequences.fa \
  --data-root data/generated \
  --kingdom eukaryotes \
  --species human \
  --model-name facebook/esm2_t6_8M_UR50D \
  --model-tag 8M
```

The resulting filename is `protein_feature_CT_<species>_<model-tag>.pt`; for the default checkpoint this is `protein_feature_CT_<species>_8M.pt`. Keep the checkpoint, `--model-tag`, `--model-size`, and `--in-channels` consistent across feature extraction, training, and evaluation. Do not relabel a feature tensor from one checkpoint as another model size.

### Prepare the NAI pair mapping

The pair file contains tab-separated nanobody ID, antigen ID, and binary label columns; a header is optional. FASTA records containing `nb` in the header are treated as nanobodies, and the remaining records as antigens. The CLI can also handle reversed pair order. It uses the bundled pair-label TSV and FASTA by default, writes to a new `data/nano/generated` directory, and refuses to overwrite an existing output directory:

```bash
python data/nano/process.py \
  --pairs data/nano/all_pair_data.pair.tsv \
  --fasta data/nano/all_pair_data.seqs.fasta \
  --output-dir data/nano/generated
```

Nanobody and antigen sequences are deduplicated separately; duplicate sequences use the lexicographically smallest original FASTA ID as the canonical ID. Negative pairs are retained only when both proteins occur in a positive pair. The output includes a stable `Protein_list.csv`, positive/negative TSV files, the legacy `Protein_list_nano.pkl`, and an input-hash/count manifest. This step does not generate train/validation/test splits or ESM-2 features. With the bundled inputs, this preprocessing produces 228 positive-participating proteins, while the checked-in legacy feature tensor has 101 rows. Regenerate all dependent features and splits before combining newly generated files with legacy artifacts.

## Training and evaluation

### Meta-train across species

Training, validation, and test species must be disjoint. The names must match the species directories and artifact prefixes. This example uses prepared data under `data/eukaryotes/`:

```bash
python train_meta.py \
  --data-root data \
  --kingdom eukaryotes \
  --train-species cerevisiae elegans melanogaster \
  --validation-species mouse \
  --test-species rat \
  --model-size 8M \
  --inner-lr 1e-3 \
  --inner-steps 5 \
  --meta-lr 1e-4 \
  --meta-batch-size 3 \
  --seed 42 \
  --out-dir output/eukaryote_run_42 \
  --no-stats-report
```

The default training limit is 20,000 epochs, with validation every two epochs and early stopping. The best checkpoint is written to `best_mhgnano.pt`; per-run evaluation metrics are appended to `maml_eval_metrics.csv` in the output directory. Run once per seed, using a distinct output directory for each checkpoint.

For a prokaryotic run, set `--kingdom prokaryotes`. For a mixed-domain run, use repeatable `--species-dir SPECIES=PATH` arguments for species outside the common kingdom directory. See `python train_meta.py --help` for model, loss, and sampling options.

`--seed` controls the training run; `--split-seed` controls preprocessing folds. They are separate settings. Each training command runs one seed and writes `best_mhgnano.pt` plus `maml_eval_metrics.csv` under its output directory.

### Evaluate a checkpoint on a target domain

The evaluator loads a saved checkpoint and reports metrics before and after inner-loop adaptation. The before-adaptation values are zero-shot results; the after-adaptation values use labeled pairs sampled from the target training split. Both are measured on the target test split. The support set contains up to 64 pairs by default, sampled in proportion to the target training split's class counts; the classification threshold is 0.5. The current inner loop updates classifier parameters only; the ESM-2 embeddings remain frozen and the hypergraph encoder stays at its checkpoint weights.

Example: evaluate an eukaryote-trained checkpoint on a prokaryotic species:

```bash
python evaluate_cross_domain_checkpoint.py \
  --data-root data \
  --species-dir 1326=data/prokaryotes/1326 \
  --train-species cerevisiae elegans melanogaster \
  --validation-species mouse \
  --test-species 1326 \
  --checkpoint-path output/eukaryote_run_42/best_mhgnano.pt \
  --support-size 64 \
  --seed 42 \
  --out-dir output/euk_to_prok_42 \
  --no-stats-report
```

For NAI evaluation, pass its data directory explicitly, for example `--species-dir nano=data/nano` and `--test-species nano`. The checkpoint, feature dimensions, and model configuration must match. `--train-species` and `--validation-species` identify the source setting for output metadata and the disjoint-species check; the evaluator loads the target data and the supplied checkpoint.

`meta_maml_nano.py` is a compatibility entry point for checkpoint evaluation; it does not train a model. For example, evaluate an eukaryote-trained checkpoint on the bundled NAI data:

```bash
python meta_maml_nano.py \
  --data-root data \
  --species-dir nano=data/nano \
  --train-species cerevisiae elegans melanogaster \
  --validation-species mouse \
  --test-species nano \
  --checkpoint-path output/eukaryote_run_42/best_mhgnano.pt \
  --support-size 64 \
  --seed 42 \
  --out-dir output/nai_eval_42 \
  --no-stats-report
```

The `before_eval` row in the output CSV records zero-shot metrics; `after_eval` records metrics after adapting on the sampled NAI support pairs. Set the source species to match the checkpoint's actual training setting. The CSV is written to `--results-csv` if supplied, otherwise `<out-dir>/maml_eval_metrics.csv`; `--no-stats-report` skips the optional report hook.
