# Data preparation and training

The same command-line workflow accepts either eukaryotic or prokaryotic STRING inputs. Prepared datasets are stored under data/eukaryotes/<species> or data/prokaryotes/<species>; the species identifier is used only in artifact filenames.

## Prepare a species dataset

    python prepare_dataset.py \
      --links data/cerevisiae.tsv \
      --fasta data/cerevisiae.fa \
      --species cerevisiae \
      --kingdom eukaryotes \
      --data-root data/generated

The command creates positive and negative pairs, maximal-clique hyperedges, deterministic train/validation/test folds, and a preprocessing manifest under data/generated/eukaryotes/cerevisiae. Existing output directories are never overwritten.

## Extract ESM-2 features

    python scripts/extract_esm2_features.py \
      --fasta data/cerevisiae.fa \
      --data-root data/generated \
      --kingdom eukaryotes \
      --species cerevisiae \
      --model-name facebook/esm2_t6_8M_UR50D \
      --model-tag 8M

## Train across species

The following example uses the prepared datasets already stored under `data/eukaryotes/<species>`.

    python train_meta.py \
      --data-root data \
      --train-species cerevisiae elegans melanogaster \
      --validation-species mouse \
      --test-species rat \
      --kingdom eukaryotes \
      --model-size 8M \
      --seed 42 \
      --out-dir output/run_42

The `--train-species`, `--validation-species`, and `--test-species` values must match the species directory names and artifact filename prefixes. When training on newly prepared files under `data/generated`, prepare and extract features for every species in the run, then use those same labels with `--data-root data/generated`. The training, validation, and test species must be disjoint. For a prokaryotic experiment, set `--kingdom prokaryotes`. For mixed-domain experiments, pass one `--species-dir SPECIES=PATH` for each species that is outside the common kingdom directory.

## Evaluate a source checkpoint on a target species

The checkpoint evaluator reports zero-shot metrics before target adaptation and few-shot metrics after adapting on a sampled support set from the target training split. It evaluates both stages on the target test split.

    python evaluate_cross_domain_checkpoint.py \
      --data-root data \
      --kingdom eukaryotes \
      --species-dir 208964=data/prokaryotes/208964 \
      --train-species cerevisiae elegans melanogaster \
      --validation-species mouse \
      --test-species 208964 \
      --checkpoint-path output/run_42/best_mhgnano.pt \
      --support-size 64 \
      --seed 42 \
      --out-dir output/transfer_eval_42

`--support-size` counts target protein pairs in total and samples positives and negatives in proportion to the target training split. Use the same source species and model dimensions used to produce the checkpoint.
