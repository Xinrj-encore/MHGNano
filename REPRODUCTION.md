# Reproduction Protocol

This file describes the data, model, seed, and statistical procedures associated with the revision. The prepared model inputs, historical splits, and sensitivity/control split collections are distributed under `data/`; the deterministic split generator creates a separate new dataset.

## Inputs and Data Processing

For each PPI species, the manuscript pipeline used STRING links and protein sequences, retained sequences of 51--800 amino acids, treated combined scores above 500 as positive and lower/equal scores as candidate negatives, capped positives at 10,000 and candidate negatives at 90,000, and required both negative-pair proteins to occur in the positive-pair protein universe. The prepared positive/negative pair tables, protein maps, clique files, features, and five-fold source split files are included in `data/zhenhe/` and `data/yuanhe/`. These supplied split files are unchanged. NAI pair/sequence inputs and prepared splits are in `data/nano/`; the retained split counts are also reported in the manuscript's NAI sample-count table.

`scripts/prepare_ppi_dataset.py` makes these filtering rules executable and records checksums and counts. It canonicalizes undirected pairs and creates deterministic stratified five-fold splits with a validation subset drawn from each fold's trainval portion. These newly generated splits are pair-disjoint by construction. They are a separate option and must not replace the supplied historical split files when reproducing the manuscript results. See `reproduction/protocol_audit/PAIR_SPLIT_AUDIT.md` for the historical split audit.

`scripts/extract_esm2_features.py` extracts frozen ESM-2 features in `Protein_list.csv` order. It uses `facebook/esm2_t6_8M_UR50D`, excludes special tokens, truncates at 800 residues, and mean-pools non-padding residue embeddings.

## Species Roles

| Domain | Meta-training species | Validation species | Held-out test species |
| --- | --- | --- | --- |
| Eukaryotic | *S. cerevisiae*, *D. melanogaster*, *M. musculus* | *D. rerio* | *C. elegans*, *H. sapiens* |
| Prokaryotic | Streptococcus, *S. enterica*, Bifidobacterium | Actinobacteria | *P. aeruginosa* |

Cross-domain experiments meta-train on the source-domain training species, select settings on the source-domain validation species, and evaluate on a held-out target species of the other domain. In released training code, source `train` pairs are support and source `val` pairs are query; held-out `train` is the adaptation support and held-out `test` is the final query. No test labels are used for model selection.

## Model and Ablation Configuration

| Setting | Main value |
| --- | --- |
| Frozen sequence encoder | `facebook/esm2_t6_8M_UR50D` |
| Protein feature pooling | Mean over residue tokens; no special tokens |
| Feature / HGNN hidden / output dimensions | 320 / 256 / 128 |
| Hypergraph encoder | 2 HGNN layers |
| Pair classifier | 512 -> 1024 -> 256 -> 1, ReLU between linear layers |
| Inner update | 5 gradient steps, learning rate `1e-3` |
| Outer optimizer | Adam, meta-learning rate `1e-4` |
| Training loss | Unweighted binary cross entropy with logits |
| Meta-batch / support-query pairs | 3 tasks / 64 support and 64 query pairs |
| Maximum epochs / early stopping | 20,000 / validation AUPRC, patience 5 |
| Threshold-dependent metrics | Fixed probability cutoff 0.5 |
| Hyperedge weighting | Uniform `e_weight=1`; vertex weight `v_weight=1` under DHG 0.9.5 defaults |

The ablation variants use the same encoder, classifier, optimizer, loss, split, stopping rule, and seed schedule as MHGNano-few; each changes only the named component:

| Variant | Change |
| --- | --- |
| MHGNano-few | Full hypergraph encoder, cross-node pair aggregation, and MAML |
| MHGNano-Max / MHGNano-Min | Replace mean protein pooling with element-wise max / min pooling |
| MHGNano w/o hg | Remove hypergraph propagation while retaining pair aggregation and meta-learning |
| MHGNano w/o agg | Remove cross-node pair aggregation and its difference/product features |
| MHGNano w/o maml | Remove the meta-learning objective; retain the base learner |

The published component ablations are reported over 30 seeds in both eukaryote-to-eukaryote and eukaryote-to-prokaryote settings. The full 30-seed list is below. The independent HIGH-PPI-MAML and TRGH-PPI-MAML baselines use five seeds and five target-support updates; their full target training split is used as adaptation support. This is a larger support budget than the 64-pair MHGNano few-shot setting, so the results should be interpreted as an additional MAML baseline comparison, not as a support-size-matched experiment.

## Random Seeds

Main 30-seed schedule (used for the manuscript main comparisons and 30-run component/scale analyses):

```text
1716116, 2336005, 2518497, 2562311, 3089950, 3280406, 3386856,
3574914, 3737690, 3757171, 3853859, 4057503, 4313174, 4556313,
4693148, 4882726, 4916639, 5019582, 5137033, 5231190, 6668044,
7160467, 7347752, 7458325, 7662046, 8238346, 8437357, 8654442,
9104266, 9725948
```

Five-seed sensitivity, class-ratio robustness, and MAML-baseline schedule:

```text
42, 123, 456, 789, 2023
```

The direct HIGH-PPI/TRGH-PPI reproduction used seeds `0--29`; those seeds are independent of the MHGNano main schedule. Keep seed-level outputs in CSV/JSON and do not average predictions before calculating the reported metrics.

## Loss, Cutoff, and Metrics

The primary PPI experiments use unweighted binary cross entropy and do not apply class weights or training-time negative resampling. The sampled class ratio is approximately 1:9 up to per-species availability/caps. Accuracy, Precision, Recall, and F1 use a fixed 0.5 probability threshold (`p >= 0.5`). AUROC and AUPRC use continuous probabilities. Query/test labels do not select the cutoff.

## Statistical Testing

- A matched pair requires the same split, target setting, and seed. Use two-sided paired Wilcoxon signed-rank tests on seed-level metric values.
- ESM-2 scale comparisons use a Friedman omnibus test per metric, then paired Wilcoxon post-hoc tests when the omnibus result is significant; BH-FDR is applied within each metric's post-hoc family.
- Ablation tables have six variants. For each transfer setting and metric, the 15 pairwise tests are BH-FDR corrected as one family. Asterisks in the full-model row identify comparisons against the three component-removal variants.
- Direct HIGH-PPI/TRGH-PPI runs use seeds 0--29, independent from the 30-seed MHGNano schedule. Use two-sided Mann--Whitney U tests and the reported BH-FDR correction; do not describe these as paired tests.
- HIGH-PPI-MAML/TRGH-PPI-MAML comparisons use five seeds and independent summary-level tests because their run schedule differs from the main 30-seed schedule.

For Figure 2, the eight MAML-by-transfer-direction comparisons form a separate BH-FDR family. Overall PPI is an aggregate of the four directional panels and is not included in this MAML testing family. The non-MAML baseline tests are corrected jointly across the six panels.

## Baseline Inclusion

An external baseline is included for a benchmark when a prediction result/checkpoint is available and its outputs can be evaluated on the target split under the common metric protocol. A missing result for a benchmark is the only omission criterion; methods are not removed based on observed performance. The direct baseline and MAML-baseline groups are kept separate because their seed schedules and adaptation protocols differ.

## Evidence and Re-run Limits

The `reproduction/` directory contains seed-level result tables, corrected summary statistics, and protocol audits. The `data/` directory contains the processed historical PPI and NAI model inputs, class-ratio splits, and pair-disjoint/randomized-hyperedge controls. Full source STRING link tables and PPI FASTA downloads are not included. The new PPI preparation workflow creates a separate deterministic pair-disjoint dataset from locally obtained STRING inputs. See `data/README.md` for each dataset collection and its role.

## Hyperedge-Size Distribution

`reproduction/hyperedge_size_distribution/` contains the species-level counts and appendix table. To recompute the CSV from the prepared historical clique pickles, run from the repository root:

```bash
python scripts/audit_hyperedge_sizes.py \
  --data-root /path/to/MHGNano/meta/data \
  --output reproduction/hyperedge_size_distribution/hyperedge_size_by_species.csv
```

The required clique pickle files are listed in `scripts/audit_hyperedge_sizes.py` and are distributed under `data/zhenhe/`. The archived CSV uses relative source paths and records the hyperedge-size counts used for the appendix table.

## Clique Biological-Coherence Analysis

`reproduction/clique_biological_validation/` contains the multi-species summary and representative enriched-clique examples. The analysis evaluates up to 1,000 maximal cliques of size at least three per species against 100 same-size random controls, using the GO annotations and statistical procedure in `scripts/validate_cliques_biological.py`. The source clique pickles, protein maps, STRING data, GO annotations, and ontology files are not redistributed. See that script's command-line help for rerunning the analysis with locally obtained inputs.
