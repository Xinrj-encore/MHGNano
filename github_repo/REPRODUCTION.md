# Reviewer-Response Reproduction Guide

This directory records the compact evidence supporting the reviewer-requested
revisions. It deliberately excludes raw biological databases, protein feature
tensors, model checkpoints, and training logs.

## Fixed Evaluation Protocol

For all reported metrics, model logits are converted to probabilities with a
sigmoid. Accuracy, Precision, Recall, and F1 use a fixed threshold of `0.5`
with `p >= 0.5`. AUROC and AUPRC are calculated from continuous probabilities.
The test/query labels are not used to choose the threshold.

The source implementation is in `src/utils/metrics.py`; the reviewer-facing
evidence is in `reproduction/protocol_audit/`.

## Split Semantics

`src/data/dataset.py` loads one selected fold and preserves the three split
files independently. The release code uses:

- source species during meta-training: `train` as support and `val` as query;
- held-out species during few-shot evaluation: `train` as support and `test`
  as query;
- held-out species during no-adaptation evaluation: `test` as query only.

The historical input files were audited and are not pair-disjoint across split
boundaries. This fact is retained in the protocol audit. The added strict 1:9
control is pair-disjoint by canonical undirected protein pair; it does not
require disjoint protein-node universes.

## Statistical Reporting

Matched results are paired only when they share the same split, target setting,
and random seed. Matched ESM-2 scale runs use Friedman omnibus tests followed
by pre-specified paired Wilcoxon tests with Benjamini-Hochberg correction.

The direct TRGH-PPI/HIGH-PPI 30-seed runs use seed identifiers `0--29`, while
the MHGNano-few reference uses a different 30-seed schedule. These comparisons
are therefore independent two-sided Mann-Whitney U tests; Welch tests are
included as sensitivity checks. They are not labelled as paired.

## Evidence Map

| Reviewer topic | Evidence |
| --- | --- |
| Common cutoff and metric protocol | `reproduction/protocol_audit/CUTOFF_PROTOCOL.md` |
| Pair overlap and strict pair-disjoint control | `reproduction/protocol_audit/PAIR_SPLIT_AUDIT.md` |
| Class-ratio robustness | `reproduction/class_imbalance/README.md` |
| TRGH-PPI and HIGH-PPI comparison | `reproduction/baseline_comparison/README.md` |
| ESM-2 scale analysis | `reproduction/esm2_scale/README.md` |
| Clique biological-coherence validation | `scripts/validate_cliques_biological.py` |

## Re-run Boundaries

The tabulated CSV files are sufficient to reproduce the reported summaries and
statistical tests. Rerunning model training additionally requires the
processed feature tensors and split files described in `README.md`; those
artifacts are not redistributed here because of size and source-data terms.
