# Class-Imbalance Robustness

This directory contains the reviewer-requested positive-to-negative ratio
analysis across four transfer settings and five seeds: `42, 123, 456, 789,
2023`.

| Ratio | Data protocol |
| --- | --- |
| 1:1 | Strict pair-disjoint control |
| 1:3 | Strict pair-disjoint control |
| 1:5 | Strict pair-disjoint control |
| 1:9 | Original-paper split used as the five-seed reference |

The 1:9 reference retains its historical-split provenance and is not silently
presented as pair-disjoint. The strict 1:9 pair-disjoint audit is documented in
`../protocol_audit/PAIR_SPLIT_AUDIT.md`.

`after_eval_raw_metrics.csv` contains the 80 after-evaluation rows. Accuracy,
Precision, Recall, and F1 use the fixed `0.5` cutoff; AUROC and AUPRC use
continuous probabilities. `after_eval_summary_mean_sd_wide.csv` is the
manuscript-facing summary, and `fig_class_imbalance_robustness.png` is the
corresponding figure.

Regenerate the long-format summary and optional figure:

```bash
python reproduction/class_imbalance/assemble_class_imbalance_robustness.py --plot
```
