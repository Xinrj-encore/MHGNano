# ESM-2 Scale Analysis

`PR-D-26-01971_table_8_30seed_long.csv` and
`PR-D-26-01971_table_8_30seed_wide.csv` retain the 30 matched seed-level
results for the frozen 8M, 35M, 150M, and 650M encoders.

The statistical protocol is:

1. Friedman omnibus test across the four matched sizes for each metric.
2. Pre-specified paired Wilcoxon post-hoc tests.
3. Benjamini-Hochberg correction of the post-hoc p-values.

The included summary files show a non-monotonic result in this frozen-feature,
limited-label setting: 8M has the highest mean Precision and Accuracy, 35M has
the highest mean Recall, F1, AUROC, and AUPRC, and 650M is not the strongest
setting. This is an empirical, protocol-specific finding rather than a claim
that larger protein language models are generally inferior.
