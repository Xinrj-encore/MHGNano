# TRGH-PPI and HIGH-PPI Reproduction

This directory records the reviewer-requested 30-seed direct, no-target-
parameter-adaptation results for TRGH-PPI and HIGH-PPI. Each method uses a
`train_positive` message-passing graph and the common fixed cutoff `0.5`.

`trgh_high_30seed_direct_raw.csv` retains the metrics, seed, target species,
threshold, graph scope, and checkpoint epoch but removes machine-specific
checkpoint paths. `trgh_high_30seed_direct_trainpositive_mean_sd.csv` provides
the corresponding mean and SD.

The comparisons with MHGNano-few in
`trgh_high_direct_vs_mhgnano_few_independent_tests.csv` are **independent**:
the two 30-seed schedules are not matched. The primary test is two-sided
Mann-Whitney U, and Welch's t-test is included as a sensitivity check. The
Benjamini-Hochberg correction is applied separately for each family of tests.

For prokaryote-to-eukaryote transfer, TRGH-PPI uses *Mus musculus* while
HIGH-PPI uses *C. elegans*; no paired TRGH-versus-HIGH conclusion is implied
for that setting.
