# MHGNano

Source and reviewer-response reproduction materials for **MHGNano:
Meta-Learning for Nanobody-antigen Interaction Pattern Recognition Using
Hypergraph**.

The maintained implementation is in [`github_repo/`](github_repo/). Start
with its [README](github_repo/README.md) for installation, processed-data
layout, training, held-out evaluation, and the fixed `0.5` cutoff protocol.

The reviewer-facing evidence is indexed in
[`github_repo/REPRODUCTION.md`](github_repo/REPRODUCTION.md), including:

- cutoff and pair-split audits;
- class-imbalance robustness at 1:1, 1:3, 1:5, and reference 1:9;
- TRGH-PPI and HIGH-PPI reproduction results;
- matched-seed ESM-2 scale statistics.

Raw feature tensors, source biological databases, checkpoints, and training
logs are intentionally excluded from version control because of size and
source-data terms.
