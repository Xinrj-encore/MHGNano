# Pair and Split Audit

Pair identity is canonicalized as `(min(protein_u, protein_v),
max(protein_u, protein_v))`; labels are retained during comparisons.

## Historical Inputs

The historical fold-0 files contain same-label repeated pairs across split
boundaries. No cross-label conflicts were found.

| Dataset | Split | Raw records | Unique canonical pairs | Positive | Negative |
| --- | --- | ---: | ---: | ---: | ---: |
| zhenhe / elegans | train | 70,400 | 64,905 | 8,610 | 56,295 |
| zhenhe / elegans | validation | 17,600 | 17,257 | 2,869 | 14,388 |
| zhenhe / elegans | test | 22,000 | 21,486 | 3,529 | 17,957 |
| yuanhe / 208964 | train | 70,400 | 62,513 | 7,727 | 54,786 |
| yuanhe / 208964 | validation | 17,600 | 17,037 | 2,874 | 14,163 |
| yuanhe / 208964 | test | 22,000 | 21,080 | 3,456 | 17,624 |

| Dataset | Train/validation overlap | Train/test overlap | Validation/test overlap |
| --- | ---: | ---: | ---: |
| zhenhe / elegans | 2,583 | 3,274 | 862 |
| yuanhe / 208964 | 3,560 | 4,483 | 1,255 |

## Strict Pair-Disjoint 1:9 Control

The later control removes canonical pair overlap across train, validation, and
test while allowing negative pairs to reuse proteins that also appear in
positive pairs.

| Dataset | Split | Raw records | Unique canonical pairs | Positive | Negative |
| --- | --- | ---: | ---: | ---: | ---: |
| zhenhe / elegans | train | 31,760 | 15,880 | 1,588 | 14,292 |
| zhenhe / elegans | validation | 7,940 | 3,970 | 397 | 3,573 |
| zhenhe / elegans | test | 9,940 | 4,970 | 497 | 4,473 |
| yuanhe / 208964 | train | 31,760 | 15,880 | 1,588 | 14,292 |
| yuanhe / 208964 | validation | 7,940 | 3,970 | 397 | 3,573 |
| yuanhe / 208964 | test | 9,940 | 4,970 | 497 | 4,473 |

All three cross-split overlap counts are zero for both datasets.

## Re-run

```bash
python reproduction/protocol_audit/hash_pair_splits.py \
  --data-root ./data/zhenhe \
  --species elegans \
  --fold 0 \
  --out pair_audit.csv
```
