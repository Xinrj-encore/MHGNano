# Cutoff and Evaluation Protocol

This document records the protocol used for the reviewer-requested cutoff
audit, generated from the August 2026 reproduction archive.

1. Model logits are converted to probabilities with the sigmoid function.
2. Accuracy, Precision, Recall, and F1 use the same fixed cutoff, `0.5`, with
   the decision rule `p >= 0.5`.
3. AUROC and AUPRC use continuous probabilities, not thresholded labels.
4. Held-out test/query labels are not used to select a cutoff.
5. For few-shot MHGNano evaluation, the target training split is support and
   the target test split is query. The inner loop adapts classifier parameters;
   the encoder is not updated in the inner loop.

## Evidence Status

| Scope | Evidence retained here | Status |
| --- | --- | --- |
| MHGNano | Release metric implementation and archived seed-level result summaries | Verified |
| TRGH-PPI and HIGH-PPI direct reproduction | 30-seed raw/summary CSVs with `threshold=0.5` | Verified |
| D-SCRIPT | Archived audit summary: default cutoff `0.5`, continuous AUROC/AUPR; one exact 0.5 score noted | Verified with boundary note |
| Other legacy baselines | Original summary evidence only | Protocol recorded; source-level artifacts are not all redistributed |

The one exact D-SCRIPT score equal to `0.5` is recorded because strict `>` and
non-strict `>=` conventions differ only at that boundary. It does not explain
the overall Recall differences reported across transfer settings.

The release implementation of this protocol is
[`src/utils/metrics.py`](../../src/utils/metrics.py). `hash_pair_splits.py`
recomputes split hashes and pair-overlap counts from locally available processed
data; it requires only the standard library and an environment capable of
unpickling NumPy objects.
