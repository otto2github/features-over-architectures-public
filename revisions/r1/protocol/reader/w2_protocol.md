# W2: second, earlier evaluation window — frozen protocol v1

PeerJ CS-2026:06:141707, first revision (R1). Frozen on 2026-10-07, before any W2 fit.
The primary-window results, the 105-fit programme, the ten-fit 42-input extension and the 20-fit E7 sensitivity were
already known. This is therefore a result-aware amendment, not a preregistration. It is reported in full whatever
its results are.

## Purpose

The primary benchmark tests on 2021–2022 with 20 positives. W2 asks whether the matched 129-input comparison between
graph and feature-only learners looks similar on an earlier, independent set of test company-years with more
positives. It adds evaluated positives; it is not a new model search.

## Window and supervision (primary label, unchanged rule)

| Partition | Fiscal years | Eligible rows | Positives |
|---|---|---:|---:|
| Training | 2010–2016 | 17,448 | 158 |
| Validation | 2017–2018 | 6,658 | 96 |
| Test | 2019–2020 | 7,309 | 40 |

Counts come from the frozen label file (`label_v1_strict_ab_primary`, Supplement Table S1c) and are checked before
any fit. Rows outside the eligible set stay unlabeled graph context, exactly as in the primary analysis.

**Disclosure that must accompany every use of W2.** The 2019–2020 rows were the validation partition of the primary
analysis. Their labels and the primary models' validation scores on them were therefore seen before this protocol.
No W2 model has been fitted or evaluated on them before this protocol. W2 test and primary test are disjoint
company-years, but the same companies recur across years, so the two windows are not independent samples of
companies.

## Fixed matrix (20 fits)

Four learners × five seeds (42, 123, 456, 789, 1024), all on the original 129 M11 inputs (`M11_COLUMN_ORDER.json`):

* RF — unweighted random forest, the primary predecessor configuration (500 trees, Gini, sqrt features, bootstrap,
  no class weights), raw inputs with missing values set to zero.
* MLP — the original `formal_rerun_core.run_mlp` training loop.
* GCN, GraphSAGE — the original `formal_rerun_core.run_gnn` training loop on the reverse-message reference graph:
  each annual stored E1–E5 record plus one reversed record with the same relation label (Eq. 2 of the manuscript);
  original 303,775-node index; no E7 links.

The original training core (SHA-256 6b7d1188…) is called verbatim; only its year constants are set to the W2
window, its seeding is replaced by strict deterministic seeding, and the annual cache is given the reverse records.
Hidden size 64, dropout 0.3, Adam, learning rate 0.0005, at most 100 epochs, validation-AUC patience 10, positive
weight N_neg/N_pos of W2 training rows (17,290/158). Scalers are fitted on W2 training rows only. No tuning, no
additional seeds, no other learners, no feature changes.

## Stage separation

Stage T reads only fiscal years 2010–2018. Each fit saves its checkpoint (RF: fitted model), validation threshold
(F1-maximizing on validation, original function) and validation history, and is locked by hash. Only after all 20
fits are locked is `ALL_20_LOCKED.json` written. Stage E then reads 2019–2020 for the first time in this run and
scores each locked model once with its own locked threshold. A failed or interrupted fit stops the job for technical
review; it is never silently rerun.

## Reported quantities

For each fit on the W2 test: ROC-AUC, AP, F1 at the locked validation threshold, P@5%, R@10FPR (archived
retained-vertex convention; the all-threshold value is also recorded), and positives within the 5% budget
(ceil(0.05 × 7,309) = 366), with cutoff-tie diagnostics.

Primary comparisons, fixed in advance: MLP−GCN, MLP−GraphSAGE, RF−GCN and RF−GraphSAGE, each as the mean of five
matched-seed differences, for all six metrics. Uncertainty: 2,000 company × matched-seed bootstrap draws,
`default_rng(241707 + 10r)`; companies resampled with all their W2 test years; five seed labels drawn with
replacement; the same draw is shared by all models and contrasts; scores and thresholds fixed. Draws lacking either
class are invalid; at least 95% must be valid. Linear 2.5/97.5 percentile intervals.

Interpretation rule, fixed in advance: an interval containing zero is "unresolved", never "equivalent". No result
from W2 replaces any primary-window result. Document-overlap (D/O) and source-status subsets are not constructed for
W2; only the full W2 test is analysed.

## Return

Aggregate files only: per-fit metrics, group means, paired contrasts, the 2000 × 4 × 6 bootstrap-difference array,
window counts, locks, input/runner identities and environment. No company keys, labels, scores or model weights.
