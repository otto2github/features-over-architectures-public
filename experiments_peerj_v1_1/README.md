# PeerJ CS revision experiments (v1.1)

This directory contains the additional experiments added during the PeerJ Computer
Science revision, addressing reviewer requests for (i) an imbalance-specialized fraud
GNN, (ii) a reproducible score-persisted headline run with a pooled paired test, and
(iii) a coarse per-architecture hyperparameter sweep. All scripts reuse the main
benchmark's data loading, time-based split, leakage control, and evaluation
(`scripts/p7_gnn/gnn_baseline_common.py` and `gnn_baseline_v1_1.py`); set
`THESIS_PROJECT_ROOT` to the project root before running.

## Scripts

| Script | Purpose | Manuscript |
|---|---|---|
| `rerun_main_persist_scores.py` | Re-runs MLP + 4 standard GNNs × 5 seeds at M11 under the canonical protocol (lr=5e-4, h=64) and **persists per-firm-year test scores**; computes the pooled five-seed-mean-score DeLong consistency check. | §V-C, Table 5c |
| `run_pcgnn.py` | A **PC-GNN-style** pick-and-choose variant in PyTorch Geometric (sigmoid-gated cosine-similarity Choose step; class-weighted BCE in place of the label-balanced sampler), run under the identical protocol × 5 seeds. | §V-C |
| `gnn_tuning_sweep.py` | Coarse per-architecture sweep over lr ∈ {1e-3, 5e-4} × hidden ∈ {64, 128} at seed=42, M11. | §V-C, Table 5d |

Run example:

```bash
cd scripts/p7_gnn
THESIS_PROJECT_ROOT=/path/to/project \
  python rerun_main_persist_scores.py --modal M11 --label-col fraud_v08_strict --lr 5e-4
```

## Result artifacts

| File | Contents |
|---|---|
| `rerun_persist_fraud_v08_strict_M11.json` | Per-seed AUC/AP/F1/P@5%/R@10FPR for the score-persisted rerun (MLP 0.7190 ± 0.0022 highest; no standard GNN exceeds it). |
| `rerun_persist_fraud_v08_strict_M11_delong.json` | Pooled five-seed-mean-score DeLong (MLP vs each GNN) on n=19,384. Reported as a lower-bar consistency check; the architecture claim rests on the five-seed paired t-test (Table 5b). |
| `pcgnn_fraud_v08_strict_M11.json` | Per-seed AUC for the PC-GNN-style variant (0.7126 ± 0.0039; above the four standard GNNs, below the MLP). |
| `tuning_sweep_fraud_v08_strict_M11_seed42.{json,csv}` | Seed-42 test AUC over the lr × hidden grid. At this seed GraphSAGE/GAT exceed the MLP in several cells; the MLP lead is established by the five-seed mean (Table 2), not by any single seed. |
| `scores_persist/{MODEL}_M11_fraud_v08_strict_{seed}_scores.npz` | Per-firm-year `y_score` / `y_true` for each (model, seed). The five MLP files underlie the Appendix A reliability diagram (Figure 1) and Table 10 calibration metrics. |

### `.npz` schema
Each file contains: `y_score` (float32, n=19,384), `y_true` (int8, n=19,384), `seed` (int64).
Test set = fiscal-year 2021–2024 firm-years under the `fraud_v08_strict` label.

## Notes
- The PC-GNN here is a **PC-GNN-style variant**, not faithful PC-GNN: it approximates the
  Choose step with a fixed cosine kernel and the Pick step with class-weighted loss,
  omitting the original's learned distance metric and label-balanced neighbor sampler.
- All neural/GNN learners use class-weighted BCE (pos_weight = N_neg/N_pos ≈ 30 on the
  training period), applied uniformly so the architecture comparison is not confounded.
- `RANDOM_SEED` / per-seed control follows the main benchmark; seeds = {42, 123, 456, 789, 1024}.
