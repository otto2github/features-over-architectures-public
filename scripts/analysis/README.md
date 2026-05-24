# Analysis Scripts

This directory holds post-experiment analysis scripts that aggregate the per-run JSONs in `../../data/results/` into the tables and figures presented in the paper.

## Workflow

The analysis side reads the raw per-run JSONs and produces:

| Output | Content | Manuscript reference |
|---|---|---|
| Main benchmark table | Five-seed mean ± std of test AUC, AP, F1@val-best, P@5%, R@10FPR for each (model, modality) cell | Table 2 of the manuscript |
| Tabular learners table | Same metrics for Lasso, Ridge, RandomForest, XGBoost | Table 3 |
| DeLong modality-progression table | Paired AUC tests for M10 vs M5, M11 vs M10, M11 vs M5 across all five learner classes | Table 4 |
| DeLong GNN vs MLP table | Paired AUC tests for each GNN against the matched-input MLP at each modality | Table 5 |
| Pure-graph diagnostic table | AUC values under three content-stripping placebo configurations | Tables 6 and 7 |
| Robustness table | M11 test AUC under each of the three label protocols (fraud_v07, fraud_v08_strict, fraud_v08_loose) | Table 9 |
| Figure 1 (calibration) | Reliability diagram + score-distribution histogram for MLP × M11 across the five seeds | Figure 1 |

The driver scripts live alongside the training pipeline under `../training/`:

- `../training/generate_latex_tables.py` reads the result JSONs and emits LaTeX-ready table files.
- `../training/generate_figures.py` reads the result JSONs and emits the figures.

## Usage

```bash
# From the repository root
cd scripts/training

# Tables
python generate_latex_tables.py --project-root <PROJECT_ROOT> --out-dir ./tables

# Figures
python generate_figures.py --project-root <PROJECT_ROOT> --out-dir ./figures
```

Both scripts read from `<PROJECT_ROOT>/data/interim/p7_gnn_results/*.json` (the canonical run-time location); the repository-root `data/results/*.json` contains the snapshot used to produce the manuscript's tables and figures.
