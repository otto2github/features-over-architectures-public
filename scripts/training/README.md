# Training Scripts

This directory contains the training-side code that produced the experimental results reported in the paper. All scripts are written for Python ≥ 3.10 with PyTorch ≥ 2.0 and PyTorch Geometric ≥ 2.4.

## Layout

| File | Purpose |
|---|---|
| `gnn_baseline_common.py` | Shared utilities: data loading, temporal train/val/test split, evaluation metrics (AUC, AP, P@5%, R@10FPR, F1@best), DeLong paired AUC test, logging, result-JSON writer. Imported by both v1.1 and v1.2 training scripts. |
| `gnn_baseline_v1_1.py` | Main benchmark runner under the v1.1 knowledge-graph configuration (five edge types E1–E5). Used to produce the main result tables and figures. |
| `gnn_baseline_v1_2.py` | Benchmark runner under the v1.2 knowledge-graph configuration. See Appendix B of the paper for the v1.2 artifact audit. |
| `prepare_node_features_v1_1.py` | One-off pipeline that converts `features_v1_1.parquet` (firm-year feature matrix) into the GNN-ready `node_features_v1_1.parquet` format used by the training scripts. Includes the modality column-prefix mapping (M1–M11). |
| `build_kg_v1_2.py` | One-off pipeline that constructs `global_edge_index_v1_2.parquet` and `node_id_index_v1_2.parquet`. See Appendix B of the paper for details on what is and is not built into the persisted artifact. |
| `generate_figures.py` | Reads the result JSONs under `../../data/results/` and produces the figures referenced in the manuscript. |
| `generate_latex_tables.py` | Reads the result JSONs under `../../data/results/` and produces LaTeX `\input{}`-able table files matching the manuscript. |
| `run_benchmark.sh` | One-command driver that re-runs the main benchmark cells, the ablation cells, and the robustness cells. |

## Project root and data layout

All training and analysis scripts resolve a `PROJECT_ROOT` directory at import time:

1. If the environment variable `PROJECT_ROOT` is set, it is used.
2. Otherwise the scripts fall back to `~/code/foa_project`.

The expected directory layout under `PROJECT_ROOT` is:

```
PROJECT_ROOT/
├── data/
│   ├── processed/
│   │   ├── kg/                              # Knowledge-graph artifacts
│   │   │   ├── node_features_v0_8.parquet   # Base v0.7 node features
│   │   │   ├── node_features_v1_1.parquet   # M1–M11 node features (produced by prepare_node_features_v1_1.py)
│   │   │   ├── node_id_index.parquet
│   │   │   ├── node_id_index_v1_2.parquet
│   │   │   ├── global_edge_index.parquet    # v1.1 edge index
│   │   │   └── global_edge_index_v1_2.parquet
│   │   └── features/
│   │       └── features_v1_1.parquet
│   └── interim/
│       └── p7_gnn_results/                  # Per-run result JSONs are written here
└── logs/                                    # Per-run log files
```

## Hyperparameters used in the paper

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Hidden dimension | 64 (main protocol); a 128-dim sensitivity check is documented in the paper |
| Max training epochs | 100 |
| Early stopping | best validation AUC across checkpoints, patience = 10 |
| Loss | unweighted binary cross-entropy |
| Random seeds (5-seed mean) | 42, 123, 456, 789, 1024 |
| GAT attention heads | 1 |
| GraphSAGE neighbor sampling | full-graph (no neighbor sampling) |
| Batch construction | full-graph forward/backward each step |

## Running the experiments

All scripts use a uv-managed virtual environment by convention but can be invoked with plain `python` if dependencies are installed in the active environment.

```bash
# Option A: one-command full re-execution
bash run_benchmark.sh

# Option B: invoke a single (model, modality, seed) combination
python gnn_baseline_v1_1.py \
  --model MLP \
  --modal M11 \
  --label-col fraud_v08_strict \
  --seed 42 \
  --epochs 100 \
  --hidden-dim 64 \
  --lr 5e-4
```

Result JSONs are written under `${PROJECT_ROOT}/data/interim/p7_gnn_results/`. The `../../data/results/` directory at the repository root contains the snapshot of the result JSONs that produced the paper's tables.

## Synthetic-sample validation

The repository's `data/` directory ships synthetic-sample inputs (five rows per modality) with the same column names and dtypes as the production data. To validate the code path end-to-end before loading any licensed CSMAR data, point `PROJECT_ROOT` at a temporary directory containing the synthetic samples and run the benchmark on a single seed and modality.
