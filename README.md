# Disclosure Features over Standard Graph Architectures: A Multi-Seed Benchmark of GNNs and a Matched-Input MLP for Post-Filing Fraud Risk Scoring in Chinese A-Share Firms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20558032.svg)](https://doi.org/10.5281/zenodo.20558032)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE-CODE)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-PAPER)

Reproducibility companion for the **PeerJ Computer Science** submission (under review):

> Cheng, Y. Q., & Cheng, X. *Disclosure Features over Standard Graph Architectures: A Multi-Seed Benchmark of GNNs and a Matched-Input MLP for Post-Filing Fraud Risk Scoring in Chinese A-Share Firms.*

This repository releases the code, per-seed results, persisted score arrays, and a synthetic data sample needed to regenerate the paper's tables and figures, subject to the licensed-data restrictions described under **Data availability** below.

## Overview

The paper asks whether standard message-passing graph neural networks (GNNs) outperform a matched-input multilayer perceptron (MLP) for *post-filing* fraud risk scoring on Chinese A-share listed firms: given the fiscal-year-*t* annual report, rank firms by their likelihood of a subsequent China Securities Regulatory Commission (CSRC) sanction for year-*t* conduct.

We benchmark **nine learners** — a matched-input MLP, four tabular learners (Lasso, Ridge, Random Forest, XGBoost), and four standard GNNs (GCN, GAT, GraphSAGE, RGCN) — across **three nested feature modalities** (M5 / M10 / M11) and **five seeds** (135 model instances) on **51,675 firm-year observations (2010–2024)**, using a time-based split (train 2010–2018 / validation 2019–2020 / test 2021–2024) with explicit leakage control.

### Headline findings

- **Positive and learner-class-invariant.** Acquiring audit-opinion, share-pledge, controller, and related-party-transaction *features* raises five-seed mean AUC for all nine learners.
- **Architectural null (heavily scoped).** Under a unified protocol, no standard GNN exceeds the matched-input MLP in five-seed mean AUC (headline MLP × M11 = **0.7172**); a PC-GNN-style imbalance-specialized variant ranks above the standard GNNs but likewise does not exceed it. The result is stated as *absence of robust evidence* of a GNN advantage, not demonstrated MLP superiority, and it is protocol-conditional.
- **Pure-graph diagnostic.** Stripping all node features yields near-random discrimination across architectures (**five-seed mean AUC 0.49–0.59**; single-seed range 0.45–0.57). Non-GNN structural baselines bound the topological signal (Node2Vec 0.6257; label propagation 0.5329).

All conclusions are protocol-conditional and do not extend to faithful PC-GNN, CARE-GNN, UD-GNN, attention-based heterogeneous models, dynamic or edge-weighted transaction graphs, or text-based models. See the paper's Limitations.

> **Scope note.** The knowledge graph (v1.1) has five edge types — equity, cross-guarantee, common-director, common-supervisor, personnel-interlock. It contains **no genuine related-party-transaction (RPT) edges**; RPT information enters only as node-level features. The v1.2 artifact is a second-protocol replication of the same five-edge graph (the planned RPT edges were never built into the persisted artifact) and provides no evidence about RPT graph edges. See Supplemental Article S1 of the paper.

## Repository structure

    .
    ├── data/                     # feature schema / data dictionary + synthetic sample (no licensed raw data)
    ├── experiments_peerj_v1_1/   # per-seed result JSON, persisted score arrays, run outputs
    ├── scripts/                  # training / evaluation / table-generation scripts
    ├── generate_synthetic_sample.py
    ├── CITATION.cff
    ├── LICENSE-CODE              # MIT (code)
    ├── LICENSE-PAPER             # CC BY 4.0 (paper text, figures, tables)
    └── README.md

*(Adjust the tree above to match your actual layout.)*

**Key scripts** (names as referenced in the paper):

- `gnn_baseline_v1_1.py` — MLP and standard-GNN (GCN/GAT/GraphSAGE/RGCN) training and evaluation on the v1.1 graph.
- `rerun_main_persist_scores.py` — score-persisting rerun; produces the authoritative per-firm-year test scores used by the pooled DeLong test and the calibration analysis.
- `run_pcgnn.py` — PC-GNN-style imbalance-specialized variant.
- `gnn_tuning_sweep.py` — single-seed learning-rate × hidden-dimension sweep (Table 5d).
- `generate_synthetic_sample.py` — generates the 50-row synthetic sample from the feature schema.
- `generate_latex_tables.py` — regenerates the manuscript tables from the released result JSON.

## Data availability

The empirical features are derived from **CSMAR** and from CSRC enforcement records, which are licensed / third-party data and **cannot be redistributed**. This repository therefore releases:

- the **feature schema / data dictionary**;
- a **50-row synthetic sample** (`generate_synthetic_sample.py`) matching the schema, for code-path testing;
- **per-seed result JSON** and **persisted score arrays** sufficient to regenerate every table and figure;
- all **training, evaluation, and table-generation code**.

Raw CSMAR feature matrices and the train/validation/test split-index files are withheld because the split indices may encode licensed firm identifiers. Researchers holding a CSMAR license can reconstruct the model inputs from the released schema and code.

## Reproducing the paper

1. Create the environment (Python 3.x with PyTorch, PyTorch Geometric, scikit-learn, XGBoost, pandas, numpy).
2. Generate the synthetic sample for a smoke test: `python generate_synthetic_sample.py`.
3. With CSMAR access, build the feature matrices per the released schema; otherwise use the released per-seed result JSON to regenerate reported outputs.
4. Regenerate the manuscript tables: `python scripts/generate_latex_tables.py`.

Random seeds used throughout: {42, 123, 456, 789, 1024}.

## Citation

If you use this repository, please cite the paper and the archived software. Citation metadata is in `CITATION.cff`. The **concept DOI (all versions)** is:

> **10.5281/zenodo.20558032**

Always cite the concept DOI so the reference resolves to the latest archived version.

## License

- **Code:** MIT — see [`LICENSE-CODE`](LICENSE-CODE).
- **Paper text, figures, and tables:** Creative Commons Attribution 4.0 International (CC BY 4.0) — see [`LICENSE-PAPER`](LICENSE-PAPER).
