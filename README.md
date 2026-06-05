[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20556287.svg)](https://doi.org/10.5281/zenodo.20556287)
# Features over Architectures

Reproducibility companion for the paper:

> **Features over Architectures: A Multi-Seed Benchmark of Standard GNNs and a
> Matched-Input MLP for Post-Filing Fraud Risk Scoring in Chinese A-Share Firms**
> Yi Qiu Cheng, Xiaorong Cheng (co-first authors)
> School of Management, Wuhan University of Technology
> Submitted to *PeerJ Computer Science* (2026).

A controlled benchmark testing whether standard graph neural networks (GNNs) outperform a
matched-input MLP for post-filing financial-fraud risk scoring on Chinese A-share listed
firms. On 51,675 firm-year observations (2010–2024; ~5% CSRC-sanction base rate) we train
nine learners across three nested feature modalities and five seeds on a time-based split
with explicit temporal-leakage control. **Main finding:** disclosure-feature acquisition
dominates standard-architecture choice — adding audit-opinion / share-pledge / controller /
related-party features lifts five-seed mean AUC for all nine learners, while no standard GNN
exceeds the matched-input MLP. A reusable pure-graph/placebo diagnostic shows the
disclosure-level static graph carries weak independent topological signal (AUC 0.45–0.57).

## Headline results (Table 2, five-seed mean test AUC at M11)

| Learner | Test AUC |
|---|---|
| **MLP (matched input)** | **0.7172 ± 0.0026** |
| GraphSAGE | 0.7073 ± 0.0065 |
| GCN | 0.7057 ± 0.0042 |
| RGCN | 0.6938 ± 0.0050 |
| GAT | 0.6882 ± 0.0257 |

Five-seed paired t-test: MLP significantly exceeds GCN and RGCN (p < 0.01) and is tied with
GraphSAGE (p = 0.056) and GAT (p = 0.074). A PC-GNN-style imbalance-specialized variant
(0.7126 ± 0.0039) ranks above the standard GNNs but still does not exceed the MLP.

## Repository layout

```
scripts/
  training/                  # main benchmark
    gnn_baseline_common.py   #   shared time-split / metrics / DeLong
    gnn_baseline_v1_1.py     #   MLP + GCN/GAT/GraphSAGE/RGCN, M5/M10/M11, five seeds
    gnn_baseline_v1_2.py     #   v1.2 replication (see data/README.md audit note)
    prepare_node_features_v1_1.py, build_kg_v1_2.py
    run_benchmark.sh
  analysis/                  # table/figure generation from the result JSON
    generate_latex_tables.py, generate_figures.py
data/
  results/                   # per-seed/run result JSON reproducing every table & figure
  README.md                  #   what each result file is; CSMAR access instructions; v1.2 audit
experiments_peerj_v1_1/      # PeerJ-revision experiments (see its own README)
  run_pcgnn.py               #   PC-GNN-style pick-and-choose variant            (§V-C)
  rerun_main_persist_scores.py  # score-persisted rerun + pooled DeLong          (Table 5c)
  gnn_tuning_sweep.py        #   coarse lr × hidden sweep                          (Table 5d)
  scores_persist/*.npz       #   per-firm-year y_score/y_true (calibration & DeLong)
  *.json / *.csv             #   result artifacts
```

## Quick start

```bash
# environment: PyTorch + PyTorch Geometric (CUDA optional), scikit-learn, xgboost, pandas
export THESIS_PROJECT_ROOT=/path/to/project   # run-time data root (holds the licensed inputs)

# main benchmark (all architectures, all modalities, five seeds)
cd scripts/training
python gnn_baseline_v1_1.py --modal M5 M10 M11 --label-col fraud_v08_strict --lr 5e-4

# PeerJ revision experiments
python ../../experiments_peerj_v1_1/rerun_main_persist_scores.py --modal M11 --lr 5e-4
python ../../experiments_peerj_v1_1/run_pcgnn.py               --modal M11 --lr 5e-4
python ../../experiments_peerj_v1_1/gnn_tuning_sweep.py        --modal M11

# regenerate tables/figures from the released result JSON
cd ../analysis
python generate_latex_tables.py --project-root /path/to/project --out-dir ./tables
```

## What is and isn't in this repository

**Included:** all experiment code; the complete set of per-seed/run **result JSON**
(`data/results/`) that reproduces every table and figure in the paper; the PeerJ-revision
experiments and their outputs (`experiments_peerj_v1_1/`); and the per-firm-year score
arrays underlying the Appendix A reliability diagram, Table 10, and the pooled DeLong
(`experiments_peerj_v1_1/scores_persist/`).

**Not included (data licensing):** the raw CSMAR firm-level financial and disclosure data is
licensed by Shenzhen GTA Education Tech and **cannot be redistributed**; this repository
therefore contains **no CSMAR raw data, no constructed feature matrices, and no split
files**. The CSRC enforcement records used for labels are public administrative-penalty
announcements. To reproduce feature extraction, obtain an institutional CSMAR licence and
follow the field list and pipeline described in `data/README.md`.

## Reproducibility

Seeds = {42, 123, 456, 789, 1024}. All neural/GNN learners use class-weighted BCE
(pos_weight = N_neg/N_pos ≈ 30 on the training period), applied uniformly so the architecture
comparison is not confounded by differential class handling. The score-persisted rerun in
`experiments_peerj_v1_1/` is the authoritative run for all score-dependent analyses (pooled
DeLong, and the Appendix A reliability diagram / Table 10 calibration).

## Citation

```bibtex
@article{cheng2026features,
  title   = {Features over Architectures: A Multi-Seed Benchmark of Standard GNNs and a
             Matched-Input MLP for Post-Filing Fraud Risk Scoring in Chinese A-Share Firms},
  author  = {Cheng, Yi Qiu and Cheng, Xiaorong},
  journal = {PeerJ Computer Science (under review)},
  year    = {2026}
}
```

## AI-use disclosure

A large language model was used only for language editing and was not used to generate
research data, code, figures, experimental results, scientific claims, or their
interpretation. The authors take full responsibility for all content.
