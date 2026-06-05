# `data/` — result artifacts and data-availability notes

This directory contains the **result artifacts** that reproduce every table and figure in
the paper, together with notes on the data layout and how to obtain the licensed raw inputs.
It does **not** contain raw CSMAR financial or disclosure features, constructed feature
matrices, or split files; see the repository root [README.md](../README.md) for the full
Data Availability Statement.

## What is here

- `results/` — per-seed and per-run **result JSON** for the full benchmark: the main
  five-seed runs (`gnn_baseline_v1_1_seed{42,123,456,789,1024}_v73.json`), the tabular
  baselines, the pure-graph/placebo diagnostic, the hidden-dimension ablation, the
  label-protocol robustness runs (`fraud_v07`, `fraud_v08_strict`, `fraud_v08_loose`), and
  the modality ablations. These JSON files are the inputs to `scripts/analysis/`
  (`generate_latex_tables.py`, `generate_figures.py`) and reproduce the manuscript tables
  and figures without any licensed data.

The PeerJ-revision experiments (score-persisted rerun, PC-GNN-style variant, tuning sweep)
and their per-firm-year score arrays live in `../experiments_peerj_v1_1/`.

## What is NOT here, and why (data licensing)

The raw firm-level financial and disclosure features used in this study are derived from
**CSMAR** (Shenzhen GTA Education Tech), under an institutional licence that **prohibits
redistribution**. Accordingly this release does **not** include:

- raw CSMAR tables or the constructed M5/M10/M11 feature matrices;
- the train/validation/test split files (they are keyed to firm-year identifiers from the
  licensed panel);
- a knowledge-graph edge index or node-feature parquet built from the licensed data.

The CSRC enforcement records used to construct the fraud labels are **public**
administrative-penalty announcements.

## How to reproduce feature extraction

1. Obtain an institutional CSMAR licence from Shenzhen GTA Education Tech.
2. Download the raw tables corresponding to the 129 M11 features (field names, dtypes, and
   source descriptions are enumerated in the manuscript, Section III and the supplementary
   feature list).
3. Run the feature-construction and graph-building scripts under `../scripts/training/`
   (`prepare_node_features_v1_1.py`, `build_kg_v1_2.py`) to produce the M5/M10/M11 feature
   matrices and the v1.1 knowledge graph, then run `gnn_baseline_v1_1.py`.

The authors can provide the feature dictionary and split-construction code to licensed
CSMAR users on request, subject to the licence terms.

## Note on the v1.2 knowledge-graph configuration

The v1.2 configuration was originally planned as v1.1 extended with a sixth edge type
representing structured related-party-transaction (RPT) edges, partitioned into five RPT
sub-categories (fund-flow, guarantee, commercial, asset, other-RPT). On audit, the persisted
v1.2 edge index contains **only the five v1.1 edge types** (E1–E5; identical edge count); the
planned RPT edges were specified in the v1.2 training script's expected-edge map but were
**not built into the persisted edge index**.

The v1.2 results in the manuscript should therefore be read as a **second independent
training-protocol replication of the v1.1 graph**, not as evidence about the predictive
contribution of RPT graph edges; the attempted leave-one-edge-type-out diagnostic is a
protocol-replication diagnostic rather than a substantive RPT-edge ablation. Full audit
details are in Appendix B of the manuscript. Building RPT graph edges from raw disclosure
records and re-running the protocol on a genuine six-edge-type graph is left to future work.
