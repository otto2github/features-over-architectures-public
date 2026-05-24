# features-over-architectures

A reproducibility companion repository for the IEEE Access submission:

**"GNNs vs. a Matched-Input MLP Baseline for Post-Filing Financial Fraud Risk Scoring on Chinese A-Share Listed Firms: A Multi-Seed Controlled Benchmark"**

by **Yi Qiu Cheng** and **Xiaorong Cheng** (co-first authors), School of Management, Wuhan University of Technology.

- **Corresponding author**: Yi Qiu Cheng — chengyq01@outlook.com — ORCID: 0009-0006-9094-3951
- **Co-first author**: Xiaorong Cheng — chengxiaorong111@outlook.com — ORCID: 0009-0002-5328-0686

## TL;DR

On a multi-seed (`seeds = {42, 123, 456, 789, 1024}`) benchmark of 51,675 firm-year observations of Chinese A-share listed firms (2010--2024), under a unified non-exhaustive hyperparameter budget (`h = 64`, `lr = 5e-4`, early stopping on validation AUC, patience = 10):

- **MLP achieves the highest five-seed mean test AUC at the saturated M11 modality** (`0.7174 ± 0.0023`). No GNN or strong tabular learner exceeds it in five-seed mean.
- A **hidden-dimension check at `h = 128`** shows that the MLP-vs-GraphSAGE rank can reverse at the point-estimate level (GraphSAGE point estimate above MLP), without retained per-seed score vectors for an inferential test, indicating that the architecture conclusion is **conditional on the main `h = 64` protocol rather than universal**.
- A **pure-graph diagnostic** that strips all node features yields AUC values in the 0.45 to 0.57 range across all four evaluated GNN architectures, evidence that the disclosure-level static knowledge graph carries weak independent signal beyond firm-level features.
- A **v1.2 knowledge-graph configuration** was developed with the intention of adding structured related-party-transaction (RPT) edges. **On audit, the persisted v1.2 edge index in this release contains only the five v1.1 edge types** (E1--E5); the planned RPT edges were specified in the v1.2 training script but were not built into the persisted artifact. The v1.2 results therefore function as a second independent training-protocol replication of the v1.1 graph (and as an artifact audit), not as a substantive edge-type contribution test. Details and full LOO replication diagnostic in Appendix B of the manuscript.

## What this repository contains

- `data/` — data dictionary, knowledge-graph schemas (v1.1, v1.2), train/validation/test split files for each label protocol, synthetic-sample schema for code validation.
- `scripts/` — training and evaluation code, score-retention re-execution script (`score_retention_demo.py`), one-command reproduction scripts.
- `results/` — per-seed test-set summary metrics (AUC, AP, F1@val-best, P@5%, R@10FPR), training run logs, hyperparameter configurations for every (architecture, modality, seed) cell.
- `paper/` — submission-related paper materials (see [Data Availability](#data-availability-statement) below for what is and is not shared here).
- `LICENSE` — MIT for code; data documentation is in `data/README.md`.

## Data Availability Statement

This release provides everything needed to reproduce the experiments **on synthetic-sample inputs** that match the production feature schema. It does **not** redistribute the raw firm-level financial and disclosure data:

- **What is shared**: code, knowledge-graph schemas, train/validation/test split files, per-seed test-set summary metrics, training logs, score-retention re-execution script, one-command reproduction scripts, synthetic-sample schema (five rows per modality covering all feature blocks).
- **What is NOT shared**: raw firm-level financial and disclosure features. These are derived from CSMAR data licensed by Shenzhen GTA Education Tech and cannot be redistributed under the data licence; users wishing to reproduce the full feature extraction must obtain an institutional CSMAR licence independently.
- **CSRC enforcement records** (used for fraud labels) are public administrative-penalty announcements and remain accessible via the official CSRC website.

## Reproducing the experiments

1. Obtain a CSMAR licence and download the raw financial / disclosure tables listed in `data/README.md`.
2. Run the feature-construction pipeline under `scripts/` against the raw tables to produce the M5/M10/M11 feature matrices.
3. Run the one-command reproduction scripts under `scripts/` to retrain each (architecture, modality, seed) cell.
4. To run the calibration / score-retention follow-up analyses without retraining everything, use `scripts/score_retention_demo.py` with the seed list specified in the manuscript.

Synthetic-sample inputs (five rows per modality, same dtypes and column names as production) are included so that the code path can be validated end-to-end before any CSMAR-licensed data are loaded.

## Citation

If you use this code or release artifact, please cite the manuscript (forthcoming, IEEE Access submission `v1.0-ieee-access`).

```bibtex
@article{cheng2026gnns,
  author = {Cheng, Yi Qiu and Cheng, Xiaorong},
  title  = {GNNs vs.\ a Matched-Input MLP Baseline for Post-Filing Financial Fraud Risk Scoring on Chinese A-Share Listed Firms: A Multi-Seed Controlled Benchmark},
  note   = {IEEE Access submission v1.0; companion repository tag v1.0-ieee-access},
  year   = {2026}
}
```

## License

Code: MIT (see `LICENSE`).
Documentation and figures: CC BY 4.0.
