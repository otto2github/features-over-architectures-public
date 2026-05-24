# `data/` — schemas, splits, synthetic samples

This directory contains everything needed to **understand** the data layout and to **execute** the code path against synthetic inputs. It does **not** contain raw CSMAR financial or disclosure features; see the repository root [README.md](../README.md) for the Data Availability Statement.

## Files

- `feature_dictionary.{csv,json}` — full 129-feature dictionary for the M5 (104), M10 (122), M11 (129) modalities with names, dtypes, and source descriptions.
- `kg_schema_v1_1.{json,md}` — knowledge-graph schema for the v1.1 configuration (five edge types: E1--E5).
- `kg_schema_v1_2.{json,md}` — knowledge-graph schema for the **v1.2 configuration as persisted in this release**. See note below.
- `split_<protocol>.parquet` — train/validation/test split files (firm-year identifiers only) for each label protocol: `fraud_v07`, `fraud_v08_strict`, `fraud_v08_loose`.
- `synthetic_sample_M5.parquet`, `synthetic_sample_M10.parquet`, `synthetic_sample_M11.parquet` — five-row synthetic samples per modality with the production schema (column names, dtypes, value ranges). Use these to validate the code path end-to-end before loading licensed CSMAR data.

## Important: v1.2 knowledge-graph artifact

The persisted v1.2 edge index in this release (`global_edge_index_v1_2.parquet`) **contains only the five v1.1 edge types** (E1--E5; 4,469,735 directed edges, identical in count to v1.1).

The original protocol planned v1.2 as v1.1 extended with a sixth edge type representing structured related-party-transaction (RPT) edges between firms, partitioned into five RPT sub-categories (fund-flow, guarantee, commercial, asset, other-RPT). On audit of the persisted artifact, however, the planned RPT edges were specified in the v1.2 training script's expected-edge map but were **not built into the persisted edge index** in this release.

The v1.2 results reported in the manuscript should therefore be read as a **second independent training-protocol replication of the v1.1 graph**, not as evidence about the predictive contribution of RPT graph edges. The leave-one-edge-type-out (LOO) diagnostic attempted under the v1.2 training script is a **protocol-replication diagnostic** rather than a substantive RPT-edge ablation. Full audit details are in Appendix B of the manuscript.

Building RPT graph edges from raw RPT disclosure records and re-running the v1.2 protocol on the resulting six-edge-type graph is left to future work.

## How to obtain the raw data

The raw firm-level financial and disclosure features used in this study are derived from CSMAR. Users wishing to reproduce the full feature extraction must:

1. Obtain an institutional CSMAR licence from Shenzhen GTA Education Tech.
2. Download the raw tables listed in `feature_dictionary.csv`.
3. Run the feature-construction pipeline under `../scripts/` to produce the M5/M10/M11 feature matrices.

The CSRC enforcement records used to construct fraud labels are public administrative-penalty announcements and remain accessible via the official CSRC website.
