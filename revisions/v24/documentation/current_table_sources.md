# Current table provenance

Table 2A feature-only rows: `aggregates/primary/source_v3_observed_condition_metrics.csv` under the primary label. GNN rows: `aggregates/direction/source_v13_per_seed_unified_metrics.csv`, `REVERSE_<model>_M11`, seeds 42/123/456/789/1024. All displayed values are rounded once from the source precision; no new fits are represented.

Table 4A preserves all original paired values from `aggregates/audits/22_M11_paired_bootstrap_comparisons.csv` and reorders the substantive learner comparisons ahead of direction ablations. Table 4B is a narrative map with no new statistic. Other numeric panels retain the same declared source logic in `tools/regenerate_main_tables.py`.

Rounding note: rounding the already printed GraphSAGE AP 0.00915 in S8a again would yield 0.0092. The unrounded retained mean is approximately 0.009146 and rounds directly to 0.0091 at four decimal places in Table 2A; the underlying value has not changed.
