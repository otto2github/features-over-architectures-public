# Source index for v18 review

The accompanying file is `PeerJ_141707_v18_Reproducibility.zip`; it is a local versioned review archive, not an assertion of an assigned DOI.

| Question / manuscript location | Existing evidence file |
|---|---|
| Main Table 2 and primary condition means | aggregates/primary/source_v3_observed_condition_metrics.csv; source_v3_primary_learner_summary_for_manuscript.csv |
| Main §5.2 MLP M10 minus M5 interval | aggregates/primary/source_v3_paired_metric_delta_company_cluster_bootstrap_ci.csv; left MLP_M10, right MLP_M5, primary label |
| Main §5.3 MLP minus weighted RF | Same paired CSV; left MLP_M11, right TABW_RandomForest_M11 |
| Main Table 4 crossed intervals | aggregates/audits/22_M11_paired_bootstrap_comparisons.csv |
| Forty temporal sensitivity fits and eight summaries | aggregates/sensitivity/; run_records/sensitivity/ |
| Source-date/value exposure | aggregates/financial_vintage/14_financial_exposure_by_block.csv; related selection/binding reports |
| Accepted as-of preparation and missing-source counts | aggregates/preparation/asof_change_summary_by_partition_label.csv; asof manifests |
| Added binary source-status scores | aggregates/source_status_diagnostics/; tools/regenerate_main_tables_v18.py |
| Document-key overlap and lag/window accounting | aggregates/audits/06_case_and_source_overlap.csv; 07_prior_positive_known_by_anchor.csv; 05_fixed_1095_counts_ONLY.csv |
| Graph coverage and degree evidence | aggregates/audits/31_graph_entity_degree_distributions.csv; 32_graph_person_hub_E3_shares.csv; 33_training_union_coverage_RECOMPUTED.csv |
| Endpoint construction | code/label_construction/ and its dependency/portability inventory |

The source-status point metrics are derivable from marginal counts, with no fitting. Their correlation with outcome does not identify which signal a fitted model actually exploits. The as-of source-absence union spans table/lag selection tasks, not necessarily final feature missingness.

Private score-based subset metrics, joint overlap-by-exposure tables, and sensitivity-family score-level intervals are not included as results because they have not been executed in this v18 revision. No missingness-neutral or case-blocked refit is represented as completed. Original empirical results and executed training core/runner/protocol bytes remain unchanged.
