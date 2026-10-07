# Condition identifiers in retained execution records

## Retained condition identifiers

For GNN records, `ORIGINAL_*` = stored-message ablation (revision rerun); `REVERSE_*` = design-aligned graph reference. `STORED_*` is the corresponding GNN ablation label in the later audit files. These are immutable execution identifiers, not claims that the original submission used the stored-message ablation. **Exception:** `ORIGINAL_MLP_M11` and `STORED_MLP_*` identify the feature-only MLP; no message-direction treatment applies to an MLP.

This mapping covers `aggregates/predecessor/direction/direction_reference_*.csv`, their reference copies in `code/predecessor/primary_diagnostics/reference/` and `code/predecessor/subset_diagnostics/baseline_audit/reference/`, and the condition fields in `aggregates/predecessor/audits/` and the sensitivity comparison files. In `aggregates/predecessor/subset_diagnostics/subset_evaluation/`, the different naming scheme `primary__GCN__M11` / `primary__SAGE__M11` identifies the reverse-message graph references; `primary__MLP__*` / `primary__RandomForest__*` identifies feature-only fits. `asof__*` and `fixed1095__*` retain their separately defined financial/window conditions. Tokens such as `ORIGINAL_SHA256`, `ORIGINAL_BYTES` and `ORIGINAL_SELECT_COLS` in source evidence or program variables are not experimental condition names. Executed CSVs and model/evaluation code remain unchanged.

The guide changes display interpretation only, not any condition, score, model or empirical value. The original-submission auxiliary loaders are not all established from surviving run records (Technical Appendix T2.1).
