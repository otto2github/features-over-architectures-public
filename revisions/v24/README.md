# Graph neural networks versus feature-only learners for Chinese A-share sanction prediction: a holder–manager graph without transaction edges
## Version identity and supersession

This is the v24-20260925 reproducibility distribution for PeerJ Computer Science manuscript CS-2026:06:141707. Version DOI: https://doi.org/10.5281/zenodo.22958631. Repository tag: `v24-cs141707-20260925`. The version DOI identifies these revision files; the historical concept DOI 10.5281/zenodo.20558032 identifies the record family, not these exact bytes.

The original endpoint, graph descriptions and headline conclusions have been superseded. Older files and releases are historical evidence, not alternative current estimates. Current aggregate regeneration and invented-input interface tests do not establish complete historical point-in-time reconstruction.


**Graph neural networks versus feature-only learners for Chinese A-share sanction prediction: a holder–manager graph without transaction edges**

This is the source/aggregate supplement for revision v24 dated 25 September 2026. It contains the completed empirical evidence and public regeneration checks; it is not a new model execution or a historical point-in-time data certification.

## Public checks

From the archive root, using Python with NumPy, pandas and scikit-learn:

```bash
python tools/verify.py
python tools/regenerate_main_tables.py --output-dir ../regenerated_tables
python tests/postprocess_numeric.py
python tests/synthetic_interfaces.py
```

Run the verifier before creating an output directory inside the archive, or write regenerated tables outside the archive. The integrity check rejects unlisted files (apart from Python bytecode caches). `verify.py --verbose` prints individual numeric-test results; by default there is a single current-release status, not a chain of legacy manuscript-version statuses. No check imports a model-training entry point, performs a fit or uses the network.

`requirements.txt` records the public-check dependency specification. The execution manifests record the separate private execution environment; a private environment package list is not distributed because it is not required for the public aggregate checks. PyArrow is needed for private Parquet workflows, not for the aggregate checks above. A synthetic interface pass is not reproduction of licensed empirical inputs.

## Contents and scopes

- `aggregates/`: unchanged executed scientific CSVs and previously completed exact complement/yield derivations.
- `run_records/`: the forty temporal-sensitivity fit records and their execution manifest.
- `code/`: available construction, training and executed evaluation implementations. Historical/redacted dependencies are identified, not guessed.
- `historical_reference/`: the byte-verified recovered reverse-expanding loader, its public-release association and its provenance limit. It is not executed by public checks.
- `display_tables/main_tables.json`: only the current eight numeric main panels and one narrative map. Tools rederive numeric cells from scientific aggregates, including four added GNN reference rows from the same existing five-fit data. Table 4B is a declared narrative map, not a new numerical result.
- `tests/`: actual numeric/interface checks on invented inputs; no synthetic replica of the empirical dataset is claimed.
- `documentation/`: data access, current source mapping, original-public-record status and analysis scope.
- `provenance/earlier_source_records/`: earlier source extraction/adaptation records, not alternative current tables or chained verification entry points.

The main text displays only crossed company-and-seed intervals. Tables 2A/3 are descriptive; their fixed-fit uncertainty is preserved in the Supplement and source CSVs. The main GNN comparison uses the documented reverse-message design; stored-message results are post-hoc ablations. No source condition or metric is relabeled as a different experiment. Read `documentation/analysis_scope.md` for analyses not performed.

## Restricted inputs

Licensed vendor rows, person identifiers, document/source masks and company-level score vectors are excluded. Data-provider modules/endpoints, fields and known snapshot identities are listed in `documentation/data_access.md` and `documentation/raw_input_fields.csv`. Access requires the appropriate independent licence/account; a new download need not reproduce undocumented historical vintages. Aggregate verification is not an editor-approved exception to data policy or evidence of redistribution rights.

## Retained execution-role identifiers

The executed programs retain their historical internal role strings: `mac` = document-mask construction, `tencent` = financial-source/mask processing, and `server` = saved-score evaluation. These names do not denote additional scientific conditions. Publication directories map the stages to `document_masks/`, `financial_masks/` and `subset_evaluation/`. Executed code bytes are preserved rather than cosmetically rewritten.

## Retained condition identifiers

For GNN records, `ORIGINAL_*` = stored-message ablation (revision rerun); `REVERSE_*` = design-aligned graph reference. `STORED_*` is the corresponding GNN ablation label in the later audit files. These are immutable execution identifiers, not claims that the original submission used the stored-message ablation. **Exception:** `ORIGINAL_MLP_M11` and `STORED_MLP_*` identify the feature-only MLP; no message-direction treatment applies to an MLP.

This mapping covers `aggregates/direction/source_v13_*.csv`, their reference copies in `code/primary_diagnostics/reference/` and `code/subset_diagnostics/baseline_audit/reference/`, and the condition fields in `aggregates/audits/` and the sensitivity comparison files. In `aggregates/subset_diagnostics/subset_evaluation/`, the different naming scheme `primary__GCN__M11` / `primary__SAGE__M11` identifies the reverse-message graph references; `primary__MLP__*` / `primary__RandomForest__*` identifies feature-only fits. `asof__*` and `fixed1095__*` retain their separately defined financial/window conditions. Tokens such as `ORIGINAL_SHA256`, `ORIGINAL_BYTES` and `ORIGINAL_SELECT_COLS` in source evidence or program variables are not experimental condition names. Executed CSVs and model/evaluation code remain unchanged.

See `documentation/condition_identifiers.md` for the file-scoped guide.

## Historical source-only scope

Three legacy `SOURCE_ONLY` planning/scaffolding files (`S0024`, `S0025`, and `S0059`) are intentionally excluded from this public v24 archive. They were not used to generate any reported v24 scientific result and contain inaccurate or irrelevant non-execution planning/template text. Their exclusion changes no scientific CSV, fit record, model output, interval, table value, or public verification input.
