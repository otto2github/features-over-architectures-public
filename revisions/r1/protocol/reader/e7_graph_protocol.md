# Conservative RPT record-key construction and alignment protocol

Reader translation of the graph-preparation protocol fixed on 6 October 2026 before any E7 fit. Earlier results (predecessor, 105-fit programme and 42-input extension) were known. This construction is a separate sensitivity, not an original-study preregistration.

## Identity amendment and retained limitation

The initial feasibility protocol required verified global counterparty-ID semantics before treating an ID as a legal-entity key. That gate remains unresolved. An explicit pre-fit amendment therefore narrows the object to conservative RPT-derived ID/name record incidence; it does not certify a globally identified transaction network. Identity-focus return SHA-256: `caa2a7bf77d96deba4293304d9e64e9d57ae268124db8676143bbbc1cdec33cb`.

## Frozen parsing and incidence rules

Bind the three retained RPT_Operation volumes, DES description and historical source identities before and after construction; historical builder scripts are inspected, not executed. Use the existing streaming XLSX reader and frozen name categories. The first physical row gives fields; rows two/three must be verified metadata. Do not install dependencies or move source exports.

Normalize provider-ID strings with NFKC and edge-space removal. Decimal parsing preserves exact numeric-cell identities; do not float-convert, drop leading zeros or casefold IDs. Normalize names with NFKC, whitespace removal and casefolding. Apply frozen missing/generic/group/aggregate-label exclusions and exclude unsafe numeric and reserved IDs. No fuzzy matching or bare-name fallback.

Only 2010–2022 records determine keys and same-year conflicts. If an ID has multiple eligible normalized names in one year, exclude all eligible incidences for that year/ID, including same-issuer and cross-issuer disagreements. Do not blacklist earlier records from future-year disagreements. Retained keys are exact `(ID, normalized_name)` pairs without issuer/year. Private keys use SHA-256 of compact UTF-8 JSON `['e7_rpt_record_v1',[ID,name]]`; this is pseudonymization. Names/aliases can split the same entity, and identical ID/name pairs can still collide.

Collapse repeated operations to unique issuer–record incidences within each report year. Do not encode amounts, economic direction, type, counts or labels. Reproduce the identity-focus same-year-conflict-excluded annual structure exactly: 274,175 forward incidences before company alignment. About 49.23% of all source records and 47.94% of primary-window records have missing/sentinel IDs. Extremely common generic labels are excluded. Available exact ID-format diagnostics do not demonstrate numeric-conversion damage in the historical builder, whose execution is not established.

## Same-year company alignment and common nodes

Bind the four original analysis inputs and retained core. Preserve original ordered M11 129 inputs, preprocessing, supervised rows and annual partitions. Map issuer companies from the frozen feature table `(firm_id,year,node_id)` to the frozen `(node_id,node_idx)` index; use the original core's `astype(str).str.zfill(6)` conversion. Keep only existing same-year company feature nodes and report absent-company/year exclusions. Do not guess exchange suffixes, create new company nodes or use labels to choose graph coverage.

Sort retained hexadecimal record keys, give them a separate `E7R:` namespace and append consecutive indices after the original 303,775 nodes. Never merge record nodes automatically with existing companies, people or institutions. Supporting features are zero. Both conditions share all new nodes and ordering; control leaves those nodes isolated.

Preserve original annual E1–E5 order, duplicates and relation codes. Base messages comprise original forward then reverse records; treatment appends sorted E7-forward then E7-reverse records. GCN/GraphSAGE use unweighted messages and the retained layer defaults. Original self-loops, invalid indices, conflicting namespaces or count discrepancies block progression. Only `edge.year == fiscal_year` participates; 2023–2024 cannot determine E7 keys or exclusions. A whole-window index is not point-in-time availability and report date is not disclosure date.

## No-fit operator preparation

Verify feature order, all input hashes, original supervision, annual edge counts and contiguous non-colliding indices. For each model and thirteen actual years, compare evaluation-forward outputs on original nodes with outputs after adding isolated zero-feature nodes (atol=rtol=1e-6); no dropout, performance computation or individual logits are exported. Repeat full 2018 control/treatment forward/backward probes with seed 42 and original dropout under strict determinism. Read no labels for those probes, create no optimizer, update no parameters and require finite gradients with identical repeated hashes. Bind six core software versions and report actual GPU/CUDA, optional backends, TF32 and layer-source identities without installing replacements.

Evaluation-forward parity does not make old predictions valid controls: additional isolated nodes alter training dropout random-number consumption. The subsequent experiment therefore requires ten new contemporaneous controls and ten E7 fits, with the original M11 supervision and training core. Its complete training protocol and all-twenty-before-test lock are specified separately in `protocol/reader/e7_training_protocol.md`.

## Restricted artifacts

Source transaction exports, the issuer/year/hashed-record bridge and the two derived edge/index Parquets remain on the authorized research servers. Hashed record keys are restricted private derivatives. Public preparation records contain aggregate coverage, fields, software/source identities and checks only; they contain no edge rows, counterparty names, individual labels or probabilities. The public archive is a reader and source supplement, not an authorization to redistribute provider data.
