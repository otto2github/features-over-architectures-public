# 105-fit supervision and financial sensitivity protocol

Reader edition of the frozen scientific requirements. Dates below refer to specification/execution chronology, not publication or independent preregistration. Literal identifiers used in hashing and record joins are intentionally retained.

**Date: 2026-09-27. Status: methodological amendment frozen; empirical execution remains on HOLD.**

This replaces freeze v1. It is a protocol, not a manuscript revision or a results report. No sensitivity empirical training or score evaluation was performed in this audit. Prior predecessor results were visible; this is a **pre-new-results amendment to an already examined benchmark**, not an original preregistration. Absence of results on the remote servers has not been independently established.

The JSON companion contains the exact feature lists, run matrix rules, source identities, seeds, contrasts and gates. `sensitivity_run_matrix.json` enumerates all 105 planned fits. The JSON and this document must agree; a disagreement is a stop condition, not permission to choose one after seeing results.

## 1. Decision and scope

Freeze v1 is **NOT APPROVED FOR EXECUTION**. Its overall sensitivity strategy is retained, but it underspecifies masking/preprocessing, overstates what a single donor completion can establish, does not define a financial-only imputation boundary consistently, and leaves several evaluation choices open.

Freeze v2 defines four bounded retrospective experiments:

| Family | Question | Models and repetitions | Fits |
|---|---|---|---:|
| A | Known-document supervision purge | MLP, unweighted RF, reverse GCN, reverse GraphSAGE; M11; five seeds | 20 |
| B | FIN87 constant-code/donor-completion stress test | MLP/RF × M5/M11 × five model seeds × three fixed donor worlds | 60 |
| B0 | Delete direct financial-block information | MLP/RF × 17-/42-column nonfinancial modalities × five seeds | 20 |
| C | Predictability of three retrospective availability flags | Unweighted RF × five seeds | 5 |
| Total | All arms reported; no favorable-seed/world selection | | **105** |

The extra donor worlds and B0 are an audit design choice, not a universal statistical requirement or a guarantee that 105 fits establish validity. All fits reuse the same eligible test rows. They create no new independent positive observations. A and B/B0/C are **not crossed**: the program cannot establish performance after simultaneous document purging and financial-signal intervention.

No multi-window, E7/RPT, new GNN architecture, tuning, seed search, PC-GNN-style model, DeLong test or restored old-endpoint placebo is authorized. Existing predecessor results remain immutable and visible.

## 2. Frozen inputs and what was actually verified

Execution-time source identities were bound before the additional fits. The publication source-copy ledger records available source identities; exact internal manuscript and administrative records are retained separately by the authors. Public verification does not require those internal records.

Private Parquet inputs, row-level prediction files, source masks and the Stage A/E/F event lineage were not available here and are **not** marked verified. Exact private hashes must be recorded in score-blind `RUN_BINDINGS_v2` before the first empirical fit. This binding may supply paths/hashes/counts only; it may not change scientific rules.

The cohort/endpoint is unchanged: primary `label_v1_strict_ab_primary`, cutoff 2026-05-08, minimum 1,095-day follow-up, train 2010–2018, validation 2019–2020, test 2021–2022. The original counts are train 24,106/254 positives, validation 7,309/40, test 8,435/20. This is a common-cutoff target, **not** a fixed 1,095-day outcome horizon.

Use the original exact sorted M5/M11 feature lists (104/129), with the explicit 87 `fin_`/`fini_` columns in the JSON. The reference inventory is copied byte-for-byte for audit purposes. All model seeds remain 42, 123, 456, 789, 1024.

## 3. Family A: retain the two-level purge, define its boundary precisely

### 3.1 All-key incidence and immutable key rule

Use the frozen `document_key` implementation, pinned source hash, and Stage A/E/F lineage. Use **every qualifying event/key for each eligible positive firm-year**, not only its earliest event, one selected row, or an already aggregated overlap flag. The earlier mask file alone is not sufficient to reconstruct the purge.

Before fitting, reconcile predecessor's original direct-key diagnostics: validation 39 identified positives/21 earlier matches; test 20 identified/13 earlier matches. Inspect number reuse/year ambiguity and multi-key rows without model scores. Do not automatically append an announcement year or invent an issuing authority. Unresolved collisions or unbound source lineage stop the experiment and require a separate pre-results identity amendment.

The key is an issuing-authority/document-number proxy, not a legal-case identifier. Exact document-key separation is not transitive case-component separation. One firm-year can link multiple documents; retaining all keys prevents an arbitrary first-key shortcut, but no legal-case-disjoint claim is permitted.

### 3.2 Masks, formed simultaneously before exclusions

Let K(r) be the set of usable keys on positive row r. Let V be all keys of the **original** validation positives and T all keys of the **original** test positives.

- Training: exclude positive r from supervised fitting if K(r) intersects V union T.
- Validation: exclude positive r from checkpoint/threshold selection if K(r) intersects T.
- Keep negatives unchanged. The qualified-event lineage does not establish the legal-case membership of every negative row; do not invent that membership.
- Retain unidentified positives; report no-key and partially-keyed positives separately. Unknown does not mean clean.
- Never turn an excluded positive into a supervised zero. Never write replacements into the frozen label file.
- The full test keys/labels and predecessor D/O membership remain unchanged. **Do not redefine D after purging.**

All four learners use identical masks. Required assertions: no usable direct key shared by retained train/retained validation; retained train/test; retained validation/test. An all-key incidence test is required, not merely matching marginal counts.

### 3.3 Preprocessing, graph context and selection

Refit scalers using **retained eligible training rows**. Recompute the neural class ratio on those same supervised rows. RF stays unweighted. Use the purged validation mask for every epoch's AUC, patience, final checkpoint and F1 threshold. Reusing the original checkpoint, old threshold or original validation AUC is forbidden.

For GNNs, keep the original annual graph and covariates, including the covariates of rows no longer supervised. Those rows can carry messages; their labels cannot enter loss, selection, label propagation or any target-derived feature. This is a **known-document-supervision-purged sensitivity**, not full graph/node deletion. Removing graph nodes would change the graph operator and answer a different question.

Preserve all negative-only training-year graphs. A training-year graph with no retained supervised rows may be skipped with its count recorded; both classes must exist across the complete training and validation partitions. No one-class threshold/default-model fallback is allowed.

The A-versus-predecessor contrast includes changes in labels used for supervision, class weight, scaler and validation selection. It is not a separately identified causal effect of case memorization. Test-document identities are used retrospectively in split construction, so this is not untouched prospective validation or proof that all training labels were historically known.

### 3.4 Planned comparisons

At M11, compute MLP−GCN, MLP−GraphSAGE, unweighted RF−GCN and unweighted RF−GraphSAGE. Also compute **purged minus predecessor original within each of the four models**, and changes in those four between-model contrasts. This supplies the intervention comparison missing from v1's explicit contrast list.

## 4. Family B: a financial-only donor stress test, not missingness removal

### 4.1 What it can and cannot identify

Featurewise donor sampling removes a *single fixed sentinel for final nonfinite FIN87 cells*. It does not make missingness statistically undetectable: inconsistent joint relationships, altered time/industry distributions, derived-field relationships, or upstream finite zero encodings can still expose source availability. It also changes substantive financial information. A performance change is therefore not an identified amount of leakage.

A simple counterexample: complete pairs satisfy X2=X1; independent marginal donors for two missing entries usually break that relation. Both marginal distributions can be correct while the missingness pattern remains inferable. More model seeds cannot repair this identification problem.

### 4.2 Exact completion rule

Use approved as-of inputs. Only the **87-column FIN87 whitelist** may change. Leave all finite financial entries, including legitimate zeros, unchanged. Convert nonfinite FIN87 cells to missing. Nonfinancial input values and their original learner-side handling are unchanged; do not donor-impute the whole 104/129-column modality. The as-of builder's upstream finite zero substitutions and industry transforms remain part of the limitations.

For each financial feature, the donor pool consists of finite values from original eligible training rows, sorted by canonical company/year. Retain row multiplicity; do not sample only unique values and do not match on labels. An empty donor pool is a hard stop.

Use three imputation seeds: **271707, 271708, 271709**. For each seed and missing cell, construct a PCG64 generator from the first 16 SHA-256 bytes of canonical JSON `["V27-B-v2", imputation_seed, feature_name, company_code, fiscal_year]`; use unbiased uint64 rejection sampling to choose a training donor index. Exact reference logic is in `protocol_reference_ops_v2.py`.

Generate all three matrices before training. A world is shared across both models, both modalities and all five model seeds. Identical financial cells must be identical between M5 and M11. Feature iteration order, row order and selecting M5 versus M11 must not change a cell's draw. The model RNG must never consume the imputation RNG.

Each world's scaler is training-only and world-specific; RF stays unscaled. Keep original supervision and validation. Save donor-world matrices/hashes privately. Never pick the best world or stop after a favorable one.

### 4.3 Comparisons and uncertainty

For each model/world: M11−M5; donor minus existing approved-as-of zero-filled inputs at M5 and M11; and the change in M11−M5. Report all per-world, per-model-seed estimates, world-level summaries and their range. Report the equal-weight mean of per-fit metrics over the three fixed worlds and five seeds; do not evaluate an averaged-score ensemble and label it mean per-fit performance.

Bootstrap comparisons are conditional on these three worlds. Give per-world intervals and a conditional interval for their mean. Do not treat 15 fitted models as 15 independent test cohorts, use Rubin's rules for these stress-test imputations, or claim the full uncertainty of historically missing values has been estimated. Three worlds are a bounded randomization sensitivity, not proof of Monte Carlo convergence.

## 5. Family B0: deterministic FIN87-block deletion

Delete all 87 financial columns, including their lagged/derived/industry-normalized fields in the whitelist. Use distinct modality identifiers: `M5_no_FIN87` (17 announcement-summary columns) and `M11_no_FIN87` (those 17 plus the same 25 audit/pledge/controller/RPT fields). Do not call these standard 104/129-dimensional M5/M11.

Run MLP and unweighted RF, both modalities, five seeds: 20 fits. Use the original cohort and supervision, and the original nonfinancial processing. The MLP's input layer changes only as required by input dimensionality; fixed hidden layers and training settings are unchanged.

This removes the *direct* FIN87 input channel without inventing missing financial values. It is not a historically clean cohort, a missing-at-random correction, or proof that remaining features lack version signals. Dropping genuine financial information also changes the question and effective capacity. Report its within-learner 42−17-column increment; compare it descriptively with B and existing as-of increments. No direct GNN conclusion is licensed by this MLP/RF-only arm.

Deleting FIN87 does not erase all financial quantities: retained blocks include `audit_fee_to_assets` and can carry correlated source-version information. The 17-/42-column variants are FIN87-deleted inputs, not certified financial-information-free inputs.

## 6. Family C: three-flag RF

Keep the three prespecified flags: `asof_any_source_missing`, `asof_any_final_financial_nonfinite`, `fin_any_no_prior_candidate`. They are extracted **before** any donor completion and joined one-to-one for every eligible train/validation/test row. Missing joins are errors, not flag=0. Reconcile the entire split-by-label source count table in the frozen archive, not only test totals.

Fit the pinned unweighted 500-tree RF with all five model seeds, original supervision and validation F1 threshold selection. Full RF parameters are locked in the JSON; materialize its complete `get_params()` under the pinned environment before fitting. Give the individual binary-flag scores without additional learned fits as context. Keep predecessor's separate four-flag report visible; do not relabel it as this three-flag model.

This is a retrospective status diagnostic. It cannot establish the direction or causal mechanism of any vendor-version effect.

## 7. Training and execution contract

Use the pinned predecessor model definitions: MLP two 64-unit hidden layers, dropout 0.3 after the first hidden activation; GCN/SAGE two layers, interlayer dropout 0.3, and reverse-message representation. Neural optimizer is Adam, learning rate 0.0005, maximum 100 epochs, validation-AUC patience 10, improvement tolerance 1e-8; restore the accepted best checkpoint. RF uses 500 trees, gini, sqrt features, bootstrap, min leaf 1/min split 2, no class weights, and the pinned configuration.

F1 threshold: existing ascending-score precision/recall candidates, first maximizing threshold, comparison `score >= threshold`. New validation thresholds are needed for every new fit. No test labels or test metrics may select a checkpoint, threshold, donor world or fit.

The archived core has three important implementation traps:

1. `set_seed()` requests deterministic algorithms with **warn_only=True**. The adapter must restore `warn_only=False` after **every** seed reset, not only at process startup.
2. The bare graph cache is stored-direction. The reverse adapter must append every reversed edge once, retaining relation labels/multiplicity. Core-file hash equality alone does not establish the correct GNN input.
3. Its budget metric uses an unstabilized `argsort`; the already-audited stable company/year metric implementation is the reporting reference. And the old convenience runner reuses an existing `result.json` without full identity checks. Neither behavior is acceptable for a new frozen run.

The old core also evaluates test metrics inside each fit and its `--no-test` branch does not itself persist the required best checkpoint. A sensitivity adapter must therefore save checkpoints, validation scores/keys, thresholds and preprocessors during **Stage T (training/validation only)**. Lock all 105 fit artifacts before **Stage E (test inference and evaluation)**. Do not launch the old sensitivity script unchanged. This package contains protocol-reference operations and tests, **not a production training runner**.

Require `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict deterministic algorithms, cuDNN benchmark off/deterministic on, plus synthetic forward/backward determinism checks covering both Linear and the GCN/SAGE operators. Fail rather than silently changing device or determinism. Preserve exact dependency/environment identities; do not upgrade libraries to match current online documentation.

## 8. Evaluation, budgets and a correction to the earlier interpretation

All new fits are evaluated on full test and frozen D/O/S/J masks. Test inference uses the same annual graph even when a reporting subset excludes a row. Small subsets remain descriptive. Do not choose a reporting subset after seeing scores.

Metrics: AUC, step-weighted AP, validation-threshold F1, P@5%, R@10FPR and hits. Use canonical company-code/year ascending order followed by stable descending score order. Budget is `ceil(0.05*N_population)`; full, D and O each have budget 422. Report cutoff tie min/max attainable hit bounds; labels may describe these bounds but may not break the actual ranking tie. R@10FPR retains the exact frozen predecessor ROC convention and is not a promised deployment FPR.

**Two ranking questions must be separate:**

- Subset-specific evaluation: restrict to D or O, then rank again within that population.
- Full-budget attribution: rank full test once, take 422 rows, then count how many selected positives belong to D and O.

Only the latter obeys `full hits = full-list D hits + full-list O hits`. It is invalid to subtract D's *reranked* hits from full hits to infer O's hits. predecessor §6.3 explicitly states this distinction. An independently reranked subset yield must not be subtracted from a full-list yield to reconstruct the other subset.

By contrast, with the identical negatives and frozen 13/7 positive split, `20*AUC_full = 13*AUC_O + 7*AUC_D` holds per fit. This identity does not apply to AP, F1 or independently reranked top-budget counts.

Predeclare D leave-one-positive-out evaluation (seven deletions), and positive-group deletion using known document/company connections. Keep the same negatives, fixed models, thresholds and graph; evaluate changed subsets only. These are influence ranges, not retraining, cross-validation, confidence intervals or exact permutation tests. Single-class metrics are NA, never imputed as 0.5.

## 9. Resampling and dependence

Primary: the predecessor crossed company × matched-seed framework, 2,000 valid draws, company seed 1807141, seed-label seed 1807142; replicate b uses base+10*b. Draw companies with replacement, retain all their eligible test years, then five matched seed labels with replacement. Share draws across contrasted models and donor worlds. Use mean per-fit metrics, not an ensemble score.

Secondary: before fitting, form connected components of test companies linked by usable test document keys; attach all rows of each company, including negatives. Bootstrap these groups with seeds 2707141/2707142, otherwise the same paired procedure. This addresses *identified* cross-company document links, not unknown legal cases or full graph dependence. Group sizes and counts must be disclosed; few groups mean unstable uncertainty, not automatic inferential validity.

Reject and log one-class resamples. Limit to 1,000 attempts per replicate; if insufficient valid draws remain, report the interval non-estimable without changing the population. Use linear percentile 2.5/97.5 bounds. Intervals are exploratory and marginal, not multiplicity-adjusted confirmatory evidence. Do not report bootstrap sign frequency as a calibrated p-value. D/S/J have descriptive summaries/influence ranges, not claims of model superiority from two or seven positives.

## 10. Multi-window decision — made now, not after A's results

**Multi-window fitting is not required for these bounded sensitivity intervention questions, and is not authorized in v2.** The decision is final for this program; it must not be reopened merely because a model order, interval or hit count is inconvenient.

This does not endorse cross-time predictive claims. More earlier test years could be useful, but merely moving the split backward while retaining labels observed up to May 2026 does not make earlier training/validation labels historically available. Common-cutoff windows also have different follow-up opportunities. Document purging does not repair either issue, and more positive firm-years do not necessarily mean more independent matters.

A future historical-prediction study would need prespecified decision dates, label availability and negative maturity at each date, eligible training/selection populations, outcome horizon, financial vintages, window list and within-window preprocessing. Pooling scores from separately fitted windows into one AUC also requires a justified comparability rule. That study is not silently bundled into this revision.

The consequence is explicit: sensitivity remains a **single-window retrospective pipeline sensitivity study** and cannot establish general new-case or historical deployment superiority. A reviewer may still request broader validation; this protocol does not promise acceptance.

## 11. Gates before any empirical execution

The JSON lists all gates. At minimum, the score-blind preflight must bind private labels/features/graphs, Stage A/E/F lineage, all-key incidence, original and purged masks, all three source flags, FIN87 donors, three completed matrices, exact software and adapted-runner identities. Reconcile baseline counts and pairwise key exclusions; confirm both classes in retained train and validation; identify unsupported rows/keys without scores.

Document mask changes, empty donors, input hash mismatches, ambiguous identity repairs, single-class validation and deterministic-kernel failures stop fitting. No ad hoc substitutes, new years, extra models, discarded seeds or extra donor worlds are permitted. Failures must be preserved. A technical retry must use the same scientific configuration with a documented fix and parity review; it does not authorize selecting a favorable prior attempt.

Permission to start training is a separate event: it requires a reviewed sensitivity runner and a passing, hashed `RUN_BINDINGS_v2` generated without model scores. This methodological audit does not assert that those private-data checks have passed.

## 12. Reporting and unchanged materials

Keep predecessor results intact. Report all planned contrasts, even reversals or losses of feature gain. Distinguish the unweighted primary RF from its existing weighting sensitivity; do not compare a new purged GNN with an old unpurged weighted RF as though only architecture changed.


The v1 originals are copied under `provenance/`. The audit report records source-derived observations separately from v2 design choices. The manifest and verification script validate this protocol package, not the truth of empirical results or remote-server readiness.

## 13. Method references (supporting principles, not journal mandates)

The core basis is the v1 protocol, predecessor clean manuscript and the exact nested archive identified in the source-binding report. The following official documentation supports general preprocessing/group-splitting principles; it does not prescribe the 105-fit design:

- scikit-learn, Common pitfalls, data leakage and controlling randomness: `https://scikit-learn.org/stable/common_pitfalls.html`
- scikit-learn, Cross-validation, grouped and time-series data: `https://scikit-learn.org/stable/modules/cross_validation.html`
- scikit-learn, Imputation of missing values: `https://scikit-learn.org/stable/modules/impute.html`

Accessed 2026-09-27. Current website version numbers are not an instruction to change the experiment environment.
