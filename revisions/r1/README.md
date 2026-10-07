# Reproducibility archive — PeerJ CS-2026:06:141707

**Graph and feature-only learning in a retrospective Chinese A-share sanction benchmark: document overlap and financial-source sensitivity**

Yi Qiu Cheng and Xiao Rong Cheng (equal first authors). This archive accompanies the first revision (R1). It contains the final frozen scientific evidence and its public verification tools, not the authors’ manuscript-editing history. The actual reserved version DOI and selected tag are recorded below; the existing source commit is bound in the final named reader asset. Identifier binding does not assert publication.

## Quick start

Python 3.13.5 was used for the checks of this build. Install the pinned public dependencies in a separate environment, then run from this directory:

```bash
python -m pip install -r requirements.txt
python tools/verify.py
python tests/test_public_interfaces.py
python tests/test_e7_public.py
python tests/test_w2_public.py
python code/w2/test_w2_synthetic.py
python tools/precision_of_contrasts.py
python tools/regenerate_tables.py --output ../regenerated_tables
```

These entry points read distributed aggregates or invented test inputs only. They do not train models, run inference, select thresholds, generate new empirical bootstrap draws or access a network. The table command writes outside the archive. Recheck the archive before making local edits; FILE_SHA256.json binds its distributed bytes.

## Contents

| Directory/file | Purpose |
|---|---|
| `aggregates/predecessor/` | Reconstructed benchmark and retained auxiliary evidence |
| `aggregates/sensitivity_fixed_score/` | 105-fit programme: frozen per-fit summaries, contrasts, intervals and replicate arrays |
| `aggregates/strict_replay/` | Existing execution-deviation and replay evidence |
| `aggregates/w2/`, `run_records/w2/`, `code/w2/` | Separate 20-fit earlier evaluation window (train 2010–2016, validation 2017–2018, test 2019–2020; 40 test positives), original 129 inputs; frozen aggregates and marked source copies |
| `aggregates/precision/`, `tools/precision_of_contrasts.py` | Bootstrap SE and approximate minimum detectable differences (MDD80) of all paired comparisons, computed from the distributed bootstrap arrays |
| `aggregates/e7/`, `run_records/e7/`, `code/e7/` | Separate 20-fit, original-129-input conservative RPT record-incidence extension; frozen aggregates and marked source copies |
| `aggregates/extension_fixed_score/` | Separate ten-fit matched 42-input extension and fixed-score summaries |
| `code/predecessor/`, `code/sensitivity/`, `code/replay/`, `code/extension/` | Construction, training and evaluation source copies; inspection/restricted-input roles are explicit |
| `config/`, `protocol/`, `run_records/` | Source schemas, analysis rules, run matrices and locks |
| `tables/current/`, `tables/expected/` | Five current main tables and six machine-readable grids |
| `tables/sensitivity_display/`, `tables/predecessor_display/` | Earlier scientific displays still referenced by the Supplement, not alternative current headline results |
| `figures/` | Two main and two supplementary figures |
| `documentation/` | Data access, scientific scope and condition/path navigation |
| `provenance/` | Current source-copy hashes and public-release identity |
| `tests/`, `tools/` | Aggregate checks, synthetic interface tests and table regeneration |

## What the checks establish

The current table verifier reconstructs six panels / 281 cells. It recalculates 1,044 interval rows / 2,088 endpoints from already retained aggregate arrays, checks existing contrast arithmetic and fit/population contracts, and verifies the manifest and original-to-public source-copy records. A separate E7 check recomputes 100 per-fit population rows into 20 group means and 60 paired metric differences, verifies 20 prescribed fits, masks, budget arithmetic and Full/D/O identities. It also recomputes all twelve E7 Full intervals (24 endpoint values) from the distributed paired bootstrap-difference array (`aggregates/e7/FULL_BOOTSTRAP_DIFFERENCES.npy`). Synthetic tests call actual metric/preparation functions on invented inputs. They are not empirical replications.

## Additional evidence in this revision

* `run_records/label_construction/label_v1_stageD_report.txt` — audit of the historical label sources: all 2,274 positives of the old announcement-year component carry the calendar year of their first penalty notice (Supplement S1.2).
* `run_records/metric_checks/R10_ALL_THRESHOLDS_CHECK.json` — R@10FPR recomputed over all observed thresholds for 511 retained test-score vectors; no full-test value differs (Technical Appendix T4.1).
* `aggregates/e7/FULL_BOOTSTRAP_DIFFERENCES.npy` — the 2,000 × 2 × 6 paired E7 bootstrap differences behind the twelve Full intervals.
* `aggregates/w2/`, `run_records/w2/`, `code/w2/`, `protocol/reader/w2_protocol.md` — the separately specified earlier evaluation window (W2): 20 fits locked before any W2 test scoring, per-fit metrics, group means, 24 paired contrast intervals and the 2,000 × 4 × 6 paired bootstrap-difference array. `tools/verify_w2.py` recomputes the group means, paired differences and all 24 intervals (48 endpoint values). The W2 test years 2019–2020 were the validation partition of the primary analysis (Supplement S25).
* `aggregates/precision/PRECISION_OF_CONTRASTS.json` — bootstrap standard errors and approximate minimum detectable differences for every paired comparison (Supplement S24).
* Private home-directory names in publication copies of source code are replaced by neutral placeholders; original execution identities remain in `provenance/source_hashes.json`.

## Scientific structure

The predecessor benchmark, the later 105-fit sensitivity programme, the isolated strict replay and the separately specified ten-fit 42-input extension are distinct. The sensitivity and 42-input programmes were specified after earlier results were known. The revision adds a separately specified 20-fit RPT record-incidence extension, specified after the preceding results were known and with its own pre-fit protocol. These are not one original preregistered programme. The earlier inference-setting deviation and component-file timing are retained in the manuscript and source evidence; packaging does not remove those qualifications.

Directory names describe analysis roles, not local editing versions. Immutable identifiers inside numerical records are retained so joins remain exact. See `documentation/scientific_provenance.json` and `documentation/condition_identifiers.md`.

## Reproduction boundary and source status

All baseline evidence needed by the public checks is directly included; there is no nested older release to download or unpack. Licensed rows, private predictions, checkpoints, component memberships and person mappings are not included. Code that requires them is source evidence, not a turnkey empirical rerun. A successful public check does not certify historical data vintages, licensing, or bitwise reproduction of model execution.

Public path adaptation and removal of administrative workflow text do not change numerical arrays or table content. Publication source copies are distinguished from exact originals in `provenance/source_hashes.json`. Original execution hashes identify original files, not modified publication copies. See `documentation/source_execution_scope.md` and `RIGHTS_AND_NOTICES.md` before reuse.

## Citation and release

Use `CITATION.cff` for the authors and title. Current release identifiers are in `provenance/release.json`; null values mean no new public identifier is claimed. The historical predecessor DOI is not the DOI of this candidate. Publish only the approved public tree/ZIP, not the separate author records or editorial correspondence.

The E7 experiment uses 265,383 annual issuer–record incidences and 62,920 new zero-feature supporting nodes; both contemporaneous conditions share 366,695 nodes. It does not validate globally unique counterparty identities, transaction amounts, direction or historical disclosure dates. See `documentation/e7_scope.md` and Supplement S23 / Technical Appendix T11. The portable `python code/e7/run_reader.py` entry point performs only aggregate verification; it does not run restricted construction or fitting sources.

<!-- R1_RELEASE_BINDING_BEGIN -->
Version DOI (registration requires Zenodo publication): 10.5281/zenodo.23181609
Git tag: cs141707-r1
Source commit binding: see provenance/release.json in the named reader asset.
<!-- R1_RELEASE_BINDING_END -->
