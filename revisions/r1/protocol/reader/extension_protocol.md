# Separate matched 42-input GNN extension protocol

Reader edition of the frozen scientific requirements. Dates below refer to specification/execution chronology, not publication or independent preregistration. Literal identifiers used in hashing and record joins are intentionally retained.

**Date: 2026-09-27. Status: frozen after all sensitivity results were known and before any extension fit.**

This is **not** an original preregistration and is **not** part of the completed sensitivity 105-fit program. It is a deliberately narrow, result-aware extension motivated by the completed sensitivity evidence: the earlier programme removed the direct FIN87 channel for MLP/RF but did not run matched reverse-GCN/GraphSAGE fits on the same 42-column representation.

## Scientific question

On the exact sensitivity `M11_no_FIN87` 42-column representation, with the original sensitivity supervision and the same holder–manager graph, how do reverse-message GCN and GraphSAGE compare with the already frozen B0 MLP and unweighted RF references?

The extension does **not** ask whether the 42 columns are historically clean. They retain known vintage/source limitations. It also does not cross FIN87 deletion with the Family-A document-supervision purge.

## Frozen fit matrix

Ten new fits only: reverse GCN and reverse GraphSAGE × seeds 42, 123, 456, 789, 1024, all using `M11_no_FIN87`. No 17-column GNN, GAT, RGCN, tuning, new seed, multi-window, E7/RPT graph, or A×B0 crossing is authorized.

## Inputs and supervision

Use the exact approved sensitivity as-of feature matrix and exact 42-column list. `M11_no_FIN87` must equal ordered M11 minus the exact FIN87 whitelist. Use the original B0 supervision: train 24,106/254 positives and validation 7,309/40 positives. Do **not** use Family-A purge masks. Fit the scaler on original eligible supervised training rows. Neural positive weight is 23,852/254.

Use the unchanged annual graph and append each stored reverse edge exactly once with the same relation label and multiplicity.

## Execution

The sensitivity GNN architecture/training settings are frozen. Stage T reads training/validation only, saves the accepted checkpoint, scaler, validation predictions and validation-selected F1 threshold, and never evaluates test labels/scores. All 10 Stage-T artifacts must be globally locked before Stage E.

Unlike the original sensitivity Stage-E entrypoint, this extension must explicitly enforce strict deterministic settings in **both** Stage T and Stage E: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=False)`, cuDNN benchmark off and deterministic on, after every seed reset.

## Evaluation and comparisons

Evaluate unchanged Full/D/O/S/J populations with the sensitivity metric/tie/budget rules. The frozen comparison references are the five-seed sensitivity B0 `M11_no_FIN87` MLP and unweighted RF predictions, verified by their Stage-E artifact hashes.

Primary matched comparisons on Full are MLP−GCN, MLP−GraphSAGE, RF−GCN and RF−GraphSAGE. Report all six metrics. D/O/S/J are descriptive; D additionally receives leave-one-positive-out fixed-score influence analysis.

Primary uncertainty uses the same 2,000 company×matched-seed design and seeds as sensitivity. Secondary company/document-component resampling is allowed only if the already materialized sensitivity component membership with SHA-256 `c8b6c4ec082d41e6264ae5667b4c3982fac43f02a6c8d53f267f824edebb102c` is bound **before these 10 fits**. If it cannot be bound before fitting, the extension must stop rather than reconstruct it after seeing extension results.

## Ancillary legacy check

After this extension design is frozen, a no-training fixed-score check may identify which of the three already frozen source-status flags applies to the single D positive hit by all five Family-A GCN seeds. Only aggregated booleans may be exported; no company or document identifier. This check cannot alter the ten-fit matrix.

## Interpretation

The extension answers a matched 42-column pipeline comparison only. It does not prove historical cleanliness, create independent positive events, identify a general architecture winner, or retroactively convert the sensitivity program into a 115-fit preregistered study.
