# Conservative RPT record-incidence extension: frozen training protocol

Reader translation of the protocol fixed on 6 October 2026 before the twenty E7 fits. Earlier results (predecessor, 105-fit programme and 42-input extension) were known. This is a separate result-aware extension, not a preregistration of the original study. Original protocol and programme identities are retained in the source ledger.

## Estimand and graph

Compare the existing E1–E5 reverse-message graph with the same graph plus conservative RPT-derived issuer–record incidences. A record key uses a provider ID and its exact normalized name. Global legal-entity semantics of provider IDs remain unverified; performance cannot validate entity matching. No amount, economic direction, transaction category or transaction count is encoded.

Keep only 2010–2022 records with usable IDs and non-generic names. Exclude every incidence for an ID in a report year if that ID has multiple eligible normalized names in that year. No future-year blacklist, fuzzy matching or name-only fallback. Provider-ID strings preserve leading zeros and case; names use NFKC normalization, whitespace removal and casefolding. Renames may split entities; identical ID/name pairs may still collide. About 47.94% of primary-window source records have missing or sentinel IDs.

The pre-alignment construction has 274,175 unique annual incidences. Align issuers only to the frozen feature table's existing same-year company nodes, independently of labels; retain 265,383 incidences and exclude 8,792 (3.2067%). No company nodes are invented. Add 62,920 separate zero-feature supporting nodes after the original 303,775 nodes. Both conditions use the identical ordered universe of 366,695 nodes, so training dropout has the same tensor shape; the new nodes are isolated in control.

Preserve all original E1–E5 records, their order, relation codes and multiplicity. Each annual control graph contains base-forward then base-reverse records. The treatment appends sorted E7-forward then E7-reverse records. GCN and GraphSAGE use unweighted messages; zero-feature or degree-one supporting nodes can still change message propagation/normalization. An annual edge uses report year, not independently verified disclosure time. The common whole-window node universe does not establish historical point-in-time availability.

Annual forward E7 counts for 2010–2022 are 8,640; 10,140; 12,054; 13,230; 15,291; 17,023; 19,645; 22,732; 24,566; 26,759; 29,976; 32,404; 32,923. See `aggregates/e7/ANNUAL_ALIGNMENT.csv` and graph-preparation records. Twenty-six actual annual original-versus-isolated-extension evaluation-forward checks and four repeated 2018 control/treatment forward/backward probes passed before fitting; probes performed no optimizer update or test-performance evaluation.

## Fits and original training core

GCN and GraphSAGE (program token SAGE), seeds 42, 123, 456, 789 and 1024, both control and E7: twenty new fits. Existing predictions from earlier stages are not reused as contemporaneous controls. No model, seed, key or exclusion rule is selected using E7 performance.

Use the original ordered M11 representation of 129 inputs and original supervision. Training 2010–2018: 24,106 rows, 254 positives, 23,852 negatives. Validation 2019–2020: 7,309 rows, 40 positives. Test 2021–2022: 8,435 rows, 20 positives. Do not substitute the as-of, fixed-1095-day, purged-supervision or 42-input variants.

Call the retained original `run_gnn` loop, model definitions, optimizer/loss, annual update order, checkpoint selection and validation F1 threshold functions. Original core SHA-256: `6b7d1188097a6e9a287b865a20ed4d5824a5e35de2d926c4da43b2ceb1c523d7`. Adapt only strict seed settings, the frozen graph cache, validation observer and output locking. Missing features are filled with zero; fit the float32 scaler only on eligible supervised training rows. Bind the scaler mean and SD identities to preparation. Neural positive weight is 23,852/254.

Two layers; hidden dimension 64; dropout 0.3; Adam learning rate 0.0005; maximum 100 epochs; patience 10; nine annual training updates per epoch. Save a checkpoint only when validation ROC-AUC exceeds its prior best by more than 1e-8. Restore that state and lock the original validation-F1 threshold. The test set selects neither checkpoint nor threshold.

Strict CUDA determinism applies to both fitting and inference: `torch.use_deterministic_algorithms(True, warn_only=False)`, CUBLAS workspace `:4096:8`, deterministic cuDNN with benchmark disabled, PyG segment matmul False. Bind the observed versions, GPU/CUDA/cuDNN, optional backends, TF32 flags and layer-source hashes. Unsupported operations halt; no relaxed determinism, CPU switch, dependency replacement or seed selection.

## Checkpoint lock and test timing

Stage T parses only 2010–2020 rows and invokes `no_test=True`; the validation observer allows validation years only. Each successful fit atomically locks its checkpoint, validation scores, scaler, training summary and validation history with hashes. All twenty successful locks and their current source bindings must verify before the global lock is written and Stage E is allowed to parse test labels or compute test performance. Preparation already checked test counts, and earlier study results were visible; this does not claim that the authors had never seen test data.

Only one worker executes at a time. A completed, unchanged fit can be reused after hash checks. A failed or interrupted uncompleted T fit retains evidence and stops for technical review; it is not silently rerun until successful. Evaluation/summary recovery uses unchanged locked models and cannot refit them. Publish the return only after all fits, evaluation, paired analysis and final source checks succeed.

## Metrics, populations and paired uncertainty

Use the original frozen reporting masks with source SHA-256 `f53fa5b7f0ee845f6075bc2329d5ef6ee9d9429d38a942a71aa228f550009810`; their reuse does not change M11 inputs or supervision. Require exact one-to-one company/year alignment. A relocated mask is accepted only if filename, size and SHA all match; changed or missing masks halt before fitting.

| Population | Rows | Positives | Negatives | 5% budget |
|---|---:|---:|---:|---:|
| Full | 8,435 | 20 | 8,415 | 422 |
| D | 8,422 | 7 | 8,415 | 422 |
| O | 8,428 | 13 | 8,415 | 422 |
| S | 7,757 | 6 | 7,751 | 388 |
| J | 7,753 | 2 | 7,751 | 388 |

Report ROC-AUC, AP, F1 at the locked validation threshold, P@5%, R@10%FPR, budget hits and cutoff ties. Canonical company/year ascending order precedes a stable descending score sort; budget is ceil(0.05 n). Preserve the original weighted-metric implementation, with only its assertion import adapted, and cross-check point metrics against sklearn and stable-budget arithmetic. Verify `20 AUC_Full = 7 AUC_D + 13 AUC_O`. Attribute Full's one 422-row ranking to D/O positive hits; separately budgeted D/O metrics are not that attribution.

The primary estimand is each model's mean of five paired Full differences, E7 minus contemporaneous control. Retain all twenty fits and all five reporting populations.

Execute 2,000 fixed-score company-and-paired-seed bootstrap draws, base seed 141707; replicate r uses `default_rng(141707 + 10*r)`. Draw the original number of company clusters with replacement, retaining all years within each company, and draw five paired seed indices with replacement. Share company and seed draws across both conditions and both models. Hold scores, models and validation thresholds fixed. Summarize model-wise mean differences for all six metrics using two-sided percentile 95% intervals with linear quantiles. Single-class draws are invalid; at least 95% valid draws are required. Complete draw arrays remain private.

D/O/S/J are descriptive only: point estimates, seed SDs and paired differences, without subset inferential intervals or significance claims. Optional 32-pattern exact seed sign-flip values on Full are descriptive, with only five pairs. Intervals are not simultaneous multiplicity-adjusted guarantees. Zero-containing intervals do not demonstrate equivalence, no causal conclusion follows, and this construction does not establish that all RPT graphs lack value.

## Public/private boundary

Only aggregate metrics, source/software identities, protocols and global-lock summaries are distributed. Raw transactions, company/record incidences and supporting keys, individual labels or probabilities, private masks, checkpoints and bootstrap draw arrays remain restricted. Hashing private record keys is pseudonymization, not permission to publish them. Public source copies document implementation and have explicit derivative identities; the portable reader wrapper runs only aggregate checks.
