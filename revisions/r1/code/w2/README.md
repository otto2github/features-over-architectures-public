# W2 (earlier evaluation window) implementation sources

Frozen protocol: `protocol/reader/w2_protocol.md`. Fit matrix: `protocol/matrices/w2_run_matrix.json`.

* `w2_worker.py` — preflight, Stage T (fits on fiscal years 2010–2018, each locked by hash) and Stage E (one scoring pass of the locked models on 2019–2020 after all 20 locks).
* `w2_statistics.py`, `w2_metrics.py` — point metrics and the company × matched-seed paired bootstrap.
* `formal_rerun_core_REFERENCE.py` — the original training core, byte-identical (SHA-256 6b7d1188…); the worker sets only its year constants and seeding.
* `test_w2_synthetic.py` — synthetic checks of the metric, window, RF Stage T and bootstrap functions on invented data (`python code/w2/test_w2_synthetic.py`; needs numpy, pandas and scikit-learn, not CUDA).

Execution-time server locations were replaced by `private_inputs/...` placeholders; changed files are marked publication derivatives in `provenance/source_hashes.json`. These copies are source evidence, not a turnkey rerun: the private labels, features and graph files are not distributed. Server launch and download helper scripts are omitted.
