# E7 implementation sources and portable reader

From the archive root:

```bash
python code/e7/run_reader.py
python code/e7/run_reader.py --list-sources
```

The default command verifies distributed E7 aggregates only. `--list-sources` shows the public source files and their execution-original versus publication-copy hashes. It does not contact a service or load restricted data.

`training/` contains the original training core and the E7 graph-cache, checkpoint-lock, weighted-metric and fixed-score analysis implementations. `construction/` contains the streaming XLSX reader, frozen name-category/identity checks, incidence construction, same-year company alignment and operator preparation. Necessary local imports are included. Execution-time locations were replaced by private-input placeholders; scientific operations and constants are retained. Changed files are marked publication derivatives in `provenance/source_hashes.json`.

These module copies are source evidence, not turnkey rerun entry points: original administrative private configurations and restricted artifacts are not supplied. Reference input bindings under `config/e7/` identify expected scientific input bytes using semantic locations. Frozen operator/software identities and checkpoint-lock summaries are under `run_records/e7/`. Source hashes establish file identity, not vendor rights or reproduction from a later download.

Do not put raw RPT records, company/record edges, supporting-node IDs, individual labels/scores, private masks, checkpoint files or per-draw company resampling indices in a public release (the model-level paired difference array `aggregates/e7/FULL_BOOTSTRAP_DIFFERENCES.npy` contains no company or row information). Hashed record keys remain private.
