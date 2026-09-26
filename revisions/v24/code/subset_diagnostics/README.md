# Evaluation-only postprocessing source (v19 distribution)

This directory contains the numerical and mask-export implementation used for the
reported postprocessing, plus its baseline audit library and source references.
No role trains models. The personal SSH/orchestration script is not distributed.

`postprocess.py` has one CLI default removed and a corresponding server argument
guard added for portability. `source_portability_manifest_v19.json` records the
executed and distributed hashes. `KIT_SOURCE_IDENTITY.json` records the executed
source identities, while `TOOL_SHA256.json` identifies this distributed directory.
No source label such as `reviewkc_v18` is a new manuscript version selector.

The self-test below requires numpy, pandas, scikit-learn and a working PyArrow
Parquet engine. It uses invented records and does not read licensed inputs:

```bash
python postprocess.py selftest
```

Authorized empirical roles require their explicit private paths. Run `--help` to
see the arguments. The `server` role requires `--project-root`, `--strict-run`,
`--financial-mask` and `--document-mask` to reproduce all subset outputs. Omission
of a mask produces a blocked subset status even if intervals can finish. Source
roles also require their original archived label/qualification stages or financial
database and accepted as-of file. Running with a vendor account or a fresh API
download does not substitute for the recorded snapshot.

Private masks remain in local PRIVATE/LOCAL_ONLY paths. Role return ZIPs contain
aggregates and input fingerprints, not individual scores or document numbers.
