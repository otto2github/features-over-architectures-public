# Source execution scope

`tools/verify.py`, `tools/regenerate_tables.py`, and `tests/test_public_interfaces.py` are the supported offline public entry points. `tests/predecessor/` also exercises actual source functions on invented arrays and is invoked by the interface runner.

`code/` contains the available construction, training, source-audit and fixed-score implementations. Some require licensed databases, private masks, saved checkpoints or prediction vectors and their original environment. Paths in publication copies may be relocated or replaced by private-input placeholders. Their original fixed hashes may refer to execution-time dependencies, not the publication copies. They have not been rerun on private inputs for this release.

Complete, redacted and recovered source representations retain their source-scope qualifications. No missing dependency or withheld literal has been supplied by guessing. `provenance/source_hashes.json` distinguishes unchanged files from derivatives; the private author mapping retains original path names as well. Historical full-repository versioning and local administrative work orders are not needed for aggregate verification and are not included.

The manifest proves the public files have not changed relative to this release. The scientific checks test existing numerical evidence. Neither proves author attestations, source-license rights, historical vendor completeness or unseen private execution history.


This revision adds `code/e7/`. Its reader wrapper is portable and reads aggregates only. The training and construction modules are marked source evidence; private locations were replaced by placeholders and their publication hashes differ where applicable. Private masks, records, input configurations and GPU environment are required for any independent empirical execution. Original runner hashes in `run_records/e7/` identify remote execution originals, not these derivative copies. No restricted E7 stage was rerun while building this public archive.
