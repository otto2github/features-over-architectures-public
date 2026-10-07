# Provenance roles

`source_hashes.json` is the current source-copy map: public paths and their actual hashes, original-byte hashes and the publication-copy status. The full original pathname map is preserved with the authors.

`predecessor_sources/` retains scientific source-identity evidence from the predecessor computations. Likewise, identity JSON under `documentation/predecessor/` and `code/*/reference/` may describe original execution dependencies, including private or historical paths. Those are source records, not instructions that every recorded dependency is a public file. Their original hashes do not become hashes of modified publication copies. Consult the current map for current paths.

`release.json` records the current candidate and separately labels the public predecessor. No parent archive needs to be embedded. Predecessor scientific evidence required by public checks was extracted and is covered file by file, not replaced by an untested external URL.
