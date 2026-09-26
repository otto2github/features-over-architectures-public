# Retained label-construction source

These complete source files expose the actual endpoint qualification and freeze logic. They are not a new label computation. The original inputs and all fixed hashes remain unchanged.

Pipeline: Stage A builds the source pair ledger; B/B1/B2 and B3-series implement successive anchor checks and public metadata retrieval; C applies the anchor and year evidence; D audits the old announcement-year source; E forms candidate qualified risk sets; F freezes the primary and sensitivity labels; G4 checks and extends the entity index.

Do not execute all versions sequentially or select the latest result by filename. Bind the chosen stage to the retained manifest and hash. Historical source functions may select the most recent completed run; such discovery is not proof that the selected input is the manuscript input. B3 programs can contact CNINFO when their command-line entry point is run.

Required private inputs include the original event-evidence package, pair ledger, CNINFO anchor metadata/cache, stage manifests and frozen feature matrix. A checksum or a test on a made-up case does not replace these inputs. No claim of a fully executable raw-source reconstruction is made from this archive alone.

The no-network demonstration is `python tests/synthetic_endpoint_demo.py`. It tests an explicitly independent synthetic endpoint convention and verifies that the retained label-source files parse and match the supplied hashes. It does not execute stage entry points or create empirical labels.
