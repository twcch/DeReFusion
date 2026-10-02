# Phase 1 outsourcing handoff

This directory contains the auditable source of the Phase 1 GPU handoff. It is
preparation material only: its existence does **not** authorize Stage B--D.

Before any training:

1. check out the commit named in `FROZEN_COMMIT.txt`;
2. verify every dataset against `reproduction/results/dataset_registry.csv`;
3. run `environment/env_fingerprint.py` and compare its hard checks with
   `environment/ENVIRONMENT_LOCK.md`;
4. run the manifest/runner dry-run and rolling-split tests;
5. obtain a written stage execution authorization after the §8 gate closes.

Do not edit manifests, data, protocol text or completed attempts on the worker.
Return every attempt, including failures. A successful attempt must contain the
six files required by the execution plan and a raw-byte SHA-256 inventory.

Use `make_package.py --output <path>` to export a deterministic source archive.
The output archive is deliberately not committed to Git.
