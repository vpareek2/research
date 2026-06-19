# Data And Experiment UX

Purpose: track prepared data, HF/local source preparation, streaming data, W&B,
profiling, and experiment reproducibility UX.

## 2026-06-19 [codex] Current state

Implemented:

- `jaxtitan data prepare <config>` for HF, parquet, jsonl, and text sources.
- `jaxtitan data inspect <manifest>` and `jaxtitan data check`.
- HF streaming training foundation with strict sequential streaming mode.
- W&B mirror as optional dashboard; local artifacts remain canonical.
- Programmatic JAX profiling artifacts under each run directory.
- Cloud smoke configs and cloud bundle runbooks exist for validation workflows.

Known constraints:

- Prepared manifests remain the most reproducible path.
- Streaming training is intentionally strict: no shuffle, no tokenization
  workers, no multi-host splitting yet.
- W&B is not a source of truth and should not be used alone for scoring.

Next actions:

- Before cloud runs, prepare or stream data on the target machine using repo
  commands rather than copying private local data unless explicitly needed.
- Record manifest paths, hashes, inspect output, and validation commands in the
  relevant stream.
