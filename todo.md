# Jaxtitan TODO

## Data

Status: foundation complete.

What is in place:

- Offline `jaxtitan data prepare` path for HF and local sources into canonical prepared-token manifests.
- `jaxtitan data inspect` and `jaxtitan data check` for local artifact UX.
- Prepared manifests remain the default, most reproducible training input.
- Grain-backed prepared runtime pipeline is the stable default path.
- HF streaming runtime training exists behind the same `TrainingDataPipeline` boundary.
- Checkpoint and resume compatibility cover both prepared and HF streaming data modes.

Remaining follow-ups:

- Run a real HF streaming smoke against a small public dataset.
- Design streaming shuffle and document-buffer streaming semantics.
- Add multi-host streaming partitioning, likely via HF `split_dataset_by_node`.
- Add streaming tokenization workers and prefetch once correctness is stable.
- Decide whether streaming eval is worth supporting or keep eval prepared-only.
- Add optional source-file hashing for local prepare if stricter provenance is needed.

