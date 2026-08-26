# Jaxtitan
**Note: In progress, readme not fully updated yet. Verified training on 1 node with correct distributed algorithms, working on performance.**
Jaxtitan is a JAX-native language-model pretraining stack inspired by TorchTitan, but designed around JAX contracts instead of a PyTorch trainer hierarchy.

The goal is production-quality research infrastructure for small and mid-scale pretraining experiments: reproducible configs, deterministic data and resume behavior, clear local artifacts, honest diagnostics, and a narrow command surface that is easy to trust.

Local artifacts are the source of truth. Dashboards, notebooks, and future registry integrations should read from the files a run writes, not replace them.

## Philosophy

Jaxtitan is built around a few hard constraints:

- One path per workflow. Config check, preflight, training, inspection, checkpoint eval, and checkpoint sampling each have one explicit command.
- TOML is the run contract. No hidden CLI flags should change stabilized experiment behavior.
- JAX state is explicit. Training state, inference state, RNG, optimizer state, dataset state, and host state have clear ownership boundaries.
- Runtime compatibility is checked before restore. Resume should fail with a Jaxtitan error, not a late Orbax or shape crash.
- Metrics are numerator/denominator first. Loss, eval, and throughput rows should be reproducible and unambiguous.
- Distributed compute should not make artifacts unreadable. Metrics, summaries, checkpoint indexes, evals, and samples stay host-global JSON records.
- No hidden fallbacks. If data, tokenizer, checkpoint, mesh, or config state is wrong, Jaxtitan should fail loudly.

## Current Surface

The installable package is `jaxtitan` under `src/jaxtitan`.

The main CLI is:

```bash
uv run jaxtitan --help
```

Supported workflows:

```bash
uv run jaxtitan config check <config.toml>
uv run jaxtitan data check <manifest.json> --tokenizer <tokenizer-id>
uv run jaxtitan run preflight <config.toml>
uv run jaxtitan run train <config.toml>
uv run jaxtitan run train --resume <config.toml>
uv run jaxtitan run inspect <run_dir>
uv run jaxtitan eval checkpoint <run_dir> --checkpoint latest
uv run jaxtitan sample checkpoint <run_dir> --checkpoint latest --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1
```

Use `--json` on commands that support machine-readable output.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Check the package and test surface:

```bash
uv run pytest -q
```

Jaxtitan expects JAX-visible devices to be real runtime state, not assumed. Use preflight before training.

## Smoke Config

The Spark smoke config lives at:

```text
configs/jaxtitan/pleias_synth_smoke.toml
```

It is a short end-to-end systems run over a local prepared-token PleIAs SYNTH sample. The config expects:

```text
data/pleias_synth_smoke_gpt2/manifest.json
```

`data/` and `runs/` are local artifacts and should not be committed.

The smoke config exercises:

- decoder model build and forward pass
- Grain-backed prepared-token pipeline
- document-buffer training order
- document boundary masks
- gradient accumulation
- block remat
- train step donation
- validation eval
- checkpoint save and retention index
- deterministic resume compatibility
- checkpoint eval
- token-native checkpoint sampling
- runtime diagnostics and live CLI health output

Run preflight:

```bash
uv run jaxtitan run preflight configs/jaxtitan/pleias_synth_smoke.toml
```

Run training:

```bash
uv run jaxtitan run train configs/jaxtitan/pleias_synth_smoke.toml
```

Resume after raising `training.target_tokens` in a copy of the config:

```bash
uv run jaxtitan run train --resume /tmp/pleias_synth_smoke_resume.toml
```

Inspect artifacts:

```bash
uv run jaxtitan run inspect runs/pleias_synth_smoke
```

Evaluate the latest checkpoint:

```bash
uv run jaxtitan eval checkpoint runs/pleias_synth_smoke --checkpoint latest --json
```

Sample token ids from the latest checkpoint:

```bash
uv run jaxtitan sample checkpoint runs/pleias_synth_smoke \
  --checkpoint latest \
  --prompt-ids "15496,11" \
  --max-new-tokens 8 \
  --temperature 1.0 \
  --top-k 1 \
  --json
```

Run checkpoint eval and sampling serially on a single accelerator. They both restore JAX state and may compile device programs.

## Config Shape

A Jaxtitan run config uses these top-level sections:

- `[run]`: run id, seed, output root
- `[model]`: decoder architecture, dtypes, remat policy
- `[optimizer]` and `[optimizer.schedule]`: optimizer backend and LR schedule
- `[data]`: prepared-token manifest, tokenizer id, data order, Grain worker policy
- `[training]`: sequence length, global batch size, accumulation, token target, logging and checkpoint cadence
- `[mesh]`: named JAX mesh axes and sizes
- `[[evals]]`: validation eval cadence and number of batches
- `[artifacts]`: local artifact policy and optional mirrors

The stabilized runtime expects prepared token manifests, not raw text. Configs point at manifest files:

```toml
[data]
train_manifest = "data/pleias_synth_smoke_gpt2/manifest.json"
tokenizer_id = "gpt2"
order = "document_buffer"
shuffle_seed = 123
worker_count = 0
worker_buffer_size = 1
prefetch = false
document_buffer_size = 128
document_refill_size = 128
```

Training uses `training.global_batch_size` as the global microbatch across the `data` mesh axis. Effective tokens per optimizer step are:

```text
global_batch_size * gradient_accumulation_steps * seq_len
```

## Data Pipeline

Jaxtitan owns the public data contract:

- prepared token manifest
- tokenizer id
- train/validation split bounds
- document offsets
- pipeline state
- batch provenance
- resume compatibility fingerprint

Grain is the packaged backend for runtime iteration. Jaxtitan does not expose Grain objects as the experiment contract.

Supported training orders:

- `sequential`: deterministic sequential records
- `shuffle`: seeded record shuffle
- `document_buffer`: seeded document-aware buffer sampling with boundary masks

`document_buffer` requires document offsets in the prepared manifest and requires:

```toml
shuffle_seed = 123
document_buffer_size = 128
document_refill_size = 128
```

Synthetic joins across documents are masked out of the loss. Document provenance and BatchHet diagnostics are written to local metrics artifacts.

## Artifacts

Every training run writes under:

```text
runs/<run_id>/
  manifest.json
  config/
    source.toml
    resolved.json
  diagnostics/
    runtime.json
  metrics/
    train.jsonl
    eval.jsonl
  checkpoints/
    index.json
    000001/
    ...
  summaries/
    final.json
  evals/
  samples/
  events.jsonl
```

Important files:

- `metrics/train.jsonl`: training loss, LR, norms, throughput, MFU, timing, data provenance
- `metrics/eval.jsonl`: validation loss and eval throughput
- `diagnostics/runtime.json`: JAX, device, mesh, data pipeline, model, sharding, package versions
- `checkpoints/index.json`: retained checkpoints, latest checkpoint, best-validation checkpoint
- `summaries/final.json`: final status and aggregate diagnostics
- `events.jsonl`: lifecycle, eval, checkpoint, failure, and resume events

Artifacts are host-global records even when compute is sharded over the JAX mesh.

## Checkpoint And Resume

Training checkpoints are Orbax-native:

- `TrainState` and RNG are JAX PyTrees.
- Dataset and host state are JSON records.
- Checkpoint metadata stores a runtime compatibility fingerprint.
- Resume allows only safe mutable controls, such as increasing `training.target_tokens` when the effective optimizer schedule is unchanged.

Resume:

```bash
uv run jaxtitan run train --resume <config.toml>
```

If model shape, optimizer identity, mesh shape, data manifest, tokenizer, seed, sequence length, batch size, or effective schedule changes, resume fails with a Jaxtitan contract error.

## Inference Boundary

Inference is separate from training state.

Checkpoint sampling restores through `InferenceState`, not by exposing optimizer or dataset state. The current sampling path is token-native:

```bash
uv run jaxtitan sample checkpoint runs/<run_id> \
  --checkpoint best \
  --prompt-ids "15496,11" \
  --max-new-tokens 32 \
  --top-k 1
```

This proves checkpoint restore, inference state, KV-cache prefill/decode, and token sampling. It does not yet provide tokenizer text UX, serving, streaming, or RL rollout APIs.

## Diagnostics

Training live output is intentionally compact:

```text
step: 10     | loss: 9.247 | grad_norm: 1.155 | mfu: 4.95% | lr: 2.975e-04 | tps: 59,660 | total_time: 39.8s
```

The full metrics row is in `metrics/train.jsonl`.

MFU is only reported when Jaxtitan has both a FLOPs estimate and a known peak FLOPs value for the selected device. Throughput is based on explicit JAX synchronization points, not asynchronous dispatch alone.

## Distributed Runtime

The current runtime is replicated data parallel over the JAX `data` mesh axis:

- `jax.jit`
- `NamedSharding`
- `PartitionSpec`
- global batch arrays
- replicated train state
- single-process execution

`fsdp` and `tp` axis names are reserved but must have size `1` until those policies are implemented.

Multi-host support is intentionally blocked until host-side data partitioning, writer election, and resume semantics are designed and tested.

## Development Rules

Keep the repo narrow:

- Do not add second stacks for the same workflow.
- Do not add hidden fallbacks.
- Do not make local runs depend on W&B or a registry.
- Do not commit `data/` or `runs/` artifacts.
- Keep configs in `configs/`.
- Prefer explicit contracts and small runtime boundaries over broad trainer classes.

Useful development checks:

```bash
uv run pytest -q
git diff --check
rg -n "from __future__ import annotations|import research|from research" src/jaxtitan tests/jaxtitan
```

## Roadmap

Near-term work should improve research value without widening the surface:

- optimizer routing and non-AdamW backends
- larger real-run comparison configs
- profiler-driven performance work
- multi-host data/runtime design
- FSDP and tensor parallel sharding policies
- registry/scoring integration built from local artifacts

Custom kernels should wait for measured bottlenecks from real run artifacts.
