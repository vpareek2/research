# Jaxtitan Cloud Validation

These configs are for first multi-GPU correctness checks, not benchmark claims.
They intentionally do not point at the local PleIAs synth dataset. Prepare or copy
a GPT-2-tokenized Jaxtitan manifest on the cloud instance and either place it at:

```text
data/cloud_smoke_gpt2/manifest.json
```

or edit each config's `data.train_manifest` to the cloud-local manifest path.

## Config Matrix

| Config | Purpose |
| --- | --- |
| `configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml` | DDP dense AdamW baseline |
| `configs/jaxtitan/cloud_dense_ddp_muon_smoke.toml` | DDP dense Muon baseline |
| `configs/jaxtitan/cloud_trinity_dense_ddp_adamw_smoke.toml` | Trinity dense recipe sanity check |
| `configs/jaxtitan/cloud_trinity_moe_smebu_ddp_muon_smoke.toml` | Trinity MoE, SMEBU, router and optimizer diagnostics |
| `configs/jaxtitan/cloud_dense_fsdp_adamw_smoke.toml` | FSDP AdamW sharded-state path |
| `configs/jaxtitan/cloud_dense_zero2_adamw_smoke.toml` | ZeRO-2 AdamW sharded optimizer path |
| `configs/jaxtitan/cloud_dense_fsdp_muon_auto_dion2_smoke.toml` | FSDP Muon intent with auto-Dion2 matrix routes |
| `configs/jaxtitan/cloud_dense_zero2_muon_auto_dion2_smoke.toml` | ZeRO-2 Muon intent with auto-Dion2 matrix routes |

The distributed configs assume four visible GPUs. For a different GPU count,
adjust `[mesh] axis_sizes` and keep `training.global_batch_size` divisible by
the `data` axis.

## Smoke Flow

Run one config at a time. Do not run eval, sample, or train commands in parallel
on the same GPU allocation.

```bash
cd /path/to/jaxtitan-repo
uv sync
uv run jaxtitan data check data/cloud_smoke_gpt2/manifest.json --tokenizer gpt2
uv run jaxtitan config check configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml
uv run jaxtitan run preflight configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml
uv run jaxtitan run inspect runs/cloud_dense_ddp_adamw_smoke
uv run jaxtitan eval checkpoint runs/cloud_dense_ddp_adamw_smoke --checkpoint latest --json
uv run jaxtitan sample checkpoint runs/cloud_dense_ddp_adamw_smoke --checkpoint latest --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json
```

Repeat the same flow for the remaining configs after the first run is clean.

## What To Check

- `run preflight` reports the expected mesh, parallelism mode, optimizer route counts, and compile contract.
- `metrics/train.jsonl` has global scalar rows with timing, MFU, router diagnostics when MoE is active, and optimizer health groups.
- `summaries/final.json` mirrors the last train row's router and optimizer health.
- `checkpoints/index.json` retains latest and best-validation checkpoints.
- `run inspect` shows readable parallelism, data pipeline, router health, optimizer health, latest checkpoint, and best checkpoint.
- `eval checkpoint` restores the checkpoint cleanly and writes `evals/checkpoints/<step>.json`.
- `sample checkpoint` restores through the inference boundary and writes `samples/checkpoints/<step>.jsonl`.

## Cloud Pass Order

1. `cloud_dense_ddp_adamw_smoke`
2. `cloud_dense_ddp_muon_smoke`
3. `cloud_dense_fsdp_adamw_smoke`
4. `cloud_dense_zero2_adamw_smoke`
5. `cloud_dense_fsdp_muon_auto_dion2_smoke`
6. `cloud_dense_zero2_muon_auto_dion2_smoke`
7. `cloud_trinity_dense_ddp_adamw_smoke`
8. `cloud_trinity_moe_smebu_ddp_muon_smoke`

Only treat performance numbers as real after a dedicated benchmark run. These
smokes are for correctness, artifact readability, resume/eval/sample restore,
and distributed optimizer sanity.
