# Jaxtitan Cloud Validation

These configs are for first multi-GPU correctness checks, not benchmark claims.
They intentionally do not point at the local PleIAs synth dataset. Prefer
preparing data from Hugging Face on the cloud instance:

```bash
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_smoke.toml
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --verify-checksums --seq-len 512
```

The older dense smoke configs still use:

```text
data/cloud_smoke_gpt2/manifest.json
```

For those configs, either prepare or copy a GPT-2-tokenized Jaxtitan manifest to
that path, or edit each config's `data.train_manifest` to the cloud-local
manifest path.

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
| `configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_adamw.toml` | 2-GPU EP dispatcher correctness with AdamW |
| `configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_muon.toml` | 2-GPU EP routed experts with per-expert Muon |
| `configs/jaxtitan/cloud_2xa100_ep_trinity_moe_fsdp_muon.toml` | 2-GPU folded FSDP+EP with dense Dion2 and routed expert Muon |
| `configs/jaxtitan/cloud_4gpu_ep_trinity_moe_ddp_muon.toml` | 4-GPU data+EP smoke |
| `configs/jaxtitan/cloud_4gpu_ep_trinity_moe_fsdp_muon.toml` | 4-GPU FSDP+EP Muon/Dion2 smoke |
| `configs/jaxtitan/cloud_4gpu_ep_trinity_moe_efsdp_adamw.toml` | 4-GPU EP plus expert-internal FSDP with AdamW |

The `cloud_2xa100_ep_*` configs are the preferred first EP checks on a two-GPU
instance. `cloud_2xa100_ep_trinity_moe_fsdp_muon.toml` uses folded FSDP+EP:
the `fsdp` axis shards dense state and also owns routed experts. The 4-GPU
FSDP+EP config keeps product-axis semantics with separate `fsdp` and `ep` axes.
The `cloud_4gpu_ep_trinity_moe_efsdp_adamw.toml` config adds `expert_fsdp=2`
to shard the internal routed expert matrix width; it intentionally uses AdamW
because exact Muon for internally sharded routed experts is a separate optimizer
task.

## Smoke Flow

Run one config at a time. Do not run eval, sample, or train commands in parallel
on the same GPU allocation.

```bash
cd /path/to/jaxtitan-repo
uv sync
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_smoke.toml
uv run jaxtitan data check data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --verify-checksums
uv run jaxtitan data check data/cloud_smoke_gpt2/manifest.json --tokenizer gpt2
uv run jaxtitan config check configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml
uv run jaxtitan run preflight configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/cloud_dense_ddp_adamw_smoke.toml
uv run jaxtitan run inspect runs/cloud_dense_ddp_adamw_smoke
uv run jaxtitan eval checkpoint runs/cloud_dense_ddp_adamw_smoke --checkpoint latest --json
uv run jaxtitan sample checkpoint runs/cloud_dense_ddp_adamw_smoke --checkpoint latest --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json
```

Repeat the same flow for the remaining configs after the first run is clean.

## Expert Parallel Flow

Run one EP config at a time. Do not run eval, sample, or train commands in
parallel on the same allocation.

```bash
cd /path/to/jaxtitan-repo
uv sync
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_smoke.toml
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --verify-checksums --seq-len 512

uv run jaxtitan config check configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_adamw.toml
uv run jaxtitan run preflight configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_adamw.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_adamw.toml
uv run jaxtitan run inspect runs/cloud_2xa100_ep_trinity_moe_ddp_adamw
uv run jaxtitan eval checkpoint runs/cloud_2xa100_ep_trinity_moe_ddp_adamw --checkpoint latest --json
uv run jaxtitan sample checkpoint runs/cloud_2xa100_ep_trinity_moe_ddp_adamw --checkpoint latest --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json

uv run jaxtitan config check configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_muon.toml
uv run jaxtitan run preflight configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_muon.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/cloud_2xa100_ep_trinity_moe_ddp_muon.toml
uv run jaxtitan run inspect runs/cloud_2xa100_ep_trinity_moe_ddp_muon
uv run jaxtitan eval checkpoint runs/cloud_2xa100_ep_trinity_moe_ddp_muon --checkpoint latest --json
uv run jaxtitan sample checkpoint runs/cloud_2xa100_ep_trinity_moe_ddp_muon --checkpoint latest --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json

uv run jaxtitan config check configs/jaxtitan/cloud_2xa100_ep_trinity_moe_fsdp_muon.toml
uv run jaxtitan run preflight configs/jaxtitan/cloud_2xa100_ep_trinity_moe_fsdp_muon.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/cloud_2xa100_ep_trinity_moe_fsdp_muon.toml
uv run jaxtitan run inspect runs/cloud_2xa100_ep_trinity_moe_fsdp_muon
uv run jaxtitan eval checkpoint runs/cloud_2xa100_ep_trinity_moe_fsdp_muon --checkpoint latest --json
uv run jaxtitan sample checkpoint runs/cloud_2xa100_ep_trinity_moe_fsdp_muon --checkpoint latest --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json
```

## What To Check

- `run preflight` reports the expected mesh, parallelism mode, optimizer route counts, and compile contract.
- `metrics/train.jsonl` has global scalar rows with timing, MFU, router diagnostics when MoE is active, and optimizer health groups.
- `summaries/final.json` mirrors the last train row's router and optimizer health.
- `checkpoints/index.json` retains latest and best-validation checkpoints.
- `run inspect` shows readable parallelism, data pipeline, router health, optimizer health, latest checkpoint, and best checkpoint.
- `eval checkpoint` restores the checkpoint cleanly and writes `evals/checkpoints/<step>.json`.
- `sample checkpoint` restores through the inference boundary and writes `samples/checkpoints/<step>.jsonl`.

For EP configs, additionally check:

- `run preflight` reports `expert_parallel=true`; product-axis configs report
  `ep=2`, while folded FSDP+EP reports `expert_parallel_axis=fsdp`.
- `diagnostics/runtime.json` includes `expert_parallel_policy` with dispatcher backend `all_to_all`.
- `optimizer` route counts show routed experts using per-expert Muon in the Muon config.
- `metrics/train.jsonl` includes `moe_router_layers`, global router health, and optimizer groups for routed experts.
- `diagnostics/profiling.json` has `status="completed"` and a Perfetto trace file.

## Cloud Pass Order

1. `cloud_dense_ddp_adamw_smoke`
2. `cloud_dense_ddp_muon_smoke`
3. `cloud_dense_fsdp_adamw_smoke`
4. `cloud_dense_zero2_adamw_smoke`
5. `cloud_dense_fsdp_muon_auto_dion2_smoke`
6. `cloud_dense_zero2_muon_auto_dion2_smoke`
7. `cloud_trinity_dense_ddp_adamw_smoke`
8. `cloud_trinity_moe_smebu_ddp_muon_smoke`
9. `cloud_2xa100_ep_trinity_moe_ddp_adamw`
10. `cloud_2xa100_ep_trinity_moe_ddp_muon`
11. `cloud_2xa100_ep_trinity_moe_fsdp_muon`
12. `cloud_4gpu_ep_trinity_moe_ddp_muon` when four GPUs are available
13. `cloud_4gpu_ep_trinity_moe_fsdp_muon` when four GPUs are available
14. `cloud_4gpu_ep_trinity_moe_efsdp_adamw` when four GPUs are available

Only treat performance numbers as real after a dedicated benchmark run. These
smokes are for correctness, artifact readability, resume/eval/sample restore,
and distributed optimizer sanity.
