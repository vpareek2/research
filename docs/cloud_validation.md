# Jaxtitan Cloud Validation

These configs are for first multi-GPU correctness checks, not benchmark claims.
They intentionally do not point at the local PleIAs synth dataset. Prefer
preparing data from Hugging Face on the cloud instance:

```bash
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_cloud_validation.toml
uv run jaxtitan data inspect data/tinystories_gpt2_cloud_validation/manifest.json --tokenizer gpt2 --verify-checksums --seq-len 1024
uv run jaxtitan data check data/tinystories_gpt2_cloud_validation/manifest.json --tokenizer gpt2 --verify-checksums
```

The older dense smoke configs still use:

```text
data/cloud_smoke_gpt2/manifest.json
```

For those configs, either prepare or copy a GPT-2-tokenized Jaxtitan manifest to
that path, or edit each config's `data.train_manifest` to the cloud-local
manifest path.

## 4-GPU Parallelism Validation Bundle

This is the current primary cloud validation matrix. It is designed for a
single-node 4x A100 80GB allocation and validates composed sharding semantics
before distributed optimizer work.

| Config | Purpose |
| --- | --- |
| `configs/jaxtitan/cloud_4gpu_dense_ddp_adamw_validation.toml` | data-parallel AdamW baseline |
| `configs/jaxtitan/cloud_4gpu_dense_fsdp_adamw_validation.toml` | dense FSDP AdamW |
| `configs/jaxtitan/cloud_4gpu_dense_zero2_adamw_validation.toml` | dense ZeRO-2 AdamW |
| `configs/jaxtitan/cloud_4gpu_dense_tp_adamw_validation.toml` | dense tensor parallelism |
| `configs/jaxtitan/cloud_4gpu_dense_tp_muon_validation.toml` | dense tensor parallelism with exact-reference Muon |
| `configs/jaxtitan/cloud_4gpu_dense_cp_adamw_validation.toml` | dense context parallelism |
| `configs/jaxtitan/cloud_4gpu_dense_tp_cp_adamw_validation.toml` | dense TP+CP composition |
| `configs/jaxtitan/cloud_4gpu_dense_fsdp_tp_adamw_validation.toml` | dense FSDP+TP composition |
| `configs/jaxtitan/cloud_4gpu_dense_fsdp_tp_muon_validation.toml` | dense FSDP+TP with exact-reference Muon |
| `configs/jaxtitan/cloud_4gpu_dense_zero2_tp_adamw_validation.toml` | dense ZeRO-2+TP composition |
| `configs/jaxtitan/cloud_4gpu_dense_zero2_tp_muon_validation.toml` | dense ZeRO-2+TP with exact-reference Muon |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_ep_adamw_validation.toml` | Trinity MoE expert parallelism |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_adamw_validation.toml` | shared-expert TP plus routed-expert EP |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_muon_validation.toml` | shared-expert TP plus routed-expert EP with Muon intent |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_cp_ep_adamw_validation.toml` | context parallelism plus routed-expert EP |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_rdep_adamw_validation.toml` | data-axis RDEP |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_folded_fsdp_ep_muon_validation.toml` | folded FSDP+EP with Muon intent |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_product_fsdp_ep_muon_validation.toml` | product-axis FSDP+EP with Muon intent |
| `configs/jaxtitan/cloud_4gpu_trinity_moe_expert_fsdp_adamw_validation.toml` | EP plus expert-internal FSDP |

TP AdamW configs remain the baseline. Muon under tensor parallelism routes
rank-2 hidden matrices through the exact reference `dist_muon_exact` backend;
the TP+Muon configs validate correctness and artifact shape, not representative
throughput.

Run each config with:

```bash
cfg=configs/jaxtitan/<config>.toml
run=<run_id>

uv run jaxtitan config check "$cfg"
uv run jaxtitan run preflight "$cfg"
uv run jaxtitan run train --overwrite "$cfg"
uv run jaxtitan run inspect "runs/$run"
uv run jaxtitan eval checkpoint "runs/$run" --checkpoint latest --json
```

Checkpoint sampling must succeed for all runs, including CP runs with
CP-sharded KV cache:

```bash
uv run jaxtitan sample checkpoint "runs/$run" \
  --checkpoint latest \
  --prompt-ids "15496,11" \
  --max-new-tokens 8 \
  --top-k 1 \
  --json
```

For CP runs, checkpoint eval must still succeed. Sampling restores the
checkpoint with a CP-sharded KV cache and pads prompt/cache lengths internally
to the CP axis multiple.

Post-run checks:

```bash
jq '.parallelism, .sharding, .runtime' "runs/$run/diagnostics/runtime.json"
jq '{status, traced_step_range, trace_files}' "runs/$run/diagnostics/profiling.json"
tail -n 1 "runs/$run/metrics/train.jsonl" | jq '{
  step,
  loss,
  total_loss,
  grad_norm,
  optimizer_route_backend_counts,
  optimizer_groups_with_zero_grad,
  optimizer_groups_with_zero_update,
  router_dead_experts_count,
  router_mean_load_cv,
  router_mean_importance_cv
}'
jq '{status, step, tokens_seen, final_eval_loss}' "runs/$run/summaries/final.json"
jq '.checkpoints' "runs/$run/checkpoints/index.json"
```

Collect lightweight result artifacts without checkpoints:

```bash
cd ~/research
stamp=$(date +%F)
out=/tmp/jaxtitan_parallel_validation_$stamp
mkdir -p "$out"

for run in \
  cloud_4gpu_dense_ddp_adamw_validation \
  cloud_4gpu_dense_fsdp_adamw_validation \
  cloud_4gpu_dense_zero2_adamw_validation \
  cloud_4gpu_dense_tp_adamw_validation \
  cloud_4gpu_dense_tp_muon_validation \
  cloud_4gpu_dense_cp_adamw_validation \
  cloud_4gpu_dense_tp_cp_adamw_validation \
  cloud_4gpu_dense_fsdp_tp_adamw_validation \
  cloud_4gpu_dense_fsdp_tp_muon_validation \
  cloud_4gpu_dense_zero2_tp_adamw_validation \
  cloud_4gpu_dense_zero2_tp_muon_validation \
  cloud_4gpu_trinity_moe_ep_adamw_validation \
  cloud_4gpu_trinity_moe_tp_ep_adamw_validation \
  cloud_4gpu_trinity_moe_tp_ep_muon_validation \
  cloud_4gpu_trinity_moe_cp_ep_adamw_validation \
  cloud_4gpu_trinity_moe_rdep_adamw_validation \
  cloud_4gpu_trinity_moe_folded_fsdp_ep_muon_validation \
  cloud_4gpu_trinity_moe_product_fsdp_ep_muon_validation \
  cloud_4gpu_trinity_moe_expert_fsdp_adamw_validation
do
  mkdir -p "$out/$run"
  cp "runs/$run/summaries/final.json" "$out/$run/final.json"
  cp "runs/$run/diagnostics/runtime.json" "$out/$run/runtime.json"
  cp "runs/$run/diagnostics/profiling.json" "$out/$run/profiling.json"
  cp "runs/$run/checkpoints/index.json" "$out/$run/checkpoints_index.json"
  tail -n 1 "runs/$run/metrics/train.jsonl" > "$out/$run/last_train.jsonl"
  find "runs/$run/profiles" -type f 2>/dev/null | sort > "$out/$run/profile_files.txt"
done

tar -czf "/tmp/jaxtitan_parallel_validation_$stamp.tgz" -C /tmp "jaxtitan_parallel_validation_$stamp"
ls -lh "/tmp/jaxtitan_parallel_validation_$stamp.tgz"
```

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
15. `cloud_4gpu_dense_tp_muon_validation`
16. `cloud_4gpu_dense_fsdp_tp_muon_validation`
17. `cloud_4gpu_dense_zero2_tp_muon_validation`
18. `cloud_4gpu_trinity_moe_tp_ep_muon_validation`

Only treat performance numbers as real after a dedicated benchmark run. These
smokes are for correctness, artifact readability, resume/eval/sample restore,
and distributed optimizer sanity.

## Acceptance Criteria

A validation run passes only if:

- `config check`, `preflight`, `train`, `inspect`, and checkpoint eval all succeed.
- Checkpoint sampling succeeds, including CP runs with the CP-sharded KV-cache
  restore path.
- `diagnostics/runtime.json` reports the expected mesh axes, parallelism flags,
  and sharding policies.
- `diagnostics/profiling.json` has `status="completed"` and at least one
  Perfetto trace file.
- Final train/eval losses are finite and no NaNs appear in metrics.
- Optimizer zero-grad/zero-update groups are zero except expected non-gradient
  state such as `moe_expert_bias`.
- Route counts match intent: TP AdamW configs use AdamW only; TP Muon configs
  route TP-sharded matrix leaves to `dist_muon_exact`; FSDP/ZeRO Muon configs
  auto-route matrix leaves to Dion2; EP Muon configs show routed/shared Muon
  routes where expected.
- MoE runs emit `moe_router_layers`; dead experts and load CV are recorded but
  are not hard failures for tiny smokes.
