# Cloud Validation Queue

Purpose: track what must be validated on cloud GPUs and what local work must
finish before spending cloud time.

## 2026-06-19 [codex] Cloud validation is deferred

Context:

- Current local branch is `master`.
- Distributed semantics have grown enough that validating one partial slice on
  cloud would give limited confidence.
- We should wait until TP is complete enough and distributed Muon/routed-expert
  optimizer policy is coherent, then validate the full stack together.

Commands:

- No cloud commands run for this entry.
- Source: discussion plus current runbook state in:
  - `runbook/streams/tp_completion.md`
  - `runbook/streams/moe_parallelism.md`
  - `docs/missing_parallelism_schemes.md`

Artifacts:

- Current local semantic validation is recorded in `runbook/streams/tp_completion.md`.
- Latest relevant commits:
  - `9a046a4 Add RDEP and tensor parallel semantics`
  - `a02e7ff Add tracked agent runbook`
  - `3dd3180 Remove legacy registry-required workflow`

Result:

- Cloud validation is intentionally blocked for now.
- Do not launch cloud validation just for current partial TP.

Prerequisites before next cloud pass:

- TP sequence-parallel semantics implemented and locally tested.
- Expert tensor parallelism semantics implemented or explicitly deferred with
  clear unsupported-mode guards.
- TP-aware Muon/Dion policy decided and locally tested.
- Routed rank-3 expert optimizer policy decided for expert-region FSDP and
  expert TP.
- Cloud configs updated so they validate meaningful combinations rather than
  obsolete partial stacks.

Cloud validation matrix to run once unblocked:

- Dense decoder:
  - DDP + AdamW baseline.
  - TP + AdamW.
  - FSDP + TP + AdamW.
  - ZeRO-2 + TP + AdamW.
- Dense Trinity:
  - DDP + AdamW.
  - TP + AdamW.
  - FSDP + TP + AdamW.
- Trinity MoE:
  - DDP + EP/RDEP + AdamW.
  - Folded FSDP+EP + AdamW.
  - Expert-region FSDP + AdamW.
  - TP combined with EP/RDEP where supported.
  - Muon/Dion combinations only after optimizer policy is exact or explicitly
    guarded.

For every cloud run, record:

- commit hash;
- config path and run id;
- hardware summary from `jax.devices()`;
- data manifest path and hash;
- preflight output;
- final `run inspect`;
- latest checkpoint eval JSON;
- checkpoint sample JSON;
- profiling metadata if enabled;
- any failed command and exact remediation.

Next:

- Continue local TP completion work first. Update this file when a prerequisite
  is finished or when the cloud matrix changes.

## 2026-06-19 [codex] Added 4-GPU parallelism validation bundle

Context:

- Added concrete 4-GPU cloud configs for DDP, FSDP, ZeRO-2, TP, CP, TP+CP,
  FSDP+TP, ZeRO-2+TP, EP, TP+EP, CP+EP, RDEP, folded FSDP+EP, product
  FSDP+EP, and expert-region FSDP.
- Added a dedicated TinyStories cloud validation data config with a larger
  validation split to avoid preflight failures from undersized val tokens.
- TP configs use AdamW because Muon remains explicitly unsupported with TP.

Commands:

```bash
cd /home/veer/Master/projects/research
for cfg in configs/jaxtitan/cloud_4gpu_*_validation.toml; do
  uv run jaxtitan config check "$cfg"
done
```

Artifacts:

- Data config: `configs/data/tinystories_gpt2_cloud_validation.toml`.
- Run configs: `configs/jaxtitan/cloud_4gpu_*_validation.toml`.
- Operator doc: `docs/cloud_validation.md`.

Result:

- All new 4-GPU validation TOMLs passed `uv run jaxtitan config check`.
- No cloud runs were launched in this entry.

Next:

- On a 4x A100 80GB instance, prepare
  `data/tinystories_gpt2_cloud_validation/manifest.json` and run the matrix in
  `docs/cloud_validation.md`.

## 2026-06-19 [codex] 4-GPU parallelism validation passed

Context:

- Validated the 4-GPU matrix from commit `1fe5e96`.
- Hardware reported by JAX on the cloud host:
  - backend `gpu`
  - process count `1`
  - devices `cuda:0` through `cuda:3`
- Prepared `data/tinystories_gpt2_cloud_validation/manifest.json` on the cloud
  host from Hugging Face TinyStories:
  - total tokens `2,000,000`
  - train tokens `1,980,000`
  - validation tokens `20,000`
  - documents `9,355`
- The HF prepare command completed and wrote valid artifacts, then hit a Python
  finalization crash from HF/PyArrow thread cleanup. `data inspect` and
  `data check --verify-checksums` both passed afterward, so the prepared
  artifacts were usable.

Commands:

```bash
cd /root/research

uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_cloud_validation.toml
uv run jaxtitan data inspect data/tinystories_gpt2_cloud_validation/manifest.json \
  --tokenizer gpt2 \
  --verify-checksums \
  --seq-len 1024

for cfg in configs/jaxtitan/cloud_4gpu_*_validation.toml; do
  uv run jaxtitan config check "$cfg"
done
```

The validation loop ran preflight, train, inspect, and checkpoint eval for each
of:

- `cloud_4gpu_dense_ddp_adamw_validation`
- `cloud_4gpu_dense_tp_adamw_validation`
- `cloud_4gpu_dense_cp_adamw_validation`
- `cloud_4gpu_dense_fsdp_adamw_validation`
- `cloud_4gpu_dense_zero2_adamw_validation`
- `cloud_4gpu_dense_tp_cp_adamw_validation`
- `cloud_4gpu_dense_fsdp_tp_adamw_validation`
- `cloud_4gpu_dense_zero2_tp_adamw_validation`
- `cloud_4gpu_trinity_moe_ep_adamw_validation`
- `cloud_4gpu_trinity_moe_tp_ep_adamw_validation`
- `cloud_4gpu_trinity_moe_cp_ep_adamw_validation`
- `cloud_4gpu_trinity_moe_rdep_adamw_validation`
- `cloud_4gpu_trinity_moe_folded_fsdp_ep_muon_validation`
- `cloud_4gpu_trinity_moe_product_fsdp_ep_muon_validation`
- `cloud_4gpu_trinity_moe_expert_fsdp_adamw_validation`

Artifacts:

- Local result bundle:
  `cloud_results/jaxtitan_parallel_validation_2026-06-19/`
- Local tarball:
  `cloud_results/jaxtitan_parallel_validation_2026-06-19.tgz`
- Each run directory in the bundle includes:
  - `final.json`
  - `runtime.json`
  - `profiling.json`
  - `checkpoints_index.json`
  - `last_train.jsonl`
  - `profile_files.txt`
- Eval artifacts reference prepared manifest hash
  `9ff47c8e9ff3c3339ebda983c82ab11d7dbd6cd174c29787dad1104714e3686d`.

Result:

- All 15 validation runs completed.
- All 15 runs wrote completed profiling metadata with three trace files and
  traced step range `{start: 4, end: 5}`.
- Dense DDP/FSDP/ZeRO-2/TP/CP combinations had no zero-gradient or zero-update
  optimizer groups.
- MoE runs had exactly one zero-gradient and zero-update group:
  `moe_expert_bias:adamw`, with weight decay disabled. This is expected for
  the SME-BU expert bias state.
- Non-CP checkpoint sampling succeeded for dense FSDP, dense ZeRO-2,
  dense FSDP+TP, dense ZeRO-2+TP, MoE EP, MoE TP+EP, MoE RDEP,
  folded FSDP+EP+Muon, product FSDP+EP+Muon, and expert-region FSDP.
- CP checkpoint sampling failed with the intended guard:
  `checkpoint sampling is not supported for context-parallel runs until CP KV-cache support lands`.

Selected final eval losses:

- Dense DDP: `6.1657`
- Dense TP: `6.2833`
- Dense CP: `6.2624`
- Dense FSDP: `6.2599`
- Dense ZeRO-2: `6.2682`
- Dense TP+CP: `6.2660`
- Dense FSDP+TP: `6.2697`
- Dense ZeRO-2+TP: `6.2641`
- MoE EP: `8.4360`
- MoE TP+EP: `8.4375`
- MoE CP+EP: `8.4366`
- MoE RDEP: `8.4324`
- MoE folded FSDP+EP+Muon: `8.3953`
- MoE product FSDP+EP+Muon: `8.3904`
- MoE expert-region FSDP: `8.4460`

Notes:

- Tiny MoE validation runs are too short to treat router load balance as a
  quality result. Router dead expert counts were recorded and are useful only
  as diagnostics for these smoke runs.
- Runtime metadata for TP+CP reports TP sequence-parallel policy as enabled
  while CP is also enabled. The run behavior passed, but diagnostics should be
  clarified later so metadata reflects CP precedence over TP sequence-parallel
  activation when both axes are configured.

Next:

- This matrix is ready to cite as the first 4-GPU cloud validation pass for
  DDP, FSDP, ZeRO-2, TP, CP, EP, RDEP, expert-region FSDP, and folded/product
  FSDP+EP compositions.
- Follow-up work should prioritize distributed optimizer policy and Muon/Dion
  cleanup under these validated sharding layouts.
