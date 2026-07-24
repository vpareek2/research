# Cloud Validation Queue

Purpose: track what must be validated on cloud GPUs and what local work must
finish before spending cloud time.

## 2026-07-24 [codex] M2 distributed Muon four-GPU queue is ready

Context:

- Local fake-device acceptance for explicit duplicated/distributed Muon TP
  modes is complete. The remaining gate is a same-node correctness and
  performance comparison, not more local algorithm work.
- Prepared four matched 64-step pairs: dense TP, dense FSDP+TP, dense
  ZeRO2+TP, and Trinity TP+EP. Each pair differs only in `muon_tp_mode` and run
  ID.
- No cloud command was run in this step.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
bash -n scripts/jaxtitan/cloud_dist_muon_m2_matrix.sh
uv run pytest -q tests/jaxtitan/test_dist_muon_m2_analysis.py
```

Cloud launch:

```bash
cd ~/research
git fetch origin
git checkout codex/distributed-muon-mode
git pull --ff-only
tmux new -s dist-muon-m2
scripts/jaxtitan/cloud_dist_muon_m2_matrix.sh --overwrite
```

Artifacts:

- Runner: `scripts/jaxtitan/cloud_dist_muon_m2_matrix.sh`.
- Analyzer: `scripts/jaxtitan/analyze_dist_muon_m2_results.py`.
- Expected lightweight output:
  `cloud_results/dist_muon_m2_YYYYMMDDTHHMMSSZ_lightweight.tgz` plus
  `.sha256`.
- Full trace trees stay under each run directory and can be analyzed on the
  instance or transferred selectively.

Result:

- All eight configs pass `jaxtitan config check`.
- Complete local suite: `691 passed, 1 skipped`.
- No GPU correctness or speedup claim exists yet.

Next:

- Prefer exactly four H100 80GB SXM/NVLink GPUs. Four H100 PCIe or four A100
  40/80GB GPUs remain useful for correctness, but performance comparison must
  be interpreted as hardware-specific.
- Copy and checksum-verify the lightweight archive before terminating the
  instance. Preserve full traces until the comparison is reviewed.

## 2026-07-21 [codex] M1 MoE H100 queue is scripted

Context:

- The next cloud allocation is for PR `#16` M1 MoE native ragged transport
  acceptance, not a new broad profiling sweep.
- Added a single runner that prepares/verifies data, runs seven short
  correctness configs followed by four 64-step profile configs, and packages
  local artifacts before the instance is terminated.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
bash -n scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh
uv run python -m py_compile scripts/jaxtitan/analyze_moe_m1_h100_results.py
```

Cloud launch:

```bash
cd ~/research
git fetch origin
git checkout codex/moe-expert-major
git pull --ff-only
tmux new -s moe-m1-h100
scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh --overwrite
```

Artifacts:

- Expected cloud output: `cloud_results/moe_m1_h100_*.tgz` and matching
  `.sha256` file.
- The archive includes hardware/topology/JAX device logs, per-run train logs,
  HLO dumps, inspect/eval/sample outputs, and the selected run directories.

Result:

- Queue is ready for the next four-H100 allocation.
- No cloud run was launched during this prep step.

Next:

- Prefer four H100 80GB SXM/NVLink for apples-to-apples comparison with
  `cloud_results/profile64_h100_sxm_2026-07-20.tgz`; four H100 PCIe is still
  useful for correctness but weaker for performance acceptance.

## 2026-07-20 [codex] Four-H100 profiling capture complete

Context:

- Ran the prepared 15-config profile matrix on four H100 80GB HBM3 GPUs with
  full `NV18` connectivity.
- Copied and checksum-verified the complete lightweight evidence bundle before
  terminating the allocation.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
sha256sum -c cloud_results/profile64_h100_sxm_2026-07-20.tgz.sha256
```

Artifacts:

- `cloud_results/profile64_h100_sxm_2026-07-20.tgz`
- SHA256:
  `87be3f018afad28f94ce6101e3bf11ef185bea0495694c927b44d69f27dffde5`
- 15 Perfetto, 15 XPlane, and 15 compressed JAX traces plus metrics,
  summaries, diagnostics, configs, events, HLO, and provenance.

Result:

- All 15 runs passed; no config, preflight, train, or trace-capture failure.
- Scalar/trace/HLO triage is recorded in
  `runbook/streams/performance_optimization.md`.
- MoE expert execution is the first optimization target, followed by exact
  distributed Muon and baseline TP/FSDP collective reduction.

Next:

- No additional cloud allocation is needed until a milestone implementation
  has local correctness coverage and a narrow before/after profile matrix.

## 2026-07-19 [codex] Four-H100 profiling matrix prepared

Context:

- Prepared a profiler-driven performance matrix after distributed correctness
  completed. These are performance measurements, not another correctness
  acceptance campaign.
- Use one four-GPU node for the entire matrix. Prefer four H100 80GB SXM GPUs
  with NVLink when the provider identifies that topology; otherwise use four
  H100 80GB PCIe GPUs matching the July acceptance hardware.
- The retained July H100 bundle contains scalar timing/profiling metadata but
  not the trace payloads. The next allocation must be kept until the complete
  `profiles/` directories have been archived, copied, and checksum-verified.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
for cfg in configs/jaxtitan/cloud_4gpu_profile64*.toml; do
  uv run jaxtitan config check "$cfg"
done

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
    tests/jaxtitan/test_config.py \
    tests/jaxtitan/test_mesh.py \
    tests/jaxtitan/test_optim.py \
    tests/jaxtitan/test_preflight.py \
    tests/jaxtitan/test_runtime_diagnostics.py
```

Primary dense matrix:

```text
configs/jaxtitan/cloud_4gpu_profile64_dense_ddp_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_ddp_muon.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_tp_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_tp_muon.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_fsdp_tp_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_fsdp_tp_muon.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_zero2_tp_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_zero2_tp_muon.toml
```

Representative MoE matrix:

```text
configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_ep_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_ep_muon.toml
configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_muon.toml
```

Optional large-dense scale guard:

```text
configs/jaxtitan/cloud_4gpu_profile64_dense_large_tp_adamw.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_large_tp_muon.toml
configs/jaxtitan/cloud_4gpu_profile64_dense_large_fsdp_tp_muon.toml
```

Artifacts:

- All runs execute 64 optimizer steps and trace steady-state steps 12-15.
- Dense primary runs use the same 1024-wide, 12-layer decoder, sequence length
  1024, global batch 4, and 262144 target tokens.
- Representative MoE runs use the 1024-wide, 12-layer, eight-expert Trinity
  model rather than the two-layer correctness model.
- Large-dense guards use the existing 2048-wide, 24-layer profile model.
- Required capture includes resolved configs, metrics, events, summaries,
  diagnostics, complete `profiles/` trees, hardware/topology output, launcher
  logs, and HLO collective summaries for the Muon TP, FSDP+TP, ZeRO-2+TP, and
  TP+EP routes.

Result:

- All 15 configs pass `jaxtitan config check` locally.
- Targeted four-device CPU suite: `225 passed`.
- Periodic eval and checkpoint work is outside the trace window; both are due
  at step 64.
- The AdamW/Muon pairs isolate optimizer cost within each parallel layout.

Next:

- On the allocated four-H100 node, record `nvidia-smi topo -m`, driver/runtime,
  clocks, and `jax.devices()` before running the matrix.
- Run config check, data checksum validation, and preflight before each train
  command. Do not terminate the node until every trace payload is present in a
  copied archive and both sides report the same SHA256.
- Analyze the traces before choosing distributed-Muon, TP/loss collectives, or
  MoE dispatch/grouped GEMM as the first optimization target.

## 2026-07-19 [codex] Distributed correctness queue is clear

Context:

- Reconciled the queue after PR `#12` merged as `a76b360`.
- The older entries below are historical snapshots; their distributed-Muon and
  TP prerequisites have since completed.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
git show --stat --oneline a76b360
sha256sum cloud_results/distributed_muon_h100_acceptance_2026-07-19.tgz
```

Artifacts:

- Rank-2 distributed-Muon acceptance bundle:
  `cloud_results/distributed_muon_h100_acceptance_2026-07-19.tgz`.
- SHA256:
  `016bec6a15d01cc1de1db8ba67e78c7326ffd0030955c6c36cf0ea43666ceef9`.
- General parallelism bundle:
  `cloud_results/jaxtitan_parallel_validation_2026-06-19.tgz`.

Result:

- The 4-GPU AdamW parallelism matrix, four-run Muon acceptance matrix, and
  three-run 64-step Muon stress matrix are complete.
- There are no pending correctness runs for currently supported dense TP,
  FSDP+TP, ZeRO-2+TP, EP, or TP+EP layouts.
- Future cloud allocations should be tied to a concrete performance hypothesis
  or a new mechanism such as matrix-axis expert tensor parallelism.

Next:

- Analyze existing profiles before launching more cloud runs.
- Do not spend cloud time repeating the completed correctness matrices unless
  a shared hot-path change invalidates them.

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
