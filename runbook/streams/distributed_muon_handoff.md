# Distributed Muon Handoff

Purpose: preserve the evidence, correction status, and remaining GPU gates for
the composed-layout Muon route.

## 2026-07-18 [codex] Production correction implemented from master

Context:

- Created `codex/distributed-muon-fix` from `master` commit `c01a5fa` and
  ported only the RCA-proven correction and durable diagnostics.
- The correction captures each selected leaf's static `NamedSharding` during
  optimizer construction. It averages non-data model axes omitted from the
  leaf partition spec at the gradient, momentum, update, and parameter
  boundaries.
- TP-sharded rank-2 matrices reconstruct the full logical matrix for Muon and
  return update/momentum to their declared shardings. Rank-3 routed expert
  matrices remain local per-expert Muon and do not claim a distributed exact
  contract.
- ZeRO-2 model leaves omit `fsdp` placement while optimizer state retains its
  intended FSDP sharding.
- Non-finite optimizer groups now write the last metrics row and an actionable
  event, then terminate before eval or checkpoint creation.

Commands:

```bash
cd /home/veer/Master/projects/research

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_optim.py tests/jaxtitan/test_train_step.py

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_runtime_training.py \
    -k 'writes_artifacts_metrics_and_summary or stops_before_checkpoint_when_optimizer_group_is_nonfinite'

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_optim.py \
    tests/jaxtitan/test_train_step.py \
    tests/jaxtitan/test_preflight.py \
    tests/jaxtitan/test_runtime_training.py \
    tests/jaxtitan/test_resume_compat.py \
    tests/jaxtitan/test_checkpoints.py

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q

for cfg in \
  configs/jaxtitan/cloud_4gpu_dense_tp_muon_validation.toml \
  configs/jaxtitan/cloud_4gpu_dense_fsdp_tp_muon_validation.toml \
  configs/jaxtitan/cloud_4gpu_dense_zero2_tp_muon_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_muon_validation.toml; do
  uv run jaxtitan config check "$cfg"
done

git diff --check
```

Artifacts:

- RCA commit: `18ea497` on `codex/distributed-muon-rca`.
- Fix base: `c01a5fa` on `master`.
- Fix branch: `codex/distributed-muon-fix`.
- Production paths:
  - `src/jaxtitan/optim/muon.py`
  - `src/jaxtitan/optim/build.py`
  - `src/jaxtitan/steps/train.py`
  - `src/jaxtitan/mesh/sharding.py`
  - `src/jaxtitan/runtime/training.py`

Result:

- Optimizer/train-step fake-device suite: `77 passed` before adding the second
  row/column parameterization; both final exact-equivalence cases passed in a
  targeted rerun.
- Runtime artifact and fail-fast tests: `2 passed`.
- Targeted optimizer/training/runtime/resume/checkpoint suite: `232 passed`.
- Full fake-device suite: `616 passed, 1 skipped`.
- All four unchanged 17-step H100 validation configs passed `config check`.
- Five-step FSDP+TP, ZeRO-2+TP, and TP+EP model tests remained finite and
  physically replica-identical.
- Five-step identical-input distributed Muon matched replicated Muon for both
  row- and column-sharded matrices.
- Runtime policy intentionally reports `exact=false` and
  `correctness_status=local_gates_passed_h100_pending` until the unchanged GPU
  acceptance matrix passes.

Next:

- Run the original four 17-step configs on one four-H100 node, then 64-step
  stress validations for FSDP+TP, ZeRO-2+TP, and TP+EP.
- Only after those gates pass, set the policy's exact claim, commit/push the
  fix branch, and open the `no-run-required` PR.

## 2026-07-18 [codex] RCA reproduced replica drift and confirmed synchronization

Context:

- Added disposable probes on `codex/distributed-muon-rca` for physical replica
  buffers and every Muon stage of each attention K/V/O leaf.
- Compared the reference route with an intervention that captures static
  shardings and averages non-data model replicas omitted from the partition
  spec.

Commands:

```bash
cd /home/veer/Master/projects/research

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_distributed_muon_rca.py tests/jaxtitan/test_optim.py

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run python tests/jaxtitan/distributed_muon_rca_probe.py \
    --rows 1024 --columns 128 --steps 5 \
    --output cloud_results/distributed_muon_rca_2026-07-18/synthetic_cpu_1024x128.json
```

Artifacts:

- `cloud_results/distributed_muon_rca_2026-07-18/synthetic_cpu_1024x128.json`
  SHA256 `4de47267fbbdb420d1386cfb02050a11fd2b2feb80b0e231f53c0c0df5dcf071`.
- Unsynchronized TP+EP trace:
  `cloud_results/distributed_muon_rca_2026-07-18/moe_tp_ep_update_trace_cpu.json`.
- Synchronized TP+EP trace SHA256
  `e1ac88358e68e771b3a46ed11e5a6c543c6e71edd66714171fd5de17783834b8`.
- Synchronized FSDP+TP trace SHA256
  `192d29cc3945049e30871f5b954b4f386283740f96a8ef47e0dbb34d810c31bd`.
- Synchronized ZeRO-2+TP trace SHA256
  `581233d0fb99c3ef277e6904443fd7048a786d29a228ad58fef58bd9f573e1d0`.

Result:

- Identical synthetic `1024x128` inputs matched TP across composed topologies
  within `8.2e-8`; matrix math alone did not reproduce the failure.
- Real TP+EP produced a step-1 K update replica difference near `1e-7`, which
  grew to `0.0024`, `0.0106`, and `0.0656` at steps 2-4.
- Dynamic tracer `.sharding` inspection inside jitted Optax did not expose a
  `NamedSharding`; the reference route's intended replica handling was a no-op.
- Static build-time shardings plus `pmean` made K/V/O updates, momentum, and
  parameters exactly replica-identical through step 4 for TP+EP, FSDP+TP, and
  ZeRO-2+TP.

Next:

- Keep the intrusive probes confined to the RCA branch.
- Validate the clean production port on fake devices and four H100s.

## 2026-07-18 [codex] Four-H100 reference-route matrix completed but failed correctness

Context:

- Compared `master` commit `c01a5fa` with frozen reference commit `87ca2da`
  using the same four Muon configs, data, seed, and four-H100 node.
- All eight run lifecycles completed, but completion status hid non-finite
  optimizer state in the composed layouts.

Commands:

```bash
cd /home/veer/Master/projects/research
sha256sum cloud_results/muon_validation_h100_2026-07-19.tgz
tar -tzf cloud_results/muon_validation_h100_2026-07-19.tgz
```

Artifacts:

- `cloud_results/muon_validation_h100_2026-07-19.tgz`
- SHA256 `234b252463a9487d2d2e0fd115a66161e9ad0b37b7ffdee368350a468e4f271a`.
- Compared commits: `c01a5fa` and `87ca2da`.

Result:

- Dense TP Muon stayed finite on both commits.
- Reference FSDP+TP and ZeRO-2+TP first became non-finite in `attention_k` at
  step 4.
- Reference TP+EP first became non-finite in `attention_k` at step 3, while
  the simultaneous `attention_v` norm was about `2.9e15`.
- Every affected K/V leaf used `P(None, 'tp')`; the additional `fsdp` or `ep`
  mesh axis was replicated and absent from the route topology.

Next:

- Preserve `codex/distributed-muon-reference` unchanged as comparison evidence.
- Do not treat low LR, zero weight decay, or FP32 Newton-Schulz alone as fixes;
  earlier runs showed they only masked or delayed corruption.
