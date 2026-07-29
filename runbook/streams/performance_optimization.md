# Performance Optimization

Purpose: turn measured Jaxtitan profiles into an ordered optimization program
without weakening correctness or adding speculative kernel surfaces.

## 2026-07-29 [codex] Gram Newton-Schulz benchmark candidate implemented

Context:

- Added Gram Newton-Schulz as a benchmark-only orthogonalization candidate on
  `codex/distributed-muon-gram-recurrence-bench`, stacked on refreshed PR
  `#21`. The production `shape_topology_v1` selector, TOML surface, runtime
  metadata, and resume fingerprint remain unchanged.
- The primary candidate accumulates the Gram recurrence with one restart
  before iteration index `2`. A no-restart variant exists only for focused
  local diagnostics. Square benchmark matrices remain standard-only.
- Transport remains independent: duplicated, direct, right/large-Gram, and
  exchange candidates retain their existing placement and collective
  contracts. Bucket compatibility now also includes orthogonalization,
  restart schedule, and expected Gram-reduction count.
- Candidate schema version `2` records transport plus orthogonalization,
  restart points, collective-volume model, optimized-HLO contract, memory,
  compile time, numerical errors, and timing. Archived candidates without
  these fields normalize to standard Newton-Schulz.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
    tests/jaxtitan/test_profile_bench.py \
    tests/jaxtitan/test_dist_muon_leaf_bench_analysis.py \
    tests/jaxtitan/test_optim.py -x

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan -x

uv run python scripts/jaxtitan/replay_dist_muon_shape_policy.py \
  --json-out /tmp/dist_muon_shape_policy_replay_gram_bench.json

uv run python scripts/jaxtitan/analyze_dist_muon_leaf_bench.py \
  cloud_results/dist_muon_leaf_bench_20260724T215030Z/benchmark/benchmark.json \
  --json-out /tmp/dist_muon_leaf_bench_schema1_compat.json

uv run python -m py_compile \
  src/jaxtitan/optim/muon.py \
  src/jaxtitan/optim/build.py \
  src/jaxtitan/runtime/muon_bench.py \
  src/jaxtitan/runtime/profile_bench.py \
  scripts/jaxtitan/analyze_dist_muon_leaf_bench.py
bash -n scripts/jaxtitan/cloud_dist_muon_leaf_bench.sh
git diff --check
```

Artifacts:

- Implementation commit: `7debd09`.
- Draft PR: `#22`, stacked on PR `#21`, labeled `no-run-required`.
- Benchmark runner: `scripts/jaxtitan/cloud_dist_muon_leaf_bench.sh`.
- Local policy replay:
  `/tmp/dist_muon_shape_policy_replay_gram_bench.json`.
- Archived candidate-schema compatibility report:
  `/tmp/dist_muon_leaf_bench_schema1_compat.json`.
- No new GPU artifact or performance claim exists.

Result:

- Standard distributed HLO retains one packed FP32 norm plus five packed Gram
  reductions. Restart-at-2 recurrence emits one norm plus two Gram reductions;
  the local no-restart diagnostic emits one norm plus one Gram reduction.
- Duplicated recurrence emits one BF16 logical-matrix gather and no Gram
  collective. Exchange emits one ordered forward/reverse all-to-all pair per
  leaf; right-Gram metadata is confined to large-Gram candidates.
- Float32 algebraic, BF16 finite/deterministic, five-step optimizer,
  sharding, replica, weight-decay, and bucket-contract tests pass across TP4,
  FSDP2xTP2, ZeRO2xTP2, and TP2xEP2.
- Full four-device Jaxtitan suite: `817 passed, 1 skipped`.
- Frozen selector replay: `63/63`, `overall_gate=True`,
  `performance_claim=false`.
- Archived 76-case benchmark compatibility: `overall_gate=True`.
- Python compilation, shell syntax, and `git diff --check` pass.

Next:

- Complete PR `#21` smoke/profile acceptance first. After it merges, refresh
  this branch from `master` without rewriting history and run the expanded
  76-case recurrence benchmark on four H100s. Keep this PR draft until the
  checksum-copied HLO/timing artifact proves or rejects recurrence promotion.

## 2026-07-29 [codex] Distributed projection dtype fix refreshed

Context:

- Refreshed PR `#17` by merging current `master` at `eaa81e8` without
  rewriting the published branch.
- The shared TP/CP projection helper had bypassed NNX dtype promotion, causing
  BF16 activations with FP32 master weights to produce FP32 outputs.
- The helper now reuses each NNX linear's `promote_dtype`, `dot_general`,
  precision, and preferred-element-type contract. KV-cache writes explicitly
  convert K/V values to the declared cache dtype before scatter.
- No precision abstraction, TOML setting, or GPU performance claim was added.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
    tests/jaxtitan/test_model.py::test_distributed_model_preserves_bfloat16_compute_dtype \
    tests/jaxtitan/test_infer.py::test_tensor_parallel_bfloat16_generation_has_dtype_safe_cache_writes

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
    tests/jaxtitan/test_model.py \
    tests/jaxtitan/test_infer.py \
    tests/jaxtitan/test_train_step.py \
    tests/jaxtitan/test_checkpoint_sample.py -x

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan -x
git diff --check
```

Artifacts:

- Branch: `codex/distributed-linear-dtype`.
- Production paths:
  `src/jaxtitan/models/execution.py` and
  `src/jaxtitan/models/components/attention.py`.
- Regression paths:
  `tests/jaxtitan/test_model.py` and `tests/jaxtitan/test_infer.py`.
- No cloud run or retained performance artifact was produced.

Result:

- TP and CP return BF16 logits with FP32 master parameters under BF16 compute.
- TP prefill/decode completes with the incompatible-scatter FutureWarning
  promoted to an error; cache keys and values remain BF16.
- Focused regressions: `3 passed`.
- Model/inference/train-step/checkpoint-sampling gate: `130 passed`.
- Complete four-device Jaxtitan suite: `709 passed, 1 skipped`.

Next:

- Mark PR `#17` ready for user-owned review and merge. Later FP8/FP4 work can
  replace the single projection boundary without changing today's TOML
  surface.

## 2026-07-24 [codex] H100 policy replay is now a local merge gate

Context:

- Added a tracked, sanitized replay fixture extracted from the
  checksum-verified 76-case H100 selector artifact. It contains the 63
  production/calibration samples, portable shape/topology inputs, accepted
  executions, candidate timing summaries, correctness/stability gates, and
  collective operand models.
- Added a local replay command that invokes the real `shape_topology_v1`
  selector. It rejects provenance drift, decision mismatches, missing or
  nonfinite candidates, correctness/stability failures, speed/MAD failures,
  invalid duplicated fallback, and selections outside the original 3%
  simplicity tie band.
- The replay output explicitly records `performance_claim=false`. Its aggregate
  summaries are regression evidence from the archived microbenchmark, not
  predictions of training-step latency.
- Split the cloud workflow into an explicit eight-step smoke phase and a
  separately authorized 64-step profile phase. No selector, Muon numerics,
  optimizer state, or training hot path changed.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

uv run python scripts/jaxtitan/replay_dist_muon_shape_policy.py \
  --json-out /tmp/dist_muon_shape_policy_replay.json

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
  tests/jaxtitan/test_dist_muon_shape_policy_replay.py \
  tests/jaxtitan/test_dist_muon_shape_policy_staging.py \
  tests/jaxtitan/test_dist_muon_shape_policy_analysis.py

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
  tests/jaxtitan/test_dist_muon_shape_policy_replay.py \
  tests/jaxtitan/test_dist_muon_shape_policy_staging.py \
  tests/jaxtitan/test_dist_muon_shape_policy_analysis.py \
  tests/jaxtitan/test_dist_muon_checkpoint_audit.py \
  tests/jaxtitan/test_config.py \
  tests/jaxtitan/test_checkpoints.py \
  tests/jaxtitan/test_checkpoint_eval.py \
  tests/jaxtitan/test_checkpoint_sample.py \
  tests/jaxtitan/test_resume_compat.py \
  tests/jaxtitan/test_optim.py

uv run pytest -q tests/jaxtitan
uv run python -m py_compile \
  scripts/jaxtitan/replay_dist_muon_shape_policy.py \
  scripts/jaxtitan/analyze_dist_muon_shape_policy_results.py \
  scripts/jaxtitan/audit_dist_muon_checkpoint.py
bash -n scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh
git diff --check
```

Artifacts:

- Replay fixture:
  `scripts/jaxtitan/dist_muon_shape_policy_h100_v1.json`.
- Replay command:
  `scripts/jaxtitan/replay_dist_muon_shape_policy.py`.
- Source artifact:
  `cloud_results/dist_muon_leaf_bench_20260724T215030Z.tgz`.
- Source SHA-256:
  `27fddd04d51cdb9f3262a3a074a2f319e8bf0f880ba80851d3a546a5bef6b1e6`.
- Local replay output:
  `/tmp/dist_muon_shape_policy_replay.json`.

Result:

- Replay gate: `63/63` decisions matched, `overall_gate=True`.
- Informational archived-calibration summaries: `1.444x` geometric mean versus
  duplicated, `1.009x` versus the previous execution, and `1.041x` worst
  measured regret. These are not end-to-end performance claims.
- Final replay/staging/analyzer tests: `61 passed`.
- Broader optimizer/config/checkpoint/resume gate: `323 passed`.
- Full Jaxtitan suite: `791 passed, 1 skipped`.
- Fixture regeneration is byte-identical, all eight smoke/profile configs
  validate, Python compilation, shell syntax, and `git diff --check` pass.
- No cloud command was run and no new GPU throughput claim exists.

Next:

- Run only the smoke phase first. Launch the 64-step profiles only when the
  smoke comparison passes and matches the exact current commit. PR #21 remains
  draft until the H100 profile acceptance gates pass.

## 2026-07-24 [codex] Shape-policy acceptance harness hardened

Context:

- Hardened the draft shape-policy PR's validation tooling without changing the
  selector, Muon numerics, acceptance configs, or runtime hot path.
- Replaced the analyzer's artifact-existence boolean with structured gates for
  every training row and optimizer group, final/eval/sample completion,
  checkpoint identity, replica audit, and the compiled HLO signature implied by
  each runtime bucket plan.
- Added a post-training checkpoint audit that reconstructs the exact training
  state, restores the latest retained checkpoint, checks complete model and
  optimizer state for finiteness, and compares physical replicas grouped by
  logical shard index with an exact-zero-difference requirement.
- Updated the cloud runner to preserve `checkpoints/index.json`, run the audit
  after checkpoint eval/sample, and package audit evidence even when its gate
  fails. The final analyzer remains the single acceptance decision.

Commands:

```bash
cd /home/veer/Master/projects/research
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
  tests/jaxtitan/test_dist_muon_shape_policy_analysis.py \
  tests/jaxtitan/test_dist_muon_checkpoint_audit.py
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
  tests/jaxtitan/test_dist_muon_shape_policy_analysis.py \
  tests/jaxtitan/test_dist_muon_checkpoint_audit.py \
  tests/jaxtitan/test_config.py \
  tests/jaxtitan/test_checkpoints.py \
  tests/jaxtitan/test_checkpoint_eval.py \
  tests/jaxtitan/test_checkpoint_sample.py \
  tests/jaxtitan/test_resume_compat.py \
  tests/jaxtitan/test_optim.py
uv run pytest -q tests/jaxtitan
bash -n scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh
uv run python -m py_compile \
  scripts/jaxtitan/analyze_dist_muon_shape_policy_results.py \
  scripts/jaxtitan/audit_dist_muon_checkpoint.py
git diff --check
```

Artifacts:

- Checkpoint audit:
  `scripts/jaxtitan/audit_dist_muon_checkpoint.py`.
- Per-run audit output:
  `replica_audit_<run_id>.json`.
- Structured comparison schema:
  `scripts/jaxtitan/analyze_dist_muon_shape_policy_results.py`, schema version
  `2`.
- Runner:
  `scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh`.
- Archived H100 HLO inspected locally:
  `cloud_results/dist_muon_m2_20260724T192739Z_lightweight.tgz`.
- Frozen baseline SHA-256 remains
  `65fb879f2636778aa5a25d6566b1538a9ea533cfceb1439428bcdbd433d2db72`.

Result:

- Hardened analyzer/audit tests: `24 passed`.
- Focused optimizer/config/checkpoint/resume gate: `280 passed`.
- Full Jaxtitan suite: `748 passed, 1 skipped`.
- The HLO parser matched all six expected buckets in the archived dense-TP
  H100 compilation dump with no planning or signature failures.
- Synthetic negative coverage rejects missing/malformed artifacts, nonfinite
  rows and optimizer groups, checkpoint mismatches, invalid HLO signatures,
  nonfinite state, missing physical replica evidence, and unequal replicas.
- No GPU run was launched and no new throughput claim is made.

Next:

- Run the unchanged four-layout shape-policy matrix on four H100 80GB GPUs.
  Acceptance additionally requires every structured artifact, HLO, and replica
  gate to pass before applying the frozen performance thresholds. Keep PR #21
  draft until that evidence exists.

## 2026-07-24 [codex] Shape/topology Muon policy ready for H100 acceptance

Context:

- Replaced the global distributed-Muon direct/exchange decision with the
  host-static `shape_topology_v1` selector. It uses only canonical matrix
  geometry, TP size, compatible-cohort population, and modeled collective
  bytes.
- Production can now select duplicated, direct small-Gram, partition-aligned
  large/right-Gram, or exchange-plus-small-Gram per compatible cohort. JAX
  execution remains one statically specialized `shard_map` per deterministic
  32 MiB-capped bucket.
- Promoted the benchmarked large/right-Gram kernel without adding a TOML
  setting, runtime autotuning, model-role rules, or accelerator-name rules.
- Added per-leaf policy/cost/sharding metadata and a narrow inference-only
  compatibility exception for legacy distributed-Muon checkpoints. Training
  resume across the policy fingerprint remains rejected.
- Added four matched 64-step acceptance configs, an HLO/evidence runner, and
  a same-node performance analyzer. No H100 production run was launched.

Commands:

```bash
cd /home/veer/Master/projects/research

uv run pytest -q tests/jaxtitan/test_optim.py \
  tests/jaxtitan/test_dist_muon_shape_policy_analysis.py
uv run pytest -q \
  tests/jaxtitan/test_config.py \
  tests/jaxtitan/test_preflight.py \
  tests/jaxtitan/test_resume_compat.py \
  tests/jaxtitan/test_checkpoints.py \
  tests/jaxtitan/test_checkpoint_eval.py \
  tests/jaxtitan/test_infer.py \
  tests/jaxtitan/test_profile_analysis.py
uv run pytest -q tests/jaxtitan/test_train_step.py
uv run pytest -q tests/jaxtitan/test_runtime_training.py -k muon
uv run pytest -q tests/jaxtitan

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_optim.py \
  -k 'shape_policy or bucketed_distributed or large_gram_bucket' \
  tests/jaxtitan/test_dist_muon_shape_policy_analysis.py \
  tests/jaxtitan/test_resume_compat.py

for cfg in configs/jaxtitan/cloud_4gpu_profile64_*_muon_shape_policy.toml; do
  uv run jaxtitan config check "$cfg"
done
bash -n scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh
uv run python -m py_compile \
  src/jaxtitan/optim/muon_policy.py \
  scripts/jaxtitan/analyze_dist_muon_shape_policy_results.py
git diff --check
```

Artifacts:

- Branch: `codex/distributed-muon-shape-policy`, base commit `eaa81e8`.
- Calibration source:
  `cloud_results/dist_muon_leaf_bench_20260724T215030Z.tgz`.
- Calibration SHA-256:
  `27fddd04d51cdb9f3262a3a074a2f319e8bf0f880ba80851d3a546a5bef6b1e6`.
- Same-node baseline:
  `cloud_results/dist_muon_m2_20260724T192739Z_lightweight.tgz`.
- Baseline SHA-256:
  `65fb879f2636778aa5a25d6566b1538a9ea533cfceb1439428bcdbd433d2db72`.
- Runner: `scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh`.
- Analyzer:
  `scripts/jaxtitan/analyze_dist_muon_shape_policy_results.py`.
- Expected candidate bundle:
  `cloud_results/dist_muon_shape_policy_YYYYMMDDTHHMMSSZ_lightweight.tgz`.

Result:

- The policy reproduces all 63 accepted production/calibration decisions with
  zero mismatches. The durable regression table covers 36 unique feature
  combinations with their 63 topology-distinct sample counts.
- Boundary tests cover both direct-pressure limits, non-divisible exchange,
  and equality favoring the redistribution-free direct/large-Gram execution.
- Full Jaxtitan gate: `730 passed, 1 skipped`.
- Final policy/bucket/analyzer/resume regression: `19 passed, 114 deselected`.
- Earlier focused gates also passed: optimizer/analyzer `86 passed`,
  compatibility/config/checkpoint/profile `251 passed`, train step `51
  passed`, and runtime-training Muon `8 passed, 60 deselected`.
- All four acceptance TOMLs pass config validation; shell syntax, Python
  compilation, and `git diff --check` pass.
- The candidate production selector has no measured training throughput yet.
  The draft PR must not claim the modeled or benchmarked selector deltas as a
  production speedup.

Next:

- On the same class of four H100 80GB NVLink node, run:

```bash
cd "$(git rev-parse --show-toplevel)"
scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh --overwrite
```

- Copy back and checksum the lightweight bundle. Keep the PR draft unless all
  correctness, HLO, per-layout, and geometric-mean performance gates pass.

## 2026-07-24 [codex] Distributed Muon leaf selector implemented

Context:

- Added a correctness-checked extension of the existing
  `jaxtitan profile bench muon` surface. It benchmarks duplicated, direct
  small-Gram, partition-aligned large/right-Gram, and exchange-plus-small-Gram
  execution without changing production routing or adding a TOML option.
- The fixed matrix covers all seven rank-2 Muon leaf classes in the current
  dense/Trinity profiles on TP4, plus K/V, O, and dense MLP confirmation on
  FSDP2xTP2 and TP2xEP2.
- Added production-sized 24-leaf K/V and 12-leaf O bucket cases. Candidate
  timings use rotating order and canonical unprofiled measurements; an
  explicitly non-canonical traced pass is optional.
- Expanded the calibration matrix before freezing a production policy:
  all seven current rank-2 role shapes now have production-sized buckets, and
  a one-factor lattice covers hidden widths `256/1024/2048`, aspect ratios
  `1/2/4`, leaf counts `1/12/24`, both canonical TP orientations, TP2/TP4,
  and TP/FSDP+TP/TP+EP topologies. The resulting fixed matrix has 76 cases and
  185 candidate programs.
- Selector output now carries architecture-independent policy features:
  short/long dimensions, aspect ratio, canonical TP dimension, TP size,
  leaf count, and aggregate matrix elements.
- The large-Gram path is benchmark-only. Its distinct BF16 multiplication
  order must clear the existing five-step numerical envelope before the
  selector can recommend it.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

bash -n scripts/jaxtitan/cloud_dist_muon_leaf_bench.sh
uv run python -m py_compile \
  src/jaxtitan/runtime/muon_bench.py \
  scripts/jaxtitan/analyze_dist_muon_leaf_bench.py
uv run pytest -q \
  tests/jaxtitan/test_profile_bench.py \
  tests/jaxtitan/test_dist_muon_leaf_bench_analysis.py \
  tests/jaxtitan/test_optim.py
uv run pytest -q tests/jaxtitan
git diff --check

cd "$(git rev-parse --show-toplevel)"
scripts/jaxtitan/cloud_dist_muon_leaf_bench.sh
```

Artifacts:

- Branch: `codex/distributed-muon-leaf-bench`, stacked on distributed-mode
  commit `4904e0d`.
- Benchmark command: `uv run jaxtitan profile bench muon`.
- Cloud runner: `scripts/jaxtitan/cloud_dist_muon_leaf_bench.sh`.
- Selector: `scripts/jaxtitan/analyze_dist_muon_leaf_bench.py`.
- Expected cloud artifact:
  `cloud_results/dist_muon_leaf_bench_YYYYMMDDTHHMMSSZ.tgz` plus `.sha256`.
- H100 artifact:
  `cloud_results/dist_muon_leaf_bench_20260724T211855Z.tgz`.
- Artifact SHA-256:
  `f8311c431dbb518ddadf943b5cec6c2f96c335a9dd6e1b0f88068d5cc970d0b0`.
- Cloud commit: `e96010d8eaf0afad073a62ae0d78707c52d159e9`.
- Expanded calibration artifact:
  `cloud_results/dist_muon_leaf_bench_20260724T215030Z.tgz`.
- Expanded artifact SHA-256:
  `27fddd04d51cdb9f3262a3a074a2f319e8bf0f880ba80851d3a546a5bef6b1e6`.
- Expanded cloud commit: `dcb00b40a216f3a75382c8856b1fa78217613ff3`.

Result:

- Targeted optimizer/benchmark/analyzer suite: `106 passed`.
- Expanded selector optimizer/benchmark/analyzer gate: `85 passed`.
- Complete Jaxtitan suite: `701 passed, 1 skipped`.
- Large/right-Gram HLO has one norm reduction and five right-Gram reductions,
  with no logical-matrix all-gather or all-to-all.
- The selector requires finite deterministic execution, exact momentum,
  physical replica equality, stable timing, and a minimum `1.05x` speedup.
  Its five-step absolute-error envelope is scaled linearly from the existing
  LR `0.001` calibration (`6e-4` update, `1.25e-3` parameter) to the production
  benchmark LR `0.02` (`1.2e-2` update, `2.5e-2` parameter).
- Four H100 80GB HBM3 GPUs with NV18 links completed all 19 cases and
  50 candidate programs; `overall_gate=True`. Every candidate was finite,
  deterministic, physically replica-equal, correctly sharded, and had exact
  momentum.
- Production-shaped K/V buckets selected `distributed_exchange`: `1.73x`
  versus duplicated on TP4 and `2.28-2.30x` on FSDP2xTP2/TP2xEP2.
- Production-shaped O buckets selected `distributed_large_gram`: `1.75x`
  versus duplicated on TP4 and `2.17x` on both composed layouts. This was
  `1.16x` faster than the current exchange execution.
- TP4 shared-MLP gate/up selected the current direct route at `1.06x`.
  Other unbucketed MLP/Q/down cases and composed dense MLP gate/up did not
  clear the `1.05x` gate and selected duplicated execution.
- No profiler trace was captured; canonical timing was unprofiled as required.
- The 19-case artifact remains valid for K/V and O. The expanded 76-case
  calibration completed with `overall_gate=True`, 76 cases, 185 candidates,
  and exactly 185 uniquely named optimized HLO files.
- Every production-sized aligned bucket selected direct execution. Square
  row-sharded buckets selected right-Gram; medium/large aspect-2/4 row-sharded
  buckets selected exchange. Small-width and singleton crossover cases provide
  the bucket-population and collective-latency breakpoints.
- A preceding successful timing capture at `20260724T213504Z` was superseded
  because 32 singleton/bucket HLO filenames collided. Its timing JSON was
  intact, but it is not the canonical artifact. The benchmark now fails if
  candidate names are non-unique or any expected HLO file is absent.

Next:

- Implement the zero-role-input shape/topology cost rule derived from the clean
  bucket evidence, then confirm it with the matched 64-step matrix.
- Confirm that policy with the existing matched 64-step training matrix before
  making any production throughput claim.

## 2026-07-24 [codex] M2 distributed Muon local acceptance

Context:

- Added an explicit Megatron-style Muon TP contract: `duplicated` preserves the
  accepted one-BF16-gather path, while `distributed` uses a global FP32 norm,
  BF16 local Gram construction, five packed TP Gram reductions, and a
  reversible tiled all-to-all for unfavorable matrix orientations.
- Consolidated all TP Muon leaves into one transform with per-leaf decay
  policy. Compatible leaves use deterministic, 32 MiB-capped reduction
  buckets. Replica synchronization and parameter/gradient/momentum/update
  shardings remain role-specific.
- Distributed execution is intentionally marked non-exact: fake-device
  calibration found bounded absolute differences from duplicated execution due
  to floating-point reduction order. No GPU performance result is claimed.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

uv run pytest -q tests/jaxtitan/test_optim.py
uv run pytest -q \
  tests/jaxtitan/test_config.py \
  tests/jaxtitan/test_preflight.py \
  tests/jaxtitan/test_resume_compat.py \
  tests/jaxtitan/test_checkpoints.py
uv run pytest -q tests/jaxtitan/test_train_step.py
uv run pytest -q tests/jaxtitan/test_runtime_training.py -k 'muon'
uv run pytest -q tests/jaxtitan

bash -n scripts/jaxtitan/cloud_dist_muon_m2_matrix.sh
uv run python -m py_compile \
  scripts/jaxtitan/analyze_dist_muon_m2_results.py
uv run pytest -q tests/jaxtitan/test_dist_muon_m2_analysis.py

for cfg in \
  configs/jaxtitan/cloud_4gpu_profile64_dense_tp_muon.toml \
  configs/jaxtitan/cloud_4gpu_profile64_dense_tp_muon_distributed.toml \
  configs/jaxtitan/cloud_4gpu_profile64_dense_fsdp_tp_muon.toml \
  configs/jaxtitan/cloud_4gpu_profile64_dense_fsdp_tp_muon_distributed.toml \
  configs/jaxtitan/cloud_4gpu_profile64_dense_zero2_tp_muon.toml \
  configs/jaxtitan/cloud_4gpu_profile64_dense_zero2_tp_muon_distributed.toml \
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_muon.toml \
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_muon_distributed.toml
do
  uv run jaxtitan config check "$cfg"
done

git diff --check
```

Artifacts:

- Branch: `codex/distributed-muon-mode`, stacked on the accepted
  `reference_once` implementation commit `3f03d7e`.
- Cloud runner:
  `scripts/jaxtitan/cloud_dist_muon_m2_matrix.sh`.
- Cloud comparison:
  `scripts/jaxtitan/analyze_dist_muon_m2_results.py`.
- Expected cloud evidence:
  `cloud_results/dist_muon_m2_YYYYMMDDTHHMMSSZ_lightweight.tgz` and its
  `.sha256`; full profiler trees remain in the eight `runs/` directories.

Result:

- Optimizer suite: `66 passed`.
- Config/preflight/resume/checkpoint suite: `197 passed`.
- Full train-step suite: `51 passed`.
- Complete fake-device Jaxtitan suite: `691 passed, 1 skipped` in `553.24s`.
- Direct and exchange HLO contain no logical-matrix all-gather, one named norm
  reduction and five named packed Gram reductions per bucket. Exchange leaves
  additionally contain one forward and one reverse all-to-all.
- Five-step tests cover TP, FSDP+TP, ZeRO2+TP, and TP+EP, adversarial replica
  disagreement, grad clipping, zero/near-zero inputs, checkpoint continuation,
  and deterministic repeated execution.

Next:

- Run the eight-config matched matrix on one four-H100 node, preferably
  SXM/NVLink. Require every run to complete train/checkpoint/eval/sample with
  finite state and require distributed median step time to beat duplicated for
  all four layouts before selecting it as the production default.

## 2026-07-21 [codex] M1 four-A100 acceptance and trace analysis

Context:

- Ran the seven-config M1 correctness matrix and four unchanged 64-step MoE
  profiles from commit `5589e6d` on four A100-SXM4-40GB devices. H100s were
  unavailable, so this is a conservative cross-hardware comparison against the
  archived four-H100 baseline, not an exact same-hardware rerun.
- Replaced the interrupted full-checkpoint package with a focused evidence
  bundle containing provenance, configs, logs, metrics, eval/sample outputs,
  diagnostics, HLO, and all four Perfetto/XPlane captures.
- Corrected the acceptance analyzer to reject pathological scatter/reduce
  kernel duration rather than any harmless scatter/reduce event. The archived
  bad path has 164-170 ms maximum kernels; the M1 candidate has 0.342-1.039 ms
  maximum kernels.

Commands:

```bash
cd ~/research
export CUDA_VISIBLE_DEVICES=0,1,2,3
tmux new-session -d -s moe-m1-h100 \
  'cd ~/research && export CUDA_VISIBLE_DEVICES=0,1,2,3 && scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh --overwrite 2>&1 | tee cloud_results/moe_m1_h100_tmux.log'

cd "$(git rev-parse --show-toplevel)"
sha256sum -c cloud_results/moe_m1_h100_20260721T214719Z_focused.tgz.sha256
uv run python scripts/jaxtitan/analyze_moe_m1_h100_results.py \
  cloud_results/moe_m1_h100_20260721T214719Z_focused.tgz \
  --json-out cloud_results/moe_m1_h100_20260721T214719Z_focused.analysis.json
uv run pytest -q \
  tests/jaxtitan/test_profile_analysis.py \
  tests/jaxtitan/test_moe_m1_analysis.py -x
```

Artifacts:

- Candidate commit: `5589e6d`.
- Focused bundle:
  `cloud_results/moe_m1_h100_20260721T214719Z_focused.tgz`.
- SHA-256:
  `a3e1f749c63b9557c5b54cbcd90f3151dda6cb566bb05d9e558a3fe029482cf9`.
- Comparison JSON:
  `cloud_results/moe_m1_h100_20260721T214719Z_focused.analysis.json`.
- Baseline: `cloud_results/profile64_h100_sxm_2026-07-20.tgz`.

Result:

- All seven correctness configurations completed training, checkpoint, eval,
  and sampling with zero nonfinite optimizer groups.
- All four profile configurations completed with profiler steps 12-15 and
  steady scalar window steps 16-63.
- Median train-step results versus the archived baseline:
  - EP AdamW: `165.594 ms`, `50.91x` speedup, reported average MFU `2.89%`.
  - EP Muon: `266.768 ms`, `31.68x` speedup, reported average MFU `1.80%`.
  - TP+EP AdamW: `271.843 ms`, `32.61x` speedup, reported average MFU `1.76%`.
  - TP+EP Muon: `528.194 ms`, `16.98x` speedup, reported average MFU `0.90%`.
- Every profile passes the `>=5x` speedup and `<=10 ms` maximum pathological
  scatter/reduce gate. Candidate maxima are `0.342`, `0.403`, `0.851`, and
  `1.039 ms`; archived baseline maxima are `164.068-170.187 ms`.
- Runtime metadata truthfully reports `stablehlo_generic`. HLO contains native
  ragged all-to-all and ragged-dot operations lowered to cuBLASLt matmuls; the
  unavailable Pallas/Triton ragged-dot lowering is not claimed.
- Remaining cost is no longer assignment-indexed expert-weight movement.
  TP+EP Muon is collective-heavy, and the unchanged non-prefetched data path
  adds roughly `31-32 ms` per step. The current dense FLOP estimator also
  undercounts top-2 routed plus shared-expert work, so absolute MoE MFU is
  directionally useful but not exact.

Next:

- Promote PR `#16` after focused checks and runbook review. Begin M2 with the
  TP/distributed-Muon collective path; treat a custom grouped-GEMM bridge as a
  later optimization if larger-model traces make generic ragged dot dominant.

## 2026-07-21 [codex] M1 H100 runner and analysis prep

Context:

- Prepared the exact cloud runner and local analysis helper for PR `#16` H100
  acceptance. No GPU run was launched in this step.
- The runner executes the unchanged short correctness matrix first, then the
  four 64-step MoE profile configs. It verifies hardware/data, dumps HLO text,
  runs inspect/eval/sample for each run, and packages the selected `runs/`
  artifacts plus provenance into `cloud_results/moe_m1_h100_*.tgz`.
- The analyzer compares the copied candidate archive against the archived July
  H100 baseline and reports per-profile median step speedup, residual
  scatter/reduce fusion count, and GEMM trace fraction.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

bash -n scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh
uv run python -m py_compile scripts/jaxtitan/analyze_moe_m1_h100_results.py

for cfg in \
  configs/jaxtitan/cloud_4gpu_trinity_moe_ep_adamw_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_adamw_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_muon_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_cp_ep_adamw_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_folded_fsdp_ep_muon_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_product_fsdp_ep_muon_validation.toml \
  configs/jaxtitan/cloud_4gpu_trinity_moe_expert_fsdp_adamw_validation.toml \
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_ep_adamw.toml \
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_ep_muon.toml \
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_adamw.toml \
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_muon.toml
do
  uv run jaxtitan config check "$cfg"
done
```

Cloud launch command:

```bash
cd ~/research
git fetch origin
git checkout codex/moe-expert-major
git pull --ff-only
tmux new -s moe-m1-h100
scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh --overwrite
```

Local analysis after copying back the generated archive and checksum:

```bash
cd "$(git rev-parse --show-toplevel)"
sha256sum -c cloud_results/moe_m1_h100_YYYYMMDDTHHMMSSZ.tgz.sha256
uv run python scripts/jaxtitan/analyze_moe_m1_h100_results.py \
  cloud_results/moe_m1_h100_YYYYMMDDTHHMMSSZ.tgz \
  --json-out cloud_results/moe_m1_h100_YYYYMMDDTHHMMSSZ.analysis.json
```

Artifacts:

- Runner: `scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh`.
- Analyzer: `scripts/jaxtitan/analyze_moe_m1_h100_results.py`.
- Existing configs are reused; no TOML changes were required.
- Baseline remains `cloud_results/profile64_h100_sxm_2026-07-20.tgz`.

Result:

- Script syntax and Python compilation passed locally.
- All seven short correctness configs and all four profile64 configs pass
  `jaxtitan config check`.
- No performance result is claimed; the GPU gate remains open.

Next:

- On the next four-H100 allocation, run the tmux launch command above, keep the
  node alive until the generated `.tgz` and `.sha256` are copied back and
  checksum-verified, then run the analyzer against the July baseline.

## 2026-07-21 [codex] M1 native ragged transport local acceptance

Context:

- Replaced M1's rectangular activation exchange with native JAX ragged
  all-to-all in both directions. Static allocation remains the dropless
  worst-case receive bound, but only live assignment slices are transferred.
- Global-expert source sorting plus exchanged count/offset metadata places
  activations directly into expert-major groups. The three gate/up/down
  `jax.lax.ragged_dot` calls are configured for GPU Pallas/Triton lowering.
- Return values are restored to source assignment order before source-local
  route weighting and deterministic top-k summation.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

uv run pytest -q tests/jaxtitan/test_model.py \
  -k 'all_to_all_expert_dispatcher' -x

uv run pytest -q \
  tests/jaxtitan/test_resume_compat.py::test_resume_rejects_pre_ragged_transport_expert_parallel_checkpoint \
  tests/jaxtitan/test_profile_bench.py -x

uv run pytest -q tests/jaxtitan/test_runtime_training.py \
  -k 'expert_parallel or per_expert_muon' -x

uv run pytest -q tests/jaxtitan -x
git diff --check
```

Artifacts:

- Branch: `codex/moe-expert-major`; draft PR: `#16`.
- Implementation commit: `2af509f`, based on the existing M1 commits
  `76b8af8` and `cbb9a5e`.
- H100 comparison baseline remains
  `cloud_results/profile64_h100_sxm_2026-07-20.tgz`.
- No new GPU profile artifact was produced locally.

Result:

- Dispatcher differential/VJP and structural gate: `10 passed` in `95.90s`.
- Resume/profile contract gate: `6 passed`; expert-parallel training subset:
  `3 passed, 64 deselected`.
- Complete Jaxtitan fake-device suite: `641 passed, 1 skipped` in `484.58s`.
- Fake CPU cannot lower native ragged all-to-all in JAX 0.10, so tests use an
  explicitly CPU-only semantic lowering. GPU execution remains native ragged
  transport; there is no TOML option or GPU fallback to the rectangular path.
- No speedup is claimed without the four-H100 artifacts. The PR remains draft.

Next:

- Run short H100 correctness for EP, TP+EP, CP+EP, folded/product FSDP+EP,
  and expert-FSDP, then the four unchanged 64-step EP/TP+EP AdamW/Muon
  profiles. Analyze steps 16-63 and profiler steps 12-15 against the archive.
- If all four profiles do not improve by at least 5x or ragged dots do not
  lower to dominant expert GEMMs, retain the semantic dispatcher and use the
  new trace to design the later custom grouped-GEMM bridge.

## 2026-07-20 [codex] M1 expert-major MoE local acceptance

Context:

- Replaced the production all-to-all MoE path with truthfully source-sharded
  sequence routing and expert-major `jax.lax.ragged_dot` execution.
- Removed assignment-indexed gate/up/down matrix gathers. The local reference,
  psum reference, and data-axis RDEP paths are unchanged.
- Restored the differential VJP guardrail that exposed the original bug and
  made the dispatcher topology part of the resume fingerprint.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest -q \
  tests/jaxtitan/test_model.py::test_all_to_all_expert_dispatcher_preserves_edge_case_outputs_and_gradients \
  tests/jaxtitan/test_model.py::test_all_to_all_expert_dispatcher_matches_multistep_reference_updates \
  tests/jaxtitan/test_model.py::test_all_to_all_expert_dispatcher_lowers_collectives \
  tests/jaxtitan/test_preflight.py::test_run_preflight_reports_expert_parallel_policy \
  tests/jaxtitan/test_preflight.py::test_run_preflight_reports_data_axis_rdep_policy \
  tests/jaxtitan/test_resume_compat.py::test_resume_fingerprint_changes_for_expert_parallel_axis \
  tests/jaxtitan/test_profile_bench.py::test_benchmark_component_emits_stable_non_gating_payload

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run jaxtitan profile bench moe --warmup 0 --iters 1 --json \
  > /tmp/jaxtitan-m1-moe-bench.json

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest -q tests/jaxtitan

git diff --check
```

Artifacts:

- Branch: `codex/moe-expert-major`, based on M0 merge `a083590`.
- Implementation commit: `76b8af8`; draft PR: `#16`.
- Dispatcher: `src/jaxtitan/models/components/moe.py`.
- Runtime contract: `src/jaxtitan/models/execution.py`.
- Directional local benchmark: `/tmp/jaxtitan-m1-moe-bench.json`; it is not a
  retained performance claim.
- H100 comparison baseline:
  `cloud_results/profile64_h100_sxm_2026-07-20.tgz`.

Result:

- Focused correctness and artifact gate: `11 passed`.
- Complete four-fake-CPU suite: `640 passed, 1 skipped` in `374.87s`.
- Exact forward/VJP coverage passes for EP4, data2-by-EP2, TP2-by-EP2,
  CP2-by-EP2, and EP2-by-expert-FSDP2 under balanced, empty-expert,
  all-to-one, and duplicate routing. Every physical replica is compared with
  the local reference at `rtol=1e-5, atol=1e-5`.
- Three successive expert-parameter updates match the replicated reference.
- The local forward/backward benchmark lowers expert work to nine direct dot
  instructions. It no longer materializes assignment-shaped expert matrices;
  the final EP replication is one all-gather.
- Pre-change EP checkpoints fail resume compatibility at the explicit
  `expert_execution` policy field. Evaluation remains independent of training
  resume compatibility.
- No H100 performance result exists yet, so the 5x M1 gate remains open.

Next:

- Run the short four-H100 correctness layouts, followed by the unchanged EP
  and TP-by-EP AdamW/Muon 64-step profiles.
- Keep the implementation PR draft until all four median step times improve by
  at least 5x and trace evidence shows GEMMs replacing the prior 67-170 ms
  scatter/reduce kernels.

## 2026-07-20 [codex] M0 local performance guardrails

Context:

- Added one read-only profile analyzer for canonical scalar windows, paired
  deltas, trace attribution, and semantic HLO summaries.
- Added opt-in four-device MoE and Muon microbenchmarks. Their timings are
  directional measurements and are explicitly not acceptance gates.
- Expanded distributed-Muon equivalence through five state updates for row and
  column shards on TP4, FSDP2-by-TP2, and TP2-by-EP2.
- Added exact MoE forward edge-case coverage for balanced, empty-expert,
  all-to-one, and duplicate routing. The known all-to-all backward defect is
  documented below and deliberately is not encoded as an xfail or accepted
  behavior; the replacement dispatcher must restore the differential VJP.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest -q \
  tests/jaxtitan/test_profile_analysis.py \
  tests/jaxtitan/test_profile_bench.py \
  tests/jaxtitan/test_cli.py

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest -q \
  tests/jaxtitan/test_model.py::test_all_to_all_expert_dispatcher_preserves_edge_case_outputs \
  tests/jaxtitan/test_optim.py::test_exact_distributed_muon_update_matches_replicated_muon_for_tp_shards

uv run jaxtitan profile analyze \
  cloud_results/profile64_h100_sxm_2026-07-20 \
  --json > /tmp/jaxtitan-profile-analysis.json

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run jaxtitan profile bench moe --warmup 0 --iters 1 --json \
  > /tmp/jaxtitan-profile-bench-moe.json

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run jaxtitan profile bench muon --warmup 0 --iters 1 --json \
  > /tmp/jaxtitan-profile-bench-muon.json

JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest -q tests/jaxtitan

git diff --check
```

Artifacts:

- Analyzer implementation: `src/jaxtitan/runtime/profile_analysis.py`.
- Benchmark implementation: `src/jaxtitan/runtime/profile_bench.py`.
- CLI surfaces: `jaxtitan profile analyze` and `jaxtitan profile bench`.
- H100 input archive:
  `cloud_results/profile64_h100_sxm_2026-07-20.tgz`, SHA256
  `87be3f018afad28f94ce6101e3bf11ef185bea0495694c927b44d69f27dffde5`.
- Foundation branch: `codex/performance-guardrails`, based on profiling merge
  `4153418`.

Result:

- Analyzer/benchmark/CLI unit tests: `35 passed`.
- Focused four-device correctness matrix: `9 passed` in `23.68s`.
- Complete four-fake-CPU Jaxtitan suite: `636 passed, 1 skipped` in
  `370.26s`.
- The analyzer found all 15 completed H100 runs, selected steps 16-63, and
  reproduced the canonical deltas, including TP Muon `+62.8%`, large TP Muon
  `+90.9%`, and MoE TP-by-EP `+5.2%` over EP.
- HLO summaries count instruction definitions rather than textual references,
  cover synchronous and asynchronous collectives, and report explicitly
  scoped first-array result shapes and byte estimates. The captured TP Muon
  train module reports 495 all-gather starts, 347 all-reduce starts, and 36
  reduce-scatters.
- Both benchmark smoke runs emitted four deterministic cases with structural
  HLO reports. No timing threshold is asserted in pytest.
- `git diff --check` passed.

Next:

- Begin M1 with a truthfully source-sharded, expert-major dispatcher. Restore
  exact forward and VJP equivalence before applying structural or H100
  performance gates.
- After M1 correctness passes locally, rerun only the representative EP and
  TP-by-EP H100 profiles and require expert GEMMs to replace the generic
  scatter/reduce hot path.

## 2026-07-19 [codex] All-to-all MoE backward RCA

Context:

- The M0 differential VJP guardrail found that the current all-to-all MoE
  dispatcher matches the local reference in the forward pass but returns
  incorrect replicated gradients for input activations and route weights.
- The dispatcher predates the guardrail. Git blame traces the rank-dependent
  route compaction and gather to `c324518` (`Add expert parallel MoE dispatch`);
  `11cbb1b` subsequently reduced the fixed source capacity without changing
  the failing differentiation pattern.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest -q \
  tests/jaxtitan/test_model.py::test_all_to_all_expert_dispatcher_preserves_edge_case_outputs_and_gradients \
  --maxfail=1 -vv

git blame -L 467,575 -- src/jaxtitan/models/components/moe.py
git log -S 'source_rank = flat_assignment_ids % ep_size' \
  --oneline --all -- src/jaxtitan/models/components/moe.py
```

Additional four-fake-CPU probes were run with `uv run python` against JAX
`0.10.0`. They inspected `jax.typeof(...)`, every physical
`addressable_shards` buffer, the VJP with `shard_map(check_vma=False)`, and an
in-memory intervention that made the replicated differentiable operands
explicitly varying before the rank-dependent gather. No probe source or
artifact was retained.

Artifacts:

- Diagnostic guardrail used for RCA (not retained as a passing foundation
  test):
  `tests/jaxtitan/test_model.py::test_all_to_all_expert_dispatcher_preserves_edge_case_outputs_and_gradients`.
- Fault site: `src/jaxtitan/models/components/moe.py`, inside
  `_all_to_all_expert_swiglu`, at the rank-dependent `order` construction and
  `flat_x[order]` / `flat_weights[order]` gathers.
- Original implementation commit: `c324518`.

Result:

- Root cause: `order` depends on `axis_index("ep")` and therefore differs by
  EP rank, but JAX's `shard_map` varying-manual-axis analysis classifies a
  gather from replicated data using those varying indices as replica
  invariant. The physical `flat_x[order]` and `flat_weights[order]` buffers
  differ even though their inferred types omit `{V:ep}`. Reverse-mode's
  `check_vma=True` optimization consequently omits the replica accumulation
  required when transposing those gathers.
- The earliest wrong values are the VJPs of `flat_x[order]` and
  `flat_weights[order]`, before expert SwiGLU math or the reverse all-to-all.
  Forward outputs are exact. Logical expert gate/up/down gradients remain
  exact because those operands are genuinely EP-sharded; replicated input and
  route-weight gradients are wrong and physically disagree across EP ranks.
- A minimal partition/gather reproducer returns the identity in the forward
  pass but gradients such as `[4, 3, 0, 0, 1, 0, 0, 0]` instead of all ones on
  the first EP rank. Disabling VMA differentiation optimization makes every
  replica return all ones. Keeping VMA enabled but truthfully marking the two
  operands varying before the gather also restores the correct VJP.
- The latter intervention matched the complete local SwiGLU reference for
  forward output and all five differentiable inputs on balanced EP4,
  data2-by-EP2, and TP2-by-EP2 probes; maximum observed error was
  `9.54e-7`. It is diagnostic evidence, not an accepted production fix.
- Post-hoc replica sum or mean does not repair the corrupted tensors. In the
  small EP4 probe, maximum errors versus reference remained `3.93` / `0.98`
  for input-gradient sum / mean and `2.07` / `0.52` for route-weight-gradient
  sum / mean. The training loop's replica averaging therefore cannot make the
  current backward pass correct.

Next:

- Reintroduce the differential VJP as a required passing correctness target in
  the replacement-dispatcher PR; do not encode the known defect as an xfail or
  as expected behavior in the foundation suite.
- Do not ship `check_vma=False` as the fix; it disables useful validation and
  produced materially heavier transpose compilation in the probe.
- Design M1 so route payloads are truthfully source-sharded or otherwise
  explicitly varying before dynamic compaction. Require exact forward and VJP
  equivalence for balanced, empty-expert, all-to-one, and duplicate-route
  cases across EP, data-by-EP, and TP-by-EP before performance profiling.

## 2026-07-20 [codex] Four-H100 profile triage and milestone roadmap

Context:

- Profiled 15 controlled 64-step configurations on one four-GPU H100 80GB
  HBM3 node with full `NV18` connectivity between every GPU pair.
- Compared AdamW and Muon under DDP, TP, FSDP+TP, ZeRO-2+TP, EP, and TP+EP.
  Added a 1.666B-parameter dense scale guard and a representative
  738.6M-parameter Trinity MoE instead of relying on the two-layer correctness
  model.
- This is performance evidence only. The accepted correctness contracts and
  July distributed-Muon acceptance remain unchanged.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
sha256sum -c cloud_results/profile64_h100_sxm_2026-07-20.tgz.sha256
mkdir -p cloud_results/profile64_h100_sxm_2026-07-20
tar -xzf cloud_results/profile64_h100_sxm_2026-07-20.tgz \
  -C cloud_results/profile64_h100_sxm_2026-07-20

find cloud_results/profile64_h100_sxm_2026-07-20 \
  -name perfetto_trace.json.gz | wc -l
find cloud_results/profile64_h100_sxm_2026-07-20 \
  -name '*.xplane.pb' | wc -l
find cloud_results/profile64_h100_sxm_2026-07-20 \
  -path '*/summaries/final.json' | wc -l
```

Scalar summaries below use the median of steps 16-63. Steps 1-2 include cold
compile/autotuning, steps 12-15 are profiler-instrumented, and steps 64-65
include final eval/checkpoint work.

Artifacts:

- Capture:
  `cloud_results/profile64_h100_sxm_2026-07-20.tgz`.
- SHA256:
  `87be3f018afad28f94ce6101e3bf11ef185bea0495694c927b44d69f27dffde5`.
- Contents: 15 Perfetto traces, 15 XPlane traces, 15 compressed JAX traces,
  all metrics/summaries/diagnostics/configs/events, 15 checkpoint indexes,
  launcher/provenance logs, and 2900 HLO text artifacts for selected Muon
  routes. Checkpoint tensors are intentionally excluded.
- Capture commit: `9548cecf2a5c1e122cfca6b93000c6db1d2258fa`.
- Prepared-data manifest SHA256:
  `d1c63b56d34b09c255afe41364d560cff8e58c8aad7b95b8d0a7533a83300e9b`.

Result:

### Executive conclusion

1. **MoE expert execution is the dominant problem by orders of magnitude.**
   The representative EP path takes 8.43 seconds of train time per step. Muon
   changes that by only 0.2%; TP+EP adds 5.2%. The path is dominated by
   per-token expert-matrix gather/scatter and reduction kernels, not NVLink.
2. **Exact distributed Muon is the largest dense-model tax.** TP Muon is 62.8%
   slower than TP AdamW on the 285.5M model and 90.9% slower on the 1.666B
   model. Full-logical-matrix reconstruction multiplies all-gather count.
3. **Ordinary TP and composed FSDP/ZeRO-2 remain collective-heavy.** TP AdamW
   is 73.4% slower than DDP AdamW at the small size. FSDP+TP and ZeRO-2+TP add
   another 35-37% over TP AdamW and are nearly identical to each other.
4. **The synchronous input path is visible once GPU steps become short.** The
   configs intentionally used `worker_count=0` and `prefetch=false`; prepared
   data costs 17-39 ms per dense step. This is 8-31% of end-to-end step time.
5. **Cold compilation is a research-iteration cost.** First-step time ranges
   from 31 seconds to 214 seconds. It is amortized in long training but makes
   short profiling and mechanism iteration expensive.

### Steady-state scalar results

`train ms` excludes data and placement. `step ms` and `tok/s` are end-to-end.

| Run | Params | Train ms | Step ms | tok/s |
|---|---:|---:|---:|---:|
| Dense DDP AdamW | 285.5M | 30.7 | 54.6 | 75,051 |
| Dense DDP Muon | 285.5M | 48.0 | 83.0 | 49,342 |
| Dense TP AdamW | 285.5M | 53.3 | 87.0 | 47,096 |
| Dense TP Muon | 285.5M | 86.8 | 128.7 | 31,824 |
| Dense FSDP+TP AdamW | 285.5M | 72.7 | 113.4 | 36,116 |
| Dense FSDP+TP Muon | 285.5M | 120.1 | 163.2 | 25,100 |
| Dense ZeRO-2+TP AdamW | 285.5M | 72.2 | 113.9 | 35,974 |
| Dense ZeRO-2+TP Muon | 285.5M | 120.6 | 164.3 | 24,936 |
| Large TP AdamW | 1.666B | 158.0 | 204.3 | 20,048 |
| Large TP Muon | 1.666B | 301.5 | 348.6 | 11,750 |
| Large FSDP+TP Muon | 1.666B | 459.1 | 503.5 | 8,136 |
| Trinity MoE EP AdamW | 738.6M | 8,430.7 | 8,489.3 | 482.5 |
| Trinity MoE EP Muon | 738.6M | 8,450.4 | 8,510.8 | 481.3 |
| Trinity MoE TP+EP AdamW | 738.6M | 8,866.0 | 8,924.0 | 459.0 |
| Trinity MoE TP+EP Muon | 738.6M | 8,966.6 | 9,017.6 | 454.2 |

The profiler adds 12-58% to the small dense trace steps, 3-6% to the large
dense trace steps, and 1-3% to MoE. Therefore trace durations identify relative
GPU work, while the non-profiled scalar window is canonical for throughput.

### Trace and HLO attribution

Perfetto GPU event sums below use GPU 0. Events on different streams may
overlap, so percentages are attribution signals rather than wall-clock shares.

| Trace | GPU event sum | Dominant attribution |
|---|---:|---|
| Dense TP AdamW | 123 ms | NCCL 66.5 ms / 54.1% |
| Dense TP Muon | 180 ms | NCCL 103.2 ms / 57.4% |
| Large TP AdamW | 357 ms | NCCL 161.8 ms / 45.3% |
| Large TP Muon | 711 ms | NCCL 376.9 ms / 53.0% |
| Large FSDP+TP Muon | 918 ms | NCCL 483.4 ms / 52.6% |
| MoE EP AdamW | 17.315 s | scatter/reduce fusions 16.761 s / 96.8%; NCCL 2.6% |
| MoE TP+EP Muon | 18.320 s | scatter/reduce fusions 17.861 s / 97.5%; NCCL 1.6% |

The MoE HLO explains the extreme result:

- `_all_to_all_expert_swiglu` routes assignments, then indexes complete expert
  matrices per assignment with `local_gate[recv_local_ids]`, `local_up[...]`,
  and `local_down[...]` before applying batched `einsum` operations.
- HLO lowers the forward/backward dot-generals into large generic input-reduce
  fusions instead of expert GEMMs. Examples include outputs
  `bf16[2,4096,2048]` and `bf16[2,4096,1024]`.
- Reverse routing lowers to a `scatter-add` fusion producing
  `bf16[4,2048,1024]`.
- The repeated kernels take 67-170 ms each. GEMM-class kernels account for
  only 0.2-0.4% of MoE GPU event time. Faster NCCL cannot materially fix this.

Dense traces show a different problem:

- In the captured TP window, AdamW executes 148 all-gathers; Muon executes
  990. All-gather GPU time rises from 15.3 ms to 48.0 ms.
- At 1.666B parameters, TP AdamW executes 434 all-gathers; TP Muon executes
  2308. All-gather time rises from 51.0 ms to 200.8 ms.
- The optimized TP Muon HLO contains 495 all-gather starts, 347 all-reduce
  starts, and 36 reduce-scatters for one compiled train step. It routes 84
  matrices through `dist_muon_exact`.
- FSDP+TP and ZeRO-2+TP Muon have effectively identical traces and scalar
  performance. Their train HLOs contain 784/783 all-reduce starts plus 366
  collective-permute starts. This is a shared composed-layout problem, not a
  ZeRO-2-only regression.

### Ordered major milestones

#### M0. Make performance analysis reproducible

- Add a small artifact analyzer that emits the canonical steady-state window,
  paired deltas, trace-window tax, top GPU kernels, and collective counts.
- Store the measurement window and hardware topology with every performance
  conclusion. Do not compare traced-step throughput against untraced steps.
- Gate: rerunning the analyzer on this archive reproduces the tables above.

#### M1. Replace per-token MoE expert-matrix execution

This is the first implementation priority and the largest available speedup.

Algorithm first:

- Route and compact tokens into expert-major buffers shaped like
  `[local_experts, capacity, hidden]` rather than gathering an entire expert
  matrix for every token assignment.
- Exchange compact token payloads and metadata once, group received tokens by
  local expert, execute gate/up/down as real per-expert or grouped GEMMs, then
  combine returned outputs by token id.
- Preserve exact routing: no silent token drops, capacity overflow, or changed
  route weights. Add uneven-load, empty-expert, duplicate-route, backward, and
  EP/TP+EP equivalence tests against the current reference path.

Lower level only after the algorithm is correct:

- If JAX `sort`/segment plus `vmap` does not produce grouped GEMMs, implement
  dispatch/combine through Pallas/Triton or JAX FFI and use CUTLASS/cuBLASLt
  grouped GEMM for expert execution.
- Do not begin with a custom all-to-all or NCCL replacement; communication is
  less than 3% of the current MoE trace.

Gate:

- Remove the 67-170 ms `input_scatter_fusion`/`input_reduce_fusion` kernels.
- Make expert GEMMs the dominant MoE GPU work.
- Achieve at least a 5x step-time improvement on the unchanged 738.6M EP and
  TP+EP configs before considering the path performance-credible.

#### M2. Remove full-matrix reconstruction from distributed Muon

Low-risk first step:

- Bucket compatible leaves so replica synchronization and reconstruction use
  fewer, larger collectives instead of hundreds of per-leaf launches.
- Keep parameter, momentum, update, and replica-axis contracts explicit.

Algorithmic target:

- Implement sharding-aware Newton-Schulz using local matrix products plus a
  reduction of the smaller Gram matrix. Keep the matrix/update sharded instead
  of all-gathering every full logical matrix for every orthogonalization step.
- Select `X X^T` or `X^T X` from logical shape and shard direction, and prove
  equivalence against the accepted full-logical-matrix reference in its
  configured precision.

Lower-level follow-up:

- Fuse the five Newton-Schulz iterations or their repeated GEMM/elementwise
  sequence only after the distributed algorithm reduces communication.

Gate:

- Preserve the existing fake-device and four-H100 correctness matrix.
- Cut TP Muon's incremental step-time tax and all-gather count by at least 50%
  on both the 285.5M and 1.666B configs.

#### M3. Reduce baseline TP and composed-layout collectives

- Bucket small reductions and avoid repeated layout conversions around
  sequence-parallel projections and vocab-parallel loss.
- For FSDP+TP, prefetch/overlap parameter all-gathers with compute and prefer
  reduce-scatter for gradients when the declared output sharding permits it.
- Investigate why TP AdamW spends 45-54% of GPU event time in NCCL even on
  full NVLink before adding compute kernels.

Gate:

- Reduce the FSDP+TP AdamW penalty over TP AdamW from 36% to below 15% on the
  primary dense config, with unchanged numerics and checkpoint compatibility.

#### M4. Hide input work and cache compilation

This can proceed alongside M1-M3 because it does not change model numerics.

- Exercise the existing prepared-data prefetch/worker path and verify exact
  ordering/resume semantics. Move normalization and next-batch preparation off
  the critical path without introducing a second data stack.
- Add a persistent JAX compilation-cache contract keyed by code/config,
  JAX/XLA version, device architecture, and mesh. Record hits/misses in runtime
  diagnostics.

Gate:

- Data preparation is fully overlapped in steady state for the dense profile
  matrix, and resume produces the same token sequence.
- A repeated unchanged preflight/train process reuses compilation artifacts;
  cache mismatch fails safely rather than loading incompatible executables.

#### M5. Promote only evidence-backed custom kernels

Current priority order:

1. MoE dispatch/combine and grouped expert GEMM.
2. Distributed/fused Newton-Schulz after M2 establishes its algorithm.
3. TP loss or projection kernels only if post-M1/M2/M3 traces still rank them.

Do not prioritize RMSNorm, generic linear, or attention kernels from this
capture; none is currently a top wall-time consumer.

Next:

- Implement M0 first so future changes have an automatic before/after report.
- Then prototype M1 on the current reference dispatcher. Keep the reference
  route available in tests, but expose no second user-facing runtime stack.
- Re-run only the four representative MoE profiles after M1; do not repeat the
  full 15-run matrix until a shared hot path changes.

## 2026-07-23 [codex] M2 Phase 0-2 local numerical gate

Context:

- Bound host-static, role-specific parameter, gradient, momentum, and update
  shardings for every `dist_muon_exact` leaf.
- Implemented the Phase 1 `reference_once` traversal: replica synchronization,
  local momentum/Nesterov, one physical two-byte logical-matrix gather,
  unchanged BF16 Newton-Schulz, and local momentum/update/decay.
- Prototyped unbucketed `gram5_direct` and `gram5_exchange`. The direct path
  failed the unchanged multi-step `rtol=1e-5, atol=1e-5` numerical gate, so the
  candidate was removed rather than exposed as an exact production execution.

Commands:

```bash
cd /home/veer/Master/projects/research
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
  tests/jaxtitan/test_optim.py::test_distributed_gram_muon_matches_reference_over_multiple_steps \
  --maxfail=2

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q \
  tests/jaxtitan/test_optim.py \
  -k 'distributed_muon_binds_role_specific or exact_distributed_muon_update or reference_once_lowers' \
  tests/jaxtitan/test_preflight.py \
  tests/jaxtitan/test_runtime_training.py

git diff --check

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_optim.py tests/jaxtitan/test_preflight.py \
  -k 'muon or optimizer_policy or preflight_auto_resolves_tensor_parallel_muon'

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_runtime_training.py \
  -k 'muon or optimizer_policy'

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_optim.py
```

Artifacts:

- Branch: `codex/distributed-muon-m2`, base commit `3000985`.
- No cloud run or W&B artifact was produced.

Result:

- Phase 0-1 focused gate: `9 passed, 145 deselected`.
- Optimizer/preflight Muon gate: `29 passed, 49 deselected`.
- Runtime-training Muon gate: `7 passed, 60 deselected`.
- Complete optimizer module: `38 passed`.
- `git diff --check` passed.
- Optimized CPU HLO contains one `u16[8,16]` all-gather for the Phase 1
  reference leaf and no FP32 full-matrix gather.
- The Phase 2 matrix `(8, 32)` with `P(None, "tp")` failed after momentum
  evolution with `assert jnp.allclose(..., rtol=1e-5, atol=1e-5)`.
- The earliest mismatch is BF16 reduction ordering: the logical reference and
  sharded partial norm/Gram reductions do not preserve the same accumulation
  tree. Newton-Schulz amplifies the rounded difference to approximately
  `1e-4` in an update. FP32 partial accumulators do not restore equivalence
  because the split GEMM and collective still change the accumulation order.
- This is not a transpose, exchange, or replica-axis bug. It is an
  incompatibility between eliminating the full gather and requiring the
  current BF16 reference result at `1e-5` for arbitrary inputs.

Next:

- Decide whether M2 should preserve exact current BF16 semantics and stop at
  `reference_once`, or define a new numerically stable distributed Muon
  reference (for example FP32 norm/Gram semantics) and validate that optimizer
  change as a scored mechanism before implementing direct/exchange buckets.
- Do not run four-GPU performance acceptance for `gram5_direct` until that
  numerical contract is decided.
