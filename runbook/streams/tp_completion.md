# Tensor Parallel Completion

Purpose: track the remaining work before Jaxtitan TP is considered complete
enough for research runs.

## 2026-07-19 [codex] Rank-2 TP correctness is complete

Context:

- Reviewed merged `master` after distributed-Muon PR `#12` landed as
  `a76b360`.
- The 2026-06-19 entry below predates sequence-parallel activations,
  TP-aware Muon, and the completed cloud matrices.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
rg -n "sequence_parallel|dist_muon_exact|routed_expert_tensor_parallel" \
  src/jaxtitan tests/jaxtitan docs
```

Artifacts:

- Merge commit: `a76b360`.
- Distributed-Muon acceptance evidence:
  `cloud_results/distributed_muon_h100_acceptance_2026-07-19.tgz`.
- Earlier full parallelism evidence:
  `cloud_results/jaxtitan_parallel_validation_2026-06-19.tgz`.

Result:

- Dense decoder and Trinity TP use row/column projection sharding,
  vocab-parallel exact loss, and sequence-parallel residual activations.
- Rank-2 Muon is exact for TP, FSDP+TP, ZeRO-2+TP, and TP+EP under the accepted
  logical-matrix contract.
- The four-H100 short and stress matrices passed with finite optimizer state,
  checkpoints, eval, sampling, and profiling.
- Routed expert tensor parallelism that shards inside each rank-3 expert matrix
  remains explicitly rejected. Expert-axis ownership with complete local
  matrices is supported.

Next action:

- Treat rank-2 TP correctness as complete.
- Use captured profiles to optimize TP/distributed-Muon performance.
- Keep matrix-axis expert tensor parallelism as a separate future semantic
  project rather than a blocker for current research runs.

## 2026-06-19 [codex] Current state after TP semantic slice

Commit: `9a046a4 Add RDEP and tensor parallel semantics`

Implemented:

- `parallelism.tensor_parallel = true` accepts a real `tp` mesh axis.
- Dense decoder and Trinity rank-2 projections carry TP layout metadata.
- Attention Q/K/V/gate and MLP gate/up/shared gate/up are column-parallel.
- Attention O and MLP down/shared down are row-parallel.
- `lm_head` is vocab-parallel over the output vocab dimension.
- Causal LM loss and z-loss are exact over vocab-sharded logical logits.
- Diagnostics, preflight, sharding summaries, and resume compatibility record
  TP and loss-parallel policy.
- Muon remains rejected with TP until exact distributed matrix optimizer
  semantics are designed.

Validation already run locally:

```sh
cd /home/veer/Master/projects/research
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q tests/jaxtitan/test_config.py \
    tests/jaxtitan/test_mesh.py \
    tests/jaxtitan/test_model.py \
    tests/jaxtitan/test_steps.py \
    tests/jaxtitan/test_train_step.py \
    tests/jaxtitan/test_preflight.py \
    tests/jaxtitan/test_resume_compat.py

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest -q

git diff --check
rg -n "^(from|import) research(\\.|\\s|$)|from __future__ import annotations" \
  src/jaxtitan tests/jaxtitan configs/jaxtitan || true
```

Observed result:

- targeted suite: `291 passed`
- full fake-device CPU suite: `568 passed, 1 skipped`
- hygiene checks passed

Remaining before TP is complete:

- sequence parallelism to reduce activation replication;
- expert tensor parallelism for routed rank-3 expert matrices;
- TP-aware Muon/Dion policy or explicit long-term AdamW-only policy;
- profile-driven chunked/fused vocab-parallel loss path;
- optional kernel/backend optimizations after profiles identify bottlenecks.

Next action:

- Finish the local TP semantic pieces before cloud validation. Cloud validation
  is tracked in `runbook/streams/cloud_validation_queue.md` and is blocked until
  TP plus distributed Muon/routed-expert optimizer policy are ready enough to
  validate as a coherent stack.
