# Tensor Parallel Completion

Purpose: track the remaining work before Jaxtitan TP is considered complete
enough for research runs.

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

- cloud validation on real GPU collectives;
- sequence parallelism to reduce activation replication;
- expert tensor parallelism for routed rank-3 expert matrices;
- TP-aware Muon/Dion policy or explicit long-term AdamW-only policy;
- profile-driven chunked/fused vocab-parallel loss path;
- optional kernel/backend optimizations after profiles identify bottlenecks.

Next action:

- Run cloud TP smoke configs once cloud GPUs are available. Record run dirs,
  inspect output, profiling metadata, and checkpoint eval/sample results here.
