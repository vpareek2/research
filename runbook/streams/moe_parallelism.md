# MoE Parallelism

Purpose: track FSDP, ZeRO-2, EP, RDEP, expert-region FSDP, and optimizer
semantics for Trinity-style MoE.

## 2026-07-19 [codex] Current supported MoE parallelism

Context:

- Reconciled this stream with the merged distributed-Muon implementation and
  June/July four-H100 results.
- The older entry below predates per-expert rank-3 Muon routing and TP+EP Muon
  acceptance.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
rg -n "rank3_expert_policy|rank3_split_matrix_policy|expert_parallel" \
  src/jaxtitan/optim src/jaxtitan/models tests/jaxtitan
```

Artifacts:

- Merge commit: `a76b360`.
- Distributed-Muon acceptance bundle:
  `cloud_results/distributed_muon_h100_acceptance_2026-07-19.tgz`.
- Full parallelism bundle:
  `cloud_results/jaxtitan_parallel_validation_2026-06-19.tgz`.

Result:

- DDP, FSDP, ZeRO-2, EP, RDEP, folded/product FSDP+EP, expert-region FSDP,
  TP+EP, and their documented AdamW/Muon routes have cloud correctness
  coverage.
- Rank-3 routed expert tensors use batched per-expert Muon when expert-axis
  ownership leaves every expert matrix complete locally.
- TP+EP Muon passed 17-step and 64-step four-H100 validation with finite
  optimizer state.
- Rank-3 routed expert tensors split along a matrix dimension remain guarded as
  unsupported. Dispatch/grouped-GEMM performance also remains reference-first.

Next actions:

- Use profiles to choose between optimizing expert dispatch/grouped GEMM and
  the rank-2 distributed-Muon path.
- Treat matrix-axis expert tensor parallelism as an explicit future mechanism,
  not an implicit fallback or blocker for current supported layouts.

## 2026-06-19 [codex] Current state

Implemented locally:

- DDP, ZeRO-2, and FSDP sharding modes.
- Folded FSDP+EP, where a shared physical axis acts as dense FSDP and routed
  expert ownership.
- Expert-region FSDP with dedicated `expert_fsdp` axis for internally sharded
  routed expert matrices.
- Data-axis RDEP semantic baseline with static route-row identity.
- AdamW works across the distributed modes covered by tests.
- Muon resolves to Dion2 for sharded dense matrix state, but routed rank-3
  expert Muon remains blocked.

Known constraints:

- Exact distributed optimizer semantics for routed rank-3 expert matrices are
  not complete.
- Expert dispatch is correctness-first JAX collectives, not a high-performance
  DeepEP/ragged/grouped-GEMM backend.
- Cloud validation is still required, but should wait until TP completion and
  distributed Muon/routed-expert optimizer policy are coherent enough to test
  together.

Next actions:

- Finish TP semantic work and exact distributed optimizer policy before the
  next cloud validation pass.
- After validation, decide whether to prioritize routed-expert optimizer
  semantics or dispatch performance based on profiles.
