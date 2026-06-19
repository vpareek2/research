# MoE Parallelism

Purpose: track FSDP, ZeRO-2, EP, RDEP, expert-region FSDP, and optimizer
semantics for Trinity-style MoE.

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
