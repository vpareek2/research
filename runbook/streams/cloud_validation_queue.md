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
