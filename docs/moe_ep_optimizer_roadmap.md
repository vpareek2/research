# MoE Expert Parallelism And Distributed Optimizer Roadmap

This document records the current design direction for scaling Jaxtitan's Trinity-style MoE path toward larger sparse models, especially the likely long-term target of a roughly 30B-total, 0.5B-active MoE. It is a roadmap, not a benchmark claim.

## Current Jaxtitan State

Jaxtitan already has the pieces needed to validate dense and small MoE correctness:

- DDP, FSDP, and ZeRO-2 parallelism modes.
- Muon for replicated rank-2 hidden matrices.
- Automatic Dion2 routing for rank-2 Muon-intent matrices when optimizer state is sharded over `fsdp`.
- AdamW fallback for embeddings, LM head, norms, router parameters, expert bias, vectors, scalars, unknown leaves, and rank-3 routed expert tensors.
- Trinity MoE components with shared experts, routed experts, AFMoE-style routing semantics, SMEBU, sequence auxiliary loss, z-loss, router diagnostics, and optimizer-health diagnostics.
- Programmatic JAX profiling and local artifact inspection.

The conservative gap is routed expert optimization. Routed expert weights are rank-3 stacks of per-expert 2D matrices. Today they fall back to AdamW because the Muon backend only handles rank-2 leaves and because flattening rank-3 expert tensors would be the wrong mathematical object.

## Recent Profile Signals

The cloud profiling smokes were correctness and instrumentation runs, not tuned throughput runs. Still, they gave useful direction:

- Data fetch and placement were not the bottleneck in the traced windows.
- FSDP and ZeRO-2 profiles were dominated by collective communication.
- FSDP with Muon intent and auto-Dion2 was substantially slower than AdamW in the small profile, which makes the distributed matrix-optimizer path an optimization target.
- Trinity MoE DDP with Muon was dominated by sparse routing/expert-path work and had poor router balance early in training.
- The captured traces included eval in the profiling window, so the next profiling pass should isolate train-only steady steps before making performance claims.

The practical takeaway is that the next large performance and scalability work should center on expert ownership, expert dispatch, and exact per-expert optimizer semantics.

## Reference Findings

### TorchTitan

The local TorchTitan reference shows the structure worth copying, even though its optimizer path is not Muon-native.

- Grouped MoE experts are stored as rank-3 tensors, one 2D matrix per expert.
- Expert parallelism shards the expert axis.
- Token dispatch is explicit: route tokens, sort or permute by expert, exchange tokens across expert ranks, run grouped expert matmuls, then combine outputs.
- FSDP placement treats expert parameters differently from non-expert parameters when expert parallelism is active.
- Optimizer support is AdamW-oriented; TorchTitan has a mixed-optimizer TODO rather than a finished Muon policy.

What matters for Jaxtitan is the separation of concerns: model components define expert tensors, the parallelism layer owns expert placement, the dispatcher owns token movement, and the optimizer sees a clear parameter layout.

### Megatron Core And Emerging Optimizers

Megatron Core now exposes Muon and a layer-wise distributed optimizer path. That is the important signal:

- Muon is not just an elementwise optimizer; it needs full matrix semantics for the orthogonalized update.
- A standard ZeRO-style elementwise optimizer shard is not enough to define exact Muon unless the algorithm has an explicit distributed matrix policy.
- Mixed optimizer policies are expected: Muon for hidden matrices and AdamW for embeddings, output heads, vectors, and other fallback leaves.
- Megatron's MoE guidance aligns with expert-axis ownership, grouped GEMM, token permutation/dispatch fusion, and distributed optimizer support.

For Jaxtitan, this supports a design where `optimizer.name = "muon"` means matrix-optimizer intent, while the resolved backend depends on the actual parameter layout.

## Core Design Rule

Do not flatten routed experts into one giant matrix.

A routed expert tensor is a stack of expert-local 2D matrices:

```text
gate: [num_experts, hidden, expert_intermediate]
up:   [num_experts, hidden, expert_intermediate]
down: [num_experts, expert_intermediate, hidden]
```

The correct Muon-like interpretation is per-expert matrix optimization. If an expert rank owns complete expert matrices, it can run exact local per-expert Muon on those matrices. If a single expert matrix is split across ranks, then the optimizer needs a separate distributed matrix algorithm such as Dion2 or a layer-wise owner policy.

Initial policy should be strict:

- Expert-axis sharded routed experts: allow per-expert Muon locally, because each owned expert matrix is complete.
- Matrix-axis sharded routed experts: reject Muon for routed experts until a tested distributed per-expert matrix optimizer exists.
- Replicated routed experts: allow per-expert Muon once rank-3 batched Muon is implemented and tested.
- Fallback AdamW remains valid for routed experts when matrix-optimizer semantics are unavailable or intentionally disabled.

## Target Architecture

The target should keep each concern isolated.

### Parallel Axes

Use plain names:

- `data`: data parallel batch sharding.
- `fsdp`: dense model and optimizer-state sharding.
- `ep`: expert parallelism over the routed expert axis.

Tensor parallelism and context parallelism stay reserved until there is a concrete need.

### Model Metadata

Extend model-owned metadata so every parameter has both optimizer metadata and layout metadata. MoE parameters need explicit expert layout information:

- whether the tensor is routed expert, shared expert, router, expert bias, dense attention, dense FFN, norm, embedding, or head;
- the expert axis, if present;
- the per-expert matrix axes;
- whether the full per-expert matrix is local under the active layout.

This keeps architecture code reusable. A future DeepSeek-style or GPT-OSS-style MoE should plug into the same metadata contract instead of requiring a second distributed stack.

### Expert Dispatcher Boundary

Introduce a dispatcher abstraction before adding custom kernels:

- local dispatcher for single-device and DDP correctness;
- explicit all-to-all dispatcher for `ep > 1`;
- later optimized dispatcher with fused permutation, grouped GEMM alignment, and optional custom kernels.

The model block should call an expert component and should not know whether tokens moved locally or through all-to-all.

### Optimizer Resolution

Keep the public user intent small:

```toml
[optimizer]
name = "muon"
```

Resolved backends should be internal and artifact-visible:

- rank-2 replicated hidden matrices: Muon;
- rank-2 FSDP-sharded hidden matrices: Dion2 or another exact distributed matrix backend;
- routed rank-3 expert tensors with complete local expert matrices: per-expert Muon;
- routed rank-3 expert tensors with matrix-axis sharding: unsupported for Muon until explicitly implemented;
- embeddings, LM head, norms, router, expert bias, vectors, and scalars: AdamW fallback.

Artifacts and resume compatibility should record the resolved policy, not just the requested optimizer name.

## Implementation Stages

### Stage 1: Expert Layout Contract

Goal: make expert ownership explicit without changing runtime behavior.

- Add `ExpertLayout` metadata for routed experts.
- Add tests that every Trinity MoE parameter is covered exactly once.
- Validate expert-axis divisibility by `ep` size.
- Validate that current DDP behavior resolves to local or replicated expert ownership.
- Keep routed expert optimizer fallback unchanged.

Exit criteria:

- Layout tests can prove whether each expert matrix is complete, expert-axis sharded, or matrix-axis sharded.
- Diagnostics can report expert layout counts.

### Stage 2: Rank-3 Per-Expert Muon

Goal: support Muon semantics for routed experts when each expert matrix is complete locally.

- Implement batched per-expert Muon for rank-3 expert tensors.
- Apply Newton-Schulz independently per expert and per matrix kind.
- Keep expert bias, router, norms, embeddings, and LM head on AdamW.
- Route rank-3 experts to per-expert Muon only when layout says full expert matrices are local.
- Add strict guards for unsupported matrix-axis sharded expert tensors.

Exit criteria:

- Tiny MoE DDP train-step tests show routed experts no longer fall back to AdamW under Muon intent.
- Optimizer health reports routed expert groups with per-expert Muon backend.
- A smoke run confirms loss, router diagnostics, checkpoint, eval, and resume still work.

### Stage 3: Expert Parallel Layout And Dispatcher

Goal: shard routed experts by expert axis and move tokens to expert owners.

- Add `ep` mesh axis support.
- Place routed expert tensors with expert-axis sharding.
- Keep dense/shared/router parameters under the existing DDP/FSDP policies.
- Add local and all-to-all dispatcher implementations behind one boundary.
- Preserve output equivalence between local routing and `ep` routing on small deterministic examples.

Exit criteria:

- One compiled MoE train step runs with `ep > 1`.
- Routed experts are expert-axis sharded and optimized by per-expert Muon locally.
- Checkpoint/restore preserves expert-sharded model and optimizer state.
- Resume rejects incompatible `ep` sizes and expert layout changes.

### Stage 4: RDEP And FSDP Interaction

Goal: support the practical large-MoE setup: data parallelism over dense state, expert parallelism over experts, and replicated or sharded dense optimizer state as configured.

- Define how `data`, `fsdp`, and `ep` compose.
- Keep dense parameters on DDP/FSDP/ZeRO-2 policies.
- Keep routed expert tensors on expert-axis ownership.
- Decide whether shared experts follow dense FFN layout or expert layout per architecture.
- Make router diagnostics aggregate globally across data and expert axes.

Exit criteria:

- MoE FSDP plus EP can preflight, train, checkpoint, eval, sample, and inspect.
- Metrics remain global and single-host in local artifacts.
- Router and optimizer diagnostics correctly aggregate across all axes.

### Stage 5: Distributed Matrix Optimizer For Split Expert Matrices

Goal: only if needed, support exact matrix-optimizer semantics when a single expert matrix is split across ranks.

- Decide between Dion2-style selected-slice updates, a layer-wise owner policy, or another exact distributed matrix backend.
- Add mathematical parity tests against a full unsharded reference.
- Keep this separate from expert-axis Muon, because it solves a different problem.

Exit criteria:

- Split-matrix routed expert optimization matches a full-matrix reference within defined tolerances.
- The runtime rejects unsupported layouts instead of silently using an approximate optimizer.

### Stage 6: Performance Kernels

Goal: optimize after semantics and artifact contracts are stable.

- Profile train-only windows before and after every kernel change.
- Prioritize token dispatch, grouped expert matmul, sparse combine, and matrix-optimizer kernels.
- Keep pure JAX fallback paths for correctness tests.
- Treat ThunderKittens or custom CUDA paths as optional accelerated backends behind the same component boundary.

Exit criteria:

- Each kernel has a correctness test against the JAX reference.
- Each performance claim is backed by local profile artifacts and run summaries.

## Testing Strategy

The testing order should follow the architecture order:

1. Metadata and layout tests.
2. Optimizer routing and per-expert Muon unit tests.
3. Dispatcher permutation and combine tests.
4. Tiny train-step equivalence tests.
5. Checkpoint/resume tests.
6. Cloud multi-GPU smoke tests.
7. Train-only profiling runs.

Key invariants:

- No hidden AdamW fallback for routed experts under Muon intent unless the artifact says so clearly.
- No approximate Muon on split matrices without an explicit tested algorithm.
- Expert bias remains non-gradient state under SMEBU.
- Router diagnostics aggregate globally and remain stable under DDP, FSDP, ZeRO-2, and EP.
- Local artifacts remain the source of truth.

## Near-Term Recommendation

The next implementation slice should be Stage 1, followed by Stage 2 before full expert parallelism. That gives Jaxtitan correct routed-expert optimizer semantics on the current DDP/local MoE path, then makes EP a placement and dispatch problem instead of mixing layout work with new optimizer math.

After that, Stage 3 is the major unlock for the 30B-total, 0.5B-active target. Expert-axis ownership is the clean path because it keeps each routed expert matrix complete on its owner rank and allows exact per-expert Muon without a custom distributed matrix optimizer.
