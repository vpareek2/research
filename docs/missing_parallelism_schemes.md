# Missing Parallelism Schemes

This document ranks the parallelism schemes Jaxtitan is missing or has not fully
validated yet. The order is based on usefulness for the near-term goal:
reproducible small-to-medium sparse pretraining, especially Trinity-style MoE
models that may fit only when dense state and routed experts are both
distributed.

## 1. Folded Expert Parallelism Plus FSDP

Status: implemented locally, covered by targeted tests and full-suite tests;
cloud validation is the remaining confidence step.

Jaxtitan currently has clean product-axis support:

```text
data x fsdp x ep
```

That is good for larger clusters, but it makes real `fsdp > 1` plus `ep > 1`
require at least four devices. A two-GPU user with a MoE that does not fit on
one GPU needs a different layout: the same physical axis should behave like
FSDP for dense/non-expert modules and like EP for routed expert modules.

The useful mental model is:

```text
2 GPUs total
dense layers: shard parameters/optimizer state over axis model
MoE routed experts: shard expert axis over axis model
```

MaxText does something close to this by letting the EP axis act like FSDP for
attention and other non-expert work. TorchTitan and Megatron also distinguish
dense and sparse/expert meshes instead of forcing every module through one
uniform topology.

Jaxtitan now supports both:

- product-axis mode for larger clusters: `data x fsdp x ep`;
- folded-axis mode for small clusters: one model axis used as FSDP for dense
  modules and EP for routed experts.

The first folded mode uses `parallelism.mode = "fsdp"` and
`parallelism.expert_parallel = true` on a mesh with `["data", "fsdp"]`. In
that layout, dense rank-2 matrices use FSDP/Dion2 policy over `fsdp`, while
routed rank-3 experts use expert-axis ownership over the same `fsdp` axis.

This unlocked the practical two-GPU MoE layout where dense state is FSDP-sharded
and routed experts are owned over the same physical axis.

## 2. Expert-Region FSDP

Status: implemented locally for AdamW-only expert-region FSDP with a dedicated
`expert_fsdp` axis. Cloud validation and exact Muon/Dion-style routed-expert
matrix optimization are the immediate follow-ups.

In large MoE systems, dense modules and expert modules often use different FSDP
groups. Dense layers can shard over one data/FSDP mesh, while expert modules
can use an expert-specific FSDP dimension after expert-axis sharding. TorchTitan
calls this `efsdp`; Megatron-FSDP documents separate dense and sparse/expert
device meshes.

The reason is memory and communication shape. Routed expert tensors are
rank-3 stacks of per-expert matrices. Sharding the expert axis solves expert
ownership, but very large individual experts may still need FSDP-style sharding
inside each expert group. That gives layouts like:

```text
expert tensor: [experts, hidden, intermediate]
EP shards experts
expert FSDP shards hidden or intermediate inside the owned experts
```

The first Jaxtitan shape is:

```text
mesh axes: data x fsdp x ep x expert_fsdp
dense/shared matrices: existing dense FSDP policy over fsdp
routed expert tensors: expert ownership over ep, internal expert width over expert_fsdp
```

This unlocks larger individual experts while keeping the model, optimizer,
checkpoint, and diagnostics contracts unified. Muon currently remains blocked
for internally sharded routed expert matrices until an exact distributed expert
matrix optimizer is added.

## 3. RDEP / Replicated Expert Parallelism

Status: semantic JAX baseline implemented locally for data-axis RDEP.

Jaxtitan now supports `parallelism.expert_parallel_axis = "data"` for Trinity
MoE. In this mode the data axis is also the routed expert-owner axis: dense and
shared paths stay ordinary data-parallel/FSDP computation, routed experts shard
over `data`, route rows are pooled across the data-axis group, owners compute
their local experts, and outputs return to the original source tokens.

This is intentionally a correctness-first `rdep_static` backend implemented
with JAX collectives. It gives us the Noumena-style route-row semantics and
artifact/fingerprint contracts before CUDA IPC, ragged dispatch, or DeepEP-like
transport work.

## 4. Optimized MoE Dispatch Backends

The current all-to-all dispatcher is correctness-first. Serious MoE training
usually needs faster dispatch/combine paths:

- sorted or grouped token dispatch;
- dropless ragged dispatch;
- DeepEP-style communication;
- custom scatter/combine VJPs;
- grouped GEMM integration.

MaxText and Megatron both treat all-to-all as the baseline and then add more
specialized communication and grouped-matmul paths for performance. Jaxtitan
should keep the dispatcher boundary stable and add optimized backends only
after profiling identifies the bottleneck.

This is ranked below folded FSDP+EP because a faster dispatcher does not help
if the user cannot fit the model in the first place.

## 5. Tensor Parallelism And Expert Tensor Parallelism

Status: semantic dense TP foundation is implemented locally and covered by
fake-device CPU tests. It is not complete until cloud validation, sequence
parallelism, expert tensor parallelism, and optimizer/kernel follow-through are
done.

Tensor parallelism shards individual dense matrix multiplies. Expert tensor
parallelism shards individual expert matrices. Jaxtitan now has the first
correctness-oriented TP slice:

- `[parallelism] tensor_parallel = true` enables a real `tp` mesh axis.
- Decoder and Trinity rank-2 dense projections have model-owned TP layout
  metadata.
- Attention Q/K/V/gate and MLP gate/up/shared gate/up are column-parallel.
- Attention O and MLP down/shared down are row-parallel.
- `lm_head` is vocab-parallel over the output vocab dimension.
- Causal LM loss and z-loss run exactly over vocab-sharded logical logits.
- Diagnostics, preflight, sharding summaries, and resume compatibility record
  the resolved TP and loss-parallel policy.

The current implementation deliberately keeps the residual stream replicated at
block boundaries. That is the right semantic baseline: it composes cleanly with
FSDP, ZeRO-2, EP, RDEP, checkpoints, eval, and sampling before we optimize
away collectives.

These become important when:

- dense attention/MLP matrices are too large for a single rank;
- a single expert matrix is too large even after expert-axis sharding;
- per-device batch is too small for FSDP or EP alone to be efficient.

Megatron treats TP and expert TP as standard scaling tools. MaxText also has
multiple TP variants, including transpose-style TP for models where the usual
MLP dimensions make the communication tradeoff different.

For Jaxtitan, TP is now a partially implemented scaling axis, not just a
reserved concept. The remaining work before calling TP complete is below.

### 5.1 Cloud Validation

The CPU fake-device tests prove sharding semantics and numerical parity, but TP
must be validated on real GPU collectives before it is trusted for research
runs.

Required cloud checks:

- dense decoder TP preflight, train, eval checkpoint, sample checkpoint, and
  inspect;
- dense Trinity TP preflight/train/eval/sample;
- Trinity MoE with shared-expert TP and routed experts replicated or EP-owned;
- TP combined with FSDP and ZeRO-2 on separate `data x fsdp x tp` meshes;
- TP combined with folded FSDP+EP and data-axis RDEP where the mesh permits it;
- profile traces confirming TP collectives appear where expected and full
  logits are not silently replicated on every rank.

Acceptance for this step is not speed. It is exactness, no hidden fallback,
clean checkpoint restore, and artifact summaries that match the physical
placement.

### 5.2 Sequence Parallelism

The current residual stream is replicated across TP ranks. That is simple and
correct, but it leaves activation memory on the table. Sequence parallelism
shards selected activation tensors over the sequence or token dimension between
TP collectives, then gathers or reduces at module boundaries that need full
hidden vectors.

This is the main missing piece for TP to become a serious memory-scaling mode.
Without it, TP reduces parameter and optimizer-state memory for rank-2
matrices, but activation memory remains closer to replicated execution.

First Jaxtitan sequence-parallel slice should:

- add an internal execution policy, not a new user knob;
- shard normalized residual activations over the `tp` axis where the model can
  preserve exact semantics;
- keep embeddings, attention masks, RoPE tables, logits, metrics, and router
  diagnostics contract-compatible;
- prove train/eval parity against replicated TP on fake CPU devices;
- record `sequence_parallel.enabled` and activation sharding policy in
  diagnostics/resume metadata.

### 5.3 Expert Tensor Parallelism

Routed expert weights are currently rank-3 tensors and intentionally stay
outside dense TP. That keeps MoE semantics simple, but it means a single expert
matrix must still fit on its owner rank except for expert-region FSDP.

Expert TP is needed when each expert is large enough that EP plus expert FSDP
is not enough or when grouped expert GEMMs become the throughput bottleneck.

The clean design is region-specific:

- dense TP handles ordinary rank-2 attention/MLP/shared-expert matrices;
- EP/RDEP decides expert ownership and dispatch;
- expert FSDP shards owned expert matrices for memory;
- expert TP shards the matrix multiply inside each owned expert group.

This should not be bolted onto the dense TP path. It needs a routed-expert
layout policy that can express expert axis, expert FSDP axis, and expert TP
matrix axis at the same time.

### 5.4 Optimizer Semantics Under TP

AdamW is elementwise and naturally shard-safe under TP. Muon is still blocked
with TP because exact orthogonalized matrix updates depend on whole-matrix
semantics. Dion2 currently resolves from sharded Muon intent for FSDP/ZeRO
matrix state, but TP changes the matrix partition geometry again.

Before Muon is allowed with TP, Jaxtitan needs one of:

- exact full-matrix Muon semantics over TP-sharded rank-2 matrices;
- a Dion-style distributed matrix optimizer with explicit support for TP
  row/column partitioning;
- a documented decision that TP+Muon uses AdamW fallback for specific matrix
  regions, with artifacts making that obvious.

This is a correctness requirement, not a performance nice-to-have. TP should
stay AdamW-only until the optimizer policy is explicit and tested.

### 5.5 Chunked / Fused Loss Path

The current vocab-parallel loss is exact and keeps the public loss API stable.
It still materializes logical `[batch, seq, vocab]` logits, physically sharded
over vocab. That is acceptable for correctness, but it is not the final
high-performance path.

A later loss slice should follow the TorchTitan-style shape:

- model can return final hidden states without immediately applying `lm_head`;
- loss code applies `lm_head` chunk-by-chunk over sequence;
- cross entropy accumulates numerator and token count without keeping the full
  logical logits tensor live;
- vocab-parallel `lm_head` and exact global softmax semantics are preserved;
- z-loss remains separately accounted for.

This should be driven by profiling. If XLA already optimizes the current path
well enough for Jaxtitan-scale runs, the chunked path can wait.

### 5.6 Kernel And Communication Optimization

The current TP helpers use normal JAX einsums plus sharding constraints. That
is correct and inspectable. It is not the final performance story.

Potential optimization targets:

- fused row/column parallel linear kernels;
- fused RMSNorm plus TP-friendly projection input layout;
- FlashAttention/attention kernels that understand TP head partitioning;
- fused vocab-parallel cross entropy;
- better collective scheduling around row-parallel reductions;
- grouped GEMM for shared/routed expert paths once expert TP exists.

These should be added only after GPU profiles identify actual bottlenecks.
The automatic kernel backend should be allowed to select optimized kernels when
available, but correctness must remain defined by the JAX reference path.

### 5.7 Completion Criteria

TP is complete for Jaxtitan when all of the following are true:

- dense decoder and dense Trinity TP run on cloud GPUs through train, eval,
  checkpoint restore, and sampling;
- Trinity MoE TP composes with EP/RDEP and FSDP policies without changing
  model semantics;
- sequence parallel reduces activation replication where it matters;
- expert TP has a clear routed-expert layout and train-step contract;
- AdamW TP is validated and Muon/Dion behavior is either exact or explicitly
  disallowed with clear artifacts;
- preflight, diagnostics, inspect, final summaries, checkpoints, and resume
  compatibility record the full TP policy;
- profiles show the expected collectives and no accidental full replication of
  large TP-sharded matrices/logits;
- full fake-device CPU tests and targeted cloud smoke runs are green.

## 6. Context / Sequence Parallelism

Context parallelism shards the sequence dimension for long-context training. It
reduces activation memory and makes very long sequence lengths feasible.

Megatron and MaxText both include CP because long contexts make attention
activations and KV-like intermediates the bottleneck. CP composes with FSDP,
TP, and EP, but it adds attention-specific communication and makes data shape
contracts more complex.

This is not urgent for Jaxtitan until the target experiments move to long
context. For now, sequence lengths are small enough that EP/FSDP/MoE work
dominates.

## 7. Pipeline Parallelism

Pipeline parallelism shards layers by depth. It is useful when the model is too
deep or too large to fit even with parameter sharding, or when the global batch
regime makes data/FSDP communication inefficient.

It is lower priority for Jaxtitan because it adds substantial scheduling,
microbatching, activation, checkpoint, and bubble-management complexity. It is
also less important for the current research target than MoE-specific
parallelism.

Pipeline parallelism should remain a later large-model feature, not part of the
next sparse-training milestone.

## Recommended Order

1. Cloud-validate folded FSDP+EP, expert-region FSDP, RDEP, and dense TP on real GPUs.
2. Finish TP as a scaling axis: sequence parallelism, expert TP, and TP-aware optimizer policy.
3. Exact distributed Muon/Dion-style optimization for internally sharded routed experts.
4. Optimized MoE dispatch backends after profiling the JAX reference path.
5. Context parallelism when long-context experiments become a priority.
6. Pipeline parallelism only when depth or global model size requires it.

The remaining theme is not another standalone mode. It is continuing to assign
different sharding semantics to dense and expert regions while keeping one
artifact, optimizer, checkpoint, and diagnostics contract.
