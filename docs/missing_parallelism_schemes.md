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

RDEP means there are multiple replicas of the expert-parallel group. It helps
when expert load imbalance or all-to-all communication makes one giant EP group
inefficient.

Instead of one EP group across every rank, the world is split into several EP
groups. Each group owns a copy of the expert set or a partition strategy, and
data is distributed across those groups. This can reduce communication radius,
improve load balance, and make expert routing less sensitive to a single hot
expert.

RDEP should come after basic EP and folded FSDP+EP are validated because it
adds another grouping dimension and makes diagnostics/checkpoint compatibility
more complex.

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

Tensor parallelism shards individual dense matrix multiplies. Expert tensor
parallelism shards individual expert matrices.

These become important when:

- dense attention/MLP matrices are too large for a single rank;
- a single expert matrix is too large even after expert-axis sharding;
- per-device batch is too small for FSDP or EP alone to be efficient.

Megatron treats TP and expert TP as standard scaling tools. MaxText also has
multiple TP variants, including transpose-style TP for models where the usual
MLP dimensions make the communication tradeoff different.

For Jaxtitan, TP is not the next step because the near-term target is sparse
models where expert ownership and FSDP solve the first memory wall. TP should
stay reserved until one matrix no longer fits or profiling shows a dense
intra-layer bottleneck that FSDP/EP cannot solve.

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

1. Exact distributed Muon/Dion-style optimization for internally sharded routed experts.
2. RDEP to improve expert load balance and reduce all-to-all pressure.
3. Optimized MoE dispatch backends after profiling the JAX reference path.
4. Tensor parallelism and expert tensor parallelism when individual matrices
   become the memory or throughput bottleneck.
5. Context parallelism when long-context experiments become a priority.
6. Pipeline parallelism only when depth or global model size requires it.

The remaining theme is not another standalone mode. It is continuing to assign
different sharding semantics to dense and expert regions while keeping one
artifact, optimizer, checkpoint, and diagnostics contract.
