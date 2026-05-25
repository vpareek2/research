# Jaxtitan Kernel Backend Design

This document describes the intended kernel backend architecture for Jaxtitan.
The goal is not to replace XLA. XLA remains the default compiler and semantic
reference. ThunderKittens kernels are an internal acceleration layer for
measured hot paths where Jaxtitan has a validated, inspectable implementation.

## Philosophy

Jaxtitan should expose a simple kernel UX:

```toml
[kernels]
enabled = true
strict = false
compile = "lazy"
```

Users should not provide custom kernel paths, register arbitrary shared
objects, or pick individual CUDA kernels. Jaxtitan resolves what it can use from
the model config, runtime shape, dtype, sharding policy, GPU architecture, and
compiled kernel cache.

The intended behavior is:

- `enabled = false`: pure JAX/XLA execution.
- `enabled = true`: use validated Jaxtitan kernels where available.
- `strict = false`: fall back to XLA for unsupported signatures and record why.
- `strict = true`: fail during preflight if a required kernel cannot be used.

This keeps one clear user path while making low-level acceleration reproducible.
Kernel activation must always be visible in local artifacts.

## Design Principles

- XLA is the compiler. ThunderKittens is the escape hatch for hot paths.
- Kernel authoring is specific, not generic. Each CUDA kernel targets a concrete
  operation or fusion boundary.
- Pure JAX implementations remain the semantic reference and correctness
  baseline.
- No kernel is considered active just because it compiles. It must be validated
  for the exact operation signature.
- Training use requires explicit differentiation support. Forward-only FFI
  kernels may be used only for inference, eval, or standalone benchmarks.
- No silent fallback. Every fallback has a reason in diagnostics.
- No public extension system in the first design. Jaxtitan-owned kernels only.

Good initial kernel targets are RMSNorm, SwiGLU or fused FFN pieces, MoE
dispatch/combine, routed expert FFN paths, attention kernels, and optimizer
math. Dense matmul is not a priority because XLA already routes that class of
work to strong vendor libraries.

## Architecture

The backend should be built around an internal registry:

```text
operation + signature + capability -> implementation
```

The signature includes at least:

- operation name, such as `rmsnorm` or `moe_dispatch_combine`;
- shape parameters that affect code generation;
- dtype and accumulation dtype;
- layout and sharding requirements;
- GPU architecture;
- forward-only, forward+backward, or fused-step capability;
- ThunderKittens commit and local patch identity.

The registry resolves a kernel plan before training starts. A resolved plan
should record:

- active ThunderKittens kernels;
- XLA fallbacks and reasons;
- unavailable or disabled kernels;
- compiled shared-object paths;
- compiler, CUDA, GPU architecture, flags, and checksums;
- whether each active kernel is valid for train, eval, inference, or benchmark
  only.

The compiled-kernel cache should live outside tracked source, for example under
`.jaxtitan/kernels/`. It should contain a manifest that is safe to copy with run
artifacts and sufficient to reproduce what binary was loaded.

## CLI And Artifact Contract

The first user-facing commands should be read-only or compile-only:

```bash
uv run jaxtitan kernels list
uv run jaxtitan kernels check configs/jaxtitan/run.toml
uv run jaxtitan kernels compile configs/jaxtitan/run.toml
```

`kernels list` reports Jaxtitan-known kernels and their validation status.
`kernels check` resolves the plan for a config without training.
`kernels compile` builds the needed shared objects into the cache.

Preflight and runtime diagnostics should include a compact kernel summary. Runs
should also write a full local artifact such as:

```text
runs/<run_id>/diagnostics/kernels.json
```

The artifact should be canonical local state, not a W&B-only view. `run inspect`
should print a compact line such as:

```text
kernels: enabled=true strict=false active=2 fallback=3 arch=SM90
```

## JAX Integration

CUDA kernels should integrate through JAX FFI/custom calls. The JAX-facing
wrapper owns shape, dtype, layout, and capability checks before dispatching to a
loaded shared object.

For training paths, the wrapper must expose correct gradients through one of:

- a custom VJP/JVP around a forward kernel;
- separate forward and backward kernels;
- a fused operation whose differentiation boundary is explicit.

Until that exists, forward-only kernels should not replace model components in
the training step. They can still be valuable for standalone correctness tests,
inference, eval-only paths, or benchmarking.

## Rollout

1. Document the architecture and keep existing execution unchanged.
2. Add kernel config/spec plumbing, registry skeleton, CLI check/list commands,
   diagnostics, and inspect output.
3. Build a single RMSNorm FFI proof of concept against the existing pure JAX
   reference.
4. Add preflight/runtime kernel-plan artifacts and strict fallback behavior.
5. Add training-capable kernels only after profiler traces show a bottleneck and
   the kernel has correctness tests against the JAX reference.
6. Add ahead-of-time compilation once lazy loading and manifest recording are
   stable.

Likely first performance work should come from profiler traces of Trinity MoE
training. MoE dispatch/combine and routed expert paths are higher-value targets
than replacing ordinary dense matmuls.

## Testing Requirements

Kernel backend tests should cover:

- config defaults and validation;
- registry resolution for supported and unsupported signatures;
- strict versus non-strict fallback behavior;
- deterministic diagnostics and `run inspect` summaries;
- shared-object manifest and checksum recording;
- JAX FFI correctness against pure JAX references;
- no training use of forward-only kernels;
- artifact compatibility across resume and checkpoint eval.

Every kernel needs a standalone CUDA test and a JAX-level comparison test. A
performance claim requires local run artifacts or benchmark output from the
target hardware; compile success alone is not evidence.

## Non-Goals

- No arbitrary user-provided kernels.
- No per-op user selection in the first version.
- No second runtime or dependency manager.
- No replacement for XLA graph compilation.
- No W&B-only kernel visibility.
- No automatic performance claims without measured artifacts.

## Current Starting Point

Jaxtitan already vendors ThunderKittens under `third_party/ThunderKittens` and
keeps local patch notes beside it. The current CUDA RMSNorm code is a standalone
proof of concept, not an active JAX training backend. The next implementation
step should be the registry/config/diagnostics skeleton, then a JAX FFI call
path for a single forward-only RMSNorm kernel.
