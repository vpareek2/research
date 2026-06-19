# Kernels And Automatic Backend

Purpose: track ThunderKittens/CUDA kernel work and the Jaxtitan automatic
kernel backend design.

## 2026-06-19 [codex] Current state

Implemented/scaffolded:

- ThunderKittens is vendored under `third_party/`.
- Kernel integration is designed as automatic best-effort selection, not a
  user-facing "custom kernel" knob.
- Runtime diagnostics expose kernel backend state, active/fallback/unavailable
  counts, and strictness.
- Current runtime remains correct through the JAX reference path when kernels
  are disabled or unavailable.

Known constraints:

- Local DGX Spark/device support can hit architecture/compiler limitations.
- Early kernels are correctness/education POCs, not serious benchmarks.
- Real performance decisions need cloud GPU profiles and representative model
  sizes.

Next actions:

- Keep kernel work behind the automatic backend contract.
- Use profiling to choose the first serious kernels: likely RMSNorm, linear,
  attention, vocab loss, or MoE dispatch/grouped GEMM.
- Do not claim speedups without local artifacts and profile evidence.
