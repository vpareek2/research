# ThunderKittens Vendoring Notes

Jaxtitan vendors ThunderKittens so CUDA kernels build from a normal repository
checkout without a separate clone step.

Upstream source:

- Repository: `https://github.com/HazyResearch/ThunderKittens`
- Commit: `41f4c2a7e4246911e4ed2b7ced8ea13bfd295e7f`

Keep local patches small and documented here. When updating ThunderKittens,
replace `third_party/ThunderKittens`, update `third_party/ThunderKittens.VERSION`,
then reapply only the still-needed compatibility patches.

Current local patches:

- `include/common/base_types.cuh`: cast `packing<char>` values to `int8`
  before constructing `int8_4`. This avoids CUDA/aarch64 narrowing errors
  on Grace Blackwell-class systems.
- `kernels/common.mk`: add `ARCH=SM121` for GB10 by compiling with
  `-DKITTENS_SM120` and CUDA `compute_121/sm_121` code generation. This keeps
  TK's existing Blackwell-family code path while producing a loadable image for
  compute capability 12.1 devices.

Known issue to watch:

- Grace Blackwell/aarch64 builds can hit `char` to signed-byte narrowing errors
  in `include/common/base_types.cuh`. See upstream issue
  `HazyResearch/ThunderKittens#150`.
