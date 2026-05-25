# Kernel Backend And EP Handoff

This is a handoff note for the next agent continuing Jaxtitan kernel/backend
and expert-parallel work. It summarizes the current implementation state,
validated facts, and the recommended next steps.

## Current Branch State

- Active branch at handoff time: `codex/kernel-backend-skeleton`.
- Recent commits on this branch:
  - `40d37f2 Add kernel backend skeleton`
  - `156a6a5 Add kernel compile cache`
  - `745269d Add RMSNorm FFI benchmark harness`
- `cloud_results/` is untracked local reference material. Do not add it unless
  explicitly asked.
- Kernel work is intentionally not wired into model training yet.

## What Exists

The kernel backend now has four pieces:

1. Kernel registry/config plumbing
   - `[kernels] enabled/strict/compile` exists.
   - `jaxtitan kernels list`, `check`, and `compile` exist.
   - Runtime/preflight/inspect diagnostics report kernel plans.

2. ThunderKittens vendoring/build path
   - ThunderKittens lives under `third_party/ThunderKittens`.
   - Local patch metadata is tracked beside it.
   - Compile cache defaults to `.jaxtitan/kernels`, with override via
     `--cache-dir`.

3. RMSNorm CUDA + JAX FFI POC
   - Standalone CUDA RMSNorm binary still builds and passes.
   - `rmsnorm_ffi.so` builds from the same CUDA dispatch path.
   - JAX FFI wrapper is forward-only, bf16-only, hidden-size-1024-only.
   - The wrapper is not imported by model code and is not train-capable.

4. RMSNorm benchmark harness
   - `jaxtitan kernels bench rmsnorm <config>` compares pure JAX RMSNorm with
     the TK FFI wrapper.
   - It writes a JSON artifact under
     `runs/<run_id>/diagnostics/kernel_benchmarks/rmsnorm.json`.
   - Benchmarks are evidence only for the tested hardware and shape.

## Validation Already Run

Focused tests:

```bash
uv run pytest -q tests/jaxtitan/test_kernels.py tests/jaxtitan/test_cli.py
```

Result at handoff: `36 passed, 1 skipped`.

Opt-in local GPU FFI tests:

```bash
JAX_PLATFORMS=cuda JAXTITAN_RUN_REAL_KERNEL_TESTS=1 \
  uv run pytest -q tests/jaxtitan/test_kernels.py -k rmsnorm
```

Result at handoff: `6 passed`.

Real compile and standalone validation:

```bash
uv run jaxtitan kernels compile \
  configs/jaxtitan/pleias_synth_smoke.toml \
  --arch SM121 \
  --cache-dir /tmp/jaxtitan-kernel-cache-stage3 \
  --json

/tmp/jaxtitan-kernel-cache-stage3/rmsnorm_test.out
```

The cache reported RMSNorm as `ffi_cached`, and the standalone CUDA test
reported `PASS`.

Real benchmark command:

```bash
JAX_PLATFORMS=cuda uv run jaxtitan kernels bench rmsnorm \
  configs/jaxtitan/pleias_synth_smoke.toml \
  --cache-dir /tmp/jaxtitan-kernel-cache-stage3 \
  --rows 1,4,17,64,256 \
  --json
```

It completed and wrote the benchmark artifact. On the local DGX Spark, TK FFI
was not broadly faster than XLA. Treat that as correctness/harness signal, not
a serious performance conclusion.

Hygiene checks passed:

```bash
git diff --check
python - <<'PY'
from pathlib import Path
bad = []
for root in ["src/jaxtitan", "tests/jaxtitan", "docs", "configs"]:
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".toml"}:
            if "future__ import annotations" in path.read_text(errors="ignore"):
                bad.append(path.as_posix())
if bad:
    raise SystemExit("\\n".join(bad))
PY
python - <<'PY'
from pathlib import Path
bad = []
for root in ["src/jaxtitan", "tests/jaxtitan"]:
    for path in Path(root).rglob("*.py"):
        text = path.read_text(errors="ignore")
        if "import research" in text or "from research" in text:
            bad.append(path.as_posix())
if bad:
    raise SystemExit("\\n".join(bad))
PY
```

## Important Constraints

- Do not enable automatic kernel use yet.
- Do not wire RMSNorm FFI into model training. It has no custom VJP/backward
  kernel.
- Do not claim performance wins without benchmark artifacts from the target
  hardware.
- Keep XLA as the semantic reference.
- Any training-capable kernel needs a gradient story and an exact fallback
  reason in diagnostics.

## Recommended Next Step

The next major work should be expert parallelism on cloud GPUs, not more local
RMSNorm work.

Reasoning:

- RMSNorm proved the CUDA/JAX FFI path.
- The real Jaxtitan hot path is expected to be Trinity MoE with FSDP/Dion2 and
  later EP/RDEP.
- EP correctness and performance require real multi-GPU collectives.
- The first useful production kernel target should be selected from EP/MoE
  profiler traces, not guessed locally.

Recommended sequence:

1. Cloud EP correctness bundle
   - Tiny Trinity MoE on 2-4 GPUs.
   - Validate expert-sharded params, token dispatch/combine, global metrics,
     checkpoint/restore, eval, and sample restore.
   - Start with deterministic/manual routing tests before real router runs.

2. Implement base EP before RDEP
   - Use JAX `Mesh`, `NamedSharding`, and explicit `PartitionSpec`.
   - Keep model components reusable and keep routing policy outside dense block
     internals.
   - Keep artifacts single-host/global unless multi-host is explicitly added.

3. Add RDEP after EP is stable
   - Replicate experts across expert groups.
   - Validate group-local routing, load metrics, and checkpoint shape/state.

4. Profile cloud Trinity MoE
   - Compare DDP, FSDP/Dion2, EP, and RDEP.
   - Use traces to pick the first serious kernel target.
   - Likely candidates: MoE dispatch/combine, routed expert FFN glue, or
     optimizer matrix update, not dense GEMM.

5. Kernelize only after measurement
   - Follow the RMSNorm pattern: pure JAX reference, standalone CUDA test, JAX
     FFI wrapper, CLI benchmark, local JSON artifact, then optional runtime use.
   - Auto backend should come only after at least one kernel is validated and
     worth activating.

## Useful Existing Docs

- `docs/kernel_backend_design.md`: kernel backend philosophy and contract.
- `docs/moe_expert_parallelism_design.md`: EP/RDEP architecture design.
- `docs/moe_ep_optimizer_roadmap.md`: optimizer implications for MoE EP.
- `docs/cloud_validation.md`: cloud validation bundle overview.
- `docs/trinity_large_training_notes.md`: Trinity training details and open
  architecture/training questions.

## Suggested First Cloud Commands

After cloning the repo and running `uv sync`, validate devices:

```bash
uv run python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("process_count:", jax.process_count())
print("process_index:", jax.process_index())
print("devices:")
for device in jax.devices():
    print(" ", device)
PY
```

Prepare or stream data through the Jaxtitan data UX. Prefer HF preparation over
copying local data unless explicitly debugging a known prepared artifact:

```bash
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_smoke.toml
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json --seq-len 1024
uv run jaxtitan data check data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --verify-checksums
```

Run cloud smoke/preflight configs from `configs/jaxtitan/` and inspect local
artifacts first. W&B, if enabled, is only a mirror.
