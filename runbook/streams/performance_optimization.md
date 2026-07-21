# Performance Optimization

Purpose: turn measured Jaxtitan profiles into an ordered optimization program
without weakening correctness or adding speculative kernel surfaces.

## 2026-07-20 [codex] Four-H100 profile triage and milestone roadmap

Context:

- Profiled 15 controlled 64-step configurations on one four-GPU H100 80GB
  HBM3 node with full `NV18` connectivity between every GPU pair.
- Compared AdamW and Muon under DDP, TP, FSDP+TP, ZeRO-2+TP, EP, and TP+EP.
  Added a 1.666B-parameter dense scale guard and a representative
  738.6M-parameter Trinity MoE instead of relying on the two-layer correctness
  model.
- This is performance evidence only. The accepted correctness contracts and
  July distributed-Muon acceptance remain unchanged.

Commands:

```bash
cd "$(git rev-parse --show-toplevel)"
sha256sum -c cloud_results/profile64_h100_sxm_2026-07-20.tgz.sha256
mkdir -p cloud_results/profile64_h100_sxm_2026-07-20
tar -xzf cloud_results/profile64_h100_sxm_2026-07-20.tgz \
  -C cloud_results/profile64_h100_sxm_2026-07-20

find cloud_results/profile64_h100_sxm_2026-07-20 \
  -name perfetto_trace.json.gz | wc -l
find cloud_results/profile64_h100_sxm_2026-07-20 \
  -name '*.xplane.pb' | wc -l
find cloud_results/profile64_h100_sxm_2026-07-20 \
  -path '*/summaries/final.json' | wc -l
```

Scalar summaries below use the median of steps 16-63. Steps 1-2 include cold
compile/autotuning, steps 12-15 are profiler-instrumented, and steps 64-65
include final eval/checkpoint work.

Artifacts:

- Capture:
  `cloud_results/profile64_h100_sxm_2026-07-20.tgz`.
- SHA256:
  `87be3f018afad28f94ce6101e3bf11ef185bea0495694c927b44d69f27dffde5`.
- Contents: 15 Perfetto traces, 15 XPlane traces, 15 compressed JAX traces,
  all metrics/summaries/diagnostics/configs/events, 15 checkpoint indexes,
  launcher/provenance logs, and 2900 HLO text artifacts for selected Muon
  routes. Checkpoint tensors are intentionally excluded.
- Capture commit: `9548cecf2a5c1e122cfca6b93000c6db1d2258fa`.
- Prepared-data manifest SHA256:
  `d1c63b56d34b09c255afe41364d560cff8e58c8aad7b95b8d0a7533a83300e9b`.

Result:

### Executive conclusion

1. **MoE expert execution is the dominant problem by orders of magnitude.**
   The representative EP path takes 8.43 seconds of train time per step. Muon
   changes that by only 0.2%; TP+EP adds 5.2%. The path is dominated by
   per-token expert-matrix gather/scatter and reduction kernels, not NVLink.
2. **Exact distributed Muon is the largest dense-model tax.** TP Muon is 62.8%
   slower than TP AdamW on the 285.5M model and 90.9% slower on the 1.666B
   model. Full-logical-matrix reconstruction multiplies all-gather count.
3. **Ordinary TP and composed FSDP/ZeRO-2 remain collective-heavy.** TP AdamW
   is 73.4% slower than DDP AdamW at the small size. FSDP+TP and ZeRO-2+TP add
   another 35-37% over TP AdamW and are nearly identical to each other.
4. **The synchronous input path is visible once GPU steps become short.** The
   configs intentionally used `worker_count=0` and `prefetch=false`; prepared
   data costs 17-39 ms per dense step. This is 8-31% of end-to-end step time.
5. **Cold compilation is a research-iteration cost.** First-step time ranges
   from 31 seconds to 214 seconds. It is amortized in long training but makes
   short profiling and mechanism iteration expensive.

### Steady-state scalar results

`train ms` excludes data and placement. `step ms` and `tok/s` are end-to-end.

| Run | Params | Train ms | Step ms | tok/s |
|---|---:|---:|---:|---:|
| Dense DDP AdamW | 285.5M | 30.7 | 54.6 | 75,051 |
| Dense DDP Muon | 285.5M | 48.0 | 83.0 | 49,342 |
| Dense TP AdamW | 285.5M | 53.3 | 87.0 | 47,096 |
| Dense TP Muon | 285.5M | 86.8 | 128.7 | 31,824 |
| Dense FSDP+TP AdamW | 285.5M | 72.7 | 113.4 | 36,116 |
| Dense FSDP+TP Muon | 285.5M | 120.1 | 163.2 | 25,100 |
| Dense ZeRO-2+TP AdamW | 285.5M | 72.2 | 113.9 | 35,974 |
| Dense ZeRO-2+TP Muon | 285.5M | 120.6 | 164.3 | 24,936 |
| Large TP AdamW | 1.666B | 158.0 | 204.3 | 20,048 |
| Large TP Muon | 1.666B | 301.5 | 348.6 | 11,750 |
| Large FSDP+TP Muon | 1.666B | 459.1 | 503.5 | 8,136 |
| Trinity MoE EP AdamW | 738.6M | 8,430.7 | 8,489.3 | 482.5 |
| Trinity MoE EP Muon | 738.6M | 8,450.4 | 8,510.8 | 481.3 |
| Trinity MoE TP+EP AdamW | 738.6M | 8,866.0 | 8,924.0 | 459.0 |
| Trinity MoE TP+EP Muon | 738.6M | 8,966.6 | 9,017.6 | 454.2 |

The profiler adds 12-58% to the small dense trace steps, 3-6% to the large
dense trace steps, and 1-3% to MoE. Therefore trace durations identify relative
GPU work, while the non-profiled scalar window is canonical for throughput.

### Trace and HLO attribution

Perfetto GPU event sums below use GPU 0. Events on different streams may
overlap, so percentages are attribution signals rather than wall-clock shares.

| Trace | GPU event sum | Dominant attribution |
|---|---:|---|
| Dense TP AdamW | 123 ms | NCCL 66.5 ms / 54.1% |
| Dense TP Muon | 180 ms | NCCL 103.2 ms / 57.4% |
| Large TP AdamW | 357 ms | NCCL 161.8 ms / 45.3% |
| Large TP Muon | 711 ms | NCCL 376.9 ms / 53.0% |
| Large FSDP+TP Muon | 918 ms | NCCL 483.4 ms / 52.6% |
| MoE EP AdamW | 17.315 s | scatter/reduce fusions 16.761 s / 96.8%; NCCL 2.6% |
| MoE TP+EP Muon | 18.320 s | scatter/reduce fusions 17.861 s / 97.5%; NCCL 1.6% |

The MoE HLO explains the extreme result:

- `_all_to_all_expert_swiglu` routes assignments, then indexes complete expert
  matrices per assignment with `local_gate[recv_local_ids]`, `local_up[...]`,
  and `local_down[...]` before applying batched `einsum` operations.
- HLO lowers the forward/backward dot-generals into large generic input-reduce
  fusions instead of expert GEMMs. Examples include outputs
  `bf16[2,4096,2048]` and `bf16[2,4096,1024]`.
- Reverse routing lowers to a `scatter-add` fusion producing
  `bf16[4,2048,1024]`.
- The repeated kernels take 67-170 ms each. GEMM-class kernels account for
  only 0.2-0.4% of MoE GPU event time. Faster NCCL cannot materially fix this.

Dense traces show a different problem:

- In the captured TP window, AdamW executes 148 all-gathers; Muon executes
  990. All-gather GPU time rises from 15.3 ms to 48.0 ms.
- At 1.666B parameters, TP AdamW executes 434 all-gathers; TP Muon executes
  2308. All-gather time rises from 51.0 ms to 200.8 ms.
- The optimized TP Muon HLO contains 495 all-gather starts, 347 all-reduce
  starts, and 36 reduce-scatters for one compiled train step. It routes 84
  matrices through `dist_muon_exact`.
- FSDP+TP and ZeRO-2+TP Muon have effectively identical traces and scalar
  performance. Their train HLOs contain 784/783 all-reduce starts plus 366
  collective-permute starts. This is a shared composed-layout problem, not a
  ZeRO-2-only regression.

### Ordered major milestones

#### M0. Make performance analysis reproducible

- Add a small artifact analyzer that emits the canonical steady-state window,
  paired deltas, trace-window tax, top GPU kernels, and collective counts.
- Store the measurement window and hardware topology with every performance
  conclusion. Do not compare traced-step throughput against untraced steps.
- Gate: rerunning the analyzer on this archive reproduces the tables above.

#### M1. Replace per-token MoE expert-matrix execution

This is the first implementation priority and the largest available speedup.

Algorithm first:

- Route and compact tokens into expert-major buffers shaped like
  `[local_experts, capacity, hidden]` rather than gathering an entire expert
  matrix for every token assignment.
- Exchange compact token payloads and metadata once, group received tokens by
  local expert, execute gate/up/down as real per-expert or grouped GEMMs, then
  combine returned outputs by token id.
- Preserve exact routing: no silent token drops, capacity overflow, or changed
  route weights. Add uneven-load, empty-expert, duplicate-route, backward, and
  EP/TP+EP equivalence tests against the current reference path.

Lower level only after the algorithm is correct:

- If JAX `sort`/segment plus `vmap` does not produce grouped GEMMs, implement
  dispatch/combine through Pallas/Triton or JAX FFI and use CUTLASS/cuBLASLt
  grouped GEMM for expert execution.
- Do not begin with a custom all-to-all or NCCL replacement; communication is
  less than 3% of the current MoE trace.

Gate:

- Remove the 67-170 ms `input_scatter_fusion`/`input_reduce_fusion` kernels.
- Make expert GEMMs the dominant MoE GPU work.
- Achieve at least a 5x step-time improvement on the unchanged 738.6M EP and
  TP+EP configs before considering the path performance-credible.

#### M2. Remove full-matrix reconstruction from distributed Muon

Low-risk first step:

- Bucket compatible leaves so replica synchronization and reconstruction use
  fewer, larger collectives instead of hundreds of per-leaf launches.
- Keep parameter, momentum, update, and replica-axis contracts explicit.

Algorithmic target:

- Implement sharding-aware Newton-Schulz using local matrix products plus a
  reduction of the smaller Gram matrix. Keep the matrix/update sharded instead
  of all-gathering every full logical matrix for every orthogonalization step.
- Select `X X^T` or `X^T X` from logical shape and shard direction, and prove
  equivalence against the accepted full-logical-matrix reference in its
  configured precision.

Lower-level follow-up:

- Fuse the five Newton-Schulz iterations or their repeated GEMM/elementwise
  sequence only after the distributed algorithm reduces communication.

Gate:

- Preserve the existing fake-device and four-H100 correctness matrix.
- Cut TP Muon's incremental step-time tax and all-gather count by at least 50%
  on both the 285.5M and 1.666B configs.

#### M3. Reduce baseline TP and composed-layout collectives

- Bucket small reductions and avoid repeated layout conversions around
  sequence-parallel projections and vocab-parallel loss.
- For FSDP+TP, prefetch/overlap parameter all-gathers with compute and prefer
  reduce-scatter for gradients when the declared output sharding permits it.
- Investigate why TP AdamW spends 45-54% of GPU event time in NCCL even on
  full NVLink before adding compute kernels.

Gate:

- Reduce the FSDP+TP AdamW penalty over TP AdamW from 36% to below 15% on the
  primary dense config, with unchanged numerics and checkpoint compatibility.

#### M4. Hide input work and cache compilation

This can proceed alongside M1-M3 because it does not change model numerics.

- Exercise the existing prepared-data prefetch/worker path and verify exact
  ordering/resume semantics. Move normalization and next-batch preparation off
  the critical path without introducing a second data stack.
- Add a persistent JAX compilation-cache contract keyed by code/config,
  JAX/XLA version, device architecture, and mesh. Record hits/misses in runtime
  diagnostics.

Gate:

- Data preparation is fully overlapped in steady state for the dense profile
  matrix, and resume produces the same token sequence.
- A repeated unchanged preflight/train process reuses compilation artifacts;
  cache mismatch fails safely rather than loading incompatible executables.

#### M5. Promote only evidence-backed custom kernels

Current priority order:

1. MoE dispatch/combine and grouped expert GEMM.
2. Distributed/fused Newton-Schulz after M2 establishes its algorithm.
3. TP loss or projection kernels only if post-M1/M2/M3 traces still rank them.

Do not prioritize RMSNorm, generic linear, or attention kernels from this
capture; none is currently a top wall-time consumer.

Next:

- Implement M0 first so future changes have an automatic before/after report.
- Then prototype M1 on the current reference dispatcher. Keep the reference
  route available in tests, but expose no second user-facing runtime stack.
- Re-run only the four representative MoE profiles after M1; do not repeat the
  full 15-run matrix until a shared hot path changes.
