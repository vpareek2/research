#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/jaxtitan/cloud_dist_muon_leaf_bench.sh [--with-trace]

Run the correctness-checked distributed-Muon leaf selector on exactly four
local GPUs. Canonical selection timing is never profiled. --with-trace performs
a second short, explicitly non-canonical traced pass for kernel inspection.
EOF
}

with_trace=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-trace)
      with_trace=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
capture_name="dist_muon_leaf_bench_${timestamp}"
capture_dir="cloud_results/${capture_name}"
benchmark_dir="${capture_dir}/benchmark"
mkdir -p "$benchmark_dir"

marker() {
  echo
  echo "================================================================================"
  echo "$1 $(date -Is)"
  echo "================================================================================"
}

marker "SETUP"
uv sync
git status -sb | tee "${capture_dir}/git_status.txt"
git rev-parse HEAD | tee "${capture_dir}/commit.txt"
nvidia-smi | tee "${capture_dir}/nvidia_smi.txt"
nvidia-smi topo -m | tee "${capture_dir}/nvidia_topology.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pstate,clocks.sm,clocks.mem \
  --format=csv,noheader | tee "${capture_dir}/nvidia_gpu_query.csv"
uv run python - <<'PY' | tee "${capture_dir}/jax_devices.txt"
import jax
import jaxlib

print(f"jax={jax.__version__}")
print(f"jaxlib={jaxlib.__version__}")
for device in jax.devices():
    print(device)
print(f"device_count={len(jax.devices())}")
PY

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -ne 4 ]]; then
  echo "expected exactly 4 visible GPUs, saw ${gpu_count}" >&2
  exit 1
fi

marker "CANONICAL_BENCHMARK"
uv run jaxtitan profile bench muon \
  --warmup 10 \
  --iters 100 \
  --artifact-dir "$benchmark_dir" \
  --json > "${capture_dir}/benchmark.stdout.json"

uv run python scripts/jaxtitan/analyze_dist_muon_leaf_bench.py \
  "${benchmark_dir}/benchmark.json" \
  --json-out "${capture_dir}/selection.json" \
  | tee "${capture_dir}/selection.txt"

if [[ "$with_trace" -eq 1 ]]; then
  marker "NON_CANONICAL_TRACE_PASS"
  uv run jaxtitan profile bench muon \
    --warmup 2 \
    --iters 5 \
    --artifact-dir "${capture_dir}/traced_benchmark" \
    --trace \
    --json > "${capture_dir}/traced_benchmark.stdout.json"
fi

marker "PACKAGE"
{
  echo "capture=${capture_name}"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "created_utc=${timestamp}"
  echo "canonical_warmup=10"
  echo "canonical_iters=100"
  echo "traced_pass=${with_trace}"
} > "${capture_dir}/provenance.txt"

archive="cloud_results/${capture_name}.tgz"
tar -czf "$archive" "$capture_dir"
sha256sum "$archive" | tee "${archive}.sha256"
du -sh "$archive"
marker "ALL_DIST_MUON_LEAF_BENCHMARKS_COMPLETE ${archive}"
