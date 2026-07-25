#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/jaxtitan/cloud_dist_muon_shape_policy_matrix.sh [--overwrite]

Run the four 64-step shape/topology Muon acceptance layouts on one four-H100
node. The script verifies hardware and data, dumps HLO, runs inspect/eval/
sample, validates the archived performance gates, and packages lightweight
evidence. Full profiler trees remain under runs/.
EOF
}

overwrite_flag=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      overwrite_flag=(--overwrite)
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
capture_name="dist_muon_shape_policy_${timestamp}"
capture_dir="cloud_results/${capture_name}"
hlo_root="${capture_dir}/hlo"
log_dir="${capture_dir}/logs"
evidence_root="${capture_dir}/runs"
mkdir -p "$hlo_root" "$log_dir" "$evidence_root"

configs=(
  configs/jaxtitan/cloud_4gpu_profile64_dense_tp_muon_shape_policy.toml
  configs/jaxtitan/cloud_4gpu_profile64_dense_fsdp_tp_muon_shape_policy.toml
  configs/jaxtitan/cloud_4gpu_profile64_dense_zero2_tp_muon_shape_policy.toml
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_muon_shape_policy.toml
)

marker() {
  echo
  echo "================================================================================"
  echo "$1 $(date -Is)"
  echo "================================================================================"
}

run_id_for_config() {
  uv run jaxtitan config check "$1" --json \
    | uv run python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])'
}

capture_hardware() {
  marker "HARDWARE"
  git status -sb | tee "${capture_dir}/git_status.txt"
  git rev-parse HEAD | tee "${capture_dir}/commit.txt"
  nvidia-smi | tee "${capture_dir}/nvidia_smi.txt"
  nvidia-smi topo -m | tee "${capture_dir}/nvidia_topology.txt"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    | tee "${capture_dir}/nvidia_gpu_query.csv"
  uv run python - <<'PY' | tee "${capture_dir}/jax_devices.txt"
import jax
for device in jax.devices():
    print(device)
print(f"device_count={len(jax.devices())}")
PY
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if [[ "$gpu_count" -ne 4 ]]; then
    echo "expected exactly 4 visible GPUs, saw ${gpu_count}" >&2
    exit 1
  fi
}

prepare_data() {
  marker "DATA"
  if [[ ! -f data/tinystories_gpt2_cloud_validation/manifest.json ]]; then
    uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_cloud_validation.toml \
      2>&1 | tee "${log_dir}/data_prepare.log"
  fi
  uv run jaxtitan data check data/tinystories_gpt2_cloud_validation/manifest.json \
    --tokenizer gpt2 --verify-checksums \
    2>&1 | tee "${log_dir}/data_check.log"
}

capture_run_evidence() {
  local run_id="$1"
  local source="runs/${run_id}"
  local destination="${evidence_root}/${run_id}"
  mkdir -p "$destination"
  for path in config.resolved.toml config diagnostics events.jsonl metrics summaries; do
    if [[ -e "${source}/${path}" ]]; then
      cp -a "${source}/${path}" "$destination/"
    fi
  done
  if [[ -f "${source}/checkpoints/index.json" ]]; then
    mkdir -p "${destination}/checkpoints"
    cp -a "${source}/checkpoints/index.json" "${destination}/checkpoints/"
  fi
  find "${source}/profiles" -type f 2>/dev/null | sort \
    > "${capture_dir}/profile_files_${run_id}.txt" || true
  du -sh "$source" | tee "${capture_dir}/du_${run_id}.txt"
}

run_one() {
  local cfg="$1"
  local run_id
  run_id="$(run_id_for_config "$cfg")"

  marker "BEGIN PROFILE64 ${run_id}"
  echo "CONFIG ${cfg}"
  uv run jaxtitan config check "$cfg"
  uv run jaxtitan run preflight "$cfg"

  local run_hlo_dir="${hlo_root}/${run_id}"
  mkdir -p "$run_hlo_dir"
  local train_xla_flags="${XLA_FLAGS:-} --xla_dump_to=${run_hlo_dir} --xla_dump_hlo_as_text"
  XLA_FLAGS="$train_xla_flags" uv run jaxtitan run train "${overwrite_flag[@]}" "$cfg" \
    2>&1 | tee "${log_dir}/train_${run_id}.log"
  local train_status="${PIPESTATUS[0]}"
  if [[ "$train_status" -ne 0 ]]; then
    marker "FAILED PROFILE64 ${run_id}"
    exit "$train_status"
  fi

  uv run jaxtitan run inspect "runs/${run_id}" \
    2>&1 | tee "${log_dir}/inspect_${run_id}.log"
  uv run jaxtitan eval checkpoint "runs/${run_id}" --checkpoint latest --json \
    > "${capture_dir}/eval_${run_id}.json"
  uv run jaxtitan sample checkpoint "runs/${run_id}" --checkpoint latest \
    --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json \
    > "${capture_dir}/sample_${run_id}.json"
  local audit_status=0
  if ! uv run python scripts/jaxtitan/audit_dist_muon_checkpoint.py \
    "runs/${run_id}" --checkpoint latest \
    --json-out "${capture_dir}/replica_audit_${run_id}.json" \
    > "${log_dir}/replica_audit_${run_id}.log"; then
    audit_status=1
  fi
  echo "REPLICA_AUDIT_EXIT=${audit_status} ${run_id}" \
    | tee -a "${log_dir}/replica_audit_${run_id}.log"
  capture_run_evidence "$run_id"
  marker "END PROFILE64 ${run_id}"
}

package_results() {
  marker "ANALYZE_AND_PACKAGE"
  uv run jaxtitan profile analyze runs --json > "${capture_dir}/profile_analysis.json"
  uv run jaxtitan profile analyze runs > "${capture_dir}/profile_analysis.txt"
  local analysis_status=0
  if ! uv run python scripts/jaxtitan/analyze_dist_muon_shape_policy_results.py \
    "${capture_dir}" --json-out "${capture_dir}/comparison.json" \
    | tee "${capture_dir}/comparison.txt"; then
    analysis_status=1
  fi
  {
    echo "capture=${capture_name}"
    echo "commit=$(git rev-parse HEAD)"
    echo "branch=$(git branch --show-current)"
    echo "created_utc=${timestamp}"
    echo "baseline_artifact_sha256=65fb879f2636778aa5a25d6566b1538a9ea533cfceb1439428bcdbd433d2db72"
  } > "${capture_dir}/provenance.txt"

  local archive="cloud_results/${capture_name}_lightweight.tgz"
  tar -czf "$archive" "$capture_dir"
  sha256sum "$archive" | tee "${archive}.sha256"
  du -sh "$archive"
  marker "ALL_DIST_MUON_SHAPE_POLICY_RUNS_COMPLETE ${archive}"
  return "$analysis_status"
}

marker "SETUP"
uv sync
capture_hardware
prepare_data
for cfg in "${configs[@]}"; do
  run_one "$cfg"
done
package_results
