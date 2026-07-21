#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/jaxtitan/cloud_moe_m1_h100_matrix.sh [--overwrite]

Run the M1 MoE expert-major H100 acceptance matrix from the repository root:
short correctness validations first, then the four 64-step profile runs.

The script verifies hardware/data, emits clear section markers, captures HLO
text under cloud_results/, runs inspect/eval/sample for each completed run,
and packages the relevant runs plus provenance into a timestamped .tgz.
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
capture_name="moe_m1_h100_${timestamp}"
capture_dir="cloud_results/${capture_name}"
hlo_root="${capture_dir}/hlo"
log_dir="${capture_dir}/logs"
mkdir -p "$hlo_root" "$log_dir"

correctness_configs=(
  configs/jaxtitan/cloud_4gpu_trinity_moe_ep_adamw_validation.toml
  configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_adamw_validation.toml
  configs/jaxtitan/cloud_4gpu_trinity_moe_tp_ep_muon_validation.toml
  configs/jaxtitan/cloud_4gpu_trinity_moe_cp_ep_adamw_validation.toml
  configs/jaxtitan/cloud_4gpu_trinity_moe_folded_fsdp_ep_muon_validation.toml
  configs/jaxtitan/cloud_4gpu_trinity_moe_product_fsdp_ep_muon_validation.toml
  configs/jaxtitan/cloud_4gpu_trinity_moe_expert_fsdp_adamw_validation.toml
)

profile_configs=(
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_ep_adamw.toml
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_ep_muon.toml
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_adamw.toml
  configs/jaxtitan/cloud_4gpu_profile64_trinity_moe_tp_ep_muon.toml
)

all_run_dirs=()

marker() {
  local message="$1"
  echo
  echo "================================================================================"
  echo "${message} $(date -Is)"
  echo "================================================================================"
}

run_id_for_config() {
  uv run jaxtitan config check "$1" --json \
    | uv run python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])'
}

check_hardware() {
  marker "HARDWARE"
  git status -sb | tee "${capture_dir}/git_status.txt"
  git rev-parse HEAD | tee "${capture_dir}/commit.txt"
  nvidia-smi | tee "${capture_dir}/nvidia_smi.txt"
  nvidia-smi topo -m | tee "${capture_dir}/nvidia_topology.txt"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    | tee "${capture_dir}/nvidia_gpu_query.csv"
  uv run python - <<'PY' | tee "cloud_results/current_jax_devices.txt" "${capture_dir}/jax_devices.txt"
import jax
for device in jax.devices():
    print(device)
print(f"device_count={len(jax.devices())}")
PY
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if [[ "$gpu_count" -lt 4 ]]; then
    echo "expected at least 4 visible GPUs, saw ${gpu_count}" >&2
    exit 1
  fi
}

prepare_data_if_needed() {
  marker "DATA"
  if [[ ! -f data/tinystories_gpt2_cloud_validation/manifest.json ]]; then
    uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_cloud_validation.toml \
      2>&1 | tee "${log_dir}/data_prepare.log"
  fi
  uv run jaxtitan data check data/tinystories_gpt2_cloud_validation/manifest.json \
    --tokenizer gpt2 --verify-checksums \
    2>&1 | tee "${log_dir}/data_check.log"
  uv run jaxtitan data inspect data/tinystories_gpt2_cloud_validation/manifest.json \
    --tokenizer gpt2 --verify-checksums --seq-len 1024 --json \
    > "${capture_dir}/data_inspect_seq1024.json"
}

run_one_config() {
  local cfg="$1"
  local phase="$2"
  local run_id
  run_id="$(run_id_for_config "$cfg")"
  all_run_dirs+=("runs/${run_id}")

  marker "BEGIN ${phase} ${run_id}"
  echo "CONFIG ${cfg}"
  uv run jaxtitan config check "$cfg"
  uv run jaxtitan run preflight "$cfg"

  local run_hlo_dir="${hlo_root}/${run_id}"
  mkdir -p "$run_hlo_dir"
  local train_xla_flags="${XLA_FLAGS:-} --xla_dump_to=${run_hlo_dir} --xla_dump_hlo_as_text"
  XLA_FLAGS="$train_xla_flags" uv run jaxtitan run train "${overwrite_flag[@]}" "$cfg" \
    2>&1 | tee "${log_dir}/train_${run_id}.log"
  local train_status="${PIPESTATUS[0]}"
  echo "TRAIN_EXIT=${train_status} ${run_id}"
  if [[ "$train_status" -ne 0 ]]; then
    marker "FAILED ${phase} ${run_id}"
    exit "$train_status"
  fi

  uv run jaxtitan run inspect "runs/${run_id}" \
    2>&1 | tee "${log_dir}/inspect_${run_id}.log"
  uv run jaxtitan eval checkpoint "runs/${run_id}" --checkpoint latest --json \
    > "${capture_dir}/eval_${run_id}.json"
  uv run jaxtitan sample checkpoint "runs/${run_id}" --checkpoint latest \
    --prompt-ids "15496,11" --max-new-tokens 8 --top-k 1 --json \
    > "${capture_dir}/sample_${run_id}.json"
  find "runs/${run_id}/profiles" -type f 2>/dev/null | sort \
    | tee "${capture_dir}/profile_files_${run_id}.txt" || true
  du -sh "runs/${run_id}" | tee "${capture_dir}/du_${run_id}.txt"
  marker "END ${phase} ${run_id}"
}

package_results() {
  marker "PACKAGE"
  {
    echo "capture=${capture_name}"
    echo "commit=$(git rev-parse HEAD)"
    echo "branch=$(git branch --show-current)"
    echo "created_utc=${timestamp}"
    echo "overwrite=$([[ ${#overwrite_flag[@]} -gt 0 ]] && echo true || echo false)"
  } > "${capture_dir}/provenance.txt"

  uv run jaxtitan profile analyze runs --json > "${capture_dir}/profile_analysis_current_runs.json" || true
  uv run jaxtitan profile analyze runs > "${capture_dir}/profile_analysis_current_runs.txt" || true

  local archive="cloud_results/${capture_name}.tgz"
  tar -czf "$archive" "${capture_dir}" "${all_run_dirs[@]}"
  sha256sum "$archive" | tee "${archive}.sha256"
  du -sh "$archive"
  marker "ALL_MOE_M1_H100_RUNS_COMPLETE ${archive}"
}

marker "SETUP"
uv sync
check_hardware
prepare_data_if_needed

for cfg in "${correctness_configs[@]}"; do
  run_one_config "$cfg" "CORRECTNESS"
done

for cfg in "${profile_configs[@]}"; do
  run_one_config "$cfg" "PROFILE64"
done

package_results
