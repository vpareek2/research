#!/usr/bin/env bash

# Source this file for copy/paste command templates. Replace placeholders before
# running. Do not store secrets or host-specific credentials here.

set -euo pipefail

jaxtitan_check() {
  uv run pytest -q
  git diff --check
  rg -n "^(from|import) research(\\.|\\s|$)|from __future__ import annotations" \
    src/jaxtitan tests/jaxtitan configs/jaxtitan || true
}

jaxtitan_fake_cpu_full() {
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    uv run pytest -q
}

jaxtitan_run_smoke() {
  local config_path="${1:?config path required}"
  uv run jaxtitan config check "${config_path}"
  uv run jaxtitan run preflight "${config_path}"
  uv run jaxtitan run train --overwrite "${config_path}"
}

jaxtitan_checkpoint_probe() {
  local run_dir="${1:?run dir required}"
  uv run jaxtitan run inspect "${run_dir}"
  uv run jaxtitan eval checkpoint "${run_dir}" --checkpoint latest --json
  uv run jaxtitan sample checkpoint "${run_dir}" \
    --checkpoint latest \
    --prompt-ids "15496,11" \
    --max-new-tokens 8 \
    --top-k 1 \
    --json
}
