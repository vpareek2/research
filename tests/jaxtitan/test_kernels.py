from pathlib import Path
import subprocess

import pytest

from jaxtitan.config import load_config
from jaxtitan.errors import ContractError
from jaxtitan.kernels import (
    compile_kernel_plan,
    enrich_kernel_plan_with_cache,
    format_kernel_plan,
    kernel_plan,
    kernel_registry_payload,
    load_cache_manifest,
    require_kernel_plan_supported,
)


MINIMAL_CONFIG = """
[run]
id = "smoke"
seed = 11
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 32000
hidden_size = 128
intermediate_size = 512
num_layers = 2
num_heads = 4
max_seq_len = 64

[optimizer]
name = "adamw"
weight_decay = 0.1

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
train_manifest = "data/train/manifest.json"
tokenizer_id = "toy-tokenizer"

[training]
seq_len = 64
global_batch_size = 2
target_tokens = 128
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]
"""


def test_kernel_registry_lists_known_internal_candidates() -> None:
    payload = kernel_registry_payload()

    ops = {item["op"] for item in payload["kernels"]}
    assert "rmsnorm" in ops
    assert "moe_dispatch_combine" in ops
    assert all(item["implementation"] == "thunderkittens" for item in payload["kernels"])


def test_kernel_plan_defaults_to_xla_when_backend_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    plan = kernel_plan(load_config(config_path), device_kind="cpu")

    assert plan["enabled"] is False
    assert plan["mode"] == "xla"
    assert plan["target_ops"] == ["rmsnorm", "attention", "swiglu"]
    assert plan["active_count"] == 0
    assert plan["fallback"] == {
        "rmsnorm": "kernels_disabled",
        "attention": "kernels_disabled",
        "swiglu": "kernels_disabled",
    }
    assert "rmsnorm: backend=xla reason=kernels_disabled" in format_kernel_plan(plan)
    enriched = enrich_kernel_plan_with_cache(plan, root=tmp_path / "kernel-cache")
    assert enriched["missing_cache"] == {"rmsnorm": "not_compiled"}
    assert enriched["decisions"][0]["cache_status"] == "missing"


def test_kernel_plan_reports_unavailable_candidates_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[kernels]
enabled = true
"""
    )

    plan = kernel_plan(load_config(config_path))

    assert plan["enabled"] is True
    assert plan["mode"] == "auto"
    assert plan["active_count"] == 0
    assert plan["fallback_count"] == 3
    assert set(plan["unavailable"]) == {"rmsnorm", "attention", "swiglu"}
    assert set(plan["fallback"].values()) == {"no_jax_ffi_implementation"}


def test_kernel_strict_mode_rejects_unavailable_candidates(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[kernels]
enabled = true
strict = true
"""
    )

    plan = kernel_plan(load_config(config_path))

    with pytest.raises(ContractError, match="kernels.strict=true"):
        require_kernel_plan_supported(plan)


def test_compile_kernel_plan_writes_cache_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    cache_dir = tmp_path / "cache"

    def fake_run(command, *, cwd, check, capture_output, text):
        output = Path(next(part.removeprefix("OUT=") for part in command if part.startswith("OUT=")))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"compiled-rmsnorm")
        return subprocess.CompletedProcess(command, 0, stdout=f"built in {cwd}", stderr="")

    monkeypatch.setattr("jaxtitan.kernels.build.subprocess.run", fake_run)
    monkeypatch.setattr("jaxtitan.kernels.build._nvcc_version", lambda: "nvcc fake")

    plan = compile_kernel_plan(load_config(config_path), arch="SM90", root=cache_dir)
    manifest = load_cache_manifest(cache_dir)

    assert manifest is not None
    assert manifest["arch"] == "SM90"
    assert manifest["artifacts"]["rmsnorm"]["path"] == "rmsnorm_test.out"
    assert plan["cached"] == {"rmsnorm": "rmsnorm_test.out"}
    assert plan["missing_cache_count"] == 0
    assert plan["decisions"][0]["cache_status"] == "cached"


def test_compile_kernel_plan_rejects_bad_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    with pytest.raises(ContractError, match="kernel ARCH"):
        compile_kernel_plan(load_config(config_path), arch="SM70", root=tmp_path / "cache")
