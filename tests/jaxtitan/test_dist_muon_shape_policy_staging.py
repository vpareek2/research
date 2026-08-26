import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest

from jaxtitan.config import load_config, run_spec_to_dict


ROOT = Path(__file__).parents[2]
ANALYZER_PATH = (
    ROOT / "scripts" / "jaxtitan" / "analyze_dist_muon_shape_policy_results.py"
)
RUNNER_PATH = ROOT / "scripts" / "jaxtitan" / "cloud_dist_muon_shape_policy_matrix.sh"
SPEC = importlib.util.spec_from_file_location(
    "analyze_dist_muon_shape_policy_results_staging",
    ANALYZER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CONFIG_PAIRS = (
    (
        "cloud_4gpu_smoke8_dense_tp_muon_shape_policy.toml",
        "cloud_4gpu_profile64_dense_tp_muon_shape_policy.toml",
    ),
    (
        "cloud_4gpu_smoke8_dense_fsdp_tp_muon_shape_policy.toml",
        "cloud_4gpu_profile64_dense_fsdp_tp_muon_shape_policy.toml",
    ),
    (
        "cloud_4gpu_smoke8_dense_zero2_tp_muon_shape_policy.toml",
        "cloud_4gpu_profile64_dense_zero2_tp_muon_shape_policy.toml",
    ),
    (
        "cloud_4gpu_smoke8_trinity_moe_tp_ep_muon_shape_policy.toml",
        "cloud_4gpu_profile64_trinity_moe_tp_ep_muon_shape_policy.toml",
    ),
)


def _without_staging_fields(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    normalized.pop("run_id")
    normalized["optimizer"]["schedule"].pop("total_steps")
    normalized["optimizer"]["adamw_fallback_schedule"].pop("total_steps")
    normalized["training"].pop("target_tokens")
    normalized["training"].pop("checkpoint_every_steps")
    normalized["profiling"].pop("enabled")
    for evaluation in normalized["evals"]:
        evaluation.pop("every_steps")
    return normalized


@pytest.mark.parametrize(("smoke_name", "profile_name"), CONFIG_PAIRS)
def test_smoke_config_diff_is_limited_to_staging_fields(
    smoke_name: str,
    profile_name: str,
) -> None:
    config_root = ROOT / "configs" / "jaxtitan"
    smoke = run_spec_to_dict(load_config(config_root / smoke_name))
    profile = run_spec_to_dict(load_config(config_root / profile_name))

    assert smoke["run_id"].startswith("cloud_4gpu_smoke8_")
    assert smoke["run_id"].endswith("_muon_shape_policy")
    assert smoke["optimizer"]["schedule"]["total_steps"] == 8
    assert smoke["optimizer"]["adamw_fallback_schedule"]["total_steps"] == 8
    assert smoke["training"]["target_tokens"] == 32768
    assert smoke["training"]["checkpoint_every_steps"] == 8
    assert smoke["profiling"]["enabled"] is False
    assert [evaluation["every_steps"] for evaluation in smoke["evals"]] == [8]
    assert _without_staging_fields(smoke) == _without_staging_fields(profile)


def _passing_smoke_gate(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "schema_version": MODULE.COMPARISON_SCHEMA_VERSION,
        "phase": "smoke",
        "policy_version": MODULE.POLICY_VERSION,
        "source_commit": commit,
        "overall_gate": True,
        "layouts": [
            {"layout": layout, "gate": True}
            for layout, _run_id in MODULE.SMOKE_RUNS
        ],
    }


def test_same_commit_complete_smoke_gate_allows_profile() -> None:
    result = MODULE.verify_smoke_gate(
        _passing_smoke_gate(),
        current_commit="a" * 40,
    )

    assert result["gate"] is True
    assert result["failures"] == []


def test_profile_gate_rejects_noncanonical_current_commit() -> None:
    result = MODULE.verify_smoke_gate(
        _passing_smoke_gate(commit="short"),
        current_commit="short",
    )

    assert result["gate"] is False
    assert any("40-character" in failure for failure in result["failures"])
    assert any("full Git SHA" in failure for failure in result["failures"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(schema_version=999),
            "schema",
        ),
        (
            lambda payload: payload.update(phase="profile"),
            "phase",
        ),
        (
            lambda payload: payload.update(policy_version="other"),
            "policy version",
        ),
        (
            lambda payload: payload.update(source_commit="b" * 40),
            "commit",
        ),
        (
            lambda payload: payload.update(overall_gate=False),
            "overall_gate",
        ),
        (
            lambda payload: payload["layouts"].pop(),
            "all four layouts",
        ),
        (
            lambda payload: payload["layouts"][0].update(gate=False),
            "layouts did not pass",
        ),
    ],
)
def test_profile_gate_rejects_invalid_smoke_evidence(mutation, message: str) -> None:
    payload = _passing_smoke_gate()
    mutation(payload)

    result = MODULE.verify_smoke_gate(payload, current_commit="a" * 40)

    assert result["gate"] is False
    assert any(message in failure for failure in result["failures"])


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--overwrite",),
        ("--phase", "profile"),
        ("--phase", "smoke", "--smoke-gate", "comparison.json"),
        ("--phase", "invalid"),
    ],
)
def test_cloud_runner_rejects_unsafe_phase_arguments(args: tuple[str, ...]) -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER_PATH), *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 2


def test_cloud_runner_help_documents_two_explicit_commands() -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER_PATH), "--help"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert "--phase smoke" in completed.stdout
    assert "--phase profile" in completed.stdout
    assert "--smoke-gate" in completed.stdout
    assert "no implicit or combined phase" in completed.stdout
