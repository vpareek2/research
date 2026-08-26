import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT_PATH = ROOT / "scripts" / "jaxtitan" / "replay_dist_muon_shape_policy.py"
FIXTURE_PATH = ROOT / "scripts" / "jaxtitan" / "dist_muon_shape_policy_h100_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "replay_dist_muon_shape_policy",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _distributed_sample(payload: dict[str, object]) -> dict[str, object]:
    return next(
        sample
        for sample in payload["samples"]
        if sample["accepted_execution"] != "duplicated"
    )


def _multi_distributed_sample(payload: dict[str, object]) -> dict[str, object]:
    return next(
        sample
        for sample in payload["samples"]
        if sample["accepted_execution"] != "duplicated"
        and sum(
            candidate["execution"] != "duplicated"
            and candidate["correctness_gate"] is True
            and candidate["stability_gate"] is True
            and candidate["speedup_vs_duplicated"] >= MODULE.MINIMUM_SPEEDUP
            for candidate in sample["candidates"]
        )
        >= 2
    )


def _selected_candidate(sample: dict[str, object]) -> dict[str, object]:
    return next(
        candidate
        for candidate in sample["candidates"]
        if candidate["execution"] == sample["accepted_execution"]
    )


def test_tracked_h100_fixture_replays_all_63_samples() -> None:
    result = MODULE.replay_fixture(_fixture())

    assert result["overall_gate"] is True
    assert result["performance_claim"] is False
    assert result["sample_count"] == 63
    assert result["decision_match_count"] == 63
    assert result["decision_mismatch_count"] == 0
    assert result["failures"] == []
    assert result["selected_execution_counts"] == {
        "distributed_direct": 32,
        "distributed_exchange": 19,
        "distributed_large_gram": 8,
        "duplicated": 4,
    }
    assert result["geomean_speedup_vs_duplicated"] > 1.0
    assert result["geomean_speedup_vs_previous"] > 1.0
    assert result["worst_measured_regret"] >= 1.0
    assert {item["key"] for item in result["breakdowns"]["topology"]} == {
        "fsdp2_tp2",
        "tp2_ep2",
        "tp4",
    }


def test_replay_output_and_format_are_deterministic() -> None:
    first = MODULE.replay_fixture(_fixture())
    second = MODULE.replay_fixture(_fixture())

    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second,
        sort_keys=True,
        allow_nan=False,
    )
    assert MODULE.format_replay(first).endswith("overall_gate=True")


def test_fixture_passes_only_portable_features_to_selector() -> None:
    seen = []

    def selector(**kwargs):
        seen.append(kwargs)
        return MODULE.select_muon_execution(**kwargs)

    fixture = _fixture()
    result = MODULE.replay_fixture(fixture, selector=selector)

    assert result["overall_gate"] is True
    assert len(seen) == 63
    assert all(
        set(kwargs)
        == {
            "requested_mode",
            "canonical_tp_dim",
            "logical_shape",
            "tp_size",
            "cohort_size",
        }
        for kwargs in seen
    )
    assert all(
        set(sample["features"]) == set(MODULE.PORTABLE_FEATURES)
        for sample in fixture["samples"]
    )
    forbidden = {"role", "tag", "variant", "device_kind", "accelerator"}
    assert all(not (set(kwargs) & forbidden) for kwargs in seen)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(schema_version=999),
            "fixture schema",
        ),
        (
            lambda payload: payload["source"].update(sha256="bad"),
            "source SHA-256",
        ),
        (
            lambda payload: payload["hardware"].update(device_count=8),
            "device_count",
        ),
        (
            lambda payload: payload["contract"].update(minimum_speedup=1.0),
            "minimum_speedup",
        ),
    ],
)
def test_replay_rejects_malformed_fixture_provenance(mutation, message: str) -> None:
    payload = _fixture()
    mutation(payload)

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert any(message in failure for failure in result["failures"])


def test_replay_rejects_missing_candidates() -> None:
    payload = _fixture()
    payload["samples"][0]["candidates"] = []

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert any("candidates must be a non-empty list" in item for item in result["failures"])


def test_replay_rejects_selector_mismatch() -> None:
    payload = _fixture()
    sample = _distributed_sample(payload)
    sample["accepted_execution"] = "duplicated"

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert result["decision_mismatch_count"] == 1
    assert any("expected duplicated" in item for item in result["failures"])


def test_replay_rejects_nonfinite_timing() -> None:
    payload = _fixture()
    candidate = _selected_candidate(_distributed_sample(payload))
    candidate["median_ms"] = math.nan

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert any("nonfinite median_ms" in item for item in result["failures"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("correctness_gate", False, "failed correctness"),
        ("stability_gate", False, "failed stability"),
        ("mad_fraction", 0.051, "exceeds 5% MAD"),
        ("speedup_vs_duplicated", 1.049, "below 1.05x speedup"),
    ],
)
def test_replay_rejects_selected_candidate_gate_failure(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _fixture()
    candidate = _selected_candidate(_distributed_sample(payload))
    candidate[field] = value

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert any(message in item for item in result["failures"])


def test_replay_rejects_candidate_outside_simplicity_tie_band() -> None:
    payload = _fixture()
    sample = _multi_distributed_sample(payload)
    selected = _selected_candidate(sample)
    fastest_other = min(
        candidate["median_ms"]
        for candidate in sample["candidates"]
        if candidate["execution"] not in {"duplicated", selected["execution"]}
        and candidate["correctness_gate"] is True
        and candidate["stability_gate"] is True
        and candidate["speedup_vs_duplicated"] >= MODULE.MINIMUM_SPEEDUP
    )
    selected["median_ms"] = fastest_other * 1.031

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert any("exceeds the 3% tie band" in item for item in result["failures"])


def test_replay_rejects_duplicated_fallback_when_distributed_candidate_passes() -> None:
    payload = _fixture()
    sample = next(
        item
        for item in payload["samples"]
        if item["accepted_execution"] == "duplicated"
    )
    candidate = next(
        item
        for item in sample["candidates"]
        if item["execution"] != "duplicated"
    )
    candidate["correctness_gate"] = True
    candidate["stability_gate"] = True
    candidate["speedup_vs_duplicated"] = 1.1

    result = MODULE.replay_fixture(payload)

    assert result["overall_gate"] is False
    assert any(
        "selected duplicated despite a passing distributed candidate" in item
        for item in result["failures"]
    )


def test_fixture_extraction_is_sanitized_and_deterministic() -> None:
    selection = {
        "schema_version": 1,
        "overall_gate": True,
        "hardware": {
            "backend": "gpu",
            "device_kind": "NVIDIA H100 80GB HBM3",
            "device_count": 4,
            "hostname": "must-not-survive",
        },
        "cases": [
            {
                "name": "case",
                "topology": "tp4",
                "role": "must-not-survive",
                "current_execution": "distributed_direct",
                "candidates": [
                    {
                        "execution": "duplicated",
                        "median_ms": 2.0,
                        "p95_ms": 2.1,
                        "mad_fraction": 0.01,
                        "correctness_gate": True,
                        "stability_gate": True,
                        "speedup_vs_duplicated": 1.0,
                        "collective_operand_model": {"kind": "gather"},
                    },
                    {
                        "execution": "distributed_direct",
                        "median_ms": 1.0,
                        "p95_ms": 1.1,
                        "mad_fraction": 0.01,
                        "correctness_gate": True,
                        "stability_gate": True,
                        "speedup_vs_duplicated": 2.0,
                        "collective_operand_model": {"kind": "gram"},
                    },
                ],
            }
        ],
        "shape_topology_samples": [
            {
                "source_case": "case",
                "source_kind": "calibration_bucket",
                "selected_execution": "distributed_direct",
                "policy_features": {
                    "short_dimension": 8,
                    "long_dimension": 16,
                    "canonical_tp_dim": 1,
                    "tp_size": 4,
                    "leaf_count": 12,
                    "role": "must-not-survive",
                },
            }
        ],
    }

    fixture = MODULE.fixture_from_selection(
        selection,
        source_artifact="artifact.tgz",
        source_sha256="sha",
    )

    assert fixture["hardware"] == {
        "backend": "gpu",
        "device_kind": "NVIDIA H100 80GB HBM3",
        "device_count": 4,
    }
    assert fixture["samples"][0]["features"] == {
        "short_dimension": 8,
        "long_dimension": 16,
        "canonical_tp_dim": 1,
        "tp_size": 4,
        "cohort_size": 12,
    }
    assert "must-not-survive" not in json.dumps(fixture, sort_keys=True)
