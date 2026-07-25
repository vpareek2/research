import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "jaxtitan"
    / "analyze_dist_muon_shape_policy_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_dist_muon_shape_policy_results",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _analysis(multiplier: float = 0.95) -> dict[str, object]:
    return {
        "runs": [
            {
                "run_id": run_id,
                "status": "completed",
                "steady": {
                    "start_step": 16,
                    "end_step": 63,
                    "row_count": 48,
                    "medians": {"train_step_sec": current_sec * multiplier},
                },
            }
            for _layout, run_id, current_sec, _duplicated_sec in MODULE.RUNS
        ]
    }


def _diagnostics() -> dict[str, dict[str, object]]:
    plans = [
        {
            "execution": execution,
            "selection": {
                "policy_version": MODULE.POLICY_VERSION,
                "eligible_executions": ["duplicated", execution],
                "selected_execution": execution,
                "selection_reason": "test",
                "modeled_costs": {"bytes": 1},
            },
            "roles": {
                "parameter": {},
                "gradient": {},
                "momentum": {},
                "update": {},
            },
        }
        for execution in sorted(MODULE.EXPECTED_EXECUTIONS)
    ]
    return {
        run_id: {
            "optimizer": {
                "dist_muon": {
                    "exact": False,
                    "shape_topology_policy": {"version": MODULE.POLICY_VERSION},
                    "leaf_execution_plans": plans,
                }
            }
        }
        for _layout, run_id, _current, _duplicated in MODULE.RUNS
    }


def test_shape_policy_analysis_requires_all_layout_and_geomean_gates() -> None:
    diagnostics = _diagnostics()
    artifacts = {run_id: True for _layout, run_id, _current, _duplicated in MODULE.RUNS}

    payload = MODULE.compare_shape_policy_results(
        _analysis(),
        diagnostics=diagnostics,
        artifact_gate=artifacts,
    )

    assert payload["overall_gate"] is True
    assert payload["geomean_speedup_vs_current"] == pytest.approx(1.0 / 0.95)
    assert all(row["gate"] for row in payload["layouts"])


def test_shape_policy_analysis_rejects_one_current_regression() -> None:
    analysis = _analysis()
    run_id = MODULE.RUNS[-1][1]
    current_sec = MODULE.RUNS[-1][2]
    next(run for run in analysis["runs"] if run["run_id"] == run_id)["steady"]["medians"][
        "train_step_sec"
    ] = current_sec * 1.02

    payload = MODULE.compare_shape_policy_results(
        analysis,
        diagnostics=_diagnostics(),
        artifact_gate={
            candidate_id: True
            for _layout, candidate_id, _current, _duplicated in MODULE.RUNS
        },
    )

    assert payload["overall_gate"] is False
    assert payload["layouts"][-1]["performance_gate"] is False


def test_shape_policy_analysis_rejects_incomplete_policy_metadata() -> None:
    diagnostics = _diagnostics()
    run_id = MODULE.RUNS[0][1]
    diagnostics[run_id]["optimizer"]["dist_muon"]["leaf_execution_plans"][0][
        "selection"
    ].pop("modeled_costs")

    payload = MODULE.compare_shape_policy_results(
        _analysis(),
        diagnostics=diagnostics,
        artifact_gate={
            candidate_id: True
            for _layout, candidate_id, _current, _duplicated in MODULE.RUNS
        },
    )

    assert payload["overall_gate"] is False
    assert payload["layouts"][0]["policy_gate"] is False
