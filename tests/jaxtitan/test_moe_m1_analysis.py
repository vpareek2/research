import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "jaxtitan"
    / "analyze_moe_m1_h100_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_moe_m1_h100_results", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _profile(*, train_step_sec: float, scatter_max_sec: float) -> dict[str, object]:
    return {
        "steady": {"medians": {"train_step_sec": train_step_sec}},
        "trace": {
            "categories": {
                "scatter_reduce_fusion": {
                    "count": 100,
                    "duration_sec": scatter_max_sec * 100,
                    "max_duration_sec": scatter_max_sec,
                },
                "gemm": {"event_sum_fraction": 0.5},
            }
        },
    }


def test_compare_gates_pathological_scatter_duration_not_harmless_count() -> None:
    baseline_runs = []
    candidate_runs = []
    for run_id in MODULE.PROFILE_RUNS:
        baseline_runs.append(
            {"run_id": run_id, **_profile(train_step_sec=1.0, scatter_max_sec=0.2)}
        )
        candidate_runs.append(
            {"run_id": run_id, **_profile(train_step_sec=0.1, scatter_max_sec=0.001)}
        )

    payload = MODULE._compare(
        {"runs": candidate_runs, "source": "candidate"},
        {"runs": baseline_runs, "source": "baseline"},
    )

    assert payload["overall_gate"] is True
    assert payload["scatter_reduce_fusion_max_sec_gate"] == pytest.approx(0.010)
    assert all(
        row["candidate_scatter_reduce_fusion_count"] == 100
        for row in payload["profile_runs"]
    )


def test_compare_rejects_slow_scatter_fusion() -> None:
    baseline_runs = []
    candidate_runs = []
    for run_id in MODULE.PROFILE_RUNS:
        baseline_runs.append(
            {"run_id": run_id, **_profile(train_step_sec=1.0, scatter_max_sec=0.2)}
        )
        candidate_runs.append(
            {"run_id": run_id, **_profile(train_step_sec=0.1, scatter_max_sec=0.02)}
        )

    payload = MODULE._compare(
        {"runs": candidate_runs, "source": "candidate"},
        {"runs": baseline_runs, "source": "baseline"},
    )

    assert payload["overall_gate"] is False
