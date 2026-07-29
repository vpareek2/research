import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "jaxtitan"
    / "analyze_dist_muon_m2_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_dist_muon_m2_results",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _analysis(*, duplicated_sec: float, distributed_sec: float) -> dict[str, object]:
    runs = []
    for _layout, duplicated_id, distributed_id in MODULE.RUN_PAIRS:
        runs.extend(
            [
                {
                    "run_id": duplicated_id,
                    "steady": {"medians": {"train_step_sec": duplicated_sec}},
                },
                {
                    "run_id": distributed_id,
                    "steady": {"medians": {"train_step_sec": distributed_sec}},
                },
            ]
        )
    return {"runs": runs}


def test_compare_profile_analysis_requires_every_distributed_layout_to_win() -> None:
    payload = MODULE.compare_profile_analysis(
        _analysis(duplicated_sec=0.5, distributed_sec=0.25)
    )

    assert payload["overall_gate"] is True
    assert all(row["speedup"] == pytest.approx(2.0) for row in payload["pairs"])


def test_compare_profile_analysis_rejects_one_slow_layout() -> None:
    analysis = _analysis(duplicated_sec=0.5, distributed_sec=0.25)
    slow_run_id = MODULE.RUN_PAIRS[-1][2]
    next(
        run for run in analysis["runs"] if run["run_id"] == slow_run_id
    )["steady"]["medians"]["train_step_sec"] = 0.6

    payload = MODULE.compare_profile_analysis(analysis)

    assert payload["overall_gate"] is False
    assert payload["pairs"][-1]["speedup"] == pytest.approx(5.0 / 6.0)
