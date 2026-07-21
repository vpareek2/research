import json

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import profile_bench


def test_benchmark_component_emits_stable_non_gating_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_bench,
        "_benchmark_moe",
        lambda *, warmup, iters: [
            {
                "name": "case",
                "compile_sec": 1.0,
                "timing_ms": {"median": 2.0, "p10": 1.0, "p90": 3.0},
            }
        ],
    )

    payload = profile_bench.benchmark_component("moe", warmup=1, iters=2)

    assert payload["component"] == "moe"
    assert payload["timing_is_acceptance_gate"] is False
    assert payload["correctness_is_checked"] is False
    assert "not an accepted correctness baseline" in payload["known_correctness_constraint"]
    assert payload["warmup"] == 1
    assert payload["iters"] == 2
    assert json.loads(profile_bench.benchmark_to_json(payload))["schema_version"] == 1
    formatted = profile_bench.format_benchmark(payload)
    assert "timing_gate=false" in formatted
    assert "correctness_check=false" in formatted


@pytest.mark.parametrize(
    ("component", "warmup", "iters", "message"),
    [
        ("unknown", 1, 1, "unsupported"),
        ("moe", -1, 1, "warmup"),
        ("muon", 1, 0, "iterations"),
    ],
)
def test_benchmark_component_validates_arguments(component: str, warmup: int, iters: int, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        profile_bench.benchmark_component(component, warmup=warmup, iters=iters)


def test_timing_summary_is_deterministic() -> None:
    assert profile_bench._timing_summary([4.0, 1.0, 3.0, 2.0]) == {
        "median": 2.5,
        "p10": pytest.approx(1.3),
        "p90": pytest.approx(3.7),
        "min": 1.0,
        "max": 4.0,
    }
