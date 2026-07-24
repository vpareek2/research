import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "jaxtitan"
    / "analyze_dist_muon_leaf_bench.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_dist_muon_leaf_bench",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(
    execution: str,
    *,
    median_ms: float,
    correct: bool = True,
    mad_ms: float = 0.01,
) -> dict[str, object]:
    return {
        "execution": execution,
        "correctness_gate": correct,
        "timing_ms": {
            "median": median_ms,
            "p95": median_ms * 1.05,
            "median_abs_deviation": mad_ms,
        },
    }


def _payload(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "component": "muon",
        "correctness_is_checked": True,
        "backend": "gpu",
        "device_kind": "H100",
        "device_count": 4,
        "cases": [
            {
                "name": "tp4_attention_kv",
                "topology": "tp4",
                "mesh": {"tp": 4},
                "role": "attention_kv",
                "shape": [1024, 256],
                "partition_spec": "P(None, 'tp')",
                "tp_partition_dim": 1,
                "canonical_tp_dim": 0,
                "candidates": candidates,
            }
        ],
    }


def test_selector_chooses_clear_fastest_correct_candidate() -> None:
    result = MODULE.analyze_benchmark(
        _payload(
            [
                _candidate("duplicated", median_ms=1.0),
                _candidate("distributed_large_gram", median_ms=0.8),
                _candidate("distributed_exchange", median_ms=0.7),
            ]
        )
    )

    assert result["overall_gate"] is True
    assert result["cases"][0]["selected_execution"] == "distributed_exchange"
    assert result["cases"][0]["selected_speedup_vs_duplicated"] == pytest.approx(1.0 / 0.7)


def test_selector_prefers_simpler_candidate_inside_tie_band() -> None:
    result = MODULE.analyze_benchmark(
        _payload(
            [
                _candidate("duplicated", median_ms=1.0),
                _candidate("distributed_large_gram", median_ms=0.71),
                _candidate("distributed_exchange", median_ms=0.70),
            ]
        )
    )

    assert result["cases"][0]["selected_execution"] == "distributed_large_gram"


def test_selector_falls_back_when_speedup_is_noise_or_candidate_is_incorrect() -> None:
    result = MODULE.analyze_benchmark(
        _payload(
            [
                _candidate("duplicated", median_ms=1.0),
                _candidate("distributed_large_gram", median_ms=0.7, correct=False),
                _candidate("distributed_exchange", median_ms=0.97),
            ]
        )
    )

    assert result["overall_gate"] is True
    assert result["cases"][0]["selected_execution"] == "duplicated"


def test_selector_rejects_current_route_correctness_failure() -> None:
    result = MODULE.analyze_benchmark(
        _payload(
            [
                _candidate("duplicated", median_ms=1.0),
                _candidate("distributed_large_gram", median_ms=0.8),
                _candidate("distributed_exchange", median_ms=0.7, correct=False),
            ]
        )
    )

    assert result["overall_gate"] is False
