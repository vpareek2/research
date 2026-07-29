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
    orthogonalization: str | None = None,
    restart_indices: tuple[int, ...] = (),
    hlo_gate: bool = True,
) -> dict[str, object]:
    candidate = {
        "execution": execution,
        "correctness_gate": correct,
        "timing_ms": {
            "median": median_ms,
            "p95": median_ms * 1.05,
            "median_abs_deviation": mad_ms,
        },
    }
    if orthogonalization is not None:
        suffix = ""
        if restart_indices:
            suffix = "_r" + "_".join(str(index) for index in restart_indices)
        elif orthogonalization == "gram_newton_schulz":
            suffix = "_no_restart"
        candidate.update(
            {
                "candidate_id": f"{execution}__{orthogonalization}{suffix}",
                "orthogonalization": orthogonalization,
                "restart_indices": list(restart_indices),
                "hlo_contract": {"gate": hlo_gate},
            }
        )
    return candidate


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
                "leaf_count": 24,
                "kind": "production_bucket",
                "candidates": candidates,
            }
        ],
    }


def _v2_payload(candidates: list[dict[str, object]]) -> dict[str, object]:
    payload = _payload(candidates)
    case = payload["cases"][0]
    assert isinstance(case, dict)
    case.update(
        {
            "candidate_schema_version": 2,
            "reference": "duplicated__standard_newton_schulz",
            "production_candidate_id": "distributed_large_gram__standard_newton_schulz",
        }
    )
    return payload


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
    assert result["production_recommendations"][0]["policy_features"] == {
        "short_dimension": 256,
        "long_dimension": 1024,
        "aspect_ratio": 4.0,
        "canonical_tp_dim": 0,
        "tp_size": 4,
        "leaf_count": 24,
        "matrix_elements_per_leaf": 262144,
        "aggregate_matrix_elements": 6291456,
    }
    assert result["shape_topology_samples"][0]["selected_execution"] == "distributed_exchange"


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


def test_schema_v2_promotes_recurrence_only_against_same_transport() -> None:
    result = MODULE.analyze_benchmark(
        _v2_payload(
            [
                _candidate(
                    "duplicated",
                    median_ms=1.0,
                    orthogonalization="standard_newton_schulz",
                ),
                _candidate(
                    "duplicated",
                    median_ms=0.94,
                    orthogonalization="gram_newton_schulz",
                    restart_indices=(2,),
                ),
                _candidate(
                    "distributed_large_gram",
                    median_ms=0.8,
                    orthogonalization="standard_newton_schulz",
                ),
                _candidate(
                    "distributed_large_gram",
                    median_ms=0.7,
                    orthogonalization="gram_newton_schulz",
                    restart_indices=(2,),
                ),
            ]
        )
    )

    case = result["cases"][0]
    assert result["schema_version"] == 2
    assert result["overall_gate"] is True
    assert (
        case["selected_candidate_id"]
        == "distributed_large_gram__gram_newton_schulz_r2"
    )
    selected = next(
        candidate
        for candidate in case["candidates"]
        if candidate["candidate_id"] == case["selected_candidate_id"]
    )
    assert selected["promotion_eligible"] is True
    assert selected["speedup_vs_same_transport_standard"] == pytest.approx(0.8 / 0.7)


def test_schema_v2_tie_band_favors_standard_orthogonalization() -> None:
    result = MODULE.analyze_benchmark(
        _v2_payload(
            [
                _candidate(
                    "duplicated",
                    median_ms=1.0,
                    orthogonalization="standard_newton_schulz",
                ),
                _candidate(
                    "distributed_large_gram",
                    median_ms=0.7,
                    orthogonalization="standard_newton_schulz",
                ),
                _candidate(
                    "distributed_large_gram",
                    median_ms=0.69,
                    orthogonalization="gram_newton_schulz",
                    restart_indices=(2,),
                ),
            ]
        )
    )

    assert (
        result["cases"][0]["selected_candidate_id"]
        == "distributed_large_gram__standard_newton_schulz"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("correct", False, "gate"),
        ("hlo", False, "gate"),
    ],
)
def test_schema_v2_recurrence_failure_blocks_benchmark(
    field: str,
    value: bool,
    message: str,
) -> None:
    recurrence = _candidate(
        "distributed_large_gram",
        median_ms=0.7,
        correct=value if field == "correct" else True,
        orthogonalization="gram_newton_schulz",
        restart_indices=(2,),
        hlo_gate=value if field == "hlo" else True,
    )
    result = MODULE.analyze_benchmark(
        _v2_payload(
            [
                _candidate(
                    "duplicated",
                    median_ms=1.0,
                    orthogonalization="standard_newton_schulz",
                ),
                _candidate(
                    "distributed_large_gram",
                    median_ms=0.8,
                    orthogonalization="standard_newton_schulz",
                ),
                recurrence,
            ]
        )
    )

    assert result["overall_gate"] is False, message


def test_schema_v2_rejects_inconsistent_candidate_identity() -> None:
    candidate = _candidate(
        "duplicated",
        median_ms=1.0,
        orthogonalization="standard_newton_schulz",
    )
    candidate["candidate_id"] = "wrong"

    with pytest.raises(ValueError, match="does not match"):
        MODULE.analyze_benchmark(_v2_payload([candidate]))
