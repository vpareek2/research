import json

import jax
import numpy as np
import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import muon_bench, profile_bench


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
    assert payload["known_correctness_constraint"] is None
    assert payload["warmup"] == 1
    assert payload["iters"] == 2
    assert json.loads(profile_bench.benchmark_to_json(payload))["schema_version"] == 2
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


def test_muon_candidate_matrix_only_offers_valid_exact_topologies() -> None:
    assert muon_bench._eligible_executions(
        shape=(1024, 4096),
        tp_partition_dim=1,
        tp_size=4,
    ) == ("duplicated", "distributed_direct")
    assert muon_bench._eligible_executions(
        shape=(1024, 256),
        tp_partition_dim=1,
        tp_size=4,
    ) == (
        "duplicated",
        "distributed_large_gram",
        "distributed_exchange",
    )
    assert muon_bench._eligible_executions(
        shape=(1025, 256),
        tp_partition_dim=1,
        tp_size=4,
    ) == ("duplicated", "distributed_large_gram")


def test_muon_timing_summary_reports_tail_and_stability() -> None:
    payload = muon_bench._timing_summary([4.0, 1.0, 3.0, 2.0])

    assert payload == {
        "median": 2.5,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "min": 1.0,
        "max": 4.0,
        "median_abs_deviation": 1.0,
    }


def test_muon_calibration_matrix_covers_width_ratio_count_and_orientation() -> None:
    cases = muon_bench._calibration_bucket_cases()

    assert {case.leaf.shape[0] for case in cases} == {256, 1024, 2048}
    assert {
        case.leaf.shape[1] / case.leaf.shape[0]
        for case in cases
    } == {1.0, 2.0, 4.0}
    assert {case.leaf_count for case in cases} == {1, 12, 24}
    assert {case.leaf.tp_partition_dim for case in cases} == {0, 1}
    assert len(cases) == 14


def test_muon_production_bucket_matrix_covers_every_rank_two_role() -> None:
    cases = muon_bench._PRODUCTION_BUCKET_CASES

    assert {case.name for case in cases} == {
        "attention_kv_bucket24",
        "attention_q_gate_bucket12",
        "attention_o_bucket12",
        "shared_mlp_gate_up_bucket20",
        "shared_mlp_down_bucket10",
        "dense_mlp_gate_up_bucket24",
        "dense_mlp_down_bucket12",
    }
    assert {case.leaf_count for case in cases}.issuperset({10, 12, 20, 24})
    assert {case.kind for case in cases} == {"production_bucket"}
    assert len({case.name for case in cases}) == len(cases)


def test_muon_policy_features_are_shape_and_topology_only() -> None:
    assert muon_bench._policy_features(
        shape=(1024, 4096),
        canonical_tp_dim=1,
        tp_size=4,
        leaf_count=24,
    ) == {
        "short_dimension": 1024,
        "long_dimension": 4096,
        "aspect_ratio": 4.0,
        "canonical_tp_dim": 1,
        "tp_size": 4,
        "leaf_count": 24,
        "matrix_elements_per_leaf": 4194304,
        "aggregate_matrix_elements": 100663296,
    }


def test_muon_artifact_manifest_requires_one_unique_hlo_per_candidate() -> None:
    cases = [
        {
            "candidates": [
                {"name": "case_duplicated"},
                {"name": "case_distributed"},
            ]
        }
    ]
    manifest = {
        "hlo_files": [
            "hlo/case_duplicated.txt",
            "hlo/case_distributed.txt",
        ]
    }

    muon_bench.validate_artifact_manifest(cases, manifest)
    with pytest.raises(ContractError, match="incomplete"):
        muon_bench.validate_artifact_manifest(
            cases,
            {"hlo_files": ["hlo/case_duplicated.txt"]},
        )
    with pytest.raises(ContractError, match="not unique"):
        muon_bench.validate_artifact_manifest(
            [{"candidates": [{"name": "same"}, {"name": "same"}]}],
            {"hlo_files": ["hlo/same.txt"]},
        )


def test_muon_benchmark_scales_reference_error_envelope_to_production_lr() -> None:
    contract = muon_bench.benchmark_contract()

    assert contract["benchmark_learning_rate"] == pytest.approx(0.02)
    assert contract["reference_calibration_learning_rate"] == pytest.approx(0.001)
    assert contract["update_atol"] == pytest.approx(0.012)
    assert contract["parameter_atol"] == pytest.approx(0.025)
    assert contract["tolerance_scaling"] == "linear_with_learning_rate"


def test_profile_benchmark_trace_requires_artifact_directory() -> None:
    with pytest.raises(ContractError, match="artifact-dir"):
        profile_bench.benchmark_component("muon", warmup=1, iters=1, trace=True)


def test_muon_leaf_benchmark_executes_all_unfavorable_candidates() -> None:
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4], dtype=object), ("tp",))

    payload = muon_bench._benchmark_leaf(
        topology=muon_bench._TP4,
        mesh=mesh,
        leaf=muon_bench._Leaf("square_row", (8, 8), 0),
        warmup=0,
        iters=1,
        artifact_root=None,
        trace=False,
    )

    assert payload["kind"] == "leaf"
    assert {candidate["execution"] for candidate in payload["candidates"]} == {
        "duplicated",
        "distributed_large_gram",
        "distributed_exchange",
    }
    assert all(candidate["timing_ms"]["median"] > 0.0 for candidate in payload["candidates"])
    assert next(
        candidate
        for candidate in payload["candidates"]
        if candidate["execution"] == "duplicated"
    )["correctness_gate"] is True


def test_muon_bucket_benchmark_packs_multiple_leaves() -> None:
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4], dtype=object), ("tp",))

    payload = muon_bench._benchmark_bucket(
        topology=muon_bench._TP4,
        mesh=mesh,
        leaf=muon_bench._Leaf("kv_bucket", (16, 8), 1),
        leaf_count=2,
        warmup=0,
        iters=1,
        artifact_root=None,
        trace=False,
    )

    assert payload["kind"] == "bucket"
    assert payload["leaf_count"] == 2
    assert next(
        candidate
        for candidate in payload["candidates"]
        if candidate["execution"] == "distributed_exchange"
    )["bucket_count"] == 1
