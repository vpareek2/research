#!/usr/bin/env python3
"""Select conservative per-leaf distributed-Muon candidates from a GPU benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MIN_SPEEDUP = 1.05
TIE_FRACTION = 0.03
MAX_MAD_FRACTION = 0.05
_SIMPLICITY = {
    "duplicated": 3,
    "distributed_direct": 0,
    "distributed_large_gram": 1,
    "distributed_exchange": 2,
}
STANDARD_ORTHOGONALIZATION = "standard_newton_schulz"
GRAM_ORTHOGONALIZATION = "gram_newton_schulz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="JSON emitted by `jaxtitan profile bench muon --json`.")
    parser.add_argument("--json-out", help="Optional machine-readable selection output.")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selection = analyze_benchmark(payload)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(format_selection(selection))
    return 0 if selection["overall_gate"] else 1


def analyze_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("component") != "muon":
        raise ValueError("distributed Muon selection requires a Muon benchmark payload")
    if payload.get("correctness_is_checked") is not True:
        raise ValueError("distributed Muon selection requires correctness-checked benchmark cases")
    if int(payload.get("device_count", 0)) != 4:
        raise ValueError("distributed Muon selection requires exactly four benchmark devices")

    rows = [_select_case(case) for case in payload.get("cases", ())]
    if not rows:
        raise ValueError("distributed Muon benchmark contains no cases")
    recommendations = _production_recommendations(rows)
    return {
        "schema_version": 2,
        "source_schema_version": payload.get("schema_version"),
        "hardware": {
            "backend": payload.get("backend"),
            "device_kind": payload.get("device_kind"),
            "device_count": payload.get("device_count"),
        },
        "selection_policy": {
            "minimum_speedup": MIN_SPEEDUP,
            "tie_fraction": TIE_FRACTION,
            "maximum_mad_fraction": MAX_MAD_FRACTION,
            "fallback": "duplicated",
            "tie_break": "simpler_collective_topology",
        },
        "cases": rows,
        "overall_gate": all(row["required_correctness_gate"] for row in rows),
        "selected_execution_counts": _counts(row["selected_execution"] for row in rows),
        "selected_candidate_counts": _counts(row["selected_candidate_id"] for row in rows),
        "production_recommendations": recommendations,
        "shape_topology_samples": _shape_topology_samples(rows),
    }


def _select_case(case: dict[str, Any]) -> dict[str, Any]:
    candidate_schema_version = int(case.get("candidate_schema_version", 1))
    normalized = [
        _normalize_candidate(candidate, schema_version=candidate_schema_version)
        for candidate in case.get("candidates", ())
    ]
    candidates = {candidate["candidate_id"]: candidate for candidate in normalized}
    if len(candidates) != len(normalized):
        raise ValueError(f"Muon benchmark case {case.get('name')!r} has duplicate candidate identities")
    reference_id = str(
        case.get(
            "reference",
            _candidate_id("duplicated", STANDARD_ORTHOGONALIZATION, ()),
        )
    )
    if reference_id == "duplicated":
        reference_id = _candidate_id("duplicated", STANDARD_ORTHOGONALIZATION, ())
    duplicated = candidates.get(reference_id)
    if duplicated is None:
        raise ValueError(f"Muon benchmark case {case.get('name')!r} is missing duplicated reference")
    duplicated_ms = _median_ms(duplicated)
    current_candidate_id = str(
        case.get(
            "production_candidate_id",
            _legacy_current_candidate_id(candidates),
        )
    )
    current = candidates.get(current_candidate_id)
    if current is None:
        raise ValueError(f"Muon benchmark case {case.get('name')!r} is missing current execution")

    same_transport_standard = {
        candidate["execution"]: candidate
        for candidate in candidates.values()
        if candidate["orthogonalization"] == STANDARD_ORTHOGONALIZATION
    }
    for candidate in candidates.values():
        if (
            candidate["orthogonalization"] == GRAM_ORTHOGONALIZATION
            and candidate["execution"] not in same_transport_standard
        ):
            raise ValueError(
                f"recurrence candidate {candidate['candidate_id']!r} is missing its "
                "same-transport standard reference"
            )
    eligible: list[tuple[str, float]] = []
    candidate_rows = []
    recurrence_correctness = []
    for candidate_id, candidate in candidates.items():
        execution = candidate["execution"]
        orthogonalization = candidate["orthogonalization"]
        timing_ms = _median_ms(candidate)
        p95_ms = float(candidate["timing_ms"]["p95"])
        mad_ms = float(candidate["timing_ms"]["median_abs_deviation"])
        if (
            not math.isfinite(p95_ms)
            or p95_ms <= 0.0
            or not math.isfinite(mad_ms)
            or mad_ms < 0.0
        ):
            raise ValueError(f"candidate {candidate_id!r} has nonfinite or invalid timing")
        mad_fraction = mad_ms / timing_ms
        stable = mad_fraction <= MAX_MAD_FRACTION
        correct = bool(candidate.get("correctness_gate"))
        speedup = duplicated_ms / timing_ms
        standard = same_transport_standard.get(execution)
        speedup_vs_same_transport_standard = (
            None if standard is None else _median_ms(standard) / timing_ms
        )
        hlo_gate = (
            True
            if candidate_schema_version == 1
            else bool(candidate.get("hlo_contract", {}).get("gate"))
        )
        recurrence = orthogonalization == GRAM_ORTHOGONALIZATION
        if recurrence:
            recurrence_correctness.append(correct and hlo_gate)
        promotion_eligible = bool(
            recurrence
            and correct
            and stable
            and hlo_gate
            and speedup_vs_same_transport_standard is not None
            and speedup_vs_same_transport_standard >= MIN_SPEEDUP
            and (execution == "duplicated" or speedup >= MIN_SPEEDUP)
        )
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "execution": execution,
                "orthogonalization": orthogonalization,
                "restart_indices": list(candidate["restart_indices"]),
                "median_ms": timing_ms,
                "p95_ms": p95_ms,
                "mad_fraction": mad_fraction,
                "speedup_vs_duplicated": speedup,
                "speedup_vs_same_transport_standard": speedup_vs_same_transport_standard,
                "correctness_gate": correct,
                "stability_gate": stable,
                "hlo_gate": hlo_gate,
                "promotion_eligible": promotion_eligible,
                "collective_operand_model": candidate.get("collective_operand_model"),
            }
        )
        if orthogonalization == STANDARD_ORTHOGONALIZATION:
            if candidate_id == reference_id or (
                execution != "duplicated"
                and correct
                and stable
                and hlo_gate
                and speedup >= MIN_SPEEDUP
            ):
                eligible.append((candidate_id, timing_ms))
        elif promotion_eligible:
            eligible.append((candidate_id, timing_ms))

    selected_candidate_id = reference_id
    reason = "no_candidate_cleared_speed_and_correctness_gates"
    if eligible:
        fastest_ms = min(timing for _candidate_id_value, timing in eligible)
        tied = [
            (candidate_id, timing)
            for candidate_id, timing in eligible
            if timing <= fastest_ms * (1.0 + TIE_FRACTION)
        ]
        selected_candidate_id, _selected_ms = min(
            tied,
            key=lambda item: (
                _candidate_simplicity(candidates[item[0]]),
                item[1],
                item[0],
            ),
        )
        reason = "fastest_clear_candidate_with_simplicity_tie_break"

    selected = candidates[selected_candidate_id]
    selected_execution = selected["execution"]
    selected_ms = _median_ms(selected)
    required_correctness_gate = bool(
        duplicated.get("correctness_gate")
        and current.get("correctness_gate")
        and (
            candidate_schema_version == 1
            or (
                duplicated.get("hlo_contract", {}).get("gate")
                and current.get("hlo_contract", {}).get("gate")
                and all(recurrence_correctness)
            )
        )
    )
    return {
        "name": case["name"],
        "kind": case.get("kind", "leaf"),
        "topology": case["topology"],
        "mesh": case["mesh"],
        "role": case["role"],
        "shape": case["shape"],
        "partition_spec": case["partition_spec"],
        "tp_partition_dim": case["tp_partition_dim"],
        "canonical_tp_dim": case["canonical_tp_dim"],
        "leaf_count": case.get("leaf_count", 1),
        "policy_features": case.get(
            "policy_features",
            _policy_features_from_case(case),
        ),
        "candidate_schema_version": candidate_schema_version,
        "current_candidate_id": current_candidate_id,
        "current_execution": current["execution"],
        "selected_candidate_id": selected_candidate_id,
        "selected_execution": selected_execution,
        "selected_orthogonalization": selected["orthogonalization"],
        "selected_restart_indices": list(selected["restart_indices"]),
        "selection_reason": reason,
        "selected_speedup_vs_duplicated": duplicated_ms / selected_ms,
        "selected_speedup_vs_current": _median_ms(current) / selected_ms,
        "required_correctness_gate": required_correctness_gate,
        "candidates": sorted(candidate_rows, key=lambda row: row["candidate_id"]),
    }


def _normalize_candidate(
    candidate: dict[str, Any],
    *,
    schema_version: int,
) -> dict[str, Any]:
    execution = str(candidate["execution"])
    orthogonalization = str(
        candidate.get("orthogonalization", STANDARD_ORTHOGONALIZATION)
    )
    restart_indices = tuple(int(value) for value in candidate.get("restart_indices", ()))
    expected_id = _candidate_id(execution, orthogonalization, restart_indices)
    candidate_id = str(candidate.get("candidate_id", expected_id))
    if schema_version >= 2:
        if candidate_id != expected_id:
            raise ValueError(
                f"candidate identity {candidate_id!r} does not match its transport "
                f"and orthogonalization contract {expected_id!r}"
            )
        if orthogonalization not in {
            STANDARD_ORTHOGONALIZATION,
            GRAM_ORTHOGONALIZATION,
        }:
            raise ValueError(f"unsupported Muon orthogonalization {orthogonalization!r}")
        if not isinstance(candidate.get("hlo_contract"), dict):
            raise ValueError(f"candidate {candidate_id!r} is missing its HLO contract")
    normalized = dict(candidate)
    normalized.update(
        {
            "candidate_id": candidate_id,
            "execution": execution,
            "orthogonalization": orthogonalization,
            "restart_indices": restart_indices,
        }
    )
    return normalized


def _candidate_id(
    execution: str,
    orthogonalization: str,
    restart_indices: tuple[int, ...],
) -> str:
    suffix = ""
    if restart_indices:
        suffix = "_r" + "_".join(str(index) for index in restart_indices)
    elif orthogonalization == GRAM_ORTHOGONALIZATION:
        suffix = "_no_restart"
    return f"{execution}__{orthogonalization}{suffix}"


def _legacy_current_candidate_id(candidates: dict[str, dict[str, Any]]) -> str:
    by_execution = {
        candidate["execution"]: candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["orthogonalization"] == STANDARD_ORTHOGONALIZATION
    }
    for execution in ("distributed_direct", "distributed_exchange", "duplicated"):
        if execution in by_execution:
            return by_execution[execution]
    return _candidate_id("duplicated", STANDARD_ORTHOGONALIZATION, ())


def _candidate_simplicity(candidate: dict[str, Any]) -> tuple[int, int]:
    return (
        0
        if candidate["orthogonalization"] == STANDARD_ORTHOGONALIZATION
        else 1,
        _SIMPLICITY[candidate["execution"]],
    )


def _median_ms(candidate: dict[str, Any]) -> float:
    value = float(candidate["timing_ms"]["median"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"candidate {candidate.get('execution')!r} has non-positive median")
    return value


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _production_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "topology": row["topology"],
            "role": row["role"],
            "shape": row["shape"],
            "partition_spec": row["partition_spec"],
            "source_case": row["name"],
            "source_kind": row["kind"],
            "policy_features": row["policy_features"],
            "selected_candidate_id": row["selected_candidate_id"],
            "selected_execution": row["selected_execution"],
            "selected_orthogonalization": row["selected_orthogonalization"],
            "selected_restart_indices": row["selected_restart_indices"],
            "speedup_vs_duplicated": row["selected_speedup_vs_duplicated"],
            "speedup_vs_current": row["selected_speedup_vs_current"],
        }
        for row in sorted(
            (item for item in rows if item["kind"] == "production_bucket"),
            key=lambda item: (
                item["policy_features"]["tp_size"],
                item["policy_features"]["canonical_tp_dim"],
                item["policy_features"]["short_dimension"],
                item["policy_features"]["aspect_ratio"],
                item["policy_features"]["leaf_count"],
                item["topology"],
            ),
        )
    ]


def _shape_topology_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_case": row["name"],
            "source_kind": row["kind"],
            "policy_features": row["policy_features"],
            "selected_candidate_id": row["selected_candidate_id"],
            "selected_execution": row["selected_execution"],
            "selected_orthogonalization": row["selected_orthogonalization"],
            "speedup_vs_duplicated": row["selected_speedup_vs_duplicated"],
            "speedup_vs_current": row["selected_speedup_vs_current"],
        }
        for row in rows
        if row["kind"] in {"production_bucket", "calibration_bucket"}
    ]


def _policy_features_from_case(case: dict[str, Any]) -> dict[str, int | float]:
    rows, columns = (int(value) for value in case["shape"])
    short_dimension = min(rows, columns)
    long_dimension = max(rows, columns)
    leaf_count = int(case.get("leaf_count", 1))
    tp_size = int(case["mesh"]["tp"])
    return {
        "short_dimension": short_dimension,
        "long_dimension": long_dimension,
        "aspect_ratio": long_dimension / short_dimension,
        "canonical_tp_dim": int(case["canonical_tp_dim"]),
        "tp_size": tp_size,
        "leaf_count": leaf_count,
        "matrix_elements_per_leaf": rows * columns,
        "aggregate_matrix_elements": rows * columns * leaf_count,
    }


def format_selection(selection: dict[str, Any]) -> str:
    lines = [
        "case                                  selected                    dup ms  selected ms  speedup",
    ]
    for case in selection["cases"]:
        selected = next(
            candidate
            for candidate in case["candidates"]
            if candidate["candidate_id"] == case["selected_candidate_id"]
        )
        duplicated = next(
            candidate
            for candidate in case["candidates"]
            if candidate["candidate_id"]
            == _candidate_id("duplicated", STANDARD_ORTHOGONALIZATION, ())
        )
        lines.append(
            f"{case['name']:<37} "
            f"{case['selected_candidate_id']:<27} "
            f"{duplicated['median_ms']:>7.3f} "
            f"{selected['median_ms']:>12.3f} "
            f"{case['selected_speedup_vs_duplicated']:>7.3f}x"
        )
    lines.append(f"overall_gate={selection['overall_gate']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
