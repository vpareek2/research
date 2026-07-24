#!/usr/bin/env python3
"""Select conservative per-leaf distributed-Muon candidates from a GPU benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIN_SPEEDUP = 1.05
TIE_FRACTION = 0.03
MAX_MAD_FRACTION = 0.05
_SIMPLICITY = {
    "distributed_direct": 0,
    "distributed_large_gram": 1,
    "distributed_exchange": 2,
}


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
        "schema_version": 1,
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
        "production_recommendations": recommendations,
    }


def _select_case(case: dict[str, Any]) -> dict[str, Any]:
    candidates = {candidate["execution"]: candidate for candidate in case.get("candidates", ())}
    duplicated = candidates.get("duplicated")
    if duplicated is None:
        raise ValueError(f"Muon benchmark case {case.get('name')!r} is missing duplicated reference")
    duplicated_ms = _median_ms(duplicated)
    current_execution = (
        "distributed_direct"
        if "distributed_direct" in candidates
        else "distributed_exchange"
    )
    current = candidates.get(current_execution)
    if current is None:
        raise ValueError(f"Muon benchmark case {case.get('name')!r} is missing current execution")

    eligible = []
    candidate_rows = []
    for execution, candidate in candidates.items():
        timing_ms = _median_ms(candidate)
        mad_fraction = float(candidate["timing_ms"]["median_abs_deviation"]) / timing_ms
        stable = mad_fraction <= MAX_MAD_FRACTION
        correct = bool(candidate.get("correctness_gate"))
        speedup = duplicated_ms / timing_ms
        candidate_rows.append(
            {
                "execution": execution,
                "median_ms": timing_ms,
                "p95_ms": float(candidate["timing_ms"]["p95"]),
                "mad_fraction": mad_fraction,
                "speedup_vs_duplicated": speedup,
                "correctness_gate": correct,
                "stability_gate": stable,
            }
        )
        if execution != "duplicated" and correct and stable and speedup >= MIN_SPEEDUP:
            eligible.append((execution, timing_ms))

    selected_execution = "duplicated"
    reason = "no_candidate_cleared_speed_and_correctness_gates"
    if eligible:
        fastest_ms = min(timing for _execution, timing in eligible)
        tied = [
            (execution, timing)
            for execution, timing in eligible
            if timing <= fastest_ms * (1.0 + TIE_FRACTION)
        ]
        selected_execution, _selected_ms = min(
            tied,
            key=lambda item: (_SIMPLICITY[item[0]], item[1], item[0]),
        )
        reason = "fastest_clear_candidate_with_simplicity_tie_break"

    selected_ms = _median_ms(candidates[selected_execution])
    required_correctness_gate = bool(
        duplicated.get("correctness_gate")
        and current.get("correctness_gate")
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
        "current_execution": current_execution,
        "selected_execution": selected_execution,
        "selection_reason": reason,
        "selected_speedup_vs_duplicated": duplicated_ms / selected_ms,
        "selected_speedup_vs_current": _median_ms(current) / selected_ms,
        "required_correctness_gate": required_correctness_gate,
        "candidates": sorted(candidate_rows, key=lambda row: row["execution"]),
    }


def _median_ms(candidate: dict[str, Any]) -> float:
    value = float(candidate["timing_ms"]["median"])
    if value <= 0.0:
        raise ValueError(f"candidate {candidate.get('execution')!r} has non-positive median")
    return value


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _production_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["topology"], row["role"])
        current = by_key.get(key)
        if current is None or (current["kind"] != "bucket" and row["kind"] == "bucket"):
            by_key[key] = row
    return [
        {
            "topology": row["topology"],
            "role": row["role"],
            "shape": row["shape"],
            "partition_spec": row["partition_spec"],
            "source_case": row["name"],
            "source_kind": row["kind"],
            "selected_execution": row["selected_execution"],
            "speedup_vs_duplicated": row["selected_speedup_vs_duplicated"],
            "speedup_vs_current": row["selected_speedup_vs_current"],
        }
        for row in sorted(by_key.values(), key=lambda item: (item["topology"], item["role"]))
    ]


def format_selection(selection: dict[str, Any]) -> str:
    lines = [
        "case                                  selected                    dup ms  selected ms  speedup",
    ]
    for case in selection["cases"]:
        selected = next(
            candidate
            for candidate in case["candidates"]
            if candidate["execution"] == case["selected_execution"]
        )
        duplicated = next(
            candidate
            for candidate in case["candidates"]
            if candidate["execution"] == "duplicated"
        )
        lines.append(
            f"{case['name']:<37} "
            f"{case['selected_execution']:<27} "
            f"{duplicated['median_ms']:>7.3f} "
            f"{selected['median_ms']:>12.3f} "
            f"{case['selected_speedup_vs_duplicated']:>7.3f}x"
        )
    lines.append(f"overall_gate={selection['overall_gate']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
