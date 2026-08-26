#!/usr/bin/env python3
"""Replay the production Muon selector against frozen H100 calibration evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
import json
import math
from pathlib import Path
from typing import Any

from jaxtitan.optim.muon_policy import (
    MUON_SHAPE_POLICY_VERSION,
    MuonExecutionDecision,
    select_muon_execution,
)


FIXTURE_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
EXPECTED_SAMPLE_COUNT = 63
EXPECTED_SOURCE_ARTIFACT = "dist_muon_leaf_bench_20260724T215030Z.tgz"
EXPECTED_SOURCE_SHA256 = "27fddd04d51cdb9f3262a3a074a2f319e8bf0f880ba80851d3a546a5bef6b1e6"
MINIMUM_SPEEDUP = 1.05
MAXIMUM_MAD_FRACTION = 0.05
TIE_FRACTION = 0.03
PORTABLE_FEATURES = (
    "short_dimension",
    "long_dimension",
    "canonical_tp_dim",
    "tp_size",
    "cohort_size",
)
DEFAULT_FIXTURE = Path(__file__).with_name("dist_muon_shape_policy_h100_v1.json")

Selector = Callable[..., MuonExecutionDecision]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        nargs="?",
        default=str(DEFAULT_FIXTURE),
        help="Tracked calibration fixture. Defaults to the fixture beside this script.",
    )
    parser.add_argument("--json-out", help="Optional machine-readable replay output.")
    parser.add_argument(
        "--extract-selection",
        help="Explicitly build a fixture from an archived selector selection.json.",
    )
    parser.add_argument("--source-artifact", help="Source archive name for fixture extraction.")
    parser.add_argument("--source-sha256", help="Source archive SHA-256 for fixture extraction.")
    args = parser.parse_args()

    if args.extract_selection:
        if not args.json_out:
            parser.error("--extract-selection requires --json-out")
        if not args.source_artifact or not args.source_sha256:
            parser.error(
                "--extract-selection requires --source-artifact and --source-sha256"
            )
        selection = _load_json(Path(args.extract_selection))
        payload = fixture_from_selection(
            selection,
            source_artifact=args.source_artifact,
            source_sha256=args.source_sha256,
        )
        _write_json(Path(args.json_out), payload)
        print(f"wrote_fixture={args.json_out} samples={len(payload['samples'])}")
        return 0

    payload = replay_fixture(_load_json(Path(args.fixture)))
    if args.json_out:
        _write_json(Path(args.json_out), payload)
    print(format_replay(payload))
    return 0 if payload["overall_gate"] else 1


def fixture_from_selection(
    selection: Mapping[str, Any],
    *,
    source_artifact: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Sanitize the accepted selector output into portable replay evidence."""

    if selection.get("schema_version") != 1:
        raise ValueError("selector fixture extraction requires selection schema version 1")
    if selection.get("overall_gate") is not True:
        raise ValueError("selector fixture extraction requires overall_gate=true")
    rows = selection.get("cases")
    samples = selection.get("shape_topology_samples")
    if not isinstance(rows, list) or not isinstance(samples, list):
        raise ValueError("selector fixture extraction requires cases and shape_topology_samples")
    rows_by_name = {row.get("name"): row for row in rows if isinstance(row, Mapping)}
    sanitized = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("selector fixture extraction found a malformed sample")
        source_case = sample.get("source_case")
        row = rows_by_name.get(source_case)
        if not isinstance(row, Mapping):
            raise ValueError(f"selector fixture sample {source_case!r} has no source case")
        features = sample.get("policy_features")
        if not isinstance(features, Mapping):
            raise ValueError(f"selector fixture sample {source_case!r} has no policy features")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"selector fixture sample {source_case!r} has no candidates")
        sanitized.append(
            {
                "source_case": str(source_case),
                "source_kind": str(sample.get("source_kind")),
                "topology": str(row.get("topology")),
                "features": {
                    "short_dimension": int(features["short_dimension"]),
                    "long_dimension": int(features["long_dimension"]),
                    "canonical_tp_dim": int(features["canonical_tp_dim"]),
                    "tp_size": int(features["tp_size"]),
                    "cohort_size": int(features["leaf_count"]),
                },
                "accepted_execution": str(sample.get("selected_execution")),
                "previous_execution": str(row.get("current_execution")),
                "candidates": [
                    {
                        "execution": str(candidate.get("execution")),
                        "median_ms": float(candidate.get("median_ms")),
                        "p95_ms": float(candidate.get("p95_ms")),
                        "mad_fraction": float(candidate.get("mad_fraction")),
                        "correctness_gate": candidate.get("correctness_gate"),
                        "stability_gate": candidate.get("stability_gate"),
                        "speedup_vs_duplicated": float(
                            candidate.get("speedup_vs_duplicated")
                        ),
                        "collective_operand_model": candidate.get(
                            "collective_operand_model"
                        ),
                    }
                    for candidate in sorted(
                        candidates, key=lambda item: str(item.get("execution"))
                    )
                ],
            }
        )
    sanitized.sort(key=lambda item: item["source_case"])
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "source": {
            "artifact": source_artifact,
            "sha256": source_sha256,
            "selection_schema_version": selection["schema_version"],
        },
        "hardware": {
            "backend": selection.get("hardware", {}).get("backend"),
            "device_kind": selection.get("hardware", {}).get("device_kind"),
            "device_count": selection.get("hardware", {}).get("device_count"),
        },
        "contract": {
            "minimum_speedup": MINIMUM_SPEEDUP,
            "maximum_mad_fraction": MAXIMUM_MAD_FRACTION,
            "tie_fraction": TIE_FRACTION,
            "timing_scope": "calibration_replay_only_not_end_to_end_acceptance",
        },
        "sample_count": len(sanitized),
        "samples": sanitized,
    }


def replay_fixture(
    fixture: Mapping[str, Any],
    *,
    selector: Selector = select_muon_execution,
) -> dict[str, Any]:
    """Apply the real host-static selector to every frozen calibration sample."""

    failures = _fixture_failures(fixture)
    raw_samples = fixture.get("samples")
    if not isinstance(raw_samples, list):
        raw_samples = []
    rows = []
    for index, sample in enumerate(raw_samples):
        rows.append(_replay_sample(index, sample, selector=selector))
    failures.extend(
        failure
        for row in rows
        for failure in row["failures"]
    )

    valid_rows = [
        row
        for row in rows
        if row.get("metrics_available")
        and all(
            _positive_finite(row.get(field))
            for field in (
                "speedup_vs_duplicated",
                "speedup_vs_previous",
                "measured_regret",
            )
        )
    ]
    decision_matches = sum(bool(row.get("decision_match")) for row in rows)
    overall_gate = not failures and decision_matches == EXPECTED_SAMPLE_COUNT
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "kind": "dist_muon_shape_policy_calibration_replay",
        "policy_version": MUON_SHAPE_POLICY_VERSION,
        "source": fixture.get("source"),
        "hardware": fixture.get("hardware"),
        "contract": fixture.get("contract"),
        "sample_count": len(rows),
        "decision_match_count": decision_matches,
        "decision_mismatch_count": len(rows) - decision_matches,
        "selected_execution_counts": dict(
            sorted(Counter(str(row.get("selected_execution")) for row in rows).items())
        ),
        "worst_measured_regret": max(
            (float(row["measured_regret"]) for row in valid_rows),
            default=None,
        ),
        "geomean_speedup_vs_duplicated": _geomean(
            [float(row["speedup_vs_duplicated"]) for row in valid_rows]
        ),
        "geomean_speedup_vs_previous": _geomean(
            [float(row["speedup_vs_previous"]) for row in valid_rows]
        ),
        "breakdowns": {
            "topology": _breakdowns(valid_rows, lambda row: str(row["topology"])),
            "canonical_orientation": _breakdowns(
                valid_rows,
                lambda row: str(row["features"]["canonical_tp_dim"]),
            ),
            "geometry": _breakdowns(
                valid_rows,
                lambda row: (
                    f"{row['features']['short_dimension']}x"
                    f"{row['features']['long_dimension']}"
                ),
            ),
        },
        "samples": rows,
        "failures": sorted(set(failures)),
        "overall_gate": overall_gate,
        "performance_claim": False,
    }


def _fixture_failures(fixture: Mapping[str, Any]) -> list[str]:
    failures = []
    if fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        failures.append(
            f"fixture schema must equal {FIXTURE_SCHEMA_VERSION}"
        )
    source = fixture.get("source")
    if not isinstance(source, Mapping):
        failures.append("fixture source must be an object")
    else:
        if source.get("artifact") != EXPECTED_SOURCE_ARTIFACT:
            failures.append("fixture source artifact does not match the frozen calibration")
        if source.get("sha256") != EXPECTED_SOURCE_SHA256:
            failures.append("fixture source SHA-256 does not match the frozen calibration")
        if source.get("selection_schema_version") != 1:
            failures.append("fixture selection schema must equal 1")
    hardware = fixture.get("hardware")
    if not isinstance(hardware, Mapping):
        failures.append("fixture hardware must be an object")
    else:
        if hardware.get("backend") != "gpu":
            failures.append("fixture hardware backend must equal gpu")
        if hardware.get("device_count") != 4:
            failures.append("fixture hardware device_count must equal 4")
        if hardware.get("device_kind") != "NVIDIA H100 80GB HBM3":
            failures.append("fixture hardware device_kind must match the H100 calibration")
    contract = fixture.get("contract")
    if not isinstance(contract, Mapping):
        failures.append("fixture contract must be an object")
    else:
        expected = {
            "minimum_speedup": MINIMUM_SPEEDUP,
            "maximum_mad_fraction": MAXIMUM_MAD_FRACTION,
            "tie_fraction": TIE_FRACTION,
            "timing_scope": "calibration_replay_only_not_end_to_end_acceptance",
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                failures.append(f"fixture contract {key} must equal {value!r}")
    samples = fixture.get("samples")
    if not isinstance(samples, list):
        failures.append("fixture samples must be a list")
    else:
        if len(samples) != EXPECTED_SAMPLE_COUNT:
            failures.append(
                f"fixture must contain exactly {EXPECTED_SAMPLE_COUNT} samples"
            )
        source_cases = [
            sample.get("source_case")
            for sample in samples
            if isinstance(sample, Mapping)
        ]
        if len(source_cases) != len(set(source_cases)):
            failures.append("fixture source cases must be unique")
        if source_cases != sorted(source_cases):
            failures.append("fixture source cases must be deterministically sorted")
    if fixture.get("sample_count") != EXPECTED_SAMPLE_COUNT:
        failures.append(
            f"fixture sample_count must equal {EXPECTED_SAMPLE_COUNT}"
        )
    return failures


def _replay_sample(
    index: int,
    sample: Any,
    *,
    selector: Selector,
) -> dict[str, Any]:
    prefix = f"sample {index}"
    failures = []
    if not isinstance(sample, Mapping):
        return _failed_row(prefix, ["sample must be an object"])
    source_case = sample.get("source_case")
    if not isinstance(source_case, str) or not source_case:
        failures.append(f"{prefix} source_case must be a non-empty string")
        source_case = prefix
    prefix = str(source_case)
    features = sample.get("features")
    if not isinstance(features, Mapping):
        return _failed_row(prefix, [f"{prefix} features must be an object"])
    if set(features) != set(PORTABLE_FEATURES):
        failures.append(
            f"{prefix} features must contain only {list(PORTABLE_FEATURES)}"
        )
    try:
        short_dimension = int(features["short_dimension"])
        long_dimension = int(features["long_dimension"])
        canonical_tp_dim = int(features["canonical_tp_dim"])
        tp_size = int(features["tp_size"])
        cohort_size = int(features["cohort_size"])
    except (KeyError, TypeError, ValueError):
        return _failed_row(prefix, failures + [f"{prefix} has invalid policy features"])
    if short_dimension <= 0 or long_dimension < short_dimension:
        failures.append(f"{prefix} dimensions must satisfy 0 < short <= long")

    candidates = sample.get("candidates")
    candidate_rows: dict[str, Mapping[str, Any]] = {}
    if not isinstance(candidates, list) or not candidates:
        failures.append(f"{prefix} candidates must be a non-empty list")
    else:
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                failures.append(f"{prefix} candidate {candidate_index} must be an object")
                continue
            execution = candidate.get("execution")
            if not isinstance(execution, str) or not execution:
                failures.append(
                    f"{prefix} candidate {candidate_index} has invalid execution"
                )
                continue
            if execution in candidate_rows:
                failures.append(f"{prefix} has duplicate candidate {execution}")
            candidate_rows[execution] = candidate
            failures.extend(_candidate_failures(prefix, execution, candidate))
    if "duplicated" not in candidate_rows:
        failures.append(f"{prefix} is missing duplicated candidate")

    try:
        decision = selector(
            requested_mode="distributed",
            canonical_tp_dim=canonical_tp_dim,
            logical_shape=(short_dimension, long_dimension),
            tp_size=tp_size,
            cohort_size=cohort_size,
        )
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        return _failed_row(
            prefix,
            failures + [f"{prefix} selector failed: {type(error).__name__}: {error}"],
        )
    accepted_execution = sample.get("accepted_execution")
    decision_match = decision.execution == accepted_execution
    if not decision_match:
        failures.append(
            f"{prefix} selected {decision.execution}, expected {accepted_execution}"
        )
    selected = candidate_rows.get(decision.execution)
    previous = candidate_rows.get(str(sample.get("previous_execution")))
    duplicated = candidate_rows.get("duplicated")
    if selected is None:
        failures.append(f"{prefix} has no candidate for selected execution {decision.execution}")
    if previous is None:
        failures.append(f"{prefix} has no candidate for previous execution")
    metrics_available = selected is not None and previous is not None and duplicated is not None
    if metrics_available:
        assert selected is not None and previous is not None and duplicated is not None
        if selected.get("correctness_gate") is not True:
            failures.append(f"{prefix} selected candidate failed correctness")
        if selected.get("stability_gate") is not True:
            failures.append(f"{prefix} selected candidate failed stability")
        selected_mad = selected.get("mad_fraction")
        selected_speedup = selected.get("speedup_vs_duplicated")
        passing = [
            candidate
            for execution, candidate in candidate_rows.items()
            if execution != "duplicated"
            and candidate.get("correctness_gate") is True
            and candidate.get("stability_gate") is True
            and _positive_finite(candidate.get("speedup_vs_duplicated"))
            and float(candidate["speedup_vs_duplicated"]) >= MINIMUM_SPEEDUP
            and _positive_finite(candidate.get("median_ms"))
        ]
        if decision.execution != "duplicated":
            if (
                not _positive_finite(selected_speedup)
                or float(selected_speedup) < MINIMUM_SPEEDUP
            ):
                failures.append(
                    f"{prefix} selected candidate is below {MINIMUM_SPEEDUP:.2f}x speedup"
                )
            if (
                not _finite_number(selected_mad)
                or float(selected_mad) > MAXIMUM_MAD_FRACTION
            ):
                failures.append(
                    f"{prefix} selected candidate exceeds {MAXIMUM_MAD_FRACTION:.0%} MAD"
                )
            if not passing:
                failures.append(f"{prefix} has no passing distributed candidate")
            else:
                fastest_ms = min(float(candidate["median_ms"]) for candidate in passing)
                if float(selected["median_ms"]) > fastest_ms * (1.0 + TIE_FRACTION):
                    failures.append(
                        f"{prefix} selected candidate exceeds the {TIE_FRACTION:.0%} tie band"
                    )
        elif passing:
            failures.append(
                f"{prefix} selected duplicated despite a passing distributed candidate"
            )
        correct_timings = [
            float(candidate["median_ms"])
            for candidate in candidate_rows.values()
            if candidate.get("correctness_gate") is True
            and candidate.get("stability_gate") is True
            and _positive_finite(candidate.get("median_ms"))
        ]
        metrics_available = (
            bool(correct_timings)
            and _positive_finite(selected.get("median_ms"))
            and _positive_finite(previous.get("median_ms"))
            and _positive_finite(duplicated.get("median_ms"))
        )
        if metrics_available:
            fastest_correct_ms = min(correct_timings)
            selected_ms = float(selected["median_ms"])
            speedup_vs_duplicated = float(duplicated["median_ms"]) / selected_ms
            speedup_vs_previous = float(previous["median_ms"]) / selected_ms
            measured_regret = selected_ms / fastest_correct_ms
        else:
            speedup_vs_duplicated = None
            speedup_vs_previous = None
            measured_regret = None
    else:
        speedup_vs_duplicated = None
        speedup_vs_previous = None
        measured_regret = None

    return {
        "source_case": source_case,
        "source_kind": sample.get("source_kind"),
        "topology": sample.get("topology"),
        "features": dict(features),
        "accepted_execution": accepted_execution,
        "previous_execution": sample.get("previous_execution"),
        "selected_execution": decision.execution,
        "selection_reason": decision.selection_reason,
        "decision_match": decision_match,
        "metrics_available": metrics_available,
        "speedup_vs_duplicated": speedup_vs_duplicated,
        "speedup_vs_previous": speedup_vs_previous,
        "measured_regret": measured_regret,
        "failures": failures,
        "gate": not failures,
    }


def _candidate_failures(
    prefix: str,
    execution: str,
    candidate: Mapping[str, Any],
) -> list[str]:
    failures = []
    for field in ("median_ms", "p95_ms", "mad_fraction", "speedup_vs_duplicated"):
        value = candidate.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{prefix} candidate {execution} has nonfinite {field}")
        elif field in {"median_ms", "p95_ms", "speedup_vs_duplicated"} and float(value) <= 0:
            failures.append(f"{prefix} candidate {execution} has nonpositive {field}")
        elif field == "mad_fraction" and float(value) < 0:
            failures.append(f"{prefix} candidate {execution} has negative mad_fraction")
    if not isinstance(candidate.get("correctness_gate"), bool):
        failures.append(f"{prefix} candidate {execution} correctness_gate must be boolean")
    if not isinstance(candidate.get("stability_gate"), bool):
        failures.append(f"{prefix} candidate {execution} stability_gate must be boolean")
    model = candidate.get("collective_operand_model")
    if not isinstance(model, Mapping) or not model:
        failures.append(f"{prefix} candidate {execution} has no collective operand model")
    return failures


def _failed_row(source_case: str, failures: list[str]) -> dict[str, Any]:
    return {
        "source_case": source_case,
        "source_kind": None,
        "topology": None,
        "features": {},
        "accepted_execution": None,
        "previous_execution": None,
        "selected_execution": None,
        "selection_reason": None,
        "decision_match": False,
        "metrics_available": False,
        "speedup_vs_duplicated": None,
        "speedup_vs_previous": None,
        "measured_regret": None,
        "failures": failures,
        "gate": False,
    }


def _breakdowns(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return [
        {
            "key": key,
            "sample_count": len(group),
            "selected_execution_counts": dict(
                sorted(Counter(row["selected_execution"] for row in group).items())
            ),
            "geomean_speedup_vs_duplicated": _geomean(
                [float(row["speedup_vs_duplicated"]) for row in group]
            ),
            "geomean_speedup_vs_previous": _geomean(
                [float(row["speedup_vs_previous"]) for row in group]
            ),
            "worst_measured_regret": max(float(row["measured_regret"]) for row in group),
        }
        for key, group in sorted(groups.items())
    ]


def _geomean(values: list[float]) -> float | None:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def format_replay(payload: Mapping[str, Any]) -> str:
    def metric(name: str) -> str:
        value = payload.get(name)
        return "n/a" if value is None else f"{float(value):.3f}x"

    lines = [
        (
            f"policy={payload['policy_version']} samples={payload['sample_count']} "
            f"matches={payload['decision_match_count']} "
            f"mismatches={payload['decision_mismatch_count']}"
        ),
        (
            f"geomean_vs_duplicated={metric('geomean_speedup_vs_duplicated')} "
            f"geomean_vs_previous={metric('geomean_speedup_vs_previous')} "
            f"worst_measured_regret={metric('worst_measured_regret')}"
        ),
    ]
    lines.extend(f"- {failure}" for failure in payload["failures"])
    lines.append("performance_claim=false")
    lines.append(f"overall_gate={payload['overall_gate']}")
    return "\n".join(lines)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_finite(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
