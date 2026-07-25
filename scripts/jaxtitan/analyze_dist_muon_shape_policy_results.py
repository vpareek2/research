#!/usr/bin/env python3
"""Validate four-layout shape/topology Muon production acceptance."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any


BASELINE_ARTIFACT_SHA256 = "65fb879f2636778aa5a25d6566b1538a9ea533cfceb1439428bcdbd433d2db72"
RUNS = (
    (
        "dense_tp",
        "cloud_4gpu_profile64_dense_tp_muon_shape_policy",
        0.07253520300002947,
        0.07915740449999475,
    ),
    (
        "dense_fsdp_tp",
        "cloud_4gpu_profile64_dense_fsdp_tp_muon_shape_policy",
        0.10777493949990458,
        0.12161696600003324,
    ),
    (
        "dense_zero2_tp",
        "cloud_4gpu_profile64_dense_zero2_tp_muon_shape_policy",
        0.10596654399978434,
        0.12134818800006997,
    ),
    (
        "trinity_tp_ep",
        "cloud_4gpu_profile64_trinity_moe_tp_ep_muon_shape_policy",
        0.2188496189999114,
        0.2657243455000753,
    ),
)
MAX_CURRENT_REGRESSION = 1.01
MIN_DUPLICATED_SPEEDUP = 1.05
MIN_GEOMEAN_CURRENT_SPEEDUP = 1.02
POLICY_VERSION = "shape_topology_v1"
EXPECTED_EXECUTIONS = {
    "distributed_direct",
    "distributed_exchange",
    "distributed_large_gram",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="Capture directory produced by the shape-policy cloud runner.")
    parser.add_argument("--json-out", help="Optional machine-readable output path.")
    args = parser.parse_args()

    capture = Path(args.capture)
    analysis = json.loads((capture / "profile_analysis.json").read_text(encoding="utf-8"))
    diagnostics = {
        run_id: json.loads(
            (capture / "runs" / run_id / "diagnostics" / "runtime.json").read_text(encoding="utf-8")
        )
        for _layout, run_id, _current, _duplicated in RUNS
    }
    artifact_gate = {
        run_id: _artifact_gate(capture, run_id)
        for _layout, run_id, _current, _duplicated in RUNS
    }
    payload = compare_shape_policy_results(
        analysis,
        diagnostics=diagnostics,
        artifact_gate=artifact_gate,
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print("layout              candidate ms  vs current  vs duplicated  gate")
    for row in payload["layouts"]:
        print(
            f"{row['layout']:<20} "
            f"{row['candidate_train_step_sec'] * 1000.0:>12.1f} "
            f"{row['speedup_vs_current']:>10.3f}x "
            f"{row['speedup_vs_duplicated']:>13.3f}x "
            f"{str(row['gate']):<5}"
        )
    print(f"geomean_speedup_vs_current={payload['geomean_speedup_vs_current']:.3f}x")
    print(f"overall_gate={payload['overall_gate']}")
    return 0 if payload["overall_gate"] else 1


def compare_shape_policy_results(
    analysis: dict[str, Any],
    *,
    diagnostics: Mapping[str, dict[str, Any]],
    artifact_gate: Mapping[str, bool],
) -> dict[str, Any]:
    runs = {run["run_id"]: run for run in analysis["runs"]}
    rows = []
    for layout, run_id, current_sec, duplicated_sec in RUNS:
        run = _required_run(runs, run_id)
        candidate_sec = _steady_train_step(run)
        speedup_vs_current = current_sec / candidate_sec
        speedup_vs_duplicated = duplicated_sec / candidate_sec
        profile_gate = (
            run.get("status") == "completed"
            and run["steady"].get("start_step") == 16
            and run["steady"].get("end_step") == 63
            and run["steady"].get("row_count") == 48
        )
        policy_gate = _policy_gate(diagnostics[run_id])
        layout_gate = (
            candidate_sec <= current_sec * MAX_CURRENT_REGRESSION
            and speedup_vs_duplicated >= MIN_DUPLICATED_SPEEDUP
        )
        rows.append(
            {
                "layout": layout,
                "run_id": run_id,
                "candidate_train_step_sec": candidate_sec,
                "current_distributed_train_step_sec": current_sec,
                "duplicated_train_step_sec": duplicated_sec,
                "speedup_vs_current": speedup_vs_current,
                "speedup_vs_duplicated": speedup_vs_duplicated,
                "profile_gate": profile_gate,
                "policy_gate": policy_gate,
                "artifact_gate": bool(artifact_gate[run_id]),
                "performance_gate": layout_gate,
                "gate": profile_gate and policy_gate and bool(artifact_gate[run_id]) and layout_gate,
            }
        )
    geomean = math.exp(
        sum(math.log(row["speedup_vs_current"]) for row in rows) / len(rows)
    )
    geomean_gate = geomean >= MIN_GEOMEAN_CURRENT_SPEEDUP
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "baseline_artifact_sha256": BASELINE_ARTIFACT_SHA256,
        "gate_contract": {
            "maximum_current_regression": MAX_CURRENT_REGRESSION,
            "minimum_duplicated_speedup": MIN_DUPLICATED_SPEEDUP,
            "minimum_geomean_current_speedup": MIN_GEOMEAN_CURRENT_SPEEDUP,
            "steady_steps": [16, 63],
        },
        "layouts": rows,
        "geomean_speedup_vs_current": geomean,
        "geomean_gate": geomean_gate,
        "overall_gate": geomean_gate and all(row["gate"] for row in rows),
    }


def _required_run(runs: Mapping[str, dict[str, Any]], run_id: str) -> dict[str, Any]:
    if run_id not in runs:
        raise ValueError(f"profile analysis is missing run {run_id!r}")
    return runs[run_id]


def _steady_train_step(run: dict[str, Any]) -> float:
    value = run.get("steady", {}).get("medians", {}).get("train_step_sec")
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"run {run.get('run_id')!r} has no positive finite steady train_step_sec")
    return float(value)


def _policy_gate(diagnostics: dict[str, Any]) -> bool:
    policy = diagnostics.get("optimizer", {}).get("dist_muon", {})
    if policy.get("exact") is not False:
        return False
    shape_policy = policy.get("shape_topology_policy", {})
    if shape_policy.get("version") != POLICY_VERSION:
        return False
    plans = policy.get("leaf_execution_plans")
    if not isinstance(plans, list) or not plans:
        return False
    selected = {plan.get("execution") for plan in plans}
    if selected != EXPECTED_EXECUTIONS:
        return False
    for plan in plans:
        selection = plan.get("selection", {})
        if selection.get("policy_version") != POLICY_VERSION:
            return False
        if selection.get("selected_execution") != plan.get("execution"):
            return False
        if not selection.get("eligible_executions"):
            return False
        if not selection.get("selection_reason"):
            return False
        if not selection.get("modeled_costs"):
            return False
        if set(plan.get("roles", {})) != {"parameter", "gradient", "momentum", "update"}:
            return False
    return True


def _artifact_gate(capture: Path, run_id: str) -> bool:
    eval_path = capture / f"eval_{run_id}.json"
    sample_path = capture / f"sample_{run_id}.json"
    hlo_root = capture / "hlo" / run_id
    if not eval_path.is_file() or not sample_path.is_file() or not hlo_root.is_dir():
        return False
    if not list(hlo_root.rglob("*.txt")):
        return False
    final_path = capture / "runs" / run_id / "summaries" / "final.json"
    if not final_path.is_file():
        return False
    final = json.loads(final_path.read_text(encoding="utf-8"))
    return (
        final.get("status") == "completed"
        and int(final.get("final_optimizer_nonfinite_group_count", -1)) == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
