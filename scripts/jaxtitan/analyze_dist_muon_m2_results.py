#!/usr/bin/env python3
"""Compare matched duplicated and distributed Muon M2 profile runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_PAIRS = (
    (
        "dense_tp",
        "cloud_4gpu_profile64_dense_tp_muon",
        "cloud_4gpu_profile64_dense_tp_muon_distributed",
    ),
    (
        "dense_fsdp_tp",
        "cloud_4gpu_profile64_dense_fsdp_tp_muon",
        "cloud_4gpu_profile64_dense_fsdp_tp_muon_distributed",
    ),
    (
        "dense_zero2_tp",
        "cloud_4gpu_profile64_dense_zero2_tp_muon",
        "cloud_4gpu_profile64_dense_zero2_tp_muon_distributed",
    ),
    (
        "trinity_tp_ep",
        "cloud_4gpu_profile64_trinity_moe_tp_ep_muon",
        "cloud_4gpu_profile64_trinity_moe_tp_ep_muon_distributed",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="Capture directory produced by cloud_dist_muon_m2_matrix.sh.")
    parser.add_argument("--json-out", help="Optional machine-readable output path.")
    args = parser.parse_args()

    capture = Path(args.capture)
    analysis = json.loads((capture / "profile_analysis.json").read_text(encoding="utf-8"))
    payload = compare_profile_analysis(analysis)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    print("layout              duplicated ms  distributed ms  speedup  gate")
    for row in payload["pairs"]:
        print(
            f"{row['layout']:<20} "
            f"{row['duplicated_train_step_sec'] * 1000.0:>13.1f} "
            f"{row['distributed_train_step_sec'] * 1000.0:>15.1f} "
            f"{row['speedup']:>7.2f}x "
            f"{str(row['gate']):<5}"
        )
    print(f"overall_gate={payload['overall_gate']}")
    return 0 if payload["overall_gate"] else 1


def compare_profile_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    runs = {run["run_id"]: run for run in analysis["runs"]}
    rows = []
    for layout, duplicated_id, distributed_id in RUN_PAIRS:
        duplicated = _required_run(runs, duplicated_id)
        distributed = _required_run(runs, distributed_id)
        duplicated_sec = _steady_metric(duplicated, "train_step_sec")
        distributed_sec = _steady_metric(distributed, "train_step_sec")
        speedup = duplicated_sec / distributed_sec if distributed_sec > 0.0 else 0.0
        row = {
            "layout": layout,
            "duplicated_run_id": duplicated_id,
            "distributed_run_id": distributed_id,
            "duplicated_train_step_sec": duplicated_sec,
            "distributed_train_step_sec": distributed_sec,
            "speedup": speedup,
            "gate": speedup > 1.0,
        }
        rows.append(row)
    return {
        "pairs": rows,
        "gate_contract": "distributed median train_step_sec must beat duplicated on every matched layout",
        "overall_gate": all(row["gate"] for row in rows),
    }


def _required_run(runs: dict[str, dict[str, Any]], run_id: str) -> dict[str, Any]:
    if run_id not in runs:
        raise SystemExit(f"profile analysis is missing run: {run_id}")
    return runs[run_id]


def _steady_metric(run: dict[str, Any], field: str) -> float:
    value = run.get("steady", {}).get("medians", {}).get(field)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"run {run['run_id']} has no steady median {field}")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
