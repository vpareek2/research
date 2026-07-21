#!/usr/bin/env python3
"""Analyze M1 MoE H100 acceptance artifacts against the July H100 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

from jaxtitan.runtime.profile_analysis import analyze_profile_root, format_profile_analysis


PROFILE_RUNS = (
    "cloud_4gpu_profile64_trinity_moe_ep_adamw",
    "cloud_4gpu_profile64_trinity_moe_ep_muon",
    "cloud_4gpu_profile64_trinity_moe_tp_ep_adamw",
    "cloud_4gpu_profile64_trinity_moe_tp_ep_muon",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="Candidate capture directory or .tgz produced by cloud_moe_m1_h100_matrix.sh.")
    parser.add_argument(
        "--baseline",
        default="cloud_results/profile64_h100_sxm_2026-07-20.tgz",
        help="Baseline capture directory or .tgz. Defaults to the archived July H100 profile bundle.",
    )
    parser.add_argument(
        "--work-dir",
        default="cloud_results/moe_m1_analysis_extract",
        help="Directory used for archive extraction.",
    )
    parser.add_argument("--json-out", help="Optional path for machine-readable comparison JSON.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    candidate_root = _materialize(Path(args.candidate), work_dir / "candidate")
    baseline_root = _materialize(Path(args.baseline), work_dir / "baseline")

    candidate = analyze_profile_root(candidate_root)
    baseline = analyze_profile_root(baseline_root)
    payload = _compare(candidate, baseline)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    print("candidate:")
    print(format_profile_analysis(candidate))
    print()
    print("baseline:")
    print(format_profile_analysis(baseline))
    print()
    print("M1 MoE profile comparison:")
    print("run                                             base ms   cand ms  speedup  gate  scatter  gemm%")
    for row in payload["profile_runs"]:
        print(
            f"{row['run_id']:<47} "
            f"{_ms(row['baseline_train_step_sec']):>8} "
            f"{_ms(row['candidate_train_step_sec']):>8} "
            f"{row['speedup']:>7.2f}x "
            f"{row['gate']:<5} "
            f"{row['candidate_scatter_reduce_fusion_count']:>7} "
            f"{row['candidate_gemm_fraction'] * 100.0:>5.1f}"
        )
    print()
    print(f"overall_gate={payload['overall_gate']}")
    return 0 if payload["overall_gate"] else 1


def _materialize(source: Path, destination: Path) -> Path:
    if source.is_dir():
        return source
    if not source.is_file():
        raise SystemExit(f"missing artifact: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if source.suffixes[-2:] == [".tar", ".gz"] or source.suffix == ".tgz":
        with tarfile.open(source, "r:gz") as archive:
            archive.extractall(destination)
        return destination
    raise SystemExit(f"unsupported artifact type: {source}")


def _compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_by_id = {run["run_id"]: run for run in candidate["runs"]}
    baseline_by_id = {run["run_id"]: run for run in baseline["runs"]}
    rows = []
    for run_id in PROFILE_RUNS:
        if run_id not in candidate_by_id:
            raise SystemExit(f"candidate is missing profile run: {run_id}")
        if run_id not in baseline_by_id:
            raise SystemExit(f"baseline is missing profile run: {run_id}")
        cand = candidate_by_id[run_id]
        base = baseline_by_id[run_id]
        cand_sec = _metric(cand, "train_step_sec")
        base_sec = _metric(base, "train_step_sec")
        speedup = base_sec / cand_sec if cand_sec > 0 else 0.0
        categories = cand.get("trace", {}).get("categories", {}) if cand.get("trace") else {}
        scatter = categories.get("scatter_reduce_fusion", {})
        gemm = categories.get("gemm", {})
        row = {
            "run_id": run_id,
            "baseline_train_step_sec": base_sec,
            "candidate_train_step_sec": cand_sec,
            "speedup": speedup,
            "candidate_scatter_reduce_fusion_count": int(scatter.get("count", 0)),
            "candidate_scatter_reduce_fusion_sec": float(scatter.get("duration_sec", 0.0)),
            "candidate_gemm_fraction": float(gemm.get("event_sum_fraction", 0.0)),
            "gate": speedup >= 5.0 and int(scatter.get("count", 0)) == 0,
        }
        rows.append(row)
    return {
        "candidate_source": candidate["source"],
        "baseline_source": baseline["source"],
        "profile_runs": rows,
        "overall_gate": all(row["gate"] for row in rows),
    }


def _metric(run: dict[str, Any], field: str) -> float:
    value = run["steady"]["medians"].get(field)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"run {run['run_id']} has no steady median {field}")
    return float(value)


def _ms(value: float) -> str:
    return f"{value * 1000.0:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
