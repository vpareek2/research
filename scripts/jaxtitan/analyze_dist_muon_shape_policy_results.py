#!/usr/bin/env python3
"""Validate four-layout shape/topology Muon production acceptance."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
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
SMOKE_RUNS = (
    ("dense_tp", "cloud_4gpu_smoke8_dense_tp_muon_shape_policy"),
    ("dense_fsdp_tp", "cloud_4gpu_smoke8_dense_fsdp_tp_muon_shape_policy"),
    ("dense_zero2_tp", "cloud_4gpu_smoke8_dense_zero2_tp_muon_shape_policy"),
    ("trinity_tp_ep", "cloud_4gpu_smoke8_trinity_moe_tp_ep_muon_shape_policy"),
)
COMPARISON_SCHEMA_VERSION = 2
SMOKE_MINIMUM_TRAIN_ROWS = 8
MAX_CURRENT_REGRESSION = 1.01
MIN_DUPLICATED_SPEEDUP = 1.05
MIN_GEOMEAN_CURRENT_SPEEDUP = 1.02
POLICY_VERSION = "shape_topology_v1"
EXPECTED_EXECUTIONS = {
    "distributed_direct",
    "distributed_exchange",
    "distributed_large_gram",
}
FINITE_TRAIN_FIELDS = (
    "loss",
    "lm_loss",
    "grad_norm",
    "param_norm",
    "update_norm",
    "train_step_sec",
    "step_sec",
)
FINITE_GROUP_FIELDS = (
    "grad_norm",
    "update_norm",
    "param_norm",
    "grad_param_ratio",
    "update_param_ratio",
)
FINITE_GROUP_FLAGS = (
    "grad_norm_finite",
    "update_norm_finite",
    "param_norm_finite",
    "grad_param_ratio_finite",
    "update_param_ratio_finite",
)
_BUCKET_PATTERN = re.compile(r'muon_bucket="(-?\d+)"')
_OP_PATTERN = re.compile(r'muon_op="([^"]+)"')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture",
        nargs="?",
        help="Capture directory produced by the shape-policy cloud runner.",
    )
    parser.add_argument("--phase", choices=("smoke", "profile"))
    parser.add_argument("--json-out", help="Optional machine-readable output path.")
    parser.add_argument(
        "--verify-smoke-gate",
        help="Validate a smoke comparison before launching profile work.",
    )
    parser.add_argument(
        "--current-commit",
        help="Current 40-character Git commit for smoke-gate verification.",
    )
    args = parser.parse_args()

    if args.verify_smoke_gate:
        if args.capture or args.phase or args.json_out:
            parser.error(
                "--verify-smoke-gate cannot be combined with capture, --phase, or --json-out"
            )
        if not args.current_commit:
            parser.error("--verify-smoke-gate requires --current-commit")
        gate = verify_smoke_gate(
            _load_json(Path(args.verify_smoke_gate)),
            current_commit=args.current_commit,
        )
        for failure in gate["failures"]:
            print(f"- {failure}")
        print(f"smoke_gate={gate['gate']}")
        return 0 if gate["gate"] else 1
    if not args.capture or not args.phase:
        parser.error("capture and --phase are required for capture analysis")

    capture = Path(args.capture)
    run_specs = _phase_run_specs(args.phase)
    analysis = (
        _load_json(capture / "profile_analysis.json")
        if args.phase == "profile"
        else {"runs": []}
    )
    diagnostics = {
        run_id: _load_json(
            capture / "runs" / run_id / "diagnostics" / "runtime.json"
        )
        for _layout, run_id, _current, _duplicated in run_specs
    }
    artifact_gate = {
        run_id: analyze_run_artifacts(
            capture,
            run_id,
            diagnostics[run_id],
            minimum_train_rows=(
                SMOKE_MINIMUM_TRAIN_ROWS if args.phase == "smoke" else None
            ),
        )
        for _layout, run_id, _current, _duplicated in run_specs
    }
    payload = compare_shape_policy_results(
        analysis,
        diagnostics=diagnostics,
        artifact_gate=artifact_gate,
        phase=args.phase,
        source_commit=_read_source_commit(capture),
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    if args.phase == "profile":
        print("layout              candidate ms  vs current  vs duplicated  gate")
        for row in payload["layouts"]:
            print(
                f"{row['layout']:<20} "
                f"{row['candidate_train_step_sec'] * 1000.0:>12.1f} "
                f"{row['speedup_vs_current']:>10.3f}x "
                f"{row['speedup_vs_duplicated']:>13.3f}x "
                f"{str(row['gate']):<5}"
            )
            for failure in row["failures"]:
                print(f"  - {failure}")
        print(
            "geomean_speedup_vs_current="
            f"{payload['geomean_speedup_vs_current']:.3f}x"
        )
    else:
        print("layout              policy  artifacts  gate")
        for row in payload["layouts"]:
            print(
                f"{row['layout']:<20} "
                f"{str(row['policy_gate']):<7} "
                f"{str(row['artifact_gate']):<10} "
                f"{str(row['gate']):<5}"
            )
            for failure in row["failures"]:
                print(f"  - {failure}")
    print(f"overall_gate={payload['overall_gate']}")
    return 0 if payload["overall_gate"] else 1


def compare_shape_policy_results(
    analysis: dict[str, Any],
    *,
    diagnostics: Mapping[str, dict[str, Any]],
    artifact_gate: Mapping[str, Mapping[str, Any]],
    phase: str = "profile",
    source_commit: str | None = None,
) -> dict[str, Any]:
    if phase not in {"smoke", "profile"}:
        raise ValueError(f"unsupported shape-policy acceptance phase {phase!r}")
    run_specs = _phase_run_specs(phase)
    runs = {run["run_id"]: run for run in analysis["runs"]}
    rows = []
    for layout, run_id, current_sec, duplicated_sec in run_specs:
        policy = analyze_policy_metadata(diagnostics[run_id])
        artifacts = dict(artifact_gate[run_id])
        failures = []
        failures.extend(policy["failures"])
        failures.extend(artifacts["failures"])
        row = {
            "layout": layout,
            "run_id": run_id,
            "policy_gate": policy["gate"],
            "artifact_gate": artifacts["gate"],
            "policy": policy,
            "artifacts": artifacts,
        }
        if phase == "profile":
            assert current_sec is not None and duplicated_sec is not None
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
            performance_gate = (
                candidate_sec <= current_sec * MAX_CURRENT_REGRESSION
                and speedup_vs_duplicated >= MIN_DUPLICATED_SPEEDUP
            )
            if not profile_gate:
                failures.append(
                    "steady profile must contain completed steps 16-63 (48 rows)"
                )
            if candidate_sec > current_sec * MAX_CURRENT_REGRESSION:
                failures.append(
                    "candidate is more than 1% slower than current distributed"
                )
            if speedup_vs_duplicated < MIN_DUPLICATED_SPEEDUP:
                failures.append("candidate is less than 5% faster than duplicated")
            row.update(
                {
                    "candidate_train_step_sec": candidate_sec,
                    "current_distributed_train_step_sec": current_sec,
                    "duplicated_train_step_sec": duplicated_sec,
                    "speedup_vs_current": speedup_vs_current,
                    "speedup_vs_duplicated": speedup_vs_duplicated,
                    "profile_gate": profile_gate,
                    "performance_gate": performance_gate,
                }
            )
            row_gate = (
                profile_gate
                and policy["gate"]
                and artifacts["gate"]
                and performance_gate
            )
        else:
            row["performance_gate"] = None
            row_gate = policy["gate"] and artifacts["gate"]
        row["failures"] = failures
        row["gate"] = row_gate
        rows.append(row)
    if phase == "profile":
        geomean = math.exp(
            sum(math.log(row["speedup_vs_current"]) for row in rows) / len(rows)
        )
        geomean_gate = geomean >= MIN_GEOMEAN_CURRENT_SPEEDUP
    else:
        geomean = None
        geomean_gate = None
    commit_gate = (
        phase != "smoke"
        or (
            isinstance(source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None
        )
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "phase": phase,
        "policy_version": POLICY_VERSION,
        "source_commit": source_commit,
        "baseline_artifact_sha256": BASELINE_ARTIFACT_SHA256,
        "gate_contract": {
            "maximum_current_regression": MAX_CURRENT_REGRESSION,
            "minimum_duplicated_speedup": MIN_DUPLICATED_SPEEDUP,
            "minimum_geomean_current_speedup": MIN_GEOMEAN_CURRENT_SPEEDUP,
            "steady_steps": [16, 63],
            "smoke_minimum_train_rows": SMOKE_MINIMUM_TRAIN_ROWS,
            "persistent_replica_max_abs_diff": 0.0,
        },
        "layouts": rows,
        "geomean_speedup_vs_current": geomean,
        "geomean_gate": geomean_gate,
        "commit_gate": commit_gate,
        "overall_gate": (
            commit_gate
            and all(row["gate"] for row in rows)
            and (geomean_gate is not False)
        ),
    }


def verify_smoke_gate(
    payload: Mapping[str, Any],
    *,
    current_commit: str,
) -> dict[str, Any]:
    failures = []
    if re.fullmatch(r"[0-9a-f]{40}", current_commit) is None:
        failures.append("current commit is not a 40-character lowercase Git SHA")
    if payload.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        failures.append(
            f"smoke comparison schema must equal {COMPARISON_SCHEMA_VERSION}"
        )
    if payload.get("phase") != "smoke":
        failures.append("comparison phase is not smoke")
    if payload.get("policy_version") != POLICY_VERSION:
        failures.append(f"smoke policy version is not {POLICY_VERSION}")
    if (
        not isinstance(payload.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["source_commit"]) is None
    ):
        failures.append("smoke comparison source_commit is not a full Git SHA")
    elif payload.get("source_commit") != current_commit:
        failures.append("smoke comparison commit does not match current HEAD")
    if payload.get("overall_gate") is not True:
        failures.append("smoke comparison overall_gate is not true")
    layouts = payload.get("layouts")
    expected_layouts = {layout for layout, _run_id in SMOKE_RUNS}
    if not isinstance(layouts, list):
        failures.append("smoke comparison layouts must be a list")
    else:
        observed_layouts = {
            row.get("layout")
            for row in layouts
            if isinstance(row, Mapping)
        }
        if observed_layouts != expected_layouts or len(layouts) != len(SMOKE_RUNS):
            failures.append("smoke comparison does not contain exactly all four layouts")
        if any(
            not isinstance(row, Mapping) or row.get("gate") is not True
            for row in layouts
        ):
            failures.append("one or more smoke layouts did not pass")
    return _gate_payload(failures)


def _phase_run_specs(
    phase: str,
) -> tuple[tuple[str, str, float | None, float | None], ...]:
    if phase == "profile":
        return RUNS
    if phase == "smoke":
        return tuple(
            (layout, run_id, None, None)
            for layout, run_id in SMOKE_RUNS
        )
    raise ValueError(f"unsupported shape-policy acceptance phase {phase!r}")


def analyze_policy_metadata(diagnostics: dict[str, Any]) -> dict[str, Any]:
    failures = []
    policy = diagnostics.get("optimizer", {}).get("dist_muon", {})
    if policy.get("exact") is not False:
        failures.append("distributed Muon metadata must retain exact=false")
    shape_policy = policy.get("shape_topology_policy", {})
    if shape_policy.get("version") != POLICY_VERSION:
        failures.append(f"missing shape policy version {POLICY_VERSION}")
    plans = policy.get("leaf_execution_plans")
    if not isinstance(plans, list) or not plans:
        return _gate_payload(["missing non-empty leaf execution plans"])
    selected = {plan.get("execution") for plan in plans}
    if selected != EXPECTED_EXECUTIONS:
        failures.append(
            f"selected executions {sorted(str(item) for item in selected)} "
            f"do not equal {sorted(EXPECTED_EXECUTIONS)}"
        )
    for index, plan in enumerate(plans):
        selection = plan.get("selection", {})
        prefix = f"leaf plan {index}"
        if selection.get("policy_version") != POLICY_VERSION:
            failures.append(f"{prefix} has wrong policy version")
        if selection.get("selected_execution") != plan.get("execution"):
            failures.append(f"{prefix} selection does not match execution")
        if not selection.get("eligible_executions"):
            failures.append(f"{prefix} has no eligible executions")
        if not selection.get("selection_reason"):
            failures.append(f"{prefix} has no selection reason")
        if not selection.get("modeled_costs"):
            failures.append(f"{prefix} has no modeled costs")
        if set(plan.get("roles", {})) != {"parameter", "gradient", "momentum", "update"}:
            failures.append(f"{prefix} is missing role sharding metadata")
    return _gate_payload(failures, leaf_count=len(plans))


def analyze_run_artifacts(
    capture: Path,
    run_id: str,
    diagnostics: dict[str, Any],
    *,
    minimum_train_rows: int | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    checks["metrics"] = _metrics_gate(
        capture / "runs" / run_id / "metrics" / "train.jsonl",
        minimum_rows=minimum_train_rows,
    )
    checks["final"] = _final_gate(
        capture / "runs" / run_id / "summaries" / "final.json",
        run_id,
    )
    checks["eval"] = _eval_gate(capture / f"eval_{run_id}.json", run_id)
    checks["sample"] = _sample_gate(capture / f"sample_{run_id}.json", run_id)
    checks["checkpoint_index"] = _checkpoint_index_gate(
        capture / "runs" / run_id / "checkpoints" / "index.json"
    )
    checks["replica_audit"] = _replica_audit_gate(
        capture / f"replica_audit_{run_id}.json",
        run_id,
        require_replicas=_plans_require_replicas(diagnostics),
    )
    checks["hlo"] = _hlo_gate(capture / "hlo" / run_id, diagnostics)
    checks["checkpoint_identity"] = _checkpoint_identity_gate(checks)
    failures = [
        f"{name}: {failure}"
        for name, check in checks.items()
        for failure in check["failures"]
    ]
    return {
        "gate": all(check["gate"] for check in checks.values()),
        "checks": checks,
        "failures": failures,
    }


def _metrics_gate(
    path: Path,
    *,
    minimum_rows: int | None = None,
) -> dict[str, Any]:
    rows, failures = _load_jsonl_checked(path, "train metrics")
    if failures:
        return _gate_payload(failures, row_count=0)
    if minimum_rows is not None and len(rows) < minimum_rows:
        failures.append(
            f"training metrics contain {len(rows)} rows, require at least {minimum_rows}"
        )
    steps = []
    for index, row in enumerate(rows):
        prefix = f"row {index}"
        step = row.get("step")
        if not isinstance(step, int):
            failures.append(f"{prefix} has no integer step")
        else:
            steps.append(step)
        for field in FINITE_TRAIN_FIELDS:
            if not _finite_number(row.get(field)):
                failures.append(f"{prefix} field {field} is not finite")
        if row.get("optimizer_nonfinite_group_count") != 0:
            failures.append(f"{prefix} reports nonfinite optimizer groups")
        if row.get("optimizer_nonfinite_groups") != []:
            failures.append(f"{prefix} must record an empty nonfinite optimizer group list")
        groups = row.get("optimizer_groups")
        if not isinstance(groups, list) or not groups:
            failures.append(f"{prefix} has no optimizer groups")
            continue
        for group_index, group in enumerate(groups):
            group_prefix = f"{prefix} optimizer group {group_index}"
            if not isinstance(group, Mapping):
                failures.append(f"{group_prefix} is not an object")
                continue
            for field in FINITE_GROUP_FIELDS:
                if not _finite_number(group.get(field)):
                    failures.append(f"{group_prefix} field {field} is not finite")
            for field in FINITE_GROUP_FLAGS:
                if group.get(field) is not True:
                    failures.append(f"{group_prefix} flag {field} is not true")
    if steps and steps != sorted(set(steps)):
        failures.append("training steps are not strictly increasing and unique")
    return _gate_payload(
        failures,
        row_count=len(rows),
        first_step=None if not steps else steps[0],
        last_step=None if not steps else steps[-1],
    )


def _final_gate(path: Path, run_id: str) -> dict[str, Any]:
    payload, failures = _load_json_checked(path, "final summary")
    if failures:
        return _gate_payload(failures)
    if payload.get("run_id") != run_id:
        failures.append("run_id does not match")
    if payload.get("status") != "completed":
        failures.append("status is not completed")
    if payload.get("final_optimizer_nonfinite_group_count") != 0:
        failures.append("final summary reports nonfinite optimizer groups")
    if payload.get("final_optimizer_nonfinite_groups") != []:
        failures.append("final summary must record an empty nonfinite optimizer group list")
    groups = payload.get("final_optimizer_groups")
    if not isinstance(groups, list) or not groups:
        failures.append("final summary has no optimizer groups")
    else:
        for index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                failures.append(f"optimizer group {index} is not an object")
                continue
            for field in FINITE_GROUP_FIELDS:
                if not _finite_number(group.get(field)):
                    failures.append(f"optimizer group {index} field {field} is not finite")
            for field in FINITE_GROUP_FLAGS:
                if group.get(field) is not True:
                    failures.append(f"optimizer group {index} flag {field} is not true")
    return _gate_payload(failures)


def _eval_gate(path: Path, run_id: str) -> dict[str, Any]:
    payload, failures = _load_json_checked(path, "checkpoint eval")
    if failures:
        return _gate_payload(failures)
    checkpoint = payload.get("checkpoint")
    evaluation = payload.get("eval")
    if not isinstance(checkpoint, Mapping):
        failures.append("checkpoint is not an object")
        checkpoint = {}
    if not isinstance(evaluation, Mapping):
        failures.append("eval is not an object")
        evaluation = {}
    if payload.get("run_id") != run_id:
        failures.append("run_id does not match")
    if payload.get("status") != "completed":
        failures.append("status is not completed")
    if not isinstance(checkpoint.get("step"), int):
        failures.append("checkpoint step is missing")
    if not _finite_number(evaluation.get("loss")):
        failures.append("eval loss is not finite")
    if not isinstance(evaluation.get("token_count"), int) or evaluation["token_count"] <= 0:
        failures.append("eval token_count is not positive")
    return _gate_payload(
        failures,
        checkpoint_step=checkpoint.get("step"),
        checkpoint_path=checkpoint.get("path"),
        checkpoint_fingerprint=checkpoint.get("runtime_fingerprint"),
    )


def _sample_gate(path: Path, run_id: str) -> dict[str, Any]:
    payload, failures = _load_json_checked(path, "checkpoint sample")
    if failures:
        return _gate_payload(failures)
    checkpoint = payload.get("checkpoint")
    sampling = payload.get("sampling")
    if not isinstance(checkpoint, Mapping):
        failures.append("checkpoint is not an object")
        checkpoint = {}
    if not isinstance(sampling, Mapping):
        failures.append("sampling is not an object")
        sampling = {}
    generated = payload.get("generated_ids")
    full_ids = payload.get("full_ids")
    prompt_ids = payload.get("prompt_ids")
    logprobs = payload.get("logprobs")
    if payload.get("run_id") != run_id:
        failures.append("run_id does not match")
    if payload.get("status") != "completed":
        failures.append("status is not completed")
    if checkpoint.get("selector") != "latest":
        failures.append("checkpoint selector is not latest")
    if not isinstance(checkpoint.get("step"), int):
        failures.append("checkpoint step is missing")
    max_new_tokens = sampling.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        failures.append("sampling max_new_tokens is not positive")
    if not isinstance(generated, list) or len(generated) != max_new_tokens:
        failures.append("generated token count does not match sampling contract")
    if not isinstance(logprobs, list) or len(logprobs) != max_new_tokens:
        failures.append("logprob count does not match sampling contract")
    elif not all(_finite_number(value) for value in logprobs):
        failures.append("sample logprobs are not finite")
    if (
        not isinstance(full_ids, list)
        or not isinstance(prompt_ids, list)
        or not isinstance(generated, list)
        or full_ids != prompt_ids + generated
    ):
        failures.append("full_ids do not equal prompt_ids plus generated_ids")
    return _gate_payload(
        failures,
        checkpoint_step=checkpoint.get("step"),
        checkpoint_path=checkpoint.get("path"),
        checkpoint_fingerprint=checkpoint.get("runtime_fingerprint"),
    )


def _checkpoint_index_gate(path: Path) -> dict[str, Any]:
    payload, failures = _load_json_checked(path, "checkpoint index")
    if failures:
        return _gate_payload(failures)
    latest_step = payload.get("latest_step")
    records = payload.get("records")
    if not isinstance(latest_step, int):
        failures.append("latest_step is missing")
    if not isinstance(records, list) or not records:
        failures.append("retained checkpoint records are missing")
    else:
        retained = [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("retained") is True
            and isinstance(record.get("step"), int)
        ]
        retained_steps = [record["step"] for record in retained]
        if not retained_steps or latest_step != max(retained_steps):
            failures.append("latest_step does not equal the newest retained checkpoint")
    return _gate_payload(
        failures,
        checkpoint_step=latest_step,
        checkpoint_path=payload.get("latest_checkpoint_path"),
    )


def _replica_audit_gate(
    path: Path,
    run_id: str,
    *,
    require_replicas: bool,
) -> dict[str, Any]:
    payload, failures = _load_json_checked(path, "replica audit")
    if failures:
        return _gate_payload(failures)
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        failures.append("checkpoint is not an object")
        checkpoint = {}
    if payload.get("run_id") != run_id:
        failures.append("run_id does not match")
    if payload.get("schema_version") != 1:
        failures.append("schema_version is not 1")
    if payload.get("overall_gate") is not True:
        failures.append("overall_gate is not true")
    if payload.get("finite") is not True:
        failures.append("persistent train state is not finite")
    if payload.get("max_replica_abs_diff") != 0.0:
        failures.append("persistent replicas disagree")
    array_count = payload.get("array_count")
    if not isinstance(array_count, int) or array_count <= 0:
        failures.append("audit contains no persistent arrays")
    sections = payload.get("sections")
    if not isinstance(sections, Mapping) or set(sections) != {"model", "optimizer"}:
        failures.append("audit must contain model and optimizer sections")
    else:
        section_array_count = 0
        section_replicated_count = 0
        for name, section in sections.items():
            if not isinstance(section, Mapping):
                failures.append(f"{name} audit section is not an object")
                continue
            if section.get("gate") is not True:
                failures.append(f"{name} audit section did not pass")
            if section.get("finite") is not True:
                failures.append(f"{name} audit section is not finite")
            if section.get("max_replica_abs_diff") != 0.0:
                failures.append(f"{name} audit section has replica disagreement")
            if not isinstance(section.get("array_count"), int) or section["array_count"] <= 0:
                failures.append(f"{name} audit section has no arrays")
            else:
                section_array_count += section["array_count"]
            if not isinstance(section.get("replicated_array_count"), int):
                failures.append(f"{name} audit section has no replicated-array count")
            else:
                section_replicated_count += section["replicated_array_count"]
            if section.get("nonfinite_paths") != []:
                failures.append(f"{name} audit section lists nonfinite paths")
            if section.get("replica_disagreement_paths") != []:
                failures.append(f"{name} audit section lists replica disagreements")
        if isinstance(array_count, int) and array_count != section_array_count:
            failures.append("audit array_count does not equal section totals")
        if (
            isinstance(payload.get("replicated_array_count"), int)
            and payload["replicated_array_count"] != section_replicated_count
        ):
            failures.append("audit replicated_array_count does not equal section totals")
    replicated_count = payload.get("replicated_array_count")
    if require_replicas and (not isinstance(replicated_count, int) or replicated_count <= 0):
        failures.append("declared replica axes produced no physically replicated arrays")
    return _gate_payload(
        failures,
        checkpoint_step=checkpoint.get("step"),
        checkpoint_path=checkpoint.get("path"),
        checkpoint_fingerprint=checkpoint.get("runtime_fingerprint"),
        array_count=array_count,
        replicated_array_count=replicated_count,
        max_replica_abs_diff=payload.get("max_replica_abs_diff"),
    )


def _checkpoint_identity_gate(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    steps = {
        name: checks[name].get("checkpoint_step")
        for name in ("eval", "sample", "checkpoint_index", "replica_audit")
    }
    paths = {
        name: checks[name].get("checkpoint_path")
        for name in ("eval", "sample", "checkpoint_index", "replica_audit")
    }
    fingerprints = {
        name: checks[name].get("checkpoint_fingerprint")
        for name in ("eval", "sample", "replica_audit")
    }
    failures = []
    if any(not isinstance(step, int) for step in steps.values()):
        failures.append(f"checkpoint steps are incomplete: {steps}")
    elif len(set(steps.values())) != 1:
        failures.append(f"checkpoint steps disagree: {steps}")
    if any(not isinstance(path, str) or not path for path in paths.values()):
        failures.append(f"checkpoint paths are incomplete: {paths}")
    elif len(set(paths.values())) != 1:
        failures.append(f"checkpoint paths disagree: {paths}")
    if any(
        not isinstance(fingerprint, str) or not fingerprint
        for fingerprint in fingerprints.values()
    ):
        failures.append(f"checkpoint fingerprints are incomplete: {fingerprints}")
    elif len(set(fingerprints.values())) != 1:
        failures.append(f"checkpoint fingerprints disagree: {fingerprints}")
    return _gate_payload(
        failures,
        checkpoint_steps=steps,
        checkpoint_paths=paths,
        checkpoint_fingerprints=fingerprints,
    )


def _hlo_gate(hlo_root: Path, diagnostics: dict[str, Any]) -> dict[str, Any]:
    plans = diagnostics.get("optimizer", {}).get("dist_muon", {}).get(
        "leaf_execution_plans", []
    )
    expected, failures = _expected_hlo_signature(plans)
    if failures:
        return _gate_payload(failures)
    if not hlo_root.is_dir():
        return _gate_payload([f"missing HLO directory {hlo_root}"])
    modules = []
    for path in sorted(hlo_root.rglob("*jit__compiled_impl.before_optimizations.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "muon_op" not in text:
            continue
        modules.append(
            {
                "path": path.name,
                "signature": _observed_hlo_signature(text),
            }
        )
    if not modules:
        return _gate_payload(["no compiled train HLO module contains Muon metadata"])
    canonical = modules[0]["signature"]
    inconsistent = [
        module["path"] for module in modules[1:] if module["signature"] != canonical
    ]
    if inconsistent:
        failures.append(f"compiled Muon HLO signatures disagree: {inconsistent}")
    if canonical != expected:
        failures.extend(_signature_failures(expected, canonical))
    return _gate_payload(
        failures,
        module_count=len(modules),
        expected_signature=expected,
        observed_signature=canonical,
    )


def _expected_hlo_signature(
    plans: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    failures = []
    buckets: dict[int, list[dict[str, Any]]] = {}
    for plan in plans:
        if plan.get("execution") == "duplicated":
            continue
        bucket_id = plan.get("bucket_id")
        if not isinstance(bucket_id, int) or bucket_id < 0:
            failures.append("distributed leaf plan has no nonnegative bucket_id")
            continue
        buckets.setdefault(bucket_id, []).append(plan)
    expected = {}
    for bucket_id, bucket_plans in sorted(buckets.items()):
        executions = {plan.get("execution") for plan in bucket_plans}
        if len(executions) != 1:
            failures.append(
                f"bucket {bucket_id} mixes executions "
                f"{sorted(str(execution) for execution in executions)}"
            )
            continue
        execution = executions.pop()
        leaf_count = len(bucket_plans)
        expected[str(bucket_id)] = {
            "norm_all_reduce": 1,
            "gram_all_reduce": 5,
            "right_gram_all_reduce": 5
            if execution == "distributed_large_gram"
            else 0,
            "exchange_forward_all_to_all": leaf_count
            if execution == "distributed_exchange"
            else 0,
            "exchange_reverse_all_to_all": leaf_count
            if execution == "distributed_exchange"
            else 0,
        }
    if not expected:
        failures.append("no distributed Muon HLO buckets were planned")
    return expected, failures


def _observed_hlo_signature(text: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for line in text.splitlines():
        bucket_match = _BUCKET_PATTERN.search(line)
        op_match = _OP_PATTERN.search(line)
        if bucket_match is None or op_match is None:
            continue
        bucket = bucket_match.group(1)
        operation = op_match.group(1)
        counter = counts.setdefault(bucket, Counter())
        if " all-reduce(" in line and operation == "norm":
            counter["norm_all_reduce"] += 1
        elif " all-reduce(" in line and operation == "gram":
            counter["gram_all_reduce"] += 1
            if 'muon_gram_side="right"' in line:
                counter["right_gram_all_reduce"] += 1
        elif " all-to-all(" in line and operation == "exchange_forward":
            counter["exchange_forward_all_to_all"] += 1
        elif " all-to-all(" in line and operation == "exchange_reverse":
            counter["exchange_reverse_all_to_all"] += 1
    fields = (
        "norm_all_reduce",
        "gram_all_reduce",
        "right_gram_all_reduce",
        "exchange_forward_all_to_all",
        "exchange_reverse_all_to_all",
    )
    return {
        bucket: {field: counter[field] for field in fields}
        for bucket, counter in sorted(counts.items(), key=lambda item: int(item[0]))
        if any(counter[field] for field in fields)
    }


def _signature_failures(
    expected: Mapping[str, Mapping[str, int]],
    observed: Mapping[str, Mapping[str, int]],
) -> list[str]:
    failures = []
    if set(observed) != set(expected):
        failures.append(
            f"HLO bucket ids {sorted(observed)} do not equal planned ids {sorted(expected)}"
        )
    for bucket in sorted(set(expected) | set(observed), key=int):
        if expected.get(bucket) != observed.get(bucket):
            failures.append(
                f"HLO bucket {bucket} signature {observed.get(bucket)} "
                f"does not equal planned {expected.get(bucket)}"
            )
    return failures


def _plans_require_replicas(diagnostics: dict[str, Any]) -> bool:
    plans = diagnostics.get("optimizer", {}).get("dist_muon", {}).get(
        "leaf_execution_plans", []
    )
    return any(
        role.get("replica_axes")
        for plan in plans
        for role in plan.get("roles", {}).values()
        if isinstance(role, Mapping)
    )


def _required_run(
    runs: Mapping[str, dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    if run_id not in runs:
        raise ValueError(f"profile analysis is missing run {run_id!r}")
    return runs[run_id]


def _steady_train_step(run: dict[str, Any]) -> float:
    value = run.get("steady", {}).get("medians", {}).get("train_step_sec")
    if not _finite_number(value) or value <= 0.0:
        raise ValueError(
            f"run {run.get('run_id')!r} has no positive finite steady train_step_sec"
        )
    return float(value)


def _read_source_commit(capture: Path) -> str | None:
    path = capture / "commit.txt"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_checked(
    path: Path,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"missing {label}: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"invalid {label}: {exc}"]
    if not isinstance(payload, dict):
        return {}, [f"{label} must be a JSON object"]
    return payload, []


def _load_jsonl_checked(
    path: Path,
    label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], [f"missing {label}: {path}"]
    rows = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                return [], [f"{label} line {line_number} is not a JSON object"]
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"invalid {label}: {exc}"]
    if not rows:
        return [], [f"{label} is empty"]
    return rows, []


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _gate_payload(failures: list[str], **details: Any) -> dict[str, Any]:
    return {
        "gate": not failures,
        "failures": failures,
        **details,
    }


if __name__ == "__main__":
    raise SystemExit(main())
