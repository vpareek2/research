import importlib.util
import json
import math
from pathlib import Path

import pytest

from jaxtitan.config import load_config, run_spec_to_dict


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "jaxtitan"
    / "analyze_dist_muon_shape_policy_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_dist_muon_shape_policy_results",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _analysis(multiplier: float = 0.95) -> dict[str, object]:
    return {
        "runs": [
            {
                "run_id": run_id,
                "status": "completed",
                "steady": {
                    "start_step": 16,
                    "end_step": 63,
                    "row_count": 48,
                    "medians": {"train_step_sec": current_sec * multiplier},
                },
            }
            for _layout, run_id, current_sec, _duplicated_sec in MODULE.RUNS
        ]
    }


def _plans(*, replicas: bool) -> list[dict[str, object]]:
    replica_axes = ["fsdp"] if replicas else []

    def plan(execution: str, bucket_id: int, path: str) -> dict[str, object]:
        return {
            "path": [path],
            "bucket_id": bucket_id,
            "execution": execution,
            "selection": {
                "policy_version": MODULE.POLICY_VERSION,
                "eligible_executions": ["duplicated", execution],
                "selected_execution": execution,
                "selection_reason": "test",
                "modeled_costs": {"bytes": 1},
            },
            "roles": {
                role: {
                    "partition_spec": "P(None, 'tp')",
                    "replica_axes": replica_axes,
                }
                for role in ("parameter", "gradient", "momentum", "update")
            },
        }

    return [
        plan("distributed_direct", 0, "direct"),
        plan("distributed_large_gram", 1, "large"),
        plan("distributed_exchange", 2, "exchange_a"),
        plan("distributed_exchange", 2, "exchange_b"),
    ]


def _diagnostics(
    run_specs=MODULE.RUNS,
) -> dict[str, dict[str, object]]:
    return {
        run_id: {
            "optimizer": {
                "dist_muon": {
                    "exact": False,
                    "shape_topology_policy": {"version": MODULE.POLICY_VERSION},
                    "leaf_execution_plans": _plans(replicas=layout != "dense_tp"),
                }
            }
        }
        for layout, run_id, _current, _duplicated in run_specs
    }


def _optimizer_group() -> dict[str, object]:
    return {
        "group": "attention_q:dist_muon",
        "grad_norm": 1.0,
        "update_norm": 0.1,
        "param_norm": 2.0,
        "grad_param_ratio": 0.5,
        "update_param_ratio": 0.05,
        "grad_norm_finite": True,
        "update_norm_finite": True,
        "param_norm_finite": True,
        "grad_param_ratio_finite": True,
        "update_param_ratio_finite": True,
    }


def _train_row() -> dict[str, object]:
    return {
        "step": 65,
        "loss": 4.0,
        "lm_loss": 4.0,
        "grad_norm": 1.0,
        "param_norm": 2.0,
        "update_norm": 0.1,
        "train_step_sec": 0.1,
        "step_sec": 0.11,
        "optimizer_nonfinite_group_count": 0,
        "optimizer_nonfinite_groups": [],
        "optimizer_groups": [_optimizer_group()],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_capture(
    root: Path,
    diagnostics: dict[str, dict[str, object]],
    *,
    run_specs=MODULE.RUNS,
    train_steps: tuple[int, ...] = (65,),
    checkpoint_step: int = 65,
) -> None:
    checkpoint_path = f"checkpoints/{checkpoint_step:06d}"
    for layout, run_id, _current, _duplicated in run_specs:
        run_root = root / "runs" / run_id
        _write_json(
            run_root / "summaries" / "final.json",
            {
                "run_id": run_id,
                "status": "completed",
                "final_optimizer_nonfinite_group_count": 0,
                "final_optimizer_nonfinite_groups": [],
                "final_optimizer_groups": [_optimizer_group()],
            },
        )
        metrics = run_root / "metrics" / "train.jsonl"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text(
            "".join(
                json.dumps({**_train_row(), "step": step}) + "\n"
                for step in train_steps
            ),
            encoding="utf-8",
        )
        _write_json(
            run_root / "checkpoints" / "index.json",
            {
                "schema_version": 1,
                "latest_step": checkpoint_step,
                "latest_checkpoint_path": checkpoint_path,
                "records": [
                    {
                        "step": checkpoint_step,
                        "retained": True,
                        "checkpoint_path": checkpoint_path,
                    },
                ],
            },
        )
        _write_json(
            root / f"eval_{run_id}.json",
            {
                "run_id": run_id,
                "status": "completed",
                "checkpoint": {
                    "step": checkpoint_step,
                    "path": checkpoint_path,
                    "runtime_fingerprint": "fingerprint",
                },
                "eval": {"loss": 4.1, "token_count": 4096},
            },
        )
        _write_json(
            root / f"sample_{run_id}.json",
            {
                "run_id": run_id,
                "status": "completed",
                "checkpoint": {
                    "step": checkpoint_step,
                    "path": checkpoint_path,
                    "runtime_fingerprint": "fingerprint",
                    "selector": "latest",
                },
                "prompt_ids": [1, 2],
                "generated_ids": [3, 4],
                "full_ids": [1, 2, 3, 4],
                "logprobs": [0.0, -0.1],
                "sampling": {"max_new_tokens": 2},
            },
        )
        _write_json(
            root / f"replica_audit_{run_id}.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "checkpoint": {
                    "step": checkpoint_step,
                    "path": checkpoint_path,
                    "runtime_fingerprint": "fingerprint",
                },
                "sections": {
                    section: {
                        "array_count": 2,
                        "replicated_array_count": (
                            0 if layout == "dense_tp" else 1
                        ),
                        "finite": True,
                        "nonfinite_paths": [],
                        "replicas_equal": True,
                        "max_replica_abs_diff": 0.0,
                        "replica_disagreement_paths": [],
                        "gate": True,
                    }
                    for section in ("model", "optimizer")
                },
                "array_count": 4,
                "finite": True,
                "replicated_array_count": 0 if layout == "dense_tp" else 2,
                "max_replica_abs_diff": 0.0,
                "overall_gate": True,
            },
        )
        hlo_root = root / "hlo" / run_id
        hlo_root.mkdir(parents=True, exist_ok=True)
        hlo = _hlo_text()
        (hlo_root / "module_1.jit__compiled_impl.before_optimizations.txt").write_text(
            hlo,
            encoding="utf-8",
        )
        (hlo_root / "module_2.jit__compiled_impl.before_optimizations.txt").write_text(
            hlo,
            encoding="utf-8",
        )
        _write_json(run_root / "diagnostics" / "runtime.json", diagnostics[run_id])


def _hlo_text() -> str:
    lines = []
    for bucket in (0, 1, 2):
        lines.append(_collective("all-reduce", bucket, "norm"))
        for _ in range(5):
            lines.append(
                _collective(
                    "all-reduce",
                    bucket,
                    "gram",
                    right=bucket == 1,
                )
            )
    for _ in range(2):
        lines.append(_collective("all-to-all", 2, "exchange_forward"))
        lines.append(_collective("all-to-all", 2, "exchange_reverse"))
    return "\n".join(lines) + "\n"


def _collective(
    operation: str,
    bucket: int,
    muon_op: str,
    *,
    right: bool = False,
) -> str:
    attributes = f'muon_bucket="{bucket}",muon_op="{muon_op}"'
    if right:
        attributes += ',muon_gram_side="right"'
    return f"  %x = f32[] {operation}(%p), frontend_attributes={{{attributes}}}"


def _artifact_results(
    capture: Path,
    diagnostics: dict[str, dict[str, object]],
    *,
    run_specs=MODULE.RUNS,
    minimum_train_rows: int | None = None,
) -> dict[str, dict[str, object]]:
    return {
        run_id: MODULE.analyze_run_artifacts(
            capture,
            run_id,
            diagnostics[run_id],
            minimum_train_rows=minimum_train_rows,
        )
        for _layout, run_id, _current, _duplicated in run_specs
    }


def test_shape_policy_analysis_accepts_complete_capture(tmp_path: Path) -> None:
    diagnostics = _diagnostics()
    _write_capture(tmp_path, diagnostics)

    payload = MODULE.compare_shape_policy_results(
        _analysis(),
        diagnostics=diagnostics,
        artifact_gate=_artifact_results(tmp_path, diagnostics),
    )

    assert payload["schema_version"] == 2
    assert payload["overall_gate"] is True
    assert payload["geomean_speedup_vs_current"] == pytest.approx(1.0 / 0.95)
    assert all(row["gate"] for row in payload["layouts"])


def test_shape_policy_smoke_accepts_eight_step_four_layout_capture(
    tmp_path: Path,
) -> None:
    run_specs = MODULE._phase_run_specs("smoke")
    diagnostics = _diagnostics(run_specs)
    _write_capture(
        tmp_path,
        diagnostics,
        run_specs=run_specs,
        train_steps=tuple(range(1, 9)),
        checkpoint_step=8,
    )

    payload = MODULE.compare_shape_policy_results(
        {"runs": []},
        diagnostics=diagnostics,
        artifact_gate=_artifact_results(
            tmp_path,
            diagnostics,
            run_specs=run_specs,
            minimum_train_rows=8,
        ),
        phase="smoke",
        source_commit="a" * 40,
    )

    assert payload["phase"] == "smoke"
    assert payload["source_commit"] == "a" * 40
    assert payload["commit_gate"] is True
    assert payload["geomean_speedup_vs_current"] is None
    assert payload["geomean_gate"] is None
    assert payload["overall_gate"] is True
    assert all(row["performance_gate"] is None for row in payload["layouts"])


def test_shape_policy_smoke_rejects_short_training_capture(tmp_path: Path) -> None:
    run_specs = MODULE._phase_run_specs("smoke")
    diagnostics = _diagnostics(run_specs)
    _write_capture(
        tmp_path,
        diagnostics,
        run_specs=run_specs,
        train_steps=tuple(range(1, 8)),
        checkpoint_step=7,
    )

    artifacts = _artifact_results(
        tmp_path,
        diagnostics,
        run_specs=run_specs,
        minimum_train_rows=8,
    )
    payload = MODULE.compare_shape_policy_results(
        {"runs": []},
        diagnostics=diagnostics,
        artifact_gate=artifacts,
        phase="smoke",
        source_commit="a" * 40,
    )

    assert payload["overall_gate"] is False
    assert all(row["artifact_gate"] is False for row in payload["layouts"])
    assert all(
        "require at least 8" in " ".join(row["failures"])
        for row in payload["layouts"]
    )


def test_shape_policy_smoke_rejects_missing_source_commit(tmp_path: Path) -> None:
    run_specs = MODULE._phase_run_specs("smoke")
    diagnostics = _diagnostics(run_specs)
    _write_capture(
        tmp_path,
        diagnostics,
        run_specs=run_specs,
        train_steps=tuple(range(1, 9)),
        checkpoint_step=8,
    )

    payload = MODULE.compare_shape_policy_results(
        {"runs": []},
        diagnostics=diagnostics,
        artifact_gate=_artifact_results(
            tmp_path,
            diagnostics,
            run_specs=run_specs,
            minimum_train_rows=8,
        ),
        phase="smoke",
    )

    assert payload["commit_gate"] is False
    assert payload["overall_gate"] is False


@pytest.mark.parametrize("corruption", ("missing_replica_audit", "missing_hlo"))
def test_shape_policy_smoke_rejects_artifact_or_hlo_failure(
    tmp_path: Path,
    corruption: str,
) -> None:
    run_specs = MODULE._phase_run_specs("smoke")
    diagnostics = _diagnostics(run_specs)
    _write_capture(
        tmp_path,
        diagnostics,
        run_specs=run_specs,
        train_steps=tuple(range(1, 9)),
        checkpoint_step=8,
    )
    _layout, run_id, _current, _duplicated = run_specs[1]
    if corruption == "missing_replica_audit":
        (tmp_path / f"replica_audit_{run_id}.json").unlink()
    else:
        for path in (tmp_path / "hlo" / run_id).iterdir():
            path.unlink()

    payload = MODULE.compare_shape_policy_results(
        {"runs": []},
        diagnostics=diagnostics,
        artifact_gate=_artifact_results(
            tmp_path,
            diagnostics,
            run_specs=run_specs,
            minimum_train_rows=8,
        ),
        phase="smoke",
        source_commit="a" * 40,
    )

    assert payload["overall_gate"] is False
    failed = next(row for row in payload["layouts"] if row["run_id"] == run_id)
    assert failed["artifact_gate"] is False
    assert failed["failures"]


def test_shape_policy_analysis_rejects_one_current_regression(tmp_path: Path) -> None:
    diagnostics = _diagnostics()
    _write_capture(tmp_path, diagnostics)
    analysis = _analysis()
    run_id = MODULE.RUNS[-1][1]
    current_sec = MODULE.RUNS[-1][2]
    next(run for run in analysis["runs"] if run["run_id"] == run_id)["steady"]["medians"][
        "train_step_sec"
    ] = current_sec * 1.02

    payload = MODULE.compare_shape_policy_results(
        analysis,
        diagnostics=diagnostics,
        artifact_gate=_artifact_results(tmp_path, diagnostics),
    )

    assert payload["overall_gate"] is False
    assert payload["layouts"][-1]["performance_gate"] is False


def test_shape_policy_analysis_rejects_incomplete_policy_metadata(tmp_path: Path) -> None:
    diagnostics = _diagnostics()
    _write_capture(tmp_path, diagnostics)
    run_id = MODULE.RUNS[0][1]
    diagnostics[run_id]["optimizer"]["dist_muon"]["leaf_execution_plans"][0][
        "selection"
    ].pop("modeled_costs")

    payload = MODULE.compare_shape_policy_results(
        _analysis(),
        diagnostics=diagnostics,
        artifact_gate=_artifact_results(tmp_path, diagnostics),
    )

    assert payload["overall_gate"] is False
    assert payload["layouts"][0]["policy_gate"] is False


@pytest.mark.parametrize(
    ("corruption", "expected_check"),
    [
        ("missing_final", "final"),
        ("missing_eval", "eval"),
        ("nonfinite_metric", "metrics"),
        ("nonfinite_optimizer_group", "metrics"),
        ("malformed_sample", "sample"),
        ("checkpoint_mismatch", "checkpoint_identity"),
        ("missing_replica_audit", "replica_audit"),
        ("nonfinite_replica_audit", "replica_audit"),
        ("replica_disagreement", "replica_audit"),
        ("missing_replica_evidence", "replica_audit"),
        ("wrong_hlo_gram_side", "hlo"),
        ("missing_hlo_exchange", "hlo"),
    ],
)
def test_shape_policy_artifact_gates_reject_corruption(
    tmp_path: Path,
    corruption: str,
    expected_check: str,
) -> None:
    diagnostics = _diagnostics()
    _write_capture(tmp_path, diagnostics)
    layout, run_id, _current, _duplicated = MODULE.RUNS[1]
    del layout

    if corruption == "missing_final":
        (tmp_path / "runs" / run_id / "summaries" / "final.json").unlink()
    elif corruption == "missing_eval":
        (tmp_path / f"eval_{run_id}.json").unlink()
    elif corruption == "nonfinite_metric":
        path = tmp_path / "runs" / run_id / "metrics" / "train.jsonl"
        row = _train_row()
        row["grad_norm"] = math.inf
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif corruption == "nonfinite_optimizer_group":
        path = tmp_path / "runs" / run_id / "metrics" / "train.jsonl"
        row = _train_row()
        row["optimizer_groups"][0]["update_norm"] = math.nan
        row["optimizer_groups"][0]["update_norm_finite"] = False
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif corruption == "malformed_sample":
        (tmp_path / f"sample_{run_id}.json").write_text("{", encoding="utf-8")
    elif corruption == "checkpoint_mismatch":
        path = tmp_path / f"sample_{run_id}.json"
        sample = json.loads(path.read_text(encoding="utf-8"))
        sample["checkpoint"]["step"] = 64
        _write_json(path, sample)
    elif corruption == "missing_replica_audit":
        (tmp_path / f"replica_audit_{run_id}.json").unlink()
    elif corruption == "nonfinite_replica_audit":
        path = tmp_path / f"replica_audit_{run_id}.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["overall_gate"] = False
        audit["finite"] = False
        audit["sections"]["optimizer"]["gate"] = False
        audit["sections"]["optimizer"]["finite"] = False
        audit["sections"]["optimizer"]["nonfinite_paths"] = [".momentum"]
        _write_json(path, audit)
    elif corruption == "replica_disagreement":
        path = tmp_path / f"replica_audit_{run_id}.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["overall_gate"] = False
        audit["max_replica_abs_diff"] = 0.25
        _write_json(path, audit)
    elif corruption == "missing_replica_evidence":
        path = tmp_path / f"replica_audit_{run_id}.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["replicated_array_count"] = 0
        _write_json(path, audit)
    elif corruption == "wrong_hlo_gram_side":
        hlo = tmp_path / "hlo" / run_id
        for path in hlo.iterdir():
            path.write_text(_hlo_text().replace('muon_gram_side="right"', ""), encoding="utf-8")
    elif corruption == "missing_hlo_exchange":
        hlo = tmp_path / "hlo" / run_id
        for path in hlo.iterdir():
            lines = [
                line
                for line in _hlo_text().splitlines()
                if 'muon_op="exchange_reverse"' not in line
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise AssertionError(corruption)

    artifacts = MODULE.analyze_run_artifacts(tmp_path, run_id, diagnostics[run_id])

    assert artifacts["gate"] is False
    assert artifacts["checks"][expected_check]["gate"] is False
    assert artifacts["failures"]


def test_shape_policy_hlo_rejects_inconsistent_duplicate_modules(tmp_path: Path) -> None:
    diagnostics = _diagnostics()
    _write_capture(tmp_path, diagnostics)
    run_id = MODULE.RUNS[0][1]
    path = (
        tmp_path
        / "hlo"
        / run_id
        / "module_2.jit__compiled_impl.before_optimizations.txt"
    )
    path.write_text(_hlo_text().replace('muon_op="norm"', 'muon_op="other"', 1), encoding="utf-8")

    result = MODULE.analyze_run_artifacts(tmp_path, run_id, diagnostics[run_id])

    assert result["checks"]["hlo"]["gate"] is False
    assert "signatures disagree" in " ".join(result["checks"]["hlo"]["failures"])


@pytest.mark.parametrize(
    ("baseline_name", "candidate_name"),
    [
        (
            "cloud_4gpu_profile64_dense_tp_muon_distributed.toml",
            "cloud_4gpu_profile64_dense_tp_muon_shape_policy.toml",
        ),
        (
            "cloud_4gpu_profile64_dense_fsdp_tp_muon_distributed.toml",
            "cloud_4gpu_profile64_dense_fsdp_tp_muon_shape_policy.toml",
        ),
        (
            "cloud_4gpu_profile64_dense_zero2_tp_muon_distributed.toml",
            "cloud_4gpu_profile64_dense_zero2_tp_muon_shape_policy.toml",
        ),
        (
            "cloud_4gpu_profile64_trinity_moe_tp_ep_muon_distributed.toml",
            "cloud_4gpu_profile64_trinity_moe_tp_ep_muon_shape_policy.toml",
        ),
    ],
)
def test_shape_policy_configs_differ_only_by_run_id(
    baseline_name: str,
    candidate_name: str,
) -> None:
    root = Path(__file__).parents[2] / "configs" / "jaxtitan"
    baseline = run_spec_to_dict(load_config(root / baseline_name))
    candidate = run_spec_to_dict(load_config(root / candidate_name))

    candidate["run_id"] = baseline["run_id"]

    assert candidate == baseline
