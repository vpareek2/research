import gzip
import json
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime.profile_analysis import (
    analyze_profile_root,
    discover_profile_runs,
    format_profile_analysis,
    profile_analysis_to_json,
    summarize_hlo_text,
    summarize_perfetto_trace,
)


def test_analyze_profile_root_selects_canonical_window_and_pairs_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "dense_ddp_adamw", optimizer="adamw", layout="ddp", scale=1.0)
    _write_run(tmp_path, "dense_tp_adamw", optimizer="adamw", layout="tp", scale=2.0)
    _write_run(tmp_path, "dense_tp_muon", optimizer="muon", layout="tp", scale=3.0, with_hlo=True)

    payload = analyze_profile_root(tmp_path)

    assert payload["run_count"] == 3
    assert [path.name for path in discover_profile_runs(tmp_path)] == [
        "dense_ddp_adamw",
        "dense_tp_adamw",
        "dense_tp_muon",
    ]
    tp_muon = next(run for run in payload["runs"] if run["run_id"] == "dense_tp_muon")
    assert tp_muon["steady"]["start_step"] == 6
    assert tp_muon["steady"]["end_step"] == 7
    assert tp_muon["steady"]["excluded_steps"] == [8]
    assert tp_muon["steady"]["medians"]["train_step_sec"] == pytest.approx(0.0195)
    assert tp_muon["trace_window"]["tax"]["train_step_sec"] == pytest.approx(1.35 / 1.95 - 1.0)
    assert tp_muon["trace"]["categories"]["nccl_all_gather"]["count"] == 1
    assert tp_muon["hlo"]["instruction_counts"] == {"all-gather-start": 1, "dot": 1}
    assert tp_muon["hlo"]["estimated_result_bytes"] == {"all-gather-start": 32, "dot": 64}
    assert {(item["kind"], item["candidate"], item["baseline"]) for item in payload["comparisons"]} == {
        ("layout", "dense_tp_adamw", "dense_ddp_adamw"),
        ("optimizer", "dense_tp_muon", "dense_tp_adamw"),
    }
    assert json.loads(profile_analysis_to_json(payload))["schema_version"] == 1
    assert "dense_tp_muon" in format_profile_analysis(payload)


def test_analyze_profile_root_rejects_incomplete_or_nonfinite_runs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "bad", optimizer="adamw", layout="ddp", scale=1.0)
    final_path = run_dir / "summaries" / "final.json"
    final = json.loads(final_path.read_text())
    final["status"] = "failed"
    final_path.write_text(json.dumps(final))

    with pytest.raises(ContractError, match="not completed"):
        analyze_profile_root(tmp_path)

    final["status"] = "completed"
    final["final_optimizer_nonfinite_group_count"] = 1
    final_path.write_text(json.dumps(final))
    with pytest.raises(ContractError, match="nonfinite"):
        analyze_profile_root(tmp_path)


def test_analyze_profile_root_reports_missing_required_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "partial"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "metrics" / "train.jsonl").write_text('{"step":1}\n')

    with pytest.raises(ContractError, match="final summary"):
        analyze_profile_root(tmp_path)


def test_summarize_hlo_text_counts_definitions_not_references() -> None:
    text = """
  %all-gather-start.1 = f32[4,2] all-gather-start(%p0), channel_id=1
  %all-gather-done.1 = f32[4,2] all-gather-done(%all-gather-start.1)
  %all-reduce.1 = bf16[8,2] all-reduce(%p1), channel_id=2
  ROOT %dot.1 = f32[4,4] dot(%all-gather-done.1, %p1)
  %scatter.1 = f32[4,4] scatter(%dot.1, %indices, %updates)
"""

    summary = summarize_hlo_text(text)

    assert summary["instruction_counts"] == {
        "all-gather-start": 1,
        "all-reduce": 1,
        "dot": 1,
        "scatter": 1,
    }
    assert summary["estimated_result_bytes"] == {
        "all-gather-start": 32,
        "all-reduce": 32,
        "dot": 64,
        "scatter": 64,
    }
    assert summary["instruction_result_shapes"]["all-reduce"] == [
        {"dtype": "bf16", "shape": [8, 2], "estimated_bytes": 32}
    ]


def test_summarize_perfetto_trace_requires_gpu_zero_metadata(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json.gz"
    with gzip.open(trace_path, "wt") as handle:
        json.dump({"traceEvents": []}, handle)

    with pytest.raises(ContractError, match="GPU:0"):
        summarize_perfetto_trace(trace_path)


def _write_run(
    root: Path,
    run_id: str,
    *,
    optimizer: str,
    layout: str,
    scale: float,
    with_hlo: bool = False,
) -> Path:
    run_dir = root / "runs" / run_id
    for child in ("metrics", "summaries", "diagnostics", "config"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "summaries" / "final.json",
        {
            "run_id": run_id,
            "status": "completed",
            "final_optimizer_nonfinite_group_count": 0,
        },
    )
    tensor_parallel = "tp" in layout
    expert_parallel = "ep" in layout
    mode = "zero2" if layout.startswith("zero2") else "fsdp" if layout.startswith("fsdp") else "ddp"
    _write_json(
        run_dir / "diagnostics" / "runtime.json",
        {
            "model": {"name": "decoder", "parameters": 100},
            "jax": {"backend": "gpu"},
            "performance": {"device_kind": "test-gpu", "device_count": 4},
            "parallelism": {"mesh": {"axis_names": ["data", "tp"], "axis_sizes": [1, 4]}},
            "profiling": {"enabled": True, "trace_start_step": 4, "trace_end_step": 5},
        },
    )
    _write_json(
        run_dir / "config" / "resolved.json",
        {
            "model": {
                "name": "decoder",
                "variant": run_id,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_layers": 1,
            },
            "optimizer": {"name": optimizer},
            "parallelism": {
                "mode": mode,
                "tensor_parallel": tensor_parallel,
                "expert_parallel": expert_parallel,
            },
        },
    )
    rows = []
    for step in range(1, 9):
        value = scale * (step / 1000.0)
        rows.append(
            {
                "step": step,
                "train_step_sec": value,
                "step_sec": value * 2,
                "train_tokens_per_sec": 1000.0 / value,
                "data_sec": value / 2,
                "placement_sec": value / 4,
            }
        )
    _write_jsonl(run_dir / "metrics" / "train.jsonl", rows)
    _write_jsonl(
        run_dir / "events.jsonl",
        [
            {"type": "run_started", "step": 0},
            {"type": "eval_started", "step": 8},
            {"type": "checkpoint_saved", "step": 8},
        ],
    )
    trace_dir = run_dir / "profiles" / "plugins" / "profile" / "test"
    trace_dir.mkdir(parents=True)
    with gzip.open(trace_dir / "perfetto_trace.json.gz", "wt") as handle:
        json.dump(
            {
                "traceEvents": [
                    {
                        "ph": "M",
                        "pid": 1,
                        "name": "process_name",
                        "args": {"name": "/device:GPU:0"},
                    },
                    {"ph": "X", "pid": 1, "name": "ncclDevKernel_AllGather_RING", "dur": 100.0},
                    {"ph": "X", "pid": 1, "name": "gemm_fusion_dot", "dur": 50.0},
                    {"ph": "X", "pid": 2, "name": "ignored", "dur": 999.0},
                ]
            },
            handle,
        )
    if with_hlo:
        hlo_dir = root / "cloud_results" / "test_hlo" / run_id
        hlo_dir.mkdir(parents=True)
        (hlo_dir / "module_1.jit__compiled_impl.sm_gpu_after_optimizations.txt").write_text(
            "%all-gather-start.1 = f32[4,2] all-gather-start(%p0)\n"
            "ROOT %dot.1 = f32[4,4] dot(%p1, %p2)\n"
        )
    return run_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
