import json
from pathlib import Path
import sqlite3

import pytest

from utils.profile_run import (
    build_nsys_plan,
    format_summary,
    load_metrics,
    main,
    run_analysis,
    run_nsys,
    summarize_nsys_sqlite,
    summarize_metrics,
    write_summary,
)


def write_metrics(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_profile_metrics(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(
        run_dir / "metrics.jsonl",
        [
            {"step": 0, "time/step_sec": 9.0},
            {
                "step": 20,
                "time/step_sec": 0.20,
                "time/train_step_sec": 0.02,
                "time/metrics_sync_sec": 0.12,
                "time/tokens_per_sec": 1000.0,
                "time/train_tokens_per_sec": 10000.0,
            },
            {
                "step": 21,
                "time/step_sec": 0.22,
                "time/train_step_sec": 0.02,
                "time/metrics_sync_sec": 0.13,
                "time/tokens_per_sec": 900.0,
                "time/train_tokens_per_sec": 10000.0,
            },
            {
                "step": 22,
                "time/step_sec": 2.0,
                "time/sample_sec": 1.7,
                "time/tokens_per_sec": 100.0,
                "sample/path": "sample.txt",
            },
        ],
    )


def write_synthetic_nsys_sqlite(path):
    with sqlite3.connect(path) as con:
        con.execute("create table StringIds (id integer primary key, value text not null)")
        con.executemany(
            "insert into StringIds (id, value) values (?, ?)",
            [
                (1, "step"),
                (2, "metrics_sync"),
                (3, "train_step"),
                (4, "gemm_kernel"),
                (5, "cuStreamSynchronize"),
                (6, "cuMemcpyDtoHAsync_v2"),
                (7, "XlaCompileBackend"),
            ],
        )
        con.execute("create table NVTX_EVENTS (start integer not null, end integer, text text, textId integer)")
        con.executemany(
            "insert into NVTX_EVENTS (start, end, text, textId) values (?, ?, ?, ?)",
            [
                (0, 100_000_000, None, 1),
                (10_000_000, 80_000_000, None, 2),
                (80_000_000, 95_000_000, None, 3),
            ],
        )
        con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer not null, end integer not null, demangledName integer)")
        con.executemany(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL (start, end, demangledName) values (?, ?, ?)",
            [
                (0, 2_000_000, 4),
                (3_000_000, 7_000_000, 4),
            ],
        )
        con.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer not null, end integer not null, nameId integer)")
        con.executemany(
            "insert into CUPTI_ACTIVITY_KIND_RUNTIME (start, end, nameId) values (?, ?, ?)",
            [
                (0, 50_000_000, 5),
                (60_000_000, 65_000_000, 6),
            ],
        )
        con.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer not null, end integer not null, bytes integer)")
        con.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY (start, end, bytes) values (0, 1000000, 1024)")
        con.execute("create table CUPTI_ACTIVITY_KIND_MEMSET (start integer not null, end integer not null, bytes integer)")
        con.execute("insert into CUPTI_ACTIVITY_KIND_MEMSET (start, end, bytes) values (0, 500000, 512)")
        con.execute("create table DIAGNOSTIC_EVENT (message text)")
        con.execute("insert into DIAGNOSTIC_EVENT (message) values ('unit warning')")


def write_nsys_config(path, *, out_dir, name="unit_nsys", enabled=True, profiler="nsys"):
    path.write_text(
        f"""
[experiment]
name = "{name}"
out_dir = "{out_dir}"

[model]
vocab_size = 128
hidden_size = 32
intermediate_size = 64
n_layers = 1
n_heads = 4
n_kv_heads = 4
seq_len = 16
theta = 10000.0
eps = 1e-6
tied = true

[train]
seed = 0
batch_size = 4
seq_len = 16
steps = 4
lr = 0.001
decay = 0.1
log_every = 1
eval_every = 10
eval_steps = 1
checkpoint_every = 10
keep_last = 1

[data]
source = "tokens"
path = "data/unit"
tokenizer = "gpt2"

[profiling]
enabled = {str(enabled).lower()}
profiler = "{profiler}"
start_step = 1
steps = 2
output_dir = "profiles"
""".lstrip(),
        encoding="utf-8",
    )


def test_load_metrics_reads_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir / "metrics.jsonl", [{"step": 0, "time/step_sec": 0.1}])

    rows = load_metrics(run_dir)

    assert rows == [{"step": 0, "time/step_sec": 0.1}]


def test_summarize_metrics_groups_timing_rows():
    rows = [
        {"step": 0, "time/step_sec": 9.0},
        {"step": 20, "time/step_sec": 0.2, "time/train_step_sec": 0.1, "time/tokens_per_sec": 100.0},
        {"step": 21, "time/step_sec": 0.3, "time/train_step_sec": 0.2, "time/tokens_per_sec": 80.0, "val/loss": 1.2},
        {"step": 22, "time/step_sec": 1.0, "time/sample_sec": 0.7, "sample/path": "sample.txt"},
        {"step": 23, "time/step_sec": 0.4, "time/checkpoint_sec": 0.2},
    ]

    summary = summarize_metrics(rows, warmup_steps=20)

    assert summary["rows_total"] == 5
    assert summary["rows_used"] == 4
    assert summary["step_min"] == 20
    assert summary["step_max"] == 23
    assert summary["groups"]["all"]["fields"]["time/step_sec"]["count"] == 4
    assert summary["groups"]["all"]["fields"]["time/step_sec"]["mean"] == pytest.approx(0.475)
    assert summary["groups"]["normal"]["rows"] == 1
    assert summary["groups"]["eval"]["rows"] == 1
    assert summary["groups"]["sample"]["rows"] == 1
    assert summary["groups"]["checkpoint"]["rows"] == 1


def test_summarize_metrics_rejects_empty_warm_rows():
    with pytest.raises(ValueError, match="No metrics rows remain"):
        summarize_metrics([{"step": 1, "time/step_sec": 0.1}], warmup_steps=20)


def test_format_and_write_summary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = summarize_metrics(
        [
            {"step": 20, "time/step_sec": 0.2, "time/train_tokens_per_sec": 1000.0},
            {"step": 21, "time/step_sec": 0.4, "time/train_tokens_per_sec": 2000.0},
        ],
        warmup_steps=20,
    )

    markdown = format_summary(run_dir, summary)
    json_path, md_path = write_summary(run_dir, summary, markdown)

    assert "Timing Summary" in markdown
    assert "time/train_tokens_per_sec" in markdown
    assert json_path == run_dir / "profiles" / "timing_summary.json"
    assert md_path == run_dir / "profiles" / "timing_summary.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["rows_used"] == 2
    assert md_path.read_text(encoding="utf-8") == markdown


def test_main_supports_summary_subcommand_and_legacy_alias(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir / "metrics.jsonl", [{"step": 0, "time/step_sec": 0.1}])

    main(["summary", str(run_dir), "--warmup-steps", "0"])
    first = capsys.readouterr().out
    main([str(run_dir), "--warmup-steps", "0"])
    second = capsys.readouterr().out

    assert "Timing Summary" in first
    assert "Timing Summary" in second


def test_build_nsys_plan_uses_temp_output_outside_runs(tmp_path):
    config_path = tmp_path / "config.toml"
    out_dir = tmp_path / "runs"
    nsys_output_dir = tmp_path / "profiles"
    write_nsys_config(config_path, out_dir=out_dir)

    plan = build_nsys_plan(config_path, nsys_output_dir=nsys_output_dir, extra_nsys_args=["--sample=none"])

    assert plan.run_dir == out_dir / "unit_nsys"
    assert plan.temp_output_prefix == nsys_output_dir / "unit_nsys"
    assert plan.temp_report_path == nsys_output_dir / "unit_nsys.nsys-rep"
    assert plan.final_report_path == out_dir / "unit_nsys" / "profiles" / "nsys.nsys-rep"
    assert plan.command[:6] == [
        "nsys",
        "profile",
        "--force-overwrite",
        "true",
        "-t",
        "cuda,nvtx,osrt,cudnn,cublas",
    ]
    assert "--sample=none" in plan.command
    output_index = plan.command.index("-o")
    assert plan.command[output_index : output_index + 6] == [
        "-o",
        str(nsys_output_dir / "unit_nsys"),
        "uv",
        "run",
        "pretrain",
        str(config_path),
    ]


def test_build_nsys_plan_rejects_non_nsys_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_nsys_config(config_path, out_dir=tmp_path / "runs", profiler="jax")

    with pytest.raises(ValueError, match="profiler = \"nsys\""):
        build_nsys_plan(config_path)


def test_run_nsys_refuses_existing_run_dir(tmp_path):
    config_path = tmp_path / "config.toml"
    out_dir = tmp_path / "runs"
    write_nsys_config(config_path, out_dir=out_dir)
    (out_dir / "unit_nsys").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="--force-run-dir"):
        run_nsys(config_path, nsys_output_dir=tmp_path / "profiles")


def test_run_nsys_force_removes_run_dir_and_copies_report(tmp_path):
    config_path = tmp_path / "config.toml"
    out_dir = tmp_path / "runs"
    write_nsys_config(config_path, out_dir=out_dir)
    run_dir = out_dir / "unit_nsys"
    run_dir.mkdir(parents=True)
    marker = run_dir / "old.txt"
    marker.write_text("old", encoding="utf-8")
    commands = []

    def fake_runner(command, check):
        commands.append((command, check))
        output_prefix = command[command.index("-o") + 1]
        report_path = output_prefix + ".nsys-rep"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("report", encoding="utf-8")

    plan = run_nsys(
        config_path,
        force_run_dir=True,
        nsys_output_dir=tmp_path / "profiles",
        runner=fake_runner,
    )

    assert commands == [(plan.command, True)]
    assert not marker.exists()
    assert plan.final_report_path.read_text(encoding="utf-8") == "report"


def test_run_nsys_dry_run_does_not_execute_or_remove_existing_run_dir(tmp_path):
    config_path = tmp_path / "config.toml"
    out_dir = tmp_path / "runs"
    write_nsys_config(config_path, out_dir=out_dir)
    run_dir = out_dir / "unit_nsys"
    run_dir.mkdir(parents=True)
    marker = run_dir / "old.txt"
    marker.write_text("old", encoding="utf-8")

    def fake_runner(command, check):
        raise AssertionError("dry-run should not execute nsys")

    plan = run_nsys(
        config_path,
        dry_run=True,
        nsys_output_dir=tmp_path / "profiles",
        runner=fake_runner,
    )

    assert plan.run_dir == run_dir
    assert marker.read_text(encoding="utf-8") == "old"


def test_main_nsys_dry_run_prints_plan(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    write_nsys_config(config_path, out_dir=tmp_path / "runs")

    main(["nsys", str(config_path), "--dry-run", "--nsys-output-dir", str(tmp_path / "profiles")])
    output = capsys.readouterr().out

    assert "NSys Profile" in output
    assert "dry run: command not executed" in output
    assert "scp" not in output


def test_run_analysis_writes_metrics_only_report(tmp_path):
    run_dir = tmp_path / "run"
    write_profile_metrics(run_dir)

    result = run_analysis(run_dir, warmup_steps=20)

    assert result.json_path == run_dir / "profiles" / "profile_report.json"
    assert result.markdown_path == run_dir / "profiles" / "profile_report.md"
    assert result.report["version"] == 1
    assert result.report["nsys_summary"] is None
    assert result.report["warnings"] == ["No NSys report found; generated metrics-only analysis."]
    assert "Host/device synchronization dominates logged steps" in result.markdown
    assert "JAX Async Interpretation" in result.markdown
    assert json.loads(result.json_path.read_text(encoding="utf-8"))["findings"]


def test_summarize_nsys_sqlite_extracts_nvtx_and_cuda_rows(tmp_path):
    sqlite_path = tmp_path / "nsys.sqlite"
    write_synthetic_nsys_sqlite(sqlite_path)

    summary = summarize_nsys_sqlite(sqlite_path)

    assert summary["nvtx_phases"]["step"]["count"] == 1
    assert summary["nvtx_phases"]["metrics_sync"]["mean"] == pytest.approx(70.0)
    assert summary["cuda"]["top_kernels"][0]["name"] == "gemm_kernel"
    assert summary["cuda"]["top_kernels"][0]["total_ms"] == pytest.approx(6.0)
    assert summary["cuda"]["top_runtime_apis"][0]["name"] == "cuStreamSynchronize"
    assert summary["cuda"]["memory"]["memcpy"]["total_bytes"] == 1024
    assert summary["diagnostics"][0]["message"] == "unit warning"
    assert summary["xla_compile_events"] == 1


def test_run_analysis_includes_synthetic_nsys_report(tmp_path):
    run_dir = tmp_path / "run"
    write_profile_metrics(run_dir)
    nsys_report = run_dir / "profiles" / "nsys.nsys-rep"
    nsys_report.parent.mkdir(parents=True, exist_ok=True)
    nsys_report.write_text("placeholder", encoding="utf-8")

    def fake_export_runner(command, check, **kwargs):
        assert command[:6] == ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true"]
        sqlite_path = Path(command[command.index("--output") + 1])
        write_synthetic_nsys_sqlite(sqlite_path)

    result = run_analysis(run_dir, warmup_steps=20, runner=fake_export_runner, nsys_executable="nsys")

    assert result.report["inputs"]["nsys_report"] == str(nsys_report)
    assert result.report["inputs"]["nsys_sqlite"] == str(run_dir / "profiles" / "nsys.sqlite")
    assert result.report["nsys_summary"]["cuda"]["top_runtime_apis"][0]["name"] == "cuStreamSynchronize"
    assert "CUDA runtime synchronization is prominent" in result.markdown


def test_main_analyze_subcommand_writes_report(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_profile_metrics(run_dir)

    main(["analyze", str(run_dir), "--warmup-steps", "20"])
    output = capsys.readouterr().out

    assert "Profile Analysis" in output
    assert (run_dir / "profiles" / "profile_report.json").exists()
    assert (run_dir / "profiles" / "profile_report.md").exists()


def test_main_nsys_analyze_runs_report_after_capture(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    out_dir = tmp_path / "runs"
    write_nsys_config(config_path, out_dir=out_dir)
    temp_profiles = tmp_path / "profiles"

    def fake_profile_runner(command, check):
        run_dir = out_dir / "unit_nsys"
        write_profile_metrics(run_dir)
        output_prefix = Path(command[command.index("-o") + 1])
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_prefix.with_suffix(".nsys-rep").write_text("placeholder", encoding="utf-8")

    def fake_export_runner(command, check, **kwargs):
        sqlite_path = Path(command[command.index("--output") + 1])
        write_synthetic_nsys_sqlite(sqlite_path)

    main(
        [
            "nsys",
            str(config_path),
            "--nsys-output-dir",
            str(temp_profiles),
            "--analyze",
            "--analyze-warmup-steps",
            "20",
        ],
        runner=fake_profile_runner,
        analysis_runner=fake_export_runner,
        nsys_executable="nsys",
    )
    output = capsys.readouterr().out

    report_path = out_dir / "unit_nsys" / "profiles" / "profile_report.json"
    assert "NSys Profile" in output
    assert "Profile Analysis" in output
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["nsys_summary"] is not None
