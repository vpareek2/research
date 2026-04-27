import json
from pathlib import Path

import pytest

from utils.profile_run import (
    build_nsys_plan,
    format_summary,
    load_metrics,
    main,
    run_nsys,
    summarize_metrics,
    write_summary,
)


def write_metrics(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


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
