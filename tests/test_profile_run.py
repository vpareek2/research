import json

import pytest

from utils.profile_run import format_summary, load_metrics, summarize_metrics, write_summary


def write_metrics(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


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
