import json
import sys

import pytest

from utils import run_registry, run_summary


def write_config(run_dir, *, steps=4):
    (run_dir / "config.toml").write_text(
        f"""
[experiment]
name = "unit"
out_dir = "{run_dir.parent}"

[model]
vocab_size = 128
hidden_size = 32
intermediate_size = 64
n_layers = 1
n_heads = 4
n_kv_heads = 1
seq_len = 8
theta = 10000.0
eps = 0.000001
tied = false

[distributed]
enabled = false
device_count = "auto"
axis_name = "data"

[train]
seed = 0
batch_size = 2
seq_len = 8
steps = {steps}
lr = 0.001
decay = 0.1
log_every = 1
eval_every = 1
eval_steps = 1
checkpoint_every = 1
keep_last = 2

[data]
source = "text"
path = "input.txt"
tokenizer = "gpt2"
val_fraction = 0.25
""",
        encoding="utf-8",
    )


def write_rows(run_dir, rows):
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_run(tmp_path, rows, *, steps=4, metadata=True):
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)
    write_config(run_dir, steps=steps)
    write_rows(run_dir, rows)
    if metadata:
        (run_dir / "metadata.json").write_text(
            json.dumps({"created_at": "2026-01-01T00:00:00+00:00", "config_path": "configs/unit.toml"}),
            encoding="utf-8",
        )
    return run_dir


def healthy_rows():
    return [
        {
            "step": 0,
            "train/loss": 4.0,
            "train/ppl": 54.6,
            "train/bpb": 2.0,
            "val/loss": 4.5,
            "val/ppl": 90.0,
            "val/bpb": 2.2,
            "val/domain/web/loss": 4.0,
            "val/domain/web/ppl": 54.6,
            "val/domain/web/bpb": 2.0,
            "val/domain/web/tokens": 16,
            "val/domain/code/loss": 5.0,
            "val/domain/code/ppl": 148.4,
            "val/domain/code/bpb": 2.5,
            "val/domain/code/tokens": 16,
            "train/tokens_seen": 16,
            "time/tokens_per_sec": 10.0,
            "time/train_tokens_per_sec": 20.0,
            "time/train_tokens_per_gpu_hour": 72000.0,
            "perf/mfu": 10.0,
            "perf/flops_per_token": 1000,
            "perf/flops_per_step": 16000,
            "perf/peak_flops_per_device": 100000.0,
            "perf/peak_flops_total": 100000.0,
            "system/device_count": 1,
            "system/device_kind": "unit gpu",
            "system/gpu_memory_used_bytes": 100,
            "system/gpu_memory_peak_bytes": 100,
            "system/gpu_utilization_pct": 30.0,
            "system/gpu_power_w": 100.0,
            "health/nan_count": 0,
            "health/loss_spike_count": 0,
            "health/grad_norm_spike_count": 0,
            "health/train_val_gap": 0.5,
        },
        {
            "step": 1,
            "train/loss": 3.0,
            "train/ppl": 20.1,
            "train/bpb": 1.5,
            "train/tokens_seen": 32,
            "time/tokens_per_sec": 30.0,
            "time/train_tokens_per_gpu_hour": 144000.0,
            "perf/mfu": 20.0,
            "system/gpu_memory_used_bytes": 200,
            "system/gpu_memory_peak_bytes": 200,
            "system/gpu_utilization_pct": 40.0,
            "system/gpu_power_w": 110.0,
            "sample/path": "sample.txt",
            "health/nan_count": 0,
            "health/loss_spike_count": 0,
            "health/grad_norm_spike_count": 0,
            "health/train_loss_slope": -1.0,
        },
        {
            "step": 2,
            "train/loss": 2.5,
            "train/ppl": 12.2,
            "train/bpb": 1.25,
            "val/loss": 3.0,
            "val/ppl": 20.1,
            "val/bpb": 1.4,
            "train/tokens_seen": 48,
            "time/train_tokens_per_sec": 40.0,
            "time/train_tokens_per_gpu_hour": 144000.0,
            "perf/mfu": 30.0,
            "system/gpu_memory_used_bytes": 300,
            "system/gpu_memory_peak_bytes": 300,
            "system/gpu_utilization_pct": 50.0,
            "system/gpu_power_w": 120.0,
            "health/nan_count": 0,
            "health/loss_spike_count": 0,
            "health/grad_norm_spike_count": 0,
            "health/train_val_gap": 0.5,
            "health/val_loss_slope": -1.5,
        },
        {
            "step": 3,
            "train/loss": 2.0,
            "train/ppl": 7.4,
            "train/bpb": 1.0,
            "val/loss": 2.8,
            "val/ppl": 16.4,
            "val/bpb": 1.3,
            "val/domain/web/loss": 3.2,
            "val/domain/web/ppl": 24.5,
            "val/domain/web/bpb": 1.6,
            "val/domain/web/tokens": 16,
            "val/domain/code/loss": 4.2,
            "val/domain/code/ppl": 66.7,
            "val/domain/code/bpb": 2.1,
            "val/domain/code/tokens": 16,
            "train/tokens_seen": 64,
            "time/tokens_per_sec": 50.0,
            "time/train_tokens_per_sec": 60.0,
            "time/train_tokens_per_gpu_hour": 216000.0,
            "time/elapsed_sec": 12.0,
            "perf/mfu": 40.0,
            "perf/flops_per_token": 1000,
            "perf/flops_per_step": 16000,
            "perf/peak_flops_per_device": 100000.0,
            "perf/peak_flops_total": 100000.0,
            "system/device_count": 1,
            "system/device_kind": "unit gpu",
            "system/gpu_memory_used_bytes": 400,
            "system/gpu_memory_peak_bytes": 400,
            "system/gpu_utilization_pct": 60.0,
            "system/gpu_power_w": 130.0,
            "health/nan_count": 0,
            "health/loss_spike_count": 0,
            "health/grad_norm_spike_count": 0,
            "health/train_val_gap": 0.8,
            "health/train_loss_slope": -0.5,
            "health/val_loss_slope": -0.2,
        },
    ]


def test_summarize_run_computes_quality_health_speed_and_checkpoint_evals(tmp_path):
    run_dir = make_run(tmp_path, healthy_rows())
    eval_dir_1 = run_dir / "evals" / "step_1"
    eval_dir_2 = run_dir / "evals" / "step_3"
    eval_dir_1.mkdir(parents=True)
    eval_dir_2.mkdir(parents=True)
    (eval_dir_1 / "metrics.json").write_text(json.dumps({
        "checkpoint_step": 1,
        "loss": 3.1,
        "ppl": 22.2,
        "bpb": 1.6,
        "domains": {"web": {"loss": 3.3, "ppl": 27.1, "bpb": 1.7}},
    }))
    (eval_dir_2 / "metrics.json").write_text(json.dumps({
        "checkpoint_step": 3,
        "loss": 2.7,
        "ppl": 14.9,
        "bpb": 1.2,
        "domains": {
            "web": {"loss": 3.0, "ppl": 20.1, "bpb": 1.5},
            "code": {"loss": 4.0, "ppl": 54.6, "bpb": 2.0},
        },
    }))

    summary = run_summary.summarize_run(run_dir)

    assert summary["run"]["name"] == "unit"
    assert summary["run"]["created_at"] == "2026-01-01T00:00:00+00:00"
    assert summary["model"]["params"] > 0
    assert summary["training"]["final_step"] == 3
    assert summary["training"]["steps_completed"] == 4
    assert summary["training"]["logged_rows"] == 4
    assert summary["training"]["tokens_seen"] == 64
    assert summary["quality"]["final_train_loss"] == 2.0
    assert summary["quality"]["best_train_loss"] == 2.0
    assert summary["quality"]["final_val_loss"] == 2.8
    assert summary["quality"]["best_val_loss"] == 2.8
    assert summary["quality"]["final_val_bpb"] == 1.3
    assert summary["health"]["nan_count"] == 0
    assert summary["health"]["final_train_val_gap"] == 0.8
    assert summary["speed"]["avg_tokens_per_sec"] == pytest.approx(30.0)
    assert summary["speed"]["avg_train_tokens_per_sec"] == pytest.approx(40.0)
    assert summary["performance"]["final_mfu"] == 40.0
    assert summary["performance"]["avg_mfu"] == pytest.approx(25.0)
    assert summary["performance"]["flops_per_token"] == 1000
    assert summary["performance"]["avg_train_tokens_per_gpu_hour"] == pytest.approx(144000.0)
    assert summary["performance"]["peak_gpu_memory_bytes"] == 400
    assert summary["performance"]["avg_gpu_utilization_pct"] == pytest.approx(45.0)
    assert summary["performance"]["avg_gpu_power_w"] == pytest.approx(115.0)
    assert summary["checkpoint_evals"]["count"] == 2
    assert summary["checkpoint_evals"]["latest"]["checkpoint_step"] == 3
    assert summary["checkpoint_evals"]["best_loss"]["checkpoint_step"] == 3
    assert summary["registry_record"]["checkpoint_eval_count"] == 2
    assert summary["registry_record"]["avg_mfu"] == pytest.approx(25.0)
    assert summary["registry_record"]["final_mfu"] == 40.0
    assert summary["registry_record"]["train_tokens_per_gpu_hour"] == pytest.approx(144000.0)
    assert summary["registry_record"]["peak_gpu_memory_bytes"] == 400
    assert summary["registry_record"]["latest_checkpoint_step"] == 3
    assert summary["registry_record"]["latest_checkpoint_loss"] == 2.7
    assert summary["registry_record"]["latest_checkpoint_bpb"] == 1.2
    assert summary["registry_record"]["best_checkpoint_loss"] == 2.7
    assert summary["registry_record"]["best_checkpoint_bpb"] == 1.2
    assert summary["registry_record"]["latest_domain_mean_loss"] == pytest.approx(3.5)
    assert summary["registry_record"]["latest_domain_worst_name"] == "code"
    assert summary["registry_record"]["latest_domain_worst_loss"] == 4.0
    assert summary["domain_validation"]["training"]["web"]["first_loss"] == 4.0
    assert summary["domain_validation"]["training"]["web"]["final_loss"] == 3.2
    assert summary["domain_validation"]["training"]["web"]["best_bpb"] == 1.6
    assert summary["domain_validation"]["training"]["web"]["delta_loss"] == pytest.approx(-0.8)
    assert summary["domain_validation"]["checkpoint_evals"]["web"]["best_loss"]["checkpoint_step"] == 3
    scorecard = run_summary.format_scorecard(summary)
    assert "Training Native Validation" in scorecard
    assert "Checkpoint Native Validation" in scorecard
    assert "Performance" in scorecard
    assert "Training Domain Validation" in scorecard
    assert "Checkpoint Domain Validation" in scorecard
    assert summary["status"] == "healthy"
    assert summary["decision_hint"] == "scale"


def test_verdict_failed_incomplete_and_unstable(tmp_path):
    failed = healthy_rows()
    failed[-1]["health/nan_count"] = 1
    assert run_summary.summarize_run(make_run(tmp_path / "failed", failed))["status"] == "failed"

    incomplete = healthy_rows()[:2]
    assert run_summary.summarize_run(make_run(tmp_path / "incomplete", incomplete))["status"] == "incomplete"

    unstable = healthy_rows()
    unstable[-1]["health/grad_norm_spike_count"] = 1
    summary = run_summary.summarize_run(make_run(tmp_path / "unstable", unstable))
    assert summary["status"] == "unstable"
    assert summary["decision_hint"] == "inspect"


def test_sparse_logging_near_final_step_counts_as_complete(tmp_path):
    rows = healthy_rows()
    rows[-1]["step"] = 90
    run_dir = make_run(tmp_path, rows, steps=100)
    text = (run_dir / "config.toml").read_text(encoding="utf-8")
    (run_dir / "config.toml").write_text(text.replace("log_every = 1", "log_every = 10"), encoding="utf-8")

    summary = run_summary.summarize_run(run_dir)

    assert summary["training"]["steps_completed"] == 91
    assert summary["training"]["logged_rows"] == 4
    assert summary["status"] == "healthy"


def test_write_summary_artifacts_and_cli_register(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path, healthy_rows())
    registry_path = tmp_path / "runs" / "registry.jsonl"
    monkeypatch.setattr(sys, "argv", ["summarize-run", str(run_dir), "--register", "--registry-path", str(registry_path)])

    run_summary.main()

    assert (run_dir / "summary" / "run_summary.json").exists()
    assert (run_dir / "summary" / "scorecard.md").exists()
    records = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["run_name"] == "unit"
    assert records[0]["status"] == "healthy"
    assert records[0]["checkpoint_eval_count"] == 0
    assert records[0]["latest_checkpoint_loss"] is None
    assert records[0]["latest_domain_mean_loss"] is None


def test_register_run_upserts_and_list_runs_prints_table(tmp_path, capsys):
    run_dir = make_run(tmp_path, healthy_rows())
    eval_dir = run_dir / "evals" / "step_3"
    eval_dir.mkdir(parents=True)
    (eval_dir / "metrics.json").write_text(json.dumps({
        "checkpoint_step": 3,
        "loss": 2.7,
        "ppl": 14.9,
        "bpb": 1.2,
        "domains": {"web": {"loss": 3.0, "ppl": 20.1, "bpb": 1.5}},
    }))
    registry_path = tmp_path / "runs" / "registry.jsonl"

    run_registry.main([str(run_dir), "--registry-path", str(registry_path)])
    run_registry.main([str(run_dir), "--registry-path", str(registry_path)])

    records = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["latest_checkpoint_loss"] == 2.7
    assert records[0]["latest_checkpoint_bpb"] == 1.2
    assert records[0]["latest_domain_mean_loss"] == 3.0

    run_registry.list_main(["--registry-path", str(registry_path)])
    output = capsys.readouterr().out
    assert "run" in output
    assert "ckpt_loss" in output
    assert "domain" in output
    assert "mfu" in output
    assert "tok/gpu-hr" in output
    assert "unit" in output
    assert "healthy" in output
