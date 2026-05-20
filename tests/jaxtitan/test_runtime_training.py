import json
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training
from jaxtitan.services import initialize_run

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_run_training_writes_artifacts_metrics_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "loop",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=1))

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    assert summary.run_id == "loop"
    assert summary.run_dir == Path("runs/loop")
    assert summary.status == "completed"
    assert summary.steps == 2
    assert summary.tokens_seen == 16
    assert summary.target_tokens == 16
    assert run_dir.is_dir()
    assert (run_dir / "config" / "source.toml").is_file()
    assert (run_dir / "config" / "resolved.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "diagnostics" / "runtime.json").is_file()

    events = _jsonl(run_dir / "events.jsonl")
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert [event["type"] for event in events] == [
        "run_initialized",
        "training_started",
        "checkpoint_saved",
        "training_completed",
    ]
    assert events[-2]["step"] == 2
    assert events[-2]["reason"] == "final"
    assert events[-2]["checkpoint_sec"] >= 0.0
    assert events[1]["execution_mode"] == "replicated_data_parallel"
    assert events[1]["metrics_scope"] == "global"
    assert events[1]["artifact_writer"] == "single_host"
    assert events[1]["model_remat"] == "none"
    assert events[1]["gradient_accumulation_steps"] == 1
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["tokens_seen"] == 16
    assert metrics[-1]["token_count"] == 8
    assert metrics[-1]["loss"] == pytest.approx(metrics[-1]["loss_sum"] / metrics[-1]["token_count"])
    assert metrics[-1]["epoch"] == 0
    assert metrics[-1]["token_start"] == 8
    assert metrics[-1]["token_end"] == 16
    assert metrics[-1]["examples"] == 2
    assert metrics[-1]["target_tokens"] == 8
    assert metrics[-1]["gradient_accumulation_steps"] == 1
    assert metrics[-1]["micro_global_batch_size"] == 2
    assert metrics[-1]["effective_global_batch_size"] == 2
    assert metrics[-1]["micro_tokens_per_step"] == 8
    assert metrics[-1]["effective_tokens_per_step"] == 8
    assert metrics[-1]["execution_mode"] == "replicated_data_parallel"
    assert metrics[-1]["metrics_scope"] == "global"
    assert metrics[-1]["artifact_writer"] == "single_host"
    assert metrics[-1]["data_sec"] >= 0.0
    assert metrics[-1]["placement_sec"] >= 0.0
    assert metrics[-1]["train_dispatch_sec"] >= 0.0
    assert metrics[-1]["metrics_sync_sec"] >= 0.0
    assert metrics[-1]["train_step_sec"] >= metrics[-1]["metrics_sync_sec"]
    assert metrics[-1]["step_sec"] >= metrics[-1]["train_step_sec"]
    assert metrics[-1]["tokens_per_sec"] > 0.0
    assert metrics[-1]["train_tokens_per_sec"] > 0.0
    assert metrics[-1]["examples_per_sec"] > 0.0
    assert metrics[-1]["flops_per_token"] == diagnostics["performance"]["flops_per_token"]
    assert metrics[-1]["flops_per_step"] == metrics[-1]["flops_per_token"] * metrics[-1]["target_tokens"]
    assert metrics[-1]["peak_flops_per_device"] == diagnostics["performance"]["peak_flops_per_device"]
    assert "device_memory_used_bytes" in metrics[-1]
    assert final["status"] == "completed"
    assert final["steps"] == metrics[-1]["step"]
    assert final["tokens_seen"] == metrics[-1]["tokens_seen"]
    assert final["final_loss"] == pytest.approx(metrics[-1]["loss"])
    assert final["total_wall_sec"] > 0.0
    assert final["avg_train_tokens_per_sec"] == pytest.approx(
        sum(row["train_tokens_per_sec"] for row in metrics) / len(metrics)
    )
    assert final["final_train_tokens_per_sec"] == pytest.approx(metrics[-1]["train_tokens_per_sec"])
    assert final["steady_train_tokens_per_sec"] == pytest.approx(metrics[-1]["train_tokens_per_sec"])
    assert final["final_mfu"] == metrics[-1]["mfu"]
    assert final["device_kind"] == diagnostics["performance"]["device_kind"]
    assert final["device_count"] == diagnostics["performance"]["device_count"]
    assert final["runtime_diagnostics_path"] == "diagnostics/runtime.json"
    assert final["execution_mode"] == "replicated_data_parallel"
    assert final["metrics_scope"] == "global"
    assert final["artifact_writer"] == "single_host"
    assert final["model_remat"] == "none"
    assert final["gradient_accumulation_steps"] == 1
    assert final["effective_global_batch_size"] == 2
    assert final["micro_tokens_per_step"] == 8
    assert final["effective_tokens_per_step"] == 8
    assert final["latest_checkpoint_path"] == "checkpoints/000002"
    assert final["best_eval_step"] is None
    assert final["best_eval_loss"] is None
    assert final["best_checkpoint_path"] is None
    assert index["latest_step"] == 2
    assert index["latest_checkpoint_path"] == "checkpoints/000002"
    assert index["best_eval_step"] is None
    assert index["records"] == [
        {
            "checkpoint_path": "checkpoints/000002",
            "eval_loss": None,
            "reason": "final",
            "retained": True,
            "step": 2,
            "tokens_seen": 16,
            "train_loss": metrics[-1]["loss"],
        }
    ]
    assert events[-1]["total_wall_sec"] == pytest.approx(final["total_wall_sec"])
    assert events[-1]["final_train_tokens_per_sec"] == pytest.approx(final["final_train_tokens_per_sec"])
    assert events[-1]["execution_mode"] == "replicated_data_parallel"
    assert events[-1]["metrics_scope"] == "global"
    assert events[-1]["artifact_writer"] == "single_host"
    assert events[-1]["model_remat"] == "none"
    assert events[-1]["gradient_accumulation_steps"] == 1
    assert events[-1]["effective_global_batch_size"] == 2
    assert events[-1]["effective_tokens_per_step"] == 8
    assert diagnostics["jax"]["backend"]
    assert diagnostics["packages"]["jaxtitan"]
    assert diagnostics["model"]["remat"] == "none"
    assert diagnostics["parallelism"]["execution_mode"] == "replicated_data_parallel"
    assert diagnostics["parallelism"]["metrics_scope"] == "global"
    assert diagnostics["parallelism"]["artifact_writer"] == "single_host"
    assert diagnostics["compile"]["train"]["donate_state"] is True
    assert diagnostics["compile"]["train"]["expected_batch_shape"] == [1, 2, 4]
    assert diagnostics["compile"]["train"]["input_shardings"]["input_ids"]["partition_spec"] == "PartitionSpec(None, 'data', None)"
    assert diagnostics["compile"]["eval"]["donate_state"] is False
    assert diagnostics["compile"]["eval"]["expected_batch_shape"] == [2, 4]
    assert diagnostics["compile"]["eval"]["input_shardings"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"
    assert diagnostics["sharding"]["batch"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"
    assert diagnostics["sharding"]["batch"]["accumulated_input_ids"]["partition_spec"] == "PartitionSpec(None, 'data', None)"
    assert diagnostics["sharding"]["train_state"]["model"]["partition_spec"] == "PartitionSpec()"
    assert (run_dir / "checkpoints" / "000002").is_dir()


def test_run_training_with_four_device_data_axis_reports_global_and_per_device_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "dp-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=65,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=32,
            log_every_steps=1,
            global_batch_size=8,
            axis_sizes=(4,),
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert summary.steps == 1
    assert summary.tokens_seen == 32
    assert metrics[-1]["token_count"] == 32
    assert metrics[-1]["target_tokens"] == 32
    assert metrics[-1]["data_axis_size"] == 4
    assert metrics[-1]["global_batch_size"] == 8
    assert metrics[-1]["per_device_batch_size"] == 2
    assert metrics[-1]["global_target_tokens"] == 32
    assert metrics[-1]["per_device_target_tokens"] == 8
    assert diagnostics["jax"]["single_process"] is True
    assert diagnostics["mesh"]["data_axis_size"] == 4
    assert diagnostics["mesh"]["global_batch_size"] == 8
    assert diagnostics["mesh"]["per_device_batch_size"] == 2
    assert diagnostics["mesh"]["global_tokens_per_step"] == 32
    assert diagnostics["mesh"]["per_device_tokens_per_step"] == 8
    assert diagnostics["mesh"]["selected_device_count"] == 4
    assert diagnostics["parallelism"]["batch"]["global_batch_size"] == 8
    assert diagnostics["parallelism"]["batch"]["per_device_batch_size"] == 2
    assert diagnostics["sharding"]["metrics"]["partition_spec"] == "PartitionSpec()"
    assert final["data_axis_size"] == 4
    assert final["global_batch_size"] == 8
    assert final["per_device_batch_size"] == 2
    assert final["selected_device_count"] == 4
    assert final["single_process"] is True


def test_run_training_with_gradient_accumulation_records_effective_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "grad-accum-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=60,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=32,
            log_every_steps=1,
            gradient_accumulation_steps=2,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert summary.steps == 2
    assert summary.tokens_seen == 32
    assert summary.gradient_accumulation_steps == 2
    assert summary.effective_global_batch_size == 4
    assert summary.effective_tokens_per_step == 16
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[0]["token_start"] == 0
    assert metrics[0]["token_end"] == 16
    assert metrics[0]["examples"] == 4
    assert metrics[0]["target_tokens"] == 16
    assert metrics[0]["token_count"] == 16
    assert metrics[0]["tokens_seen"] == 16
    assert metrics[0]["gradient_accumulation_steps"] == 2
    assert metrics[0]["micro_global_batch_size"] == 2
    assert metrics[0]["effective_global_batch_size"] == 4
    assert metrics[0]["micro_tokens_per_step"] == 8
    assert metrics[0]["effective_tokens_per_step"] == 16
    assert metrics[-1]["tokens_seen"] == 32
    assert final["gradient_accumulation_steps"] == 2
    assert final["effective_global_batch_size"] == 4
    assert final["effective_tokens_per_step"] == 16
    assert diagnostics["mesh"]["gradient_accumulation_steps"] == 2
    assert diagnostics["mesh"]["effective_global_batch_size"] == 4
    assert diagnostics["compile"]["train"]["expected_batch_shape"] == [2, 2, 4]
    assert diagnostics["parallelism"]["batch"]["gradient_accumulation_steps"] == 2
    assert diagnostics["parallelism"]["batch"]["effective_tokens_per_step"] == 16


def test_run_training_with_block_remat_completes_and_records_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "remat-loop",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=35,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            gradient_accumulation_steps=2,
            remat="block",
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")

    assert summary.steps == 1
    assert summary.tokens_seen == 16
    assert summary.model_remat == "block"
    assert metrics[-1]["gradient_accumulation_steps"] == 2
    assert final["model_remat"] == "block"
    assert diagnostics["model"]["remat"] == "block"
    assert diagnostics["compile"]["train"]["donate_state"] is True
    assert diagnostics["compile"]["train"]["expected_batch_shape"] == [2, 2, 4]


def test_run_training_resumes_four_device_data_axis_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "dp-resume",
        shard_token_groups=(tuple(range(0, 120)),),
        train_tokens=100,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=32,
            log_every_steps=1,
            global_batch_size=8,
            axis_sizes=(4,),
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=64,
            log_every_steps=1,
            global_batch_size=8,
            axis_sizes=(4,),
        )
    )

    summary = run_training(config_path, resume=True)

    metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "train.jsonl")
    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert summary.steps == 2
    assert summary.tokens_seen == 64
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["data_axis_size"] == 4
    assert any(event["type"] == "training_resumed" for event in events)


def test_run_training_logs_final_row_even_when_not_on_log_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "final-row",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=5))

    run_training(config_path)

    metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "train.jsonl")
    assert [row["step"] for row in metrics] == [2]
    assert metrics[-1]["tokens_seen"] == 16


def test_run_training_saves_interval_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "checkpoint-interval",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1)
    )

    run_training(config_path)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    checkpoint_events = [event for event in events if event["type"] == "checkpoint_saved"]
    assert [event["step"] for event in checkpoint_events] == [1, 2]
    assert [event["reason"] for event in checkpoint_events] == ["interval", "interval"]
    assert (tmp_path / "runs" / "loop" / "checkpoints" / "000001").is_dir()
    assert (tmp_path / "runs" / "loop" / "checkpoints" / "000002").is_dir()


def test_run_training_stops_after_crossing_target_token_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "cross-target",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=10, log_every_steps=1))

    summary = run_training(config_path)

    assert summary.steps == 2
    assert summary.tokens_seen == 16
    assert summary.target_tokens == 10


def test_run_training_records_failure_when_dataset_exhausts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "exhaust",
        shard_token_groups=(tuple(range(0, 20)),),
        train_tokens=17,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=24, log_every_steps=1))

    with pytest.raises(ContractError, match="prepared train split ended"):
        run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    assert events[-1]["type"] == "training_failed"
    assert events[-1]["error_type"] == "ContractError"
    assert [row["step"] for row in metrics] == [1, 2]
    assert not (run_dir / "summaries" / "final.json").exists()


def test_run_training_fails_when_dataset_exhausts_inside_accumulation_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "exhaust-accum",
        shard_token_groups=(tuple(range(0, 20)),),
        train_tokens=12,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            gradient_accumulation_steps=2,
        )
    )

    with pytest.raises(ContractError, match="prepared train split ended"):
        run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    assert events[-1]["type"] == "training_failed"
    assert events[-1]["error_type"] == "ContractError"
    assert not (run_dir / "metrics" / "train.jsonl").exists()
    assert not (run_dir / "summaries" / "final.json").exists()


def test_run_training_rejects_multi_process_before_creating_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("multi-process", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=1))
    monkeypatch.setattr("jaxtitan.mesh.sharding.jax.process_count", lambda: 2)

    with pytest.raises(ContractError, match="exactly one process"):
        run_training(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_training_with_validation_eval_writes_eval_metrics_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "eval-loop",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            eval_every_steps=1,
            eval_num_batches=2,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    eval_metrics = _jsonl(run_dir / "metrics" / "eval.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    events = _jsonl(run_dir / "events.jsonl")
    assert [row["step"] for row in eval_metrics] == [1, 2]
    assert eval_metrics[-1]["eval_name"] == "validation"
    assert eval_metrics[-1]["num_batches"] == 2
    assert eval_metrics[-1]["token_count"] == 16
    assert eval_metrics[-1]["token_start"] == 25
    assert eval_metrics[-1]["token_end"] == 41
    assert eval_metrics[-1]["examples"] == 4
    assert eval_metrics[-1]["target_tokens"] == 16
    assert eval_metrics[-1]["eval_sec"] > 0.0
    assert eval_metrics[-1]["eval_tokens_per_sec"] > 0.0
    assert eval_metrics[-1]["eval_examples_per_sec"] > 0.0
    assert final["final_eval_loss"] == pytest.approx(eval_metrics[-1]["loss"])
    assert final["final_eval_token_count"] == eval_metrics[-1]["token_count"]
    assert final["final_eval_num_batches"] == eval_metrics[-1]["num_batches"]
    assert final["best_eval_loss"] == pytest.approx(index["best_eval_loss"])
    assert final["best_checkpoint_path"] == index["best_checkpoint_path"]
    assert index["latest_step"] == 2
    assert index["latest_checkpoint_path"] == "checkpoints/000002"
    assert len(index["records"]) == 1
    assert index["records"][0]["step"] == 2
    assert index["records"][0]["reason"] == "final"
    assert index["records"][0]["eval_loss"] == pytest.approx(eval_metrics[-1]["loss"])
    assert index["records"][0]["train_loss"] == pytest.approx(_jsonl(run_dir / "metrics" / "train.jsonl")[-1]["loss"])
    assert summary.final_eval_loss == pytest.approx(eval_metrics[-1]["loss"])
    assert [event["type"] for event in events if event["type"].startswith("eval_")] == [
        "eval_started",
        "eval_completed",
        "eval_started",
        "eval_completed",
    ]
    eval_completed = [event for event in events if event["type"] == "eval_completed"]
    assert eval_completed[-1]["eval_sec"] == pytest.approx(eval_metrics[-1]["eval_sec"])


def test_run_training_runs_final_validation_when_cadence_misses_final_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "eval-final",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=24,
            log_every_steps=1,
            eval_every_steps=2,
            eval_num_batches=1,
        )
    )

    run_training(config_path)

    eval_metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "eval.jsonl")
    assert [row["step"] for row in eval_metrics] == [2, 3]


def test_run_training_scores_checkpoint_even_when_eval_cadence_does_not_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "checkpoint-score",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            checkpoint_every_steps=1,
            eval_every_steps=10,
            eval_num_batches=1,
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    assert [row["step"] for row in _jsonl(run_dir / "metrics" / "eval.jsonl")] == [1]
    assert index["records"][0]["step"] == 1
    assert index["records"][0]["eval_loss"] is not None
    assert [event["type"] for event in events[-4:]] == [
        "eval_started",
        "eval_completed",
        "checkpoint_saved",
        "training_completed",
    ]


def test_run_training_eval_uses_validation_manifest_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    train_manifest = prepared_dataset_factory(
        "eval-train",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    validation_manifest = prepared_dataset_factory(
        "eval-validation",
        shard_token_groups=(tuple(range(100, 140)),),
        train_tokens=10,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            train_manifest,
            target_tokens=8,
            log_every_steps=1,
            eval_every_steps=1,
            eval_num_batches=1,
            validation_manifest=validation_manifest,
        )
    )

    run_training(config_path)

    eval_metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "eval.jsonl")
    assert eval_metrics[-1]["token_start"] == 10
    assert eval_metrics[-1]["token_end"] == 18


def test_run_training_rejects_unsupported_eval_runtime_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "bad-eval",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            eval_every_steps=1,
            eval_name="perplexity",
        )
    )

    with pytest.raises(ContractError, match="validation"):
        run_training(config_path)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert events[-1]["type"] == "training_failed"


def test_run_training_rejects_multiple_eval_runtime_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "multi-eval",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            eval_every_steps=1,
            second_eval=True,
        )
    )

    with pytest.raises(ContractError, match="exactly one eval"):
        run_training(config_path)


def test_run_training_eval_failure_records_eval_and_training_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "small-val",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            eval_every_steps=1,
            eval_num_batches=1,
        )
    )

    with pytest.raises(ContractError, match="val split"):
        run_training(config_path)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert events[-3]["type"] == "eval_started"
    assert events[-2]["type"] == "eval_failed"
    assert events[-1]["type"] == "training_failed"
    assert not (tmp_path / "runs" / "loop" / "metrics" / "eval.jsonl").exists()


def test_run_training_resume_continues_from_latest_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    first = run_training(config_path)
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=5, checkpoint_every_steps=2)
    )

    resumed = run_training(config_path, resume=True)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    assert first.steps == 1
    assert resumed.steps == 2
    assert resumed.tokens_seen == 16
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["token_start"] == 8
    assert metrics[-1]["token_end"] == 16
    assert "training_resumed" in [event["type"] for event in events]
    resumed_event = next(event for event in events if event["type"] == "training_resumed")
    assert resumed_event["checkpoint_step"] == 1
    assert resumed_event["compat_checked"] is True
    assert resumed_event["runtime_fingerprint"]
    assert resumed_event["dataset_token_offset"] == 8
    assert (run_dir / "checkpoints" / "000002").is_dir()


def test_run_training_resume_with_eval_preserves_train_cursor_and_restarts_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "resume-eval",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            checkpoint_every_steps=1,
            eval_every_steps=1,
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            eval_every_steps=1,
        )
    )

    run_training(config_path, resume=True)

    run_dir = tmp_path / "runs" / "loop"
    train_metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    eval_metrics = _jsonl(run_dir / "metrics" / "eval.jsonl")
    resumed_event = next(event for event in _jsonl(run_dir / "events.jsonl") if event["type"] == "training_resumed")
    assert resumed_event["dataset_token_offset"] == 8
    assert [row["token_start"] for row in train_metrics] == [0, 8]
    assert [row["step"] for row in eval_metrics] == [1, 2]
    assert [row["token_start"] for row in eval_metrics] == [25, 25]
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    assert [record["step"] for record in index["records"]] == [1, 2]
    assert index["latest_step"] == 2


def test_run_training_retains_latest_and_best_checkpoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "retention",
        shard_token_groups=(tuple(range(0, 60)),),
        train_tokens=33,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=24,
            log_every_steps=1,
            checkpoint_every_steps=1,
            eval_every_steps=1,
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    assert (run_dir / index["latest_checkpoint_path"]).is_dir()
    assert index["best_checkpoint_path"] is not None
    assert (run_dir / index["best_checkpoint_path"]).is_dir()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"hidden_size": 16}, r"compatibility\.model\.hidden_size"),
        ({"remat": "block"}, r"compatibility\.model\.remat"),
        ({"weight_decay": 0.2}, r"compatibility\.optimizer\.weight_decay"),
        ({"seq_len": 2}, r"compatibility\.training\.seq_len"),
        ({"global_batch_size": 1}, r"compatibility\.training\.global_batch_size"),
        ({"gradient_accumulation_steps": 2}, r"compatibility\.training\.gradient_accumulation_steps"),
        ({"axis_sizes": (2,)}, r"compatibility\.mesh\.axis_sizes"),
        ({"seed": 9}, r"compatibility\.seed"),
    ],
)
def test_run_training_resume_rejects_unsafe_config_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
    change: dict,
    match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "unsafe-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1, **change)
    )

    with pytest.raises(ContractError, match=match):
        run_training(config_path, resume=True)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert events[-2]["type"] == "checkpoint_restore_failed"
    assert events[-1]["type"] == "training_failed"


def test_run_training_resume_rejects_data_manifest_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_manifest = prepared_dataset_factory(
        "data-resume-first",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    second_manifest = prepared_dataset_factory(
        "data-resume-second",
        shard_token_groups=(tuple(range(10, 40)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(first_manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(second_manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1)
    )

    with pytest.raises(ContractError, match=r"compatibility\.data\.train_manifest"):
        run_training(config_path, resume=True)


def test_run_training_resume_rejects_tokenizer_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "tokenizer-first",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    manifest_json = json.loads(manifest.read_text())
    manifest_json["tokenizer"]["name"] = "other-tokenizer"
    manifest.write_text(json.dumps(manifest_json, sort_keys=True))
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            tokenizer_id="other-tokenizer",
        )
    )

    with pytest.raises(ContractError, match=r"compatibility\.data\.tokenizer_id"):
        run_training(config_path, resume=True)


def test_run_training_resume_rejects_auto_cosine_schedule_target_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "cosine-auto-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
        )
    )

    with pytest.raises(ContractError, match=r"compatibility\.optimizer\.schedule\.total_steps"):
        run_training(config_path, resume=True)


def test_run_training_resume_allows_explicit_cosine_schedule_target_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "cosine-explicit-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
            total_steps=2,
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
            total_steps=2,
        )
    )

    resumed = run_training(config_path, resume=True)

    assert resumed.steps == 2
    assert resumed.tokens_seen == 16


def test_run_training_resume_records_restore_failure_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "restore-failure",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=1))
    initialize_run(config_path)

    with pytest.raises(ContractError, match="no checkpoints"):
        run_training(config_path, resume=True)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert [event["type"] for event in events] == [
        "run_initialized",
        "training_started",
        "checkpoint_restore_failed",
        "training_failed",
    ]
    assert events[1]["resume"] is True


def test_cli_run_train_succeeds_for_tiny_run(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = prepared_dataset_factory(
        "cli-loop",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, log_every_steps=1))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "runs/loop"
    assert (tmp_path / "runs" / "loop" / "metrics" / "train.jsonl").is_file()
    assert (tmp_path / "runs" / "loop" / "summaries" / "final.json").is_file()


def test_cli_run_train_resume_succeeds(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = prepared_dataset_factory(
        "cli-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    first = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1)
    )

    resumed = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", "--resume", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "train.jsonl")
    assert first.returncode == 0
    assert resumed.returncode == 0
    assert resumed.stdout.strip() == "runs/loop"
    assert [row["step"] for row in metrics] == [1, 2]


def test_cli_run_train_invalid_data_fails_cleanly(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(tmp_path / "missing" / "manifest.json", target_tokens=8, log_every_steps=1))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "manifest does not exist" in result.stderr
    assert "Traceback" not in result.stderr


def _training_config(
    train_manifest: Path | str,
    *,
    target_tokens: int,
    log_every_steps: int,
    checkpoint_every_steps: int = 10,
    seed: int = 7,
    hidden_size: int = 8,
    remat: str = "none",
    weight_decay: float = 0.0,
    schedule_name: str = "constant",
    total_steps: int | None = None,
    tokenizer_id: str = "toy-tokenizer",
    seq_len: int = 4,
    global_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    axis_sizes: tuple[int, ...] = (1,),
    validation_manifest: Path | str | None = None,
    eval_every_steps: int | None = None,
    eval_num_batches: int = 1,
    eval_name: str = "validation",
    second_eval: bool = False,
) -> str:
    total_steps_line = "" if total_steps is None else f"total_steps = {total_steps}\n"
    validation_manifest_line = (
        "" if validation_manifest is None else f'validation_manifest = "{Path(validation_manifest).as_posix()}"\n'
    )
    eval_block = ""
    if eval_every_steps is not None:
        eval_block = f"""
[[evals]]
name = "{eval_name}"
every_steps = {eval_every_steps}
num_batches = {eval_num_batches}
"""
        if second_eval:
            eval_block += """
[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""
    return f"""
[run]
id = "loop"
seed = {seed}
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 64
hidden_size = {hidden_size}
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"
remat = "{remat}"

[optimizer]
name = "adamw"
weight_decay = {weight_decay}

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = 0.001
{total_steps_line}

[data]
train_manifest = "{Path(train_manifest).as_posix()}"
tokenizer_id = "{tokenizer_id}"
{validation_manifest_line}

[training]
seq_len = {seq_len}
global_batch_size = {global_batch_size}
gradient_accumulation_steps = {gradient_accumulation_steps}
target_tokens = {target_tokens}
log_every_steps = {log_every_steps}
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = ["data"]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]
{eval_block}
"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
