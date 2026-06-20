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
    assert events[1]["data_pipeline_backend"] == "grain"
    assert events[1]["data_pipeline_order"] == "sequential"
    assert events[1]["data_pipeline_shuffle_seed"] is None
    assert events[1]["data_pipeline_worker_count"] == 0
    assert events[1]["data_pipeline_worker_buffer_size"] == 1
    assert events[1]["data_pipeline_prefetch"] is False
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["tokens_seen"] == 16
    assert metrics[-1]["token_count"] == 8
    assert metrics[-1]["loss"] == pytest.approx(metrics[-1]["loss_sum"] / metrics[-1]["token_count"])
    assert metrics[-1]["epoch"] == 0
    assert metrics[-1]["token_start"] == 8
    assert metrics[-1]["token_end"] == 16
    assert metrics[-1]["examples"] == 2
    assert metrics[-1]["target_tokens"] == 8
    assert metrics[-1]["document_aware"] is False
    assert metrics[-1]["documents_touched"] is None
    assert metrics[-1]["document_min"] is None
    assert metrics[-1]["document_max"] is None
    assert metrics[-1]["gradient_accumulation_steps"] == 1
    assert metrics[-1]["micro_global_batch_size"] == 2
    assert metrics[-1]["effective_global_batch_size"] == 2
    assert metrics[-1]["micro_tokens_per_step"] == 8
    assert metrics[-1]["effective_tokens_per_step"] == 8
    assert metrics[-1]["execution_mode"] == "replicated_data_parallel"
    assert metrics[-1]["metrics_scope"] == "global"
    assert metrics[-1]["artifact_writer"] == "single_host"
    assert metrics[-1]["data_pipeline_backend"] == "grain"
    assert metrics[-1]["data_order"] == "sequential"
    assert metrics[-1]["data_worker_count"] == 0
    assert metrics[-1]["data_worker_buffer_size"] == 1
    assert metrics[-1]["data_prefetch"] is False
    assert metrics[-1]["microbatch_loss_mean"] == pytest.approx(metrics[-1]["loss"])
    assert metrics[-1]["microbatch_loss_max"] == pytest.approx(metrics[-1]["loss"])
    assert metrics[-1]["batch_het"] == pytest.approx(0.0)
    assert metrics[-1]["optimizer_groups"]
    assert sum(group["leaf_count"] for group in metrics[-1]["optimizer_groups"]) == diagnostics["model"]["parameter_leaves"]
    assert sum(group["parameter_count"] for group in metrics[-1]["optimizer_groups"]) == diagnostics["model"]["parameters"]
    assert metrics[-1]["optimizer_grad_norm_max_group"]
    assert metrics[-1]["optimizer_update_norm_max_group"]
    assert metrics[-1]["optimizer_update_param_ratio_max"] >= 0.0
    assert metrics[-1]["optimizer_update_param_ratio_mean"] >= 0.0
    assert metrics[-1]["optimizer_groups_with_zero_grad"] >= 0
    assert metrics[-1]["optimizer_groups_with_zero_update"] >= 0
    assert metrics[-1]["optimizer_route_backend_counts"] == {"adamw": diagnostics["model"]["parameter_leaves"]}
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
    assert final["final_batch_het"] == pytest.approx(metrics[-1]["batch_het"])
    assert final["avg_batch_het"] == pytest.approx(sum(row["batch_het"] for row in metrics) / len(metrics))
    assert final["final_optimizer_groups"] == metrics[-1]["optimizer_groups"]
    assert final["final_optimizer_grad_norm_max_group"] == metrics[-1]["optimizer_grad_norm_max_group"]
    assert final["final_optimizer_update_norm_max_group"] == metrics[-1]["optimizer_update_norm_max_group"]
    assert final["final_optimizer_update_param_ratio_max"] == metrics[-1]["optimizer_update_param_ratio_max"]
    assert final["final_optimizer_update_param_ratio_mean"] == metrics[-1]["optimizer_update_param_ratio_mean"]
    assert final["final_optimizer_groups_with_zero_grad"] == metrics[-1]["optimizer_groups_with_zero_grad"]
    assert final["final_optimizer_groups_with_zero_update"] == metrics[-1]["optimizer_groups_with_zero_update"]
    assert final["final_optimizer_route_backend_counts"] == metrics[-1]["optimizer_route_backend_counts"]
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
    assert final["data_pipeline_backend"] == "grain"
    assert final["data_pipeline_order"] == "sequential"
    assert final["data_pipeline_shuffle_seed"] is None
    assert final["data_pipeline_worker_count"] == 0
    assert final["data_pipeline_worker_buffer_size"] == 1
    assert final["data_pipeline_prefetch"] is False
    assert final["data_pipeline_state_schema_version"] == 2
    assert final["data_document_aware"] is False
    assert final["data_document_count"] is None
    assert final["data_document_buffer_size"] is None
    assert final["data_document_refill_size"] is None
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
    assert events[-1]["final_optimizer_route_backend_counts"] == metrics[-1]["optimizer_route_backend_counts"]
    assert diagnostics["jax"]["backend"]
    assert diagnostics["packages"]["jaxtitan"]
    assert diagnostics["packages"]["grain"]
    assert diagnostics["model"]["remat"] == "none"
    assert diagnostics["optimizer"]["name"] == "adamw"
    assert diagnostics["optimizer"]["route_counts"] == {"adamw": diagnostics["model"]["parameter_leaves"]}
    assert diagnostics["data_pipeline"]["backend"] == "grain"
    assert diagnostics["data_pipeline"]["order"] == "sequential"
    assert diagnostics["data_pipeline"]["shuffle_seed"] is None
    assert diagnostics["data_pipeline"]["worker_count"] == 0
    assert diagnostics["data_pipeline"]["worker_buffer_size"] == 1
    assert diagnostics["data_pipeline"]["prefetch"] is False
    assert diagnostics["data_pipeline"]["state_schema_version"] == 2
    assert diagnostics["data_pipeline"]["document_aware"] is False
    assert diagnostics["data_pipeline"]["document_count"] is None
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


def test_run_training_writes_hf_streaming_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_hf_stream(monkeypatch, [{"text": "hello world " * 32}])
    config_path = tmp_path / "streaming.toml"
    config_path.write_text(_streaming_training_config(target_tokens=4))

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "streaming-loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    checkpoint_index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert summary.status == "completed"
    assert summary.tokens_seen == 4
    assert metrics[-1]["data_pipeline_backend"] == "hf_streaming"
    assert metrics[-1]["data_order"] == "sequential"
    assert metrics[-1]["token_start"] == 0
    assert metrics[-1]["token_end"] == 4
    assert metrics[-1]["target_tokens"] == 4
    assert diagnostics["data_pipeline"]["backend"] == "hf_streaming"
    assert diagnostics["data_pipeline"]["source"]["revision"] == "abc123"
    assert diagnostics["data_pipeline"]["exact_resume"] is True
    assert final["data_pipeline_backend"] == "hf_streaming"
    assert checkpoint_index["latest_checkpoint_path"] == "checkpoints/000001"
    assert manifest["data"]["mode"] == "hf_streaming"
    assert manifest["data"]["hf_streaming"]["dataset"] == "mock/dataset"


def test_run_training_resumes_hf_streaming_checkpoint_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    rows = [{"text": "alpha beta gamma delta epsilon " * 32}]
    _patch_hf_stream(monkeypatch, rows)
    config_path = tmp_path / "streaming.toml"
    config_path.write_text(_streaming_training_config(target_tokens=4, checkpoint_every_steps=1))
    first = run_training(config_path)
    config_path.write_text(_streaming_training_config(target_tokens=8, checkpoint_every_steps=1))

    resumed = run_training(config_path, resume=True)

    run_dir = tmp_path / "runs" / "streaming-loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    events = _jsonl(run_dir / "events.jsonl")
    assert first.steps == 1
    assert resumed.steps == 2
    assert resumed.tokens_seen == 8
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["token_start"] == 4
    assert metrics[-1]["token_end"] == 8
    resumed_event = next(event for event in events if event["type"] == "training_resumed")
    assert resumed_event["dataset_token_offset"] == 4
    assert resumed_event["runtime_fingerprint"]


def test_run_training_wandb_mirror_logs_metrics_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    manifest = prepared_dataset_factory(
        "wandb-loop",
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
            artifacts_block=_wandb_artifacts_block(),
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metadata = json.loads((run_dir / "diagnostics" / "wandb.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    logged_payloads = [payload for payload, _step in fake_wandb.logs]

    assert fake_wandb.init_calls[0]["project"] == "jaxtitan-test"
    assert fake_wandb.init_calls[0]["entity"] == "test-entity"
    assert fake_wandb.init_calls[0]["group"] == "unit"
    assert fake_wandb.init_calls[0]["tags"] == ["fake", "runtime"]
    assert fake_wandb.init_calls[0]["mode"] == "offline"
    assert fake_wandb.init_calls[0]["resume"] == "allow"
    assert metadata["wandb_run_id"] == fake_wandb.init_calls[0]["id"]
    assert metadata["project"] == "jaxtitan-test"
    assert diagnostics["wandb"]["wandb_run_id"] == metadata["wandb_run_id"]
    assert any(payload.get("event/training_started") == 1 for payload in logged_payloads)
    assert any("train/loss" in payload and "data/tokens_seen" in payload for payload in logged_payloads)
    assert any("eval/loss" in payload for payload in logged_payloads)
    assert any(payload.get("event/checkpoint_saved") == 1 for payload in logged_payloads)
    assert any("final/optimizer_groups" in payload for payload in logged_payloads)
    assert fake_wandb.finished is True


def test_run_training_wandb_failure_writes_local_event_and_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_wandb = _FakeWandbModule(fail_on_key="train/loss")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    manifest = prepared_dataset_factory(
        "wandb-failure",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            artifacts_block=_wandb_artifacts_block(),
        )
    )

    with pytest.raises(ContractError, match="W&B mirror failed during train_metrics"):
        run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    assert any(event["type"] == "wandb_failed" and event["phase"] == "train_metrics" for event in events)
    assert events[-1]["type"] == "training_failed"
    assert not (run_dir / "summaries" / "final.json").exists()


def test_run_training_wandb_resume_reuses_saved_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", first_wandb)
    manifest = prepared_dataset_factory(
        "wandb-resume",
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
            artifacts_block=_wandb_artifacts_block(),
        )
    )
    run_training(config_path)
    metadata = json.loads((tmp_path / "runs" / "loop" / "diagnostics" / "wandb.json").read_text())
    second_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", second_wandb)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            artifacts_block=_wandb_artifacts_block(),
        )
    )

    run_training(config_path, resume=True)

    assert second_wandb.init_calls[0]["id"] == metadata["wandb_run_id"]
    assert second_wandb.init_calls[0]["resume"] == "must"


def test_run_training_wandb_resume_requires_saved_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    manifest = prepared_dataset_factory(
        "wandb-missing-metadata",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            artifacts_block=_wandb_artifacts_block(),
        )
    )

    with pytest.raises(ContractError, match="diagnostics/wandb.json is missing"):
        run_training(config_path, resume=True)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert events[-1]["type"] == "wandb_failed"
    assert events[-1]["phase"] == "init"


def test_run_training_captures_configured_jax_profile_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_profiler = _patch_fake_jax_profiler(monkeypatch)
    manifest = prepared_dataset_factory(
        "profiled-loop",
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
            profiling_block=_profiling_block(trace_start_step=1, trace_steps=1),
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    profiling = json.loads((run_dir / "diagnostics" / "profiling.json").read_text())
    runtime = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert fake_profiler.start_calls == [
        {
            "log_dir": "runs/loop/profiles",
            "create_perfetto_link": False,
            "create_perfetto_trace": True,
        }
    ]
    assert fake_profiler.stop_count == 1
    assert fake_profiler.annotations == [
        "train_loop_step",
        "data",
        "placement",
        "train_step",
        "metrics_sync",
        "eval",
        "checkpoint",
    ]
    assert [event["type"] for event in events if event["type"].startswith("profiling_")] == [
        "profiling_trace_started",
        "profiling_trace_completed",
    ]
    assert profiling["status"] == "completed"
    assert profiling["traced_step_range"] == {"start": 1, "end": 1}
    assert profiling["trace_files"] == ["profiles/trace.trace.json.gz"]
    assert runtime["profiling"]["enabled"] is True
    assert runtime["profiling"]["trace_start_step"] == 1
    assert runtime["profiling"]["trace_steps"] == 1


def test_run_training_profiler_failure_writes_event_and_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_fake_jax_profiler(monkeypatch, fail_start=True)
    manifest = prepared_dataset_factory(
        "profile-failure",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            profiling_block=_profiling_block(trace_start_step=1, trace_steps=1),
        )
    )

    with pytest.raises(ContractError, match="JAX profiling failed during start"):
        run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    profiling = json.loads((run_dir / "diagnostics" / "profiling.json").read_text())
    assert events[-2]["type"] == "profiling_failed"
    assert events[-1]["type"] == "training_failed"
    assert profiling["status"] == "failed"
    assert profiling["error"]["phase"] == "start"
    assert not (run_dir / "summaries" / "final.json").exists()


def test_run_training_resume_rejects_past_profiling_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_profiler = _patch_fake_jax_profiler(monkeypatch)
    manifest = prepared_dataset_factory(
        "profile-resume",
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
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            profiling_block=_profiling_block(trace_start_step=1, trace_steps=1),
        )
    )

    with pytest.raises(ContractError, match="profiling window"):
        run_training(config_path, resume=True)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert fake_profiler.start_calls == []
    assert any(event["type"] == "profiling_failed" and event["phase"] == "resume" for event in events)
    assert events[-1]["type"] == "training_failed"


def test_run_training_records_muon_optimizer_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "muon-loop",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            optimizer_name="muon",
        )
    )

    run_training(config_path)

    diagnostics = json.loads((tmp_path / "runs" / "loop" / "diagnostics" / "runtime.json").read_text())
    assert diagnostics["optimizer"]["name"] == "muon"
    assert diagnostics["optimizer"]["route_counts"] == {"adamw": 7, "muon": 7}
    assert diagnostics["optimizer"]["muon"]["scale_mode"] == "match_rms_adamw"
    assert diagnostics["optimizer"]["distributed_policy"]["zero2_fsdp"] == "auto_dion2"
    assert diagnostics["optimizer"]["muon"]["newton_schulz_precision"] == "bfloat16"
    assert diagnostics["optimizer"]["muon"]["distributed_policy"] == "replicated_or_auto_dion2_when_sharded"
    assert diagnostics["optimizer"]["auto_routing"]["active"] is False


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


def test_run_training_accepts_fsdp_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "fsdp-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=65,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode="fsdp",
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    checkpoint_index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())
    assert summary.execution_mode == "fsdp"
    assert diagnostics["parallelism"]["mode"] == "fsdp"
    assert diagnostics["parallelism"]["execution_mode"] == "fsdp"
    assert diagnostics["sharding"]["model_state"]["mode"] == "fsdp"
    assert diagnostics["sharding"]["model_state"]["fsdp_sharded_leaves"] > 0
    assert diagnostics["compile"]["train"]["input_shardings"]["state"]["mode"] == "from_template"
    assert checkpoint_index["latest_step"] == 1
    assert metadata["compatibility"]["parallelism"]["mode"] == "fsdp"


def test_run_training_accepts_zero2_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "zero2-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=65,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode="zero2",
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    checkpoint_index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())
    assert summary.execution_mode == "zero2"
    assert diagnostics["parallelism"]["mode"] == "zero2"
    assert diagnostics["sharding"]["model_state"]["fsdp_sharded_leaves"] == 0
    assert diagnostics["sharding"]["optimizer_state"]["fsdp_sharded_leaves"] > 0
    assert diagnostics["sharding"]["gradients"]["fsdp_sharded_leaves"] > 0
    assert checkpoint_index["latest_step"] == 1
    assert metadata["compatibility"]["parallelism"]["mode"] == "zero2"


@pytest.mark.parametrize("mode", ["fsdp", "zero2"])
def test_run_training_auto_resolves_sharded_muon_to_dion2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
    mode: str,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        f"{mode}-muon-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=65,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            optimizer_name="muon",
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode=mode,
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())
    assert diagnostics["optimizer"]["name"] == "muon"
    assert diagnostics["optimizer"]["route_counts"] == {"adamw": 7, "dion2": 7}
    assert diagnostics["optimizer"]["auto_routing"]["active"] is True
    assert diagnostics["optimizer"]["dion2"]["orthogonalizer"] == "polar_express"
    assert metadata["compatibility"]["optimizer"]["policy"]["auto_routing"]["active"] is True
    assert metadata["compatibility"]["parallelism"]["mode"] == mode


def test_run_training_auto_resolves_tensor_parallel_muon_to_exact_distributed_muon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "tp-muon-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=65,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            optimizer_name="muon",
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            axis_names=("data", "tp"),
            axis_sizes=(2, 2),
            tensor_parallel=True,
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    row = _jsonl(run_dir / "metrics" / "train.jsonl")[-1]
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())

    assert diagnostics["optimizer"]["name"] == "muon"
    assert diagnostics["optimizer"]["route_counts"] == {"adamw": 7, "dist_muon_exact": 7}
    assert diagnostics["optimizer"]["auto_routing"]["active"] is True
    assert diagnostics["optimizer"]["dist_muon_exact"]["exact"] is True
    assert row["optimizer_route_backend_counts"] == {"adamw": 7, "dist_muon_exact": 7}
    assert final["final_optimizer_route_backend_counts"] == row["optimizer_route_backend_counts"]
    assert metadata["compatibility"]["optimizer"]["policy"]["auto_routing"] == {
        "active": True,
        "muon_sharded_matrix_backend": "dion2",
        "muon_tp_sharded_matrix_backend": "dist_muon_exact",
    }
    assert metadata["compatibility"]["parallelism"]["tensor_parallel"] is True
    assert metadata["compatibility"]["parallelism"]["tensor_parallel_policy"]["optimizer"] == "muon_routes_to_dist_muon_exact"


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
    assert metrics[0]["microbatch_loss_max"] >= metrics[0]["microbatch_loss_mean"]
    assert metrics[0]["batch_het"] == pytest.approx(metrics[0]["microbatch_loss_max"] - metrics[0]["microbatch_loss_mean"])
    assert metrics[-1]["tokens_seen"] == 32
    assert final["gradient_accumulation_steps"] == 2
    assert final["effective_global_batch_size"] == 4
    assert final["effective_tokens_per_step"] == 16
    assert diagnostics["mesh"]["gradient_accumulation_steps"] == 2
    assert diagnostics["mesh"]["effective_global_batch_size"] == 4
    assert diagnostics["compile"]["train"]["expected_batch_shape"] == [2, 2, 4]
    assert diagnostics["parallelism"]["batch"]["gradient_accumulation_steps"] == 2
    assert diagnostics["parallelism"]["batch"]["effective_tokens_per_step"] == 16


def test_run_training_with_shuffle_records_loader_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "shuffle-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=60,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            data_order="shuffle",
            shuffle_seed=123,
            prefetch=True,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert summary.steps == 2
    assert metrics[-1]["data_order"] == "shuffle"
    assert metrics[-1]["data_prefetch"] is True
    assert metrics[-1]["data_pipeline_backend"] == "grain"
    assert diagnostics["data_pipeline"]["order"] == "shuffle"
    assert diagnostics["data_pipeline"]["shuffle_seed"] == 123
    assert diagnostics["data_pipeline"]["prefetch"] is True
    assert final["data_pipeline_order"] == "shuffle"
    assert final["data_pipeline_shuffle_seed"] == 123
    assert final["data_pipeline_prefetch"] is True


def test_run_training_with_document_buffer_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "document-buffer-loop",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=48,
        document_offsets=(0, 3, 6, 9, 12, 20, 32, 48, 80),
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=12,
            log_every_steps=1,
            data_order="document_buffer",
            shuffle_seed=123,
            document_buffer_size=3,
            document_refill_size=2,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert summary.data_pipeline_order == "document_buffer"
    assert summary.data_document_buffer_size == 3
    assert summary.data_document_refill_size == 2
    assert metrics[-1]["data_order"] == "document_buffer"
    assert metrics[-1]["document_aware"] is True
    assert metrics[-1]["documents_touched"] >= 1
    assert metrics[-1]["token_count"] <= metrics[-1]["target_tokens"]
    assert diagnostics["data_pipeline"]["order"] == "document_buffer"
    assert diagnostics["data_pipeline"]["document_buffer_size"] == 3
    assert diagnostics["data_pipeline"]["document_refill_size"] == 2
    assert final["data_pipeline_order"] == "document_buffer"
    assert final["data_document_buffer_size"] == 3
    assert final["data_document_refill_size"] == 2
    assert final["final_batch_het"] == pytest.approx(metrics[-1]["batch_het"])


def test_run_training_records_document_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "documents-loop",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
        document_offsets=(0, 6, 12, 25, 50),
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=1, eval_every_steps=1))

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    eval_metrics = _jsonl(run_dir / "metrics" / "eval.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())

    assert summary.data_document_aware is True
    assert summary.data_document_count == 4
    assert metrics[0]["document_aware"] is True
    assert metrics[0]["documents_touched"] == 1
    assert metrics[0]["document_min"] == 0
    assert metrics[0]["document_max"] == 0
    assert metrics[-1]["documents_touched"] == 2
    assert metrics[-1]["document_min"] == 1
    assert metrics[-1]["document_max"] == 2
    assert eval_metrics[-1]["document_aware"] is True
    assert eval_metrics[-1]["documents_touched"] == 1
    assert eval_metrics[-1]["document_min"] == 3
    assert eval_metrics[-1]["document_max"] == 3
    assert diagnostics["data_pipeline"]["document_aware"] is True
    assert diagnostics["data_pipeline"]["document_count"] == 4
    assert diagnostics["data_pipeline"]["document_offsets_path"] == "document_offsets.u64"
    assert final["data_document_aware"] is True
    assert final["data_document_count"] == 4
    assert final["data_document_offsets_path"] == "document_offsets.u64"


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


def test_run_training_writes_moe_router_layer_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "moe-router-diagnostics",
        shard_token_groups=(tuple(range(0, 40)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            model_name="trinity",
            num_layers=2,
            trinity_moe_balance_name="none",
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())
    row = metrics[-1]
    layers = row["moe_router_layers"]

    assert summary.final_moe_router_layers == layers
    assert len(layers) == 1
    assert layers[0]["layer_index"] == 0
    assert layers[0]["total_assignments"] == 2 * 4 * 2
    assert sum(layers[0]["expert_counts"]) == 2 * 4 * 2
    assert sum(layers[0]["importance"]) == pytest.approx(2 * 4)
    assert layers[0]["experts_active"] <= 3
    assert layers[0]["load_p10"] <= layers[0]["load_p50"] <= layers[0]["load_p90"]
    assert layers[0]["importance_p10"] <= layers[0]["importance_p50"] <= layers[0]["importance_p90"]
    assert row["router_dead_experts_count"] >= 0
    assert row["router_mean_load_cv"] == pytest.approx(layers[0]["load_cv"])
    assert row["router_mean_importance_entropy"] == pytest.approx(layers[0]["importance_entropy"])
    assert row["smebu_bias_norm"] is None
    assert row["smebu_momentum_norm"] is None
    optimizer_groups = row["optimizer_groups"]
    tags = {group["tag"] for group in optimizer_groups}
    assert {"moe_router", "moe_expert_bias", "moe_gate", "moe_up", "moe_down"}.issubset(tags)
    assert row["optimizer_route_backend_counts"]["adamw"] == sum(group["leaf_count"] for group in optimizer_groups)
    assert final["final_moe_router_layers"] == layers
    assert final["final_router_mean_load_cv"] == row["router_mean_load_cv"]
    assert final["final_router_dead_experts_count"] == row["router_dead_experts_count"]
    assert final["final_router_mean_importance_cv"] == row["router_mean_importance_cv"]
    assert final["final_optimizer_groups"] == optimizer_groups
    assert final["final_optimizer_route_backend_counts"] == row["optimizer_route_backend_counts"]


def test_run_training_records_trinity_moe_per_expert_muon_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "moe-muon-routes",
        shard_token_groups=(tuple(range(0, 40)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            model_name="trinity",
            num_layers=2,
            optimizer_name="muon",
            trinity_moe_balance_name="none",
        )
    )

    run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    row = _jsonl(run_dir / "metrics" / "train.jsonl")[-1]
    routes = {(route["tag"], route["backend"]) for route in diagnostics["optimizer"]["routes"]}
    groups = {(group["tag"], group["backend"]) for group in row["optimizer_groups"]}

    for tag in ("moe_gate", "moe_up", "moe_down"):
        assert (tag, "muon") in routes
        assert (tag, "muon") in groups
    assert diagnostics["optimizer"]["muon"]["rank3_expert_policy"] == "per_expert_full_matrix_when_complete_local"
    assert diagnostics["optimizer"]["route_counts"]["muon"] > 0
    assert row["optimizer_route_backend_counts"]["muon"] > 0


def test_run_training_accepts_expert_parallel_moe_muon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "moe-ep-muon",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=50,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            model_name="trinity",
            num_layers=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            optimizer_name="muon",
            axis_names=("data", "ep"),
            axis_sizes=(1, 4),
            expert_parallel=True,
            trinity_moe_balance_name="none",
            trinity_moe_num_experts=4,
            checkpoint_every_steps=1,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    row = _jsonl(run_dir / "metrics" / "train.jsonl")[-1]
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())

    assert summary.execution_mode == "replicated_data_parallel+ep"
    assert diagnostics["parallelism"]["expert_parallel"] is True
    assert diagnostics["parallelism"]["expert_parallel_policy"]["dispatcher_backend"] == "all_to_all"
    assert diagnostics["parallelism"]["expert_parallel_policy"]["capacity_policy"] == "strict_dropless_static_source_buckets"
    assert diagnostics["parallelism"]["mesh"]["ep_axis_size"] == 4
    assert diagnostics["sharding"]["model_state"]["ep_sharded_leaves"] == 3
    assert diagnostics["sharding"]["optimizer_state"]["ep_sharded_leaves"] == 3
    assert diagnostics["optimizer"]["route_counts"]["muon"] > 0
    assert row["optimizer_route_backend_counts"]["muon"] > 0
    assert metadata["compatibility"]["parallelism"]["expert_parallel"] is True
    assert metadata["compatibility"]["parallelism"]["expert_parallel_policy"]["dispatcher_backend"] == "all_to_all"


def test_run_training_accepts_folded_fsdp_expert_parallel_moe_muon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "folded-moe-ep-muon",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=50,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            model_name="trinity",
            num_layers=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            optimizer_name="muon",
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode="fsdp",
            expert_parallel=True,
            trinity_moe_balance_name="none",
            trinity_moe_num_experts=4,
            checkpoint_every_steps=1,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    row = _jsonl(run_dir / "metrics" / "train.jsonl")[-1]
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())

    assert summary.execution_mode == "fsdp+ep"
    assert diagnostics["parallelism"]["expert_parallel_policy"]["axis"] == "fsdp"
    assert diagnostics["parallelism"]["expert_parallel_policy"]["axis_sharing"] == "shared_with_fsdp"
    assert diagnostics["sharding"]["model_state"]["expert_parallel_axis"] == "fsdp"
    assert diagnostics["sharding"]["model_state"]["ep_sharded_leaves"] == 3
    assert diagnostics["optimizer"]["route_counts"]["dion2"] > 0
    assert diagnostics["optimizer"]["route_counts"]["muon"] > 0
    assert row["optimizer_route_backend_counts"]["dion2"] > 0
    assert row["optimizer_route_backend_counts"]["muon"] > 0
    assert metadata["compatibility"]["parallelism"]["expert_parallel_policy"]["axis"] == "fsdp"
    assert metadata["compatibility"]["parallelism"]["expert_parallel_policy"]["axis_sharing"] == "shared_with_fsdp"


def test_run_training_accepts_expert_region_fsdp_moe_adamw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "expert-fsdp-runtime",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=50,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            model_name="trinity",
            num_layers=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            optimizer_name="adamw",
            axis_names=("data", "fsdp", "ep", "expert_fsdp"),
            axis_sizes=(1, 1, 2, 2),
            parallelism_mode="fsdp",
            expert_parallel=True,
            trinity_moe_balance_name="none",
            trinity_moe_num_experts=4,
            checkpoint_every_steps=1,
        )
    )

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    diagnostics = json.loads((run_dir / "diagnostics" / "runtime.json").read_text())
    metadata = json.loads((run_dir / "checkpoints" / "000001" / "metadata" / "metadata").read_text())

    assert summary.execution_mode == "fsdp+ep"
    assert diagnostics["parallelism"]["expert_parallel_policy"]["expert_fsdp_axis"] == "expert_fsdp"
    assert diagnostics["parallelism"]["expert_parallel_policy"]["expert_fsdp_axis_size"] == 2
    assert diagnostics["parallelism"]["mesh"]["expert_parallel_axis"] == "ep"
    assert diagnostics["parallelism"]["mesh"]["expert_fsdp_axis"] == "expert_fsdp"
    assert diagnostics["sharding"]["model_state"]["expert_fsdp_sharded_leaves"] == 3
    assert diagnostics["optimizer"]["route_counts"]["adamw"] > 0
    assert metadata["compatibility"]["parallelism"]["expert_fsdp_policy"]["axis"] == "expert_fsdp"
    assert metadata["compatibility"]["parallelism"]["expert_fsdp_policy"]["axis_size"] == 2


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
        ({"data_order": "shuffle", "shuffle_seed": 1}, r"compatibility\.data\.training_pipeline\.order"),
        ({"worker_count": 1}, r"compatibility\.data\.training_pipeline\.worker_count"),
        ({"worker_buffer_size": 2}, r"compatibility\.data\.training_pipeline\.worker_buffer_size"),
        ({"prefetch": True}, r"compatibility\.data\.training_pipeline\.prefetch"),
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
    assert "JAX TITAN TRAINING" in result.stdout
    assert "step: 0      | compiling train step..." in result.stdout
    assert "step: 1" in result.stdout
    assert "loss:" in result.stdout
    assert "grad_norm:" in result.stdout
    assert "mfu:" in result.stdout
    assert "lr:" in result.stdout
    assert "tps:" in result.stdout
    assert "total_time:" in result.stdout
    assert "batch_het=" not in result.stdout
    assert "docs=" not in result.stdout
    assert "run_dir | runs/loop" in result.stdout
    assert (tmp_path / "runs" / "loop" / "metrics" / "train.jsonl").is_file()
    assert (tmp_path / "runs" / "loop" / "summaries" / "final.json").is_file()


def test_cli_run_train_existing_dir_fails_without_overwrite(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = prepared_dataset_factory(
        "cli-existing",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, log_every_steps=1))
    run_dir = tmp_path / "runs" / "loop"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "keep.txt"
    sentinel.write_text("keep")

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "run directory already exists" in result.stderr
    assert "--overwrite" in result.stderr
    assert sentinel.read_text() == "keep"
    assert "Traceback" not in result.stderr


def test_cli_run_train_overwrite_replaces_existing_dir(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = prepared_dataset_factory(
        "cli-overwrite",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, log_every_steps=1))
    run_dir = tmp_path / "runs" / "loop"
    run_dir.mkdir(parents=True)
    (run_dir / "old.txt").write_text("old")

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", "--overwrite", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "JAX TITAN TRAINING" in result.stdout
    assert not (run_dir / "old.txt").exists()
    assert (run_dir / "metrics" / "train.jsonl").is_file()
    assert (run_dir / "summaries" / "final.json").is_file()


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
    assert "JAX TITAN TRAINING" in resumed.stdout
    assert "resume: true" in resumed.stdout
    assert "resumed:" in resumed.stdout
    assert "run_dir | runs/loop" in resumed.stdout
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
    model_name: str = "decoder",
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_layers: int = 1,
    num_heads: int = 2,
    n_kv_heads: int = 1,
    remat: str = "none",
    optimizer_name: str = "adamw",
    weight_decay: float = 0.0,
    schedule_name: str = "constant",
    total_steps: int | None = None,
    tokenizer_id: str = "toy-tokenizer",
    seq_len: int = 4,
    global_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    axis_names: tuple[str, ...] = ("data",),
    axis_sizes: tuple[int, ...] = (1,),
    parallelism_mode: str = "ddp",
    validation_manifest: Path | str | None = None,
    eval_every_steps: int | None = None,
    eval_num_batches: int = 1,
    eval_name: str = "validation",
    second_eval: bool = False,
    data_order: str = "sequential",
    shuffle_seed: int | None = None,
    worker_count: int = 0,
    worker_buffer_size: int = 1,
    prefetch: bool = False,
    document_buffer_size: int | None = None,
    document_refill_size: int | None = None,
    trinity_moe_balance_name: str | None = None,
    trinity_moe_num_experts: int = 3,
    expert_parallel: bool = False,
    tensor_parallel: bool = False,
    artifacts_block: str = "",
    profiling_block: str = "",
) -> str:
    total_steps_line = "" if total_steps is None else f"total_steps = {total_steps}\n"
    validation_manifest_line = (
        "" if validation_manifest is None else f'validation_manifest = "{Path(validation_manifest).as_posix()}"\n'
    )
    shuffle_seed_line = "" if shuffle_seed is None else f"shuffle_seed = {shuffle_seed}\n"
    document_buffer_size_line = "" if document_buffer_size is None else f"document_buffer_size = {document_buffer_size}\n"
    document_refill_size_line = "" if document_refill_size is None else f"document_refill_size = {document_refill_size}\n"
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
    trinity_block = ""
    if model_name == "trinity":
        balance_name = "none" if trinity_moe_balance_name is None else trinity_moe_balance_name
        trinity_block = f"""
[model.trinity]
initial_dense_layers = 1
local_window = {seq_len}
local_layers_per_global = 1

[model.trinity.moe]
num_experts = {trinity_moe_num_experts}
top_k = 2

[model.trinity.moe.balance]
name = "{balance_name}"
"""
    return f"""
[run]
id = "loop"
seed = {seed}
output_dir = "runs"

[model]
name = "{model_name}"
variant = "tiny"
vocab_size = 64
hidden_size = {hidden_size}
intermediate_size = {intermediate_size}
num_layers = {num_layers}
num_heads = {num_heads}
n_kv_heads = {n_kv_heads}
max_seq_len = 4
compute_dtype = "float32"
remat = "{remat}"
{trinity_block}

[optimizer]
name = "{optimizer_name}"
weight_decay = {weight_decay}

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = 0.001
{total_steps_line}

[data]
train_manifest = "{Path(train_manifest).as_posix()}"
tokenizer_id = "{tokenizer_id}"
{validation_manifest_line}
order = "{data_order}"
{shuffle_seed_line}worker_count = {worker_count}
worker_buffer_size = {worker_buffer_size}
prefetch = {str(prefetch).lower()}
{document_buffer_size_line}{document_refill_size_line}

[training]
seq_len = {seq_len}
global_batch_size = {global_batch_size}
gradient_accumulation_steps = {gradient_accumulation_steps}
target_tokens = {target_tokens}
log_every_steps = {log_every_steps}
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = [{", ".join(f'"{name}"' for name in axis_names)}]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]

[parallelism]
mode = "{parallelism_mode}"
expert_parallel = {str(expert_parallel).lower()}
tensor_parallel = {str(tensor_parallel).lower()}
{artifacts_block}
{profiling_block}
{eval_block}
"""


class _FakeHFIterable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.index = 0

    def __iter__(self) -> "_FakeHFIterable":
        return self

    def __next__(self) -> dict[str, object]:
        if self.index >= len(self.rows):
            raise StopIteration
        row = self.rows[self.index]
        self.index += 1
        return row

    def state_dict(self) -> dict[str, int]:
        return {"index": self.index}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.index = int(state.get("index", 0))


def _patch_hf_stream(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    monkeypatch.setattr("jaxtitan.data.streaming._load_hf_dataset", lambda _source: _FakeHFIterable(list(rows)))


def _streaming_training_config(*, target_tokens: int, checkpoint_every_steps: int = 10) -> str:
    return f"""
[run]
id = "streaming-loop"
seed = 7
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 50257
hidden_size = 8
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"

[optimizer]
name = "adamw"
weight_decay = 0.0

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
mode = "hf_streaming"
tokenizer_id = "gpt2"
order = "sequential"

[data.hf_streaming]
dataset = "mock/dataset"
split = "train"
revision = "abc123"
text_column = "text"
append_eot = true

[training]
seq_len = 4
global_batch_size = 1
target_tokens = {target_tokens}
log_every_steps = 1
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = ["data"]
axis_sizes = [1]

[parallelism]
mode = "ddp"
"""


def _wandb_artifacts_block() -> str:
    return """
[artifacts]
wandb_enabled = true
wandb_project = "jaxtitan-test"
wandb_entity = "test-entity"
wandb_group = "unit"
wandb_tags = ["fake", "runtime"]
wandb_mode = "offline"
"""


def _profiling_block(*, trace_start_step: int, trace_steps: int) -> str:
    return f"""
[profiling]
enabled = true
trace_start_step = {trace_start_step}
trace_steps = {trace_steps}
create_perfetto_trace = true
create_perfetto_link = false
"""


class _FakeJaxProfiler:
    def __init__(self, *, fail_start: bool = False, fail_stop: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.start_calls: list[dict[str, object]] = []
        self.stop_count = 0
        self.annotations: list[str] = []
        self._last_log_dir: Path | None = None

    def start_trace(
        self,
        log_dir: str,
        *,
        create_perfetto_link: bool = False,
        create_perfetto_trace: bool = False,
        **_kwargs: object,
    ) -> None:
        if self.fail_start:
            raise RuntimeError("fake start failure")
        self.start_calls.append(
            {
                "log_dir": log_dir,
                "create_perfetto_link": create_perfetto_link,
                "create_perfetto_trace": create_perfetto_trace,
            }
        )
        self._last_log_dir = Path(log_dir)

    def stop_trace(self) -> None:
        if self.fail_stop:
            raise RuntimeError("fake stop failure")
        self.stop_count += 1
        if self._last_log_dir is not None:
            self._last_log_dir.mkdir(parents=True, exist_ok=True)
            (self._last_log_dir / "trace.trace.json.gz").write_bytes(b"fake trace")

    def trace_annotation(self, name: str):
        parent = self

        class _Annotation:
            def __enter__(self) -> None:
                parent.annotations.append(name)

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        return _Annotation()


def _patch_fake_jax_profiler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_start: bool = False,
    fail_stop: bool = False,
) -> _FakeJaxProfiler:
    fake = _FakeJaxProfiler(fail_start=fail_start, fail_stop=fail_stop)
    monkeypatch.setattr("jaxtitan.runtime.profiling.jax.profiler.start_trace", fake.start_trace)
    monkeypatch.setattr("jaxtitan.runtime.profiling.jax.profiler.stop_trace", fake.stop_trace)
    monkeypatch.setattr("jaxtitan.runtime.profiling.jax.profiler.TraceAnnotation", fake.trace_annotation)
    return fake


class _FakeWandbRun:
    def __init__(self, parent: "_FakeWandbModule", run_id: str) -> None:
        self.parent = parent
        self.id = run_id
        self.name = f"run-{run_id}"
        self.url = f"https://wandb.test/{run_id}"
        self.summary: dict[str, object] = {}

    def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
        if self.parent.fail_on_key is not None and self.parent.fail_on_key in payload:
            raise RuntimeError(f"fake W&B failure for {self.parent.fail_on_key}")
        self.parent.logs.append((dict(payload), step))

    def finish(self) -> None:
        self.parent.finished = True


class _FakeWandbModule:
    def __init__(self, *, fail_on_key: str | None = None) -> None:
        self.fail_on_key = fail_on_key
        self.init_calls: list[dict[str, object]] = []
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.finished = False

    def init(self, **kwargs: object) -> _FakeWandbRun:
        self.init_calls.append(dict(kwargs))
        return _FakeWandbRun(self, str(kwargs["id"]))

    class Table:
        def __init__(self, *, columns, data) -> None:
            self.columns = columns
            self.data = data


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
