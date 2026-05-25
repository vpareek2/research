import json
from pathlib import Path

import jax
import pytest

from jaxtitan.errors import ConfigError, ContractError
from jaxtitan.runtime.preflight import format_preflight_report, preflight_report_to_json, run_preflight

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_run_preflight_validates_full_runtime_path_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("preflight", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, eval_every_steps=1, eval_num_batches=2))

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)
    decoded = json.loads(preflight_report_to_json(report))

    assert payload["status"] == "passed"
    assert payload["run_id"] == "loop"
    assert payload["run_dir"] == "runs/loop"
    assert payload["data"]["train_split_tokens"] == 25
    assert payload["data"]["pipeline"]["backend"] == "grain"
    assert payload["data"]["pipeline"]["order"] == "sequential"
    assert payload["data"]["pipeline"]["shuffle_seed"] is None
    assert payload["data"]["pipeline"]["worker_count"] == 0
    assert payload["data"]["pipeline"]["worker_buffer_size"] == 1
    assert payload["data"]["pipeline"]["prefetch"] is False
    assert payload["data"]["pipeline"]["document_aware"] is False
    assert payload["data"]["pipeline"]["document_count"] is None
    assert payload["data"]["first_batch"]["target_tokens"] == 8
    assert payload["data"]["first_batch"]["document_aware"] is False
    assert payload["data"]["first_batch"]["documents_touched"] is None
    assert payload["devices"]["selected_device_count"] == 1
    assert payload["mesh"]["axis_names"] == ["data"]
    assert payload["model"]["parameters"] > 0
    assert payload["model"]["remat"] == "none"
    assert payload["optimizer"]["name"] == "adamw"
    assert payload["training"]["estimated_steps"] == 2
    assert payload["training"]["compile"] == "passed"
    assert payload["training"]["first_step_train_tokens_per_sec"] > 0.0
    assert "first_step_mfu" in payload["training"]
    assert payload["eval"]["name"] == "validation"
    assert payload["eval"]["num_batches"] == 2
    assert payload["eval"]["compile"] == "passed"
    assert payload["diagnostics"]["run_id"] == "loop"
    assert payload["diagnostics"]["data_pipeline"]["backend"] == "grain"
    assert payload["diagnostics"]["data_pipeline"]["worker_buffer_size"] == 1
    assert payload["diagnostics"]["packages"]["grain"]
    assert payload["diagnostics"]["model"]["remat"] == "none"
    assert payload["diagnostics"]["optimizer"]["name"] == "adamw"
    assert payload["diagnostics"]["optimizer"]["route_counts"] == {"adamw": payload["model"]["parameter_leaves"]}
    assert payload["diagnostics"]["jax"]["backend"]
    assert payload["compile"]["train"]["donate_state"] is True
    assert payload["compile"]["train"]["expected_batch_shape"] == [1, 2, 4]
    assert payload["compile"]["train"]["input_shardings"]["input_ids"]["partition_spec"] == "PartitionSpec(None, 'data', None)"
    assert payload["compile"]["eval"]["donate_state"] is False
    assert payload["compile"]["eval"]["expected_batch_shape"] == [2, 4]
    assert payload["compile"]["eval"]["input_shardings"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"
    assert payload["diagnostics"]["compile"] == payload["compile"]
    assert payload["profiling"]["enabled"] is False
    assert payload["diagnostics"]["profiling"] == payload["profiling"]
    assert payload["kernels"]["enabled"] is False
    assert payload["kernels"]["active_count"] == 0
    assert payload["kernels"]["fallback"]["rmsnorm"] == "kernels_disabled"
    assert payload["diagnostics"]["kernels"] == payload["kernels"]
    assert payload["parallelism"]["execution_mode"] == "replicated_data_parallel"
    assert payload["parallelism"]["metrics_scope"] == "global"
    assert payload["parallelism"]["artifact_writer"] == "single_host"
    assert payload["sharding"]["batch"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"
    assert payload["sharding"]["metrics"]["partition_spec"] == "PartitionSpec()"
    assert payload["observed_sharding"]["first_train_batch"]["input_ids"]["global_shape"] == [1, 2, 4]
    assert payload["observed_sharding"]["train_state"]["replicated_leaf_count"] > 0
    assert payload["diagnostics"]["performance"]["flops_per_token"] > 0
    assert payload["diagnostics"]["device_telemetry"]["device_memory_used_bytes"] is None or isinstance(
        payload["diagnostics"]["device_telemetry"]["device_memory_used_bytes"], int
    )
    assert decoded == payload
    assert "preflight: passed" in text
    assert "mesh:" in text
    assert "parallelism:" in text
    assert "mode=replicated_data_parallel" in text
    assert "devices:" in text
    assert "runtime:" in text
    assert "pipeline=grain" in text
    assert "documents=False" in text
    assert "compile=passed" in text
    assert "compile contract:" in text
    assert "donate_train=True" in text
    assert "profiling: enabled=False" in text
    assert "kernels: enabled=False strict=False compile=lazy active=0" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_document_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "documents",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
        document_offsets=(0, 6, 12, 25, 50),
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, eval_every_steps=1, eval_num_batches=1))

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["data"]["pipeline"]["document_aware"] is True
    assert payload["data"]["pipeline"]["document_count"] == 4
    assert payload["data"]["pipeline"]["document_offsets_path"] == "document_offsets.u64"
    assert payload["data"]["first_batch"]["document_aware"] is True
    assert payload["data"]["first_batch"]["documents_touched"] == 1
    assert payload["observed_sharding"]["first_train_batch"]["doc_ids"]["global_shape"] == [1, 2]
    assert payload["eval"]["document_aware"] is True
    assert payload["eval"]["documents_touched"] == 1
    assert payload["diagnostics"]["data_pipeline"]["document_aware"] is True
    assert "documents=True count=4" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_block_remat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("remat", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, remat="block"))

    report = run_preflight(config_path)

    assert report.payload["model"]["remat"] == "block"
    assert report.payload["diagnostics"]["model"]["remat"] == "block"
    assert report.payload["training"]["compile"] == "passed"
    assert "remat=block" in format_preflight_report(report)
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_accepts_dense_trinity_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("trinity-preflight", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "trinity.toml"
    config_path.write_text(_preflight_config(manifest, model_name="trinity", num_layers=2, trinity=True))

    report = run_preflight(config_path)

    assert report.payload["status"] == "passed"
    assert report.payload["model"]["name"] == "trinity"
    assert report.payload["training"]["compile"] == "passed"
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_accepts_muon_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("muon", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, optimizer_name="muon", adamw_fallback_peak_lr=0.001))

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["optimizer"]["name"] == "muon"
    assert payload["optimizer"]["peak_lr"] == 0.02
    assert payload["optimizer"]["adamw_fallback_peak_lr"] == 0.001
    assert payload["optimizer"]["policy"]["adamw_fallback_schedule"]["peak_lr"] == 0.001
    assert payload["optimizer"]["policy"]["route_counts"] == {"adamw": 7, "muon": 7}
    assert payload["optimizer"]["policy"]["distributed_policy"]["zero2_fsdp"] == "auto_dion2"
    assert payload["optimizer"]["policy"]["muon"]["newton_schulz_precision"] == "bfloat16"
    assert payload["optimizer"]["policy"]["muon"]["distributed_policy"] == "replicated_or_auto_dion2_when_sharded"
    assert payload["optimizer"]["policy"]["auto_routing"]["active"] is False
    assert payload["optimizer"]["policy"]["fallback_counts"] == {
        "embedding": 1,
        "lm_head": 1,
        "norm": 5,
    }
    assert payload["diagnostics"]["optimizer"] == payload["optimizer"]["policy"]
    assert "adamw_fallback_peak_lr=0.001" in text
    assert "routes={'adamw': 7, 'muon': 7}" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_profiling_config_without_collecting_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("profiled-preflight", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            profiling_block="""
[profiling]
enabled = true
trace_start_step = 2
trace_steps = 1
create_perfetto_trace = true
create_perfetto_link = false
""",
        )
    )

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["profiling"] == {
        "schema_version": 1,
        "enabled": True,
        "trace_start_step": 2,
        "trace_steps": 1,
        "trace_end_step": 2,
        "create_perfetto_trace": True,
        "create_perfetto_link": False,
        "trace_dir": "profiles",
    }
    assert payload["diagnostics"]["profiling"] == payload["profiling"]
    assert "profiling: enabled=True start=2 steps=1 perfetto_trace=True" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_strict_unavailable_kernels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("strict-kernels", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            profiling_block="""
[kernels]
enabled = true
strict = true
""",
        )
    )

    with pytest.raises(ContractError, match="kernels.strict=true"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


@pytest.mark.parametrize("mode", ["fsdp", "zero2"])
def test_run_preflight_auto_resolves_sharded_muon_to_dion2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
    mode: str,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("muon-sharded", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            optimizer_name="muon",
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode=mode,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            target_tokens=16,
        )
    )

    report = run_preflight(config_path)
    payload = report.payload

    assert payload["optimizer"]["name"] == "muon"
    assert payload["optimizer"]["policy"]["route_counts"] == {"adamw": 7, "dion2": 7}
    assert payload["optimizer"]["policy"]["auto_routing"] == {
        "active": True,
        "muon_sharded_matrix_backend": "dion2",
    }
    assert payload["optimizer"]["policy"]["dion2"]["fraction"] == 0.25
    assert {route["backend"] for route in payload["optimizer"]["policy"]["routes"]} == {"adamw", "dion2"}
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_accepts_four_device_data_axis_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("dp-preflight", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, axis_sizes=(4,), global_batch_size=8, target_tokens=32))

    payload = run_preflight(config_path).payload

    assert payload["devices"]["selected_device_count"] == 4
    assert payload["devices"]["global_device_count"] >= 4
    assert payload["devices"]["process_count"] == 1
    assert payload["devices"]["single_process"] is True
    assert payload["mesh"]["data_axis_size"] == 4
    assert payload["training"]["global_batch_size"] == 8
    assert payload["training"]["per_device_batch_size"] == 2
    assert payload["training"]["per_device_target_tokens"] == 8
    assert payload["diagnostics"]["mesh"]["per_device_batch_size"] == 2
    assert payload["observed_sharding"]["first_train_batch"]["input_ids"]["unique_addressable_shard_shapes"] == [[1, 2, 4]]
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_accepts_fsdp_parallelism_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("fsdp-preflight", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode="fsdp",
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            target_tokens=16,
        )
    )

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["parallelism"]["mode"] == "fsdp"
    assert payload["parallelism"]["execution_mode"] == "fsdp"
    assert payload["mesh"]["fsdp_axis_size"] == 4
    assert payload["diagnostics"]["parallelism"]["mode"] == "fsdp"
    assert payload["sharding"]["model_state"]["mode"] == "fsdp"
    assert payload["sharding"]["model_state"]["fsdp_sharded_leaves"] > 0
    assert payload["sharding"]["model_state"]["replicated_leaves"] > 0
    assert payload["compile"]["train"]["input_shardings"]["state"]["mode"] == "from_template"
    assert payload["training"]["compile"] == "passed"
    assert "mode=fsdp" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_accepts_zero2_parallelism_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("zero2-preflight", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            axis_names=("data", "fsdp"),
            axis_sizes=(1, 4),
            parallelism_mode="zero2",
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            n_kv_heads=4,
            global_batch_size=4,
            target_tokens=16,
        )
    )

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["parallelism"]["mode"] == "zero2"
    assert payload["parallelism"]["execution_mode"] == "zero2"
    assert payload["sharding"]["model_state"]["mode"] == "zero2"
    assert payload["sharding"]["model_state"]["fsdp_sharded_leaves"] == 0
    assert payload["sharding"]["optimizer_state"]["fsdp_sharded_leaves"] > 0
    assert payload["sharding"]["gradients"]["fsdp_sharded_leaves"] > 0
    assert payload["training"]["compile"] == "passed"
    assert "mode=zero2" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_expert_parallel_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("ep-preflight", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            optimizer_name="muon",
            axis_names=("data", "ep"),
            axis_sizes=(1, 4),
            global_batch_size=4,
            target_tokens=16,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=4,
            n_kv_heads=4,
            model_name="trinity",
            trinity=True,
            trinity_moe=True,
            expert_parallel=True,
        )
    )

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["parallelism"]["mode"] == "ddp"
    assert payload["parallelism"]["expert_parallel"] is True
    assert payload["parallelism"]["expert_parallel_policy"] == {
        "enabled": True,
        "axis": "ep",
        "axis_size": 4,
        "num_experts": 4,
        "experts_per_rank": 1,
        "dispatcher_backend": "all_to_all",
        "capacity_policy": "strict_dropless_static_source_buckets",
        "token_partition": "assignment_index_mod_ep",
        "combine_policy": "reverse_all_to_all_then_psum",
    }
    assert payload["parallelism"]["execution_mode"] == "replicated_data_parallel+ep"
    assert payload["mesh"]["ep_axis_size"] == 4
    assert payload["sharding"]["model_state"]["ep_sharded_leaves"] == 3
    assert payload["sharding"]["optimizer_state"]["ep_sharded_leaves"] == 3
    assert payload["sharding"]["reserved"]["ep"]["dispatcher_backend"] == "all_to_all"
    assert payload["optimizer"]["policy"]["route_counts"]["muon"] >= 3
    assert "mode=replicated_data_parallel+ep" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_gradient_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("accum", shard_token_groups=(tuple(range(0, 50)),), train_tokens=35)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, target_tokens=32, gradient_accumulation_steps=2))

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)

    assert payload["data"]["first_batch"]["target_tokens"] == 16
    assert payload["training"]["estimated_steps"] == 2
    assert payload["training"]["gradient_accumulation_steps"] == 2
    assert payload["training"]["micro_global_batch_size"] == 2
    assert payload["training"]["effective_global_batch_size"] == 4
    assert payload["training"]["micro_tokens_per_step"] == 8
    assert payload["training"]["effective_tokens_per_step"] == 16
    assert payload["compile"]["train"]["expected_batch_shape"] == [2, 2, 4]
    assert payload["parallelism"]["batch"]["gradient_accumulation_steps"] == 2
    assert payload["observed_sharding"]["first_train_batch"]["input_ids"]["global_shape"] == [2, 2, 4]
    assert "grad_accum=2" in text
    assert "effective_batch=4" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_shuffle_loader_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("shuffle", shard_token_groups=(tuple(range(0, 60)),), train_tokens=40)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            data_order="shuffle",
            shuffle_seed=123,
            worker_buffer_size=2,
            prefetch=True,
        )
    )

    payload = run_preflight(config_path).payload

    assert payload["data"]["pipeline"]["order"] == "shuffle"
    assert payload["data"]["pipeline"]["shuffle_seed"] == 123
    assert payload["data"]["pipeline"]["worker_buffer_size"] == 2
    assert payload["data"]["pipeline"]["prefetch"] is True
    assert payload["diagnostics"]["data_pipeline"] == payload["data"]["pipeline"]
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_reports_document_buffer_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "document-buffer",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=48,
        document_offsets=(0, 3, 6, 9, 12, 20, 32, 48, 80),
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            data_order="document_buffer",
            shuffle_seed=123,
            document_buffer_size=3,
            document_refill_size=2,
        )
    )

    payload = run_preflight(config_path).payload

    assert payload["data"]["pipeline"]["order"] == "document_buffer"
    assert payload["data"]["pipeline"]["document_buffer_size"] == 3
    assert payload["data"]["pipeline"]["document_refill_size"] == 2
    assert payload["data"]["first_batch"]["document_aware"] is True
    assert payload["training"]["first_step_loss"] > 0.0
    assert payload["diagnostics"]["data_pipeline"] == payload["data"]["pipeline"]
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_skips_eval_when_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("no-eval", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))

    report = run_preflight(config_path)

    assert report.payload["eval"] is None
    assert "eval: skipped" in format_preflight_report(report)
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_resolves_auto_schedule_total_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("cosine", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, schedule_name="cosine", target_tokens=16))

    report = run_preflight(config_path)

    assert report.payload["optimizer"]["schedule"] == "cosine"
    assert report.payload["optimizer"]["total_steps"] == 2
    assert report.payload["training"]["estimated_steps"] == 2


def test_run_preflight_rejects_existing_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("existing", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))
    (tmp_path / "runs" / "loop").mkdir(parents=True)

    with pytest.raises(ContractError, match="run directory already exists"):
        run_preflight(config_path)


@pytest.mark.parametrize(
    ("config_kwargs", "match"),
    [
        ({"tokenizer_id": "wrong-tokenizer"}, "does not match config tokenizer"),
        ({"eval_name": "perplexity", "eval_every_steps": 1}, "validation"),
        ({"second_eval": True, "eval_every_steps": 1}, "exactly one eval"),
        ({"optimizer_name": "soap"}, "no Jaxtitan runtime adapter"),
    ],
)
def test_run_preflight_rejects_invalid_runtime_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
    config_kwargs,
    match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("invalid", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, **config_kwargs))

    with pytest.raises(ContractError, match=match):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_insufficient_local_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("devices", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, axis_sizes=(8,), global_batch_size=8, target_tokens=32))

    with pytest.raises(ContractError, match="only 4 local device"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_non_divisible_global_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("nondivisible", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, axis_sizes=(4,), global_batch_size=6, target_tokens=24))

    with pytest.raises(ConfigError, match="data axis size"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_reserved_parallel_axes_greater_than_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    require_fake_devices()
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("reserved-axis", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _preflight_config(
            manifest,
            axis_names=("data", "tp"),
            axis_sizes=(4, 2),
            global_batch_size=8,
            target_tokens=32,
        )
    )

    with pytest.raises(ContractError, match="reserved for later"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_multi_process_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("multi-process", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))
    monkeypatch.setattr("jaxtitan.mesh.sharding.jax.process_count", lambda: 2)

    with pytest.raises(ContractError, match="exactly one process"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_missing_train_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(tmp_path / "missing" / "manifest.json"))

    with pytest.raises(ContractError, match="manifest does not exist"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_accepts_hf_streaming_train_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_hf_stream(monkeypatch, [{"text": "hello world " * 32}])
    config_path = tmp_path / "streaming.toml"
    config_path.write_text(_streaming_preflight_config())

    report = run_preflight(config_path)
    payload = report.payload

    assert payload["status"] == "passed"
    assert payload["data"]["mode"] == "hf_streaming"
    assert payload["data"]["train_manifest"] is None
    assert payload["data"]["train_split_tokens"] is None
    assert payload["data"]["pipeline"]["backend"] == "hf_streaming"
    assert payload["data"]["pipeline"]["source"]["dataset"] == "mock/dataset"
    assert payload["data"]["pipeline"]["exact_resume"] is True
    assert payload["data"]["first_batch"]["target_tokens"] == 4


def test_run_preflight_rejects_too_small_train_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("small-train", shard_token_groups=(tuple(range(0, 20)),), train_tokens=8)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, target_tokens=8))

    with pytest.raises(ContractError, match="train split has"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_too_small_validation_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("small-val", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, eval_every_steps=1, eval_num_batches=1))

    with pytest.raises(ContractError, match="val split has"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def _preflight_config(
    train_manifest: Path,
    *,
    target_tokens: int = 16,
    tokenizer_id: str = "toy-tokenizer",
    schedule_name: str = "constant",
    optimizer_name: str = "adamw",
    adamw_fallback_peak_lr: float | None = None,
    axis_names: tuple[str, ...] = ("data",),
    axis_sizes: tuple[int, ...] = (1,),
    parallelism_mode: str = "ddp",
    global_batch_size: int = 2,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_layers: int = 1,
    num_heads: int = 2,
    n_kv_heads: int = 1,
    model_name: str = "decoder",
    trinity: bool = False,
    trinity_moe: bool = False,
    expert_parallel: bool = False,
    gradient_accumulation_steps: int = 1,
    remat: str = "none",
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
    profiling_block: str = "",
) -> str:
    shuffle_seed_line = "" if shuffle_seed is None else f"shuffle_seed = {shuffle_seed}\n"
    document_buffer_size_line = "" if document_buffer_size is None else f"document_buffer_size = {document_buffer_size}\n"
    document_refill_size_line = "" if document_refill_size is None else f"document_refill_size = {document_refill_size}\n"
    fallback_schedule_block = ""
    if adamw_fallback_peak_lr is not None:
        fallback_schedule_block = f"""
[optimizer.adamw_fallback_schedule]
name = "{schedule_name}"
peak_lr = {adamw_fallback_peak_lr}
"""
    trinity_block = ""
    if trinity:
        trinity_block = """
[model.trinity]
initial_dense_layers = 1
local_window = 4
local_layers_per_global = 1
"""
        if trinity_moe:
            trinity_block += """
[model.trinity.moe]
num_experts = 4
top_k = 2
"""
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
seed = 7
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
weight_decay = 0.0

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = {0.02 if optimizer_name == "muon" else 0.001}
{fallback_schedule_block}

[data]
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "{tokenizer_id}"
order = "{data_order}"
{shuffle_seed_line}worker_count = {worker_count}
worker_buffer_size = {worker_buffer_size}
prefetch = {str(prefetch).lower()}
{document_buffer_size_line}{document_refill_size_line}

[training]
seq_len = 4
global_batch_size = {global_batch_size}
gradient_accumulation_steps = {gradient_accumulation_steps}
target_tokens = {target_tokens}
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = [{", ".join(f'"{name}"' for name in axis_names)}]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]

[parallelism]
mode = "{parallelism_mode}"
expert_parallel = {str(expert_parallel).lower()}
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


def _streaming_preflight_config() -> str:
    return """
[run]
id = "loop"
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
target_tokens = 4
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]
"""
