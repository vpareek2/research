import pytest
import jax
import numpy as np

from jaxtitan.batch import Batch
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_accumulated_batch, place_batch
from jaxtitan.runtime import diagnostics
from jaxtitan.runtime.diagnostics import (
    PhaseTimer,
    compile_contract_summary,
    enrich_train_metrics,
    estimate_flops_per_token,
    peak_flops_for_device,
    placed_array_summary,
    sample_device_telemetry,
    sharding_policy_summary,
    training_diagnostics_summary,
)
from jaxtitan.specs.data import DataSpec
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.specs.run import RunSpec, TrainingSpec

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_phase_timer_records_named_phases_and_manual_additions(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 1.0, 1.25, 2.0])
    monkeypatch.setattr(diagnostics.time, "perf_counter", lambda: next(ticks))

    timer = PhaseTimer()
    with timer.phase("data"):
        pass
    timer.add("sync", 0.5)

    assert timer.seconds("data") == pytest.approx(0.25)
    assert timer.seconds("sync") == pytest.approx(0.5)
    assert timer.total_sec() == pytest.approx(2.0)


def test_estimate_flops_per_token_counts_decoder_shapes() -> None:
    assert estimate_flops_per_token(_tiny_model_spec(), seq_len=4) == 6912


def test_peak_flops_for_known_and_unknown_devices() -> None:
    assert peak_flops_for_device("NVIDIA H100 80GB HBM3") == 989e12
    assert peak_flops_for_device("NVIDIA GeForce RTX 4090") == 165.2e12
    assert peak_flops_for_device("mystery accelerator") is None
    assert peak_flops_for_device(None) is None


def test_enrich_train_metrics_reports_timing_throughput_and_mfu() -> None:
    row = _base_row()
    runtime = {
        "performance": {
            "device_kind": "test",
            "device_count": 2,
            "flops_per_token": 100,
            "peak_flops_per_device": 1000.0,
            "peak_flops_total": 2000.0,
        },
        "device_telemetry": _empty_telemetry(),
    }

    enriched = enrich_train_metrics(
        row,
        timings={
            "data_sec": 0.5,
            "placement_sec": 0.25,
            "train_dispatch_sec": 0.1,
            "metrics_sync_sec": 0.9,
            "train_step_sec": 1.0,
            "step_sec": 2.0,
        },
        runtime=runtime,
        telemetry={**_empty_telemetry(), "device_memory_used_bytes": 123},
    )

    assert enriched["tokens_per_sec"] == pytest.approx(10.0)
    assert enriched["train_tokens_per_sec"] == pytest.approx(20.0)
    assert enriched["examples_per_sec"] == pytest.approx(1.0)
    assert enriched["flops_per_step"] == 2000
    assert enriched["flops_per_sec"] == pytest.approx(2000.0)
    assert enriched["mfu"] == pytest.approx(100.0)
    assert enriched["device_memory_used_bytes"] == 123
    assert enriched["execution_mode"] == "replicated_data_parallel"
    assert enriched["metrics_scope"] == "global"
    assert enriched["artifact_writer"] == "single_host"


def test_enrich_train_metrics_keeps_mfu_null_when_peak_flops_unknown() -> None:
    runtime = {
        "performance": {
            "device_kind": "unknown",
            "device_count": 1,
            "flops_per_token": 100,
            "peak_flops_per_device": None,
            "peak_flops_total": None,
        },
        "device_telemetry": _empty_telemetry(),
    }

    enriched = enrich_train_metrics(
        _base_row(),
        timings={"train_step_sec": 1.0, "step_sec": 1.0},
        runtime=runtime,
    )

    assert enriched["mfu"] is None
    assert enriched["peak_flops_total"] is None


def test_sample_device_telemetry_returns_explicit_nulls_when_unavailable() -> None:
    telemetry = sample_device_telemetry([_FakeDevice(None)], nvml_provider=None)

    assert set(telemetry) == set(_empty_telemetry())
    assert all(value is None for value in telemetry.values())


def test_sample_device_telemetry_aggregates_jax_memory_and_nvml() -> None:
    telemetry = sample_device_telemetry(
        [
            _FakeDevice({"bytes_in_use": 10, "peak_bytes_in_use": 20, "bytes_limit": 100}),
            _FakeDevice({"bytes_in_use": 15, "peak_bytes_in_use": 30, "bytes_limit": 100}),
        ],
        nvml_provider=_FakeNvml(),
    )

    assert telemetry["device_memory_used_bytes"] == 25
    assert telemetry["device_memory_peak_bytes"] == 30
    assert telemetry["device_memory_limit_bytes"] == 200
    assert telemetry["gpu_memory_used_bytes"] == 400
    assert telemetry["gpu_memory_total_bytes"] == 2000
    assert telemetry["gpu_utilization_pct"] == pytest.approx(50.0)
    assert telemetry["gpu_memory_utilization_pct"] == pytest.approx(25.0)
    assert telemetry["gpu_power_w"] == pytest.approx(200.0)
    assert telemetry["gpu_temperature_c"] == pytest.approx(61.0)


def test_sharding_policy_summary_records_data_parallel_policy() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)

    summary = sharding_policy_summary(plan)

    assert summary["batch"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"
    assert summary["batch"]["loss_mask"]["partition_spec"] == "PartitionSpec('data', None)"
    assert summary["batch"]["accumulated_input_ids"]["partition_spec"] == "PartitionSpec(None, 'data', None)"
    assert summary["batch"]["accumulated_loss_mask"]["partition_spec"] == "PartitionSpec(None, 'data', None)"
    assert summary["train_state"]["model"]["partition_spec"] == "PartitionSpec()"
    assert summary["optimizer_state"]["partition_spec"] == "PartitionSpec()"
    assert summary["metrics"]["partition_spec"] == "PartitionSpec()"
    assert summary["checkpoint"]["restore_template"]["partition_spec"] == "PartitionSpec()"
    assert summary["reserved"] == {"fsdp": None, "ep": None, "tp": None, "kv_cache": None}


def test_compile_contract_summary_records_donation_shapes_and_shardings() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    spec = RunSpec(
        run_id="diagnostics",
        seed=0,
        output_dir="runs",
        model=_tiny_model_spec(),
        optimizer=OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1e-3)),
        data=DataSpec(train_manifest="manifest.json"),
        mesh=MeshSpec(axis_names=("data",), axis_sizes=(4,)),
        training=TrainingSpec(
            seq_len=4,
            global_batch_size=8,
            target_tokens=64,
            gradient_accumulation_steps=2,
        ),
    )

    summary = compile_contract_summary(spec, plan)

    assert summary["train"]["donate_state"] is True
    assert summary["train"]["expected_batch_shape"] == [2, 8, 4]
    assert summary["train"]["input_shardings"]["state"]["partition_spec"] == "PartitionSpec()"
    assert summary["train"]["input_shardings"]["input_ids"]["partition_spec"] == "PartitionSpec(None, 'data', None)"
    assert summary["eval"]["donate_state"] is False
    assert summary["eval"]["expected_batch_shape"] == [8, 4]
    assert summary["eval"]["input_shardings"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"


def test_placed_array_summary_reports_global_and_shard_shapes() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.arange(32, dtype=np.int32).reshape(8, 4),
        target_ids=np.arange(32, 64, dtype=np.int32).reshape(8, 4),
        loss_mask=np.ones((8, 4), dtype=np.bool_),
    )

    summary = placed_array_summary(place_batch(batch, plan).input_ids)

    assert summary["global_shape"] == [8, 4]
    assert summary["dtype"] == "int32"
    assert summary["addressable_shard_count"] == 4
    assert summary["unique_addressable_shard_shapes"] == [[2, 4]]
    assert summary["sharding"]["partition_spec"] == "PartitionSpec('data', None)"


def test_placed_array_summary_reports_accumulated_batch_shards() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.arange(64, dtype=np.int32).reshape(2, 8, 4),
        target_ids=np.arange(64, 128, dtype=np.int32).reshape(2, 8, 4),
        loss_mask=np.ones((2, 8, 4), dtype=np.bool_),
    )

    summary = placed_array_summary(place_accumulated_batch(batch, plan).input_ids)

    assert summary["global_shape"] == [2, 8, 4]
    assert summary["addressable_shard_count"] == 4
    assert summary["unique_addressable_shard_shapes"] == [[2, 2, 4]]
    assert summary["sharding"]["partition_spec"] == "PartitionSpec(None, 'data', None)"


def test_training_diagnostics_summary_uses_logged_rows_and_steady_state() -> None:
    runtime = {
        "performance": {
            "device_kind": "NVIDIA H100",
            "device_count": 2,
            "flops_per_token": 100,
            "peak_flops_per_device": 1000.0,
            "peak_flops_total": 2000.0,
        }
    }
    rows = [
        {"train_tokens_per_sec": 10.0, "mfu": 1.0},
        {"train_tokens_per_sec": 20.0, "mfu": 2.0},
        {"train_tokens_per_sec": 30.0, "mfu": 3.0},
    ]

    summary = training_diagnostics_summary(rows, total_wall_sec=12.5, runtime=runtime)

    assert summary["total_wall_sec"] == 12.5
    assert summary["avg_train_tokens_per_sec"] == pytest.approx(20.0)
    assert summary["final_train_tokens_per_sec"] == pytest.approx(30.0)
    assert summary["steady_train_tokens_per_sec"] == pytest.approx(25.0)
    assert summary["avg_mfu"] == pytest.approx(2.0)
    assert summary["final_mfu"] == pytest.approx(3.0)
    assert summary["runtime_diagnostics_path"] == "diagnostics/runtime.json"
    assert summary["execution_mode"] == "replicated_data_parallel"
    assert summary["metrics_scope"] == "global"
    assert summary["artifact_writer"] == "single_host"


class _FakeDevice:
    platform = "gpu"
    device_kind = "NVIDIA H100 80GB HBM3"
    id = 0
    process_index = 0

    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


class _FakeMemory:
    used = 200
    total = 1000


class _FakeUtil:
    gpu = 50
    memory = 25


class _FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def nvmlInit(self) -> None:
        return None

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> _FakeMemory:
        return _FakeMemory()

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> _FakeUtil:
        return _FakeUtil()

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return 100_000

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> float:
        return 60.0 + handle


def _tiny_model_spec() -> ModelSpec:
    return ModelSpec(
        name="decoder",
        variant="tiny",
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        n_kv_heads=1,
        max_seq_len=4,
        compute_dtype="float32",
    )


def _base_row() -> dict:
    return {
        "schema_version": 1,
        "step": 1,
        "tokens_seen": 20,
        "loss_sum": 2.0,
        "token_count": 20,
        "loss": 0.1,
        "lr": 0.001,
        "grad_norm": 1.0,
        "param_norm": 2.0,
        "update_norm": 3.0,
        "epoch": 0,
        "token_start": 0,
        "token_end": 20,
        "examples": 2,
        "target_tokens": 20,
    }


def _empty_telemetry() -> dict:
    return {
        "device_memory_used_bytes": None,
        "device_memory_peak_bytes": None,
        "device_memory_limit_bytes": None,
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_utilization_pct": None,
        "gpu_memory_utilization_pct": None,
        "gpu_power_w": None,
        "gpu_temperature_c": None,
    }
