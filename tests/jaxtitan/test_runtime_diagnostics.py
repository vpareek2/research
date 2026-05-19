import pytest

from jaxtitan.runtime import diagnostics
from jaxtitan.runtime.diagnostics import (
    PhaseTimer,
    enrich_train_metrics,
    estimate_flops_per_token,
    peak_flops_for_device,
    sample_device_telemetry,
    training_diagnostics_summary,
)
from jaxtitan.specs.model import ModelSpec


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
