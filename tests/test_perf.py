from types import SimpleNamespace

import pytest

from config import ModelConfig
from utils.perf import NvmlMonitor, PerfMonitor, estimate_flops_per_token, peak_flops_for_device


def tiny_model_config(**overrides):
    values = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        seq_len=8,
        theta=10000.0,
        eps=1e-6,
        tied=False,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_estimate_flops_per_token_counts_transformer_and_lm_head_matmuls():
    assert estimate_flops_per_token(tiny_model_config()) == 79872
    assert estimate_flops_per_token(tiny_model_config(tied=True)) == 55296


def test_peak_flops_for_known_and_unknown_devices():
    assert peak_flops_for_device("NVIDIA H100 80GB HBM3") == 989e12
    assert peak_flops_for_device("NVIDIA A100-SXM4-80GB") == 312e12
    assert peak_flops_for_device("NVIDIA GB10") == 125e12
    assert peak_flops_for_device("mystery accelerator") is None
    assert peak_flops_for_device(None) is None


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def nvmlInit(self):
        self.initialized = True

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=(handle + 1) * 100, total=(handle + 1) * 1000)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=50 + handle * 10, memory=20 + handle * 10)

    def nvmlDeviceGetPowerUsage(self, handle):
        return (100 + handle * 50) * 1000

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 60 + handle


class NoMemoryNvml(FakeNvml):
    def nvmlDeviceGetMemoryInfo(self, handle):
        raise RuntimeError("memory unsupported")


def test_nvml_monitor_rolls_up_device_samples():
    monitor = NvmlMonitor(2, provider=FakeNvml())
    metrics = monitor.sample()

    assert metrics["system/gpu_memory_used_bytes"] == 300
    assert metrics["system/gpu_memory_total_bytes"] == 3000
    assert metrics["system/gpu_memory_peak_bytes"] == 300
    assert metrics["system/gpu_utilization_pct"] == 55.0
    assert metrics["system/gpu_memory_utilization_pct"] == 25.0
    assert metrics["system/gpu_power_w"] == 250.0
    assert metrics["system/gpu_temperature_c"] == 61.0


def test_nvml_monitor_keeps_supported_fields_when_memory_is_missing():
    monitor = NvmlMonitor(1, provider=NoMemoryNvml())
    metrics = monitor.sample()

    assert "system/gpu_memory_used_bytes" not in metrics
    assert metrics["system/gpu_utilization_pct"] == 50.0
    assert metrics["system/gpu_power_w"] == 100.0
    assert metrics["system/gpu_temperature_c"] == 60.0


def test_perf_monitor_enriches_metrics_and_handles_unknown_peak():
    cfg = tiny_model_config()
    monitor = PerfMonitor(
        model_config=cfg,
        device_count=2,
        device_kind="NVIDIA H100 80GB HBM3",
        peak_flops_per_device=peak_flops_for_device("H100"),
        nvml=NvmlMonitor(2, provider=FakeNvml()),
    )
    metrics = {
        "time/train_step_sec": 0.5,
        "time/train_tokens_per_sec": 1000.0,
        "time/tokens_per_sec": 500.0,
    }

    monitor.enrich(metrics)

    flops_per_token = estimate_flops_per_token(cfg)
    assert metrics["perf/flops_per_token"] == flops_per_token
    assert metrics["perf/flops_per_step"] == flops_per_token * 500
    assert metrics["perf/flops_per_sec"] == flops_per_token * 1000.0
    assert metrics["perf/mfu"] == pytest.approx(100.0 * flops_per_token * 1000.0 / (989e12 * 2))
    assert metrics["time/tokens_per_gpu_hour"] == 900000.0
    assert metrics["time/train_tokens_per_gpu_hour"] == 1800000.0
    assert metrics["system/gpu_memory_peak_bytes"] == 300

    unknown = PerfMonitor(cfg, device_count=1, device_kind="unknown", peak_flops_per_device=None, nvml=None)
    unknown_metrics = {"time/train_step_sec": 1.0, "time/train_tokens_per_sec": 10.0}
    unknown.enrich(unknown_metrics)
    assert unknown_metrics["perf/mfu"] is None
