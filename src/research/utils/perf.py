"""
Training performance and hardware telemetry helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research.config import ModelConfig


SECONDS_PER_GPU_HOUR = 3600.0


# Hardcoded BF16 peak FLOPs. Table order matters: more specific patterns first.
BF16_PEAK_FLOPS_TABLE = (
    # NVIDIA Blackwell
    (("rtx pro 6000", "blackwell"), 1.0e15),
    (("gb200",), 2.5e15),
    (("grace blackwell",), 2.5e15),
    (("b200",), 2.25e15),
    (("b100",), 1.8e15),
    # NVIDIA Hopper
    (("h200", "nvl"), 836e12),
    (("h200", "pcie"), 836e12),
    (("h200",), 989e12),
    (("h100", "nvl"), 835e12),
    (("h100", "pcie"), 756e12),
    (("h100",), 989e12),
    (("h800", "nvl"), 989e12),
    (("h800",), 756e12),
    # NVIDIA Ampere data center
    (("a100",), 312e12),
    (("a800",), 312e12),
    (("a40",), 149.7e12),
    (("a30",), 165e12),
    # NVIDIA Ada data center
    (("l40s",), 362e12),
    (("l40-s",), 362e12),
    (("l40 s",), 362e12),
    (("l4",), 121e12),
    # AMD CDNA accelerators
    (("mi355",), 2.5e15),
    (("mi325",), 1.3074e15),
    (("mi300x",), 1.3074e15),
    (("mi300a",), 980.6e12),
    (("mi250x",), 383e12),
    (("mi250",), 362.1e12),
    # Consumer RTX
    (("5090",), 209.5e12),
    (("4090",), 165.2e12),
    (("3090",), 71e12),
    # GB10 does not have a clean official dense BF16 peak spec published in
    # the usual NVIDIA tables. Use the common estimate from the 1 PFLOP sparse
    # FP4 headline so MFU remains useful for DGX Spark / GB10 run comparisons.
    (("gb10",), 125e12),
)


def estimate_flops_per_token(config: ModelConfig) -> int:
    head_dim = config.hidden_size // config.n_heads
    kv_width = config.n_kv_heads * head_dim

    q_proj = config.hidden_size * config.hidden_size
    k_proj = config.hidden_size * kv_width
    v_proj = config.hidden_size * kv_width
    o_proj = config.hidden_size * config.hidden_size
    mlp = 3 * config.hidden_size * config.intermediate_size
    lm_head = 0 if config.tied else config.hidden_size * config.vocab_size

    matmul_compute_params = config.n_layers * (q_proj + k_proj + v_proj + o_proj + mlp) + lm_head
    attention_flops = config.n_layers * 12 * config.n_heads * head_dim * config.seq_len
    return 6 * matmul_compute_params + attention_flops


def peak_flops_for_device(device_name: str | None) -> float | None:
    if not device_name:
        return None
    normalized = device_name.lower().replace("nvidia", "").strip()
    for patterns, flops in BF16_PEAK_FLOPS_TABLE:
        if all(pattern in normalized for pattern in patterns):
            return flops
    return None


@dataclass
class PerfMonitor:
    model_config: ModelConfig
    device_count: int
    device_kind: str
    peak_flops_per_device: float | None = None
    nvml: "NvmlMonitor | None" = None

    @classmethod
    def from_distributed(cls, model_config: ModelConfig, distributed, *, nvml_provider: Any | None = None) -> "PerfMonitor":
        devices = list(distributed.mesh.devices.flat)
        device_kind = _device_kind(devices[0]) if devices else "unknown"
        peak_flops = peak_flops_for_device(device_kind)
        return cls(
            model_config=model_config,
            device_count=distributed.device_count,
            device_kind=device_kind,
            peak_flops_per_device=peak_flops,
            nvml=NvmlMonitor(distributed.device_count, provider=nvml_provider),
        )

    def enrich(self, metrics: dict) -> dict:
        flops_per_token = estimate_flops_per_token(self.model_config)
        flops_per_step = None
        train_tokens_per_sec = _number(metrics.get("time/train_tokens_per_sec"))
        step_sec = _number(metrics.get("time/train_step_sec"))
        if train_tokens_per_sec is not None and step_sec is not None:
            flops_per_step = flops_per_token * train_tokens_per_sec * step_sec

        loop_tokens_per_sec = _number(metrics.get("time/tokens_per_sec"))
        flops_per_sec = flops_per_token * train_tokens_per_sec if train_tokens_per_sec is not None else None
        peak_total = self.peak_flops_per_device * self.device_count if self.peak_flops_per_device is not None else None

        metrics["system/device_count"] = self.device_count
        metrics["system/device_kind"] = self.device_kind
        metrics["perf/flops_per_token"] = flops_per_token
        metrics["perf/flops_per_step"] = flops_per_step
        metrics["perf/flops_per_sec"] = flops_per_sec
        metrics["perf/peak_flops_per_device"] = self.peak_flops_per_device
        metrics["perf/peak_flops_total"] = peak_total
        metrics["perf/mfu"] = 100.0 * flops_per_sec / peak_total if flops_per_sec is not None and peak_total else None
        metrics["time/tokens_per_gpu_hour"] = (
            loop_tokens_per_sec * SECONDS_PER_GPU_HOUR / self.device_count if loop_tokens_per_sec is not None else None
        )
        metrics["time/train_tokens_per_gpu_hour"] = (
            train_tokens_per_sec * SECONDS_PER_GPU_HOUR / self.device_count if train_tokens_per_sec is not None else None
        )

        if self.nvml is not None:
            metrics.update(self.nvml.sample())
        return metrics


class NvmlMonitor:
    def __init__(self, device_count: int, *, provider: Any | None = None):
        self.device_count = device_count
        self.provider = provider if provider is not None else _load_nvml()
        self.handles = []
        self.peak_memory_used = 0
        if self.provider is None:
            return
        try:
            self.provider.nvmlInit()
            self.handles = [self.provider.nvmlDeviceGetHandleByIndex(index) for index in range(device_count)]
        except Exception:
            self.provider = None
            self.handles = []

    def sample(self) -> dict:
        if self.provider is None or not self.handles:
            return {}
        memory_used = 0
        memory_total = 0
        gpu_utils = []
        memory_utils = []
        power_w = 0.0
        temperatures = []

        for handle in self.handles:
            try:
                memory = self.provider.nvmlDeviceGetMemoryInfo(handle)
                memory_used += int(memory.used)
                memory_total += int(memory.total)
            except Exception:
                pass
            try:
                util = self.provider.nvmlDeviceGetUtilizationRates(handle)
                gpu_utils.append(float(util.gpu))
                memory_utils.append(float(util.memory))
            except Exception:
                pass
            try:
                power_w += float(self.provider.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                pass
            try:
                temperatures.append(float(self.provider.nvmlDeviceGetTemperature(handle, self.provider.NVML_TEMPERATURE_GPU)))
            except Exception:
                pass

        self.peak_memory_used = max(self.peak_memory_used, memory_used)
        metrics = {}
        if memory_total:
            metrics["system/gpu_memory_used_bytes"] = memory_used
            metrics["system/gpu_memory_total_bytes"] = memory_total
            metrics["system/gpu_memory_peak_bytes"] = self.peak_memory_used
        if gpu_utils:
            metrics["system/gpu_utilization_pct"] = sum(gpu_utils) / len(gpu_utils)
        if memory_utils:
            metrics["system/gpu_memory_utilization_pct"] = sum(memory_utils) / len(memory_utils)
        if power_w:
            metrics["system/gpu_power_w"] = power_w
        if temperatures:
            metrics["system/gpu_temperature_c"] = max(temperatures)
        return metrics


def _load_nvml():
    try:
        import pynvml
    except Exception:
        return None
    return pynvml


def _device_kind(device) -> str:
    return str(getattr(device, "device_kind", None) or getattr(device, "platform", None) or device)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
