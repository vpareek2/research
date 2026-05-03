"""
Baseline-relative run scoring.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from research.utils.run_summary import DEFAULT_REGISTRY_PATH, SUMMARY_DIR_NAME, SUMMARY_JSON_NAME, summarize_run


BASE_SCORE = 25.0
MIN_RATIO = 0.5
MAX_RATIO = 1.5


def score_summary(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    run_inputs = _score_inputs(summary)
    baseline_inputs = _score_inputs(baseline)
    missing = sorted(set(run_inputs["missing"] + [f"baseline:{name}" for name in baseline_inputs["missing"]]))

    if summary.get("status") == "failed":
        missing.append("status")

    if missing:
        return _empty_score(summary, baseline, missing)

    quality = _quality_score(run_inputs, baseline_inputs)
    training = _training_efficiency_score(run_inputs, baseline_inputs, quality["value"])
    inference = _inference_efficiency_score(run_inputs, baseline_inputs)
    health = _health_score(summary)
    final_score = BASE_SCORE * (
        0.40 * quality["value"]
        + 0.25 * training["value"]
        + 0.20 * inference["value"]
        + 0.15 * health["value"]
    )

    return {
        "schema_version": 1,
        "eligible": True,
        "missing": [],
        "base_score": BASE_SCORE,
        "final_score": final_score,
        "baseline_run_name": _run_name(baseline),
        "baseline_run_dir": _run_dir(baseline),
        "quality": quality,
        "training_efficiency": training,
        "inference_efficiency": inference,
        "health": health,
    }


def attach_score(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summary)
    summary["score"] = score_summary(summary, baseline)
    return summary


def load_summary_or_summarize(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = run_dir / SUMMARY_DIR_NAME / SUMMARY_JSON_NAME
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return summarize_run(run_dir)


def select_baseline_summary(
    current_summary: dict[str, Any],
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    baseline_run: str | Path | None = None,
) -> dict[str, Any]:
    if baseline_run is not None:
        current_run_dir = _run_dir(current_summary)
        if current_run_dir is not None and _same_path(Path(baseline_run), Path(str(current_run_dir))):
            return current_summary
        return load_summary_or_summarize(baseline_run)

    for record in _load_registry(Path(registry_path)):
        if record.get("score_eligible") and _is_number(record.get("score")) and record.get("run_dir"):
            path = Path(str(record["run_dir"]))
            if path.exists():
                return load_summary_or_summarize(path)
    return current_summary


def _score_inputs(summary: dict[str, Any]) -> dict[str, Any]:
    missing = []
    core = _latest_full_core(summary)
    native_val_bpb = _native_val_bpb(summary)
    domain_bpbs = _domain_bpbs(summary)
    epiplexity = _number(_nested(summary, ["epiplexity", "train_bpb_auc_per_byte"]))
    avg_mfu = _first_number(
        _nested(summary, ["performance", "steady_train_mfu"]),
        _nested(summary, ["performance", "avg_mfu"]),
    )
    flops_per_token = _number(_nested(summary, ["performance", "flops_per_token"]))
    peak_flops_total = _number(_nested(summary, ["performance", "peak_flops_total"]))
    tokens_seen = _number(_nested(summary, ["training", "tokens_seen"]))
    train_tps = _first_number(
        _nested(summary, ["speed", "steady_train_tokens_per_sec"]),
        _nested(summary, ["speed", "avg_train_tokens_per_sec"]),
    )
    decode_tps = _number(_nested(summary, ["inference_benchmark", "latest", "decode_tokens_per_sec"]))
    prefill_tps = _number(_nested(summary, ["inference_benchmark", "latest", "prefill_tokens_per_sec"]))
    ttft = _number(_nested(summary, ["inference_benchmark", "latest", "ttft_sec"]))

    required = {
        "full_core": core,
        "native_val_bpb": native_val_bpb,
        "domain_mean_bpb": mean(domain_bpbs) if domain_bpbs else None,
        "domain_worst_bpb": max(domain_bpbs) if domain_bpbs else None,
        "train_epiplexity_bpb_auc_per_byte": epiplexity,
        "avg_mfu": avg_mfu,
        "flops_per_token": flops_per_token,
        "peak_flops_total": peak_flops_total,
        "tokens_seen": tokens_seen,
        "avg_train_tokens_per_sec": train_tps,
        "decode_tokens_per_sec": decode_tps,
        "prefill_tokens_per_sec": prefill_tps,
        "ttft_sec": ttft,
    }
    can_be_zero = {"full_core", "train_epiplexity_bpb_auc_per_byte"}
    for name, value in required.items():
        if value is None or (value <= 0 and name not in can_be_zero):
            missing.append(name)

    return {
        **required,
        "train_compute_flops": flops_per_token * tokens_seen if flops_per_token and tokens_seen else None,
        "tokens_per_peak_flop": train_tps / peak_flops_total if train_tps and peak_flops_total else None,
        "missing": missing,
    }


def _quality_score(run: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    components = {
        "core": _higher_better(run["full_core"] + 1.0, baseline["full_core"] + 1.0),
        "native_val_bpb": _lower_better(run["native_val_bpb"], baseline["native_val_bpb"]),
        "domain_mean_bpb": _lower_better(run["domain_mean_bpb"], baseline["domain_mean_bpb"]),
        "domain_worst_bpb": _lower_better(run["domain_worst_bpb"], baseline["domain_worst_bpb"]),
        "epiplexity": _higher_better(
            run["train_epiplexity_bpb_auc_per_byte"],
            baseline["train_epiplexity_bpb_auc_per_byte"],
        ),
    }
    value = (
        0.50 * components["core"]
        + 0.23 * components["native_val_bpb"]
        + 0.17 * components["domain_mean_bpb"]
        + 0.05 * components["domain_worst_bpb"]
        + 0.05 * components["epiplexity"]
    )
    return {"value": value, "components": components}


def _training_efficiency_score(run: dict[str, Any], baseline: dict[str, Any], quality_value: float) -> dict[str, Any]:
    baseline_compute = baseline["train_compute_flops"]
    run_compute = run["train_compute_flops"]
    components = {
        "mfu": _higher_better(run["avg_mfu"], baseline["avg_mfu"]),
        "quality_per_compute": _clamp(quality_value * baseline_compute / run_compute),
        "tokens_per_peak_flop": _higher_better(run["tokens_per_peak_flop"], baseline["tokens_per_peak_flop"]),
    }
    value = 0.50 * components["mfu"] + 0.30 * components["quality_per_compute"] + 0.20 * components["tokens_per_peak_flop"]
    return {"value": value, "components": components}


def _inference_efficiency_score(run: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    components = {
        "decode": _higher_better(run["decode_tokens_per_sec"], baseline["decode_tokens_per_sec"]),
        "prefill": _higher_better(run["prefill_tokens_per_sec"], baseline["prefill_tokens_per_sec"]),
        "ttft": _lower_better(run["ttft_sec"], baseline["ttft_sec"]),
    }
    value = 0.55 * components["decode"] + 0.30 * components["prefill"] + 0.15 * components["ttft"]
    return {"value": value, "components": components}


def _health_score(summary: dict[str, Any]) -> dict[str, Any]:
    health = summary.get("health", {})
    status = summary.get("status")
    nan_count = int(_number(health.get("nan_count")) or 0)
    loss_spikes = int(_number(health.get("loss_spike_count")) or 0)
    grad_spikes = int(_number(health.get("grad_norm_spike_count")) or 0)
    final_bpb = _native_val_bpb(summary)
    best_bpb = _number(_nested(summary, ["quality", "best_val_bpb"]))
    val_regression = 0.0
    if final_bpb is not None and best_bpb is not None and best_bpb > 0:
        val_regression = max(0.0, final_bpb / best_bpb - 1.0)

    value = 1.0
    if status == "failed" or nan_count > 0:
        value = 0.0
    elif status == "incomplete":
        value = 0.25
    elif status == "unstable":
        value = 0.75

    value -= min(0.25, 0.05 * loss_spikes)
    value -= min(0.15, 0.03 * grad_spikes)
    value -= min(0.20, val_regression)
    value = max(0.0, value)
    return {
        "value": value,
        "status": status,
        "nan_count": nan_count,
        "loss_spike_count": loss_spikes,
        "grad_norm_spike_count": grad_spikes,
        "val_regression": val_regression,
    }


def _latest_full_core(summary: dict[str, Any]) -> float | None:
    latest = _nested(summary, ["benchmark_core", "latest"])
    if not isinstance(latest, dict):
        return None
    max_per_task = latest.get("max_per_task")
    if max_per_task not in {-1, None}:
        return None
    return _number(latest.get("core"))


def _native_val_bpb(summary: dict[str, Any]) -> float | None:
    return _number(_nested(summary, ["checkpoint_evals", "latest", "bpb"])) or _number(
        _nested(summary, ["quality", "final_val_bpb"])
    )


def _domain_bpbs(summary: dict[str, Any]) -> list[float]:
    latest_eval = _nested(summary, ["checkpoint_evals", "latest"])
    domains = latest_eval.get("domains") if isinstance(latest_eval, dict) else None
    if isinstance(domains, dict):
        values = [_number(metrics.get("bpb")) for metrics in domains.values() if isinstance(metrics, dict)]
        values = [value for value in values if value is not None]
        if values:
            return values

    training_domains = _nested(summary, ["domain_validation", "training"])
    if isinstance(training_domains, dict):
        values = [_number(metrics.get("final_bpb")) for metrics in training_domains.values() if isinstance(metrics, dict)]
        return [value for value in values if value is not None]
    return []


def _empty_score(summary: dict[str, Any], baseline: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eligible": False,
        "missing": sorted(set(missing)),
        "base_score": BASE_SCORE,
        "final_score": None,
        "baseline_run_name": _run_name(baseline),
        "baseline_run_dir": _run_dir(baseline),
    }


def _higher_better(value: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if value == 0 else MAX_RATIO
    return _clamp(value / baseline)


def _lower_better(value: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if value == 0 else MIN_RATIO
    return _clamp(baseline / value)


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return MIN_RATIO
    return min(MAX_RATIO, max(MIN_RATIO, value))


def _nested(row: dict[str, Any], path: list[str]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _is_number(value: Any) -> bool:
    return _number(value) is not None


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_name(summary: dict[str, Any]) -> str | None:
    return _nested(summary, ["run", "name"])


def _run_dir(summary: dict[str, Any]) -> str | None:
    return _nested(summary, ["run", "run_dir"])
