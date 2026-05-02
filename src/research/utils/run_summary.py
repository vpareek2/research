"""
Post-run summary and scorecard generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from research.config import load_config
from research.utils.param_count import count_params


SUMMARY_DIR_NAME = "summary"
SUMMARY_JSON_NAME = "run_summary.json"
SCORECARD_NAME = "scorecard.md"
DEFAULT_REGISTRY_PATH = Path("runs") / "registry.jsonl"
VAL_REGRESSION_THRESHOLD = 0.05


def load_run_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics rows found in {path}")
    return rows


def load_checkpoint_eval_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    evals_dir = Path(run_dir) / "evals"
    if not evals_dir.exists():
        return []

    rows = []
    for path in sorted(evals_dir.glob("step_*/metrics.json"), key=_eval_metrics_sort_key):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_core_eval_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    evals_dir = Path(run_dir) / "evals"
    if not evals_dir.exists():
        return []

    rows = []
    for path in sorted(evals_dir.glob("step_*/core_metrics.json"), key=_eval_metrics_sort_key):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_inference_benchmark_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    evals_dir = Path(run_dir) / "evals"
    if not evals_dir.exists():
        return []

    rows = []
    for path in sorted(evals_dir.glob("step_*/inference_metrics.json"), key=_eval_metrics_sort_key):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.toml")
    metadata = _load_optional_json(run_dir / "metadata.json")
    rows = load_run_rows(run_dir)
    eval_rows = load_checkpoint_eval_metrics(run_dir)
    core_rows = load_core_eval_metrics(run_dir)
    inference_rows = load_inference_benchmark_metrics(run_dir)
    final_row = rows[-1]
    val_rows = [row for row in rows if "val/loss" in row]
    train_tokens_target = config.train.steps * config.train.batch_size * config.train.seq_len
    param_count = count_params(config.model).total
    health = _health_summary(rows)
    quality = _quality_summary(rows, val_rows)
    checkpoint_evals = _checkpoint_eval_summary(eval_rows)
    domain_validation = _domain_validation_summary(rows, eval_rows)
    performance = _performance_summary(rows)
    core_benchmark = _core_benchmark_summary(core_rows)
    inference_benchmark = _inference_benchmark_summary(inference_rows)
    epiplexity = _epiplexity_summary(rows, val_rows, config.train)
    status, decision_hint = _verdict(config, rows, final_row, val_rows, health)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "name": config.experiment.name,
            "run_dir": str(run_dir),
            "created_at": metadata.get("created_at"),
            "config_path": metadata.get("config_path"),
        },
        "model": {
            "params": param_count,
            "layers": config.model.n_layers,
            "hidden_size": config.model.hidden_size,
            "intermediate_size": config.model.intermediate_size,
            "heads": config.model.n_heads,
            "kv_heads": config.model.n_kv_heads,
            "seq_len": config.model.seq_len,
            "vocab_size": config.model.vocab_size,
            "tied_embeddings": config.model.tied,
        },
        "training": {
            "configured_steps": config.train.steps,
            "configured_tokens": train_tokens_target,
            "steps_completed": min(int(final_row["step"]) + 1, config.train.steps),
            "logged_rows": len(rows),
            "final_step": int(final_row["step"]),
            "tokens_seen": _number_or_none(final_row.get("train/tokens_seen")),
            "batch_size": config.train.batch_size,
            "seq_len": config.train.seq_len,
            "log_every": config.train.log_every,
        },
        "quality": quality,
        "health": health,
        "speed": {
            "avg_tokens_per_sec": _mean_metric(rows, "time/tokens_per_sec"),
            "avg_train_tokens_per_sec": _mean_metric(rows, "time/train_tokens_per_sec"),
            "final_elapsed_sec": _number_or_none(final_row.get("time/elapsed_sec")),
        },
        "performance": performance,
        "checkpoint_evals": checkpoint_evals,
        "domain_validation": domain_validation,
        "benchmark_core": core_benchmark,
        "inference_benchmark": inference_benchmark,
        "epiplexity": epiplexity,
        "status": status,
        "decision_hint": decision_hint,
    }
    summary["registry_record"] = registry_record(summary)
    return summary


def format_scorecard(summary: dict[str, Any]) -> str:
    run = summary["run"]
    model = summary["model"]
    training = summary["training"]
    quality = summary["quality"]
    health = summary["health"]
    speed = summary["speed"]
    performance = summary.get("performance", {})
    evals = summary["checkpoint_evals"]
    domains = summary.get("domain_validation", {})
    core_benchmark = summary.get("benchmark_core", {})
    inference_benchmark = summary.get("inference_benchmark", {})
    epiplexity = summary.get("epiplexity", {})
    score = summary.get("score", {})

    lines = [
        "# Run Scorecard",
        "",
        f"- run: `{run['name']}`",
        f"- run_dir: `{run['run_dir']}`",
        f"- status: `{summary['status']}`",
        f"- decision_hint: `{summary['decision_hint']}`",
    ]

    if score:
        lines.extend(
            [
                f"- score: `{_format_optional(score.get('final_score'))}`",
                f"- score_eligible: `{score.get('eligible')}`",
                f"- baseline: `{_format_optional(score.get('baseline_run_name'))}`",
            ]
        )
        if not score.get("eligible") and score.get("missing"):
            lines.append(f"- score_missing: `{', '.join(score['missing'])}`")

    if score.get("eligible"):
        lines.extend(
            [
                "",
                "## Run Score",
                "",
                f"- final_score: `{_format_optional(score.get('final_score'))}`",
                f"- quality: `{_format_optional(_nested_metric(score, ['quality', 'value']))}`",
                f"- training_efficiency: `{_format_optional(_nested_metric(score, ['training_efficiency', 'value']))}`",
                f"- inference_efficiency: `{_format_optional(_nested_metric(score, ['inference_efficiency', 'value']))}`",
                f"- health: `{_format_optional(_nested_metric(score, ['health', 'value']))}`",
            ]
        )

    lines.extend(
        [
            "",
        "## Model",
        "",
        f"- params: `{model['params']}`",
        f"- layers: `{model['layers']}`",
        f"- hidden_size: `{model['hidden_size']}`",
        f"- heads: `{model['heads']}`",
        f"- kv_heads: `{model['kv_heads']}`",
        f"- seq_len: `{model['seq_len']}`",
        f"- vocab_size: `{model['vocab_size']}`",
        "",
        "## Training",
        "",
        f"- final_step: `{training['final_step']}`",
        f"- steps_completed: `{training['steps_completed']}`",
        f"- tokens_seen: `{_format_optional(training['tokens_seen'])}`",
        f"- configured_tokens: `{training['configured_tokens']}`",
        "",
        "## Training Native Validation",
        "",
        f"- final_train_loss: `{_format_optional(quality['final_train_loss'])}`",
        f"- best_train_loss: `{_format_optional(quality['best_train_loss'])}`",
        f"- first_val_loss: `{_format_optional(quality['first_val_loss'])}`",
        f"- final_val_loss: `{_format_optional(quality['final_val_loss'])}`",
        f"- best_val_loss: `{_format_optional(quality['best_val_loss'])}`",
        f"- final_val_bpb: `{_format_optional(quality['final_val_bpb'])}`",
        f"- best_val_bpb: `{_format_optional(quality['best_val_bpb'])}`",
        "",
        "## Health",
        "",
        f"- nan_count: `{health['nan_count']}`",
        f"- loss_spike_count: `{health['loss_spike_count']}`",
        f"- grad_norm_spike_count: `{health['grad_norm_spike_count']}`",
        f"- final_train_val_gap: `{_format_optional(health['final_train_val_gap'])}`",
        f"- final_train_loss_slope: `{_format_optional(health['final_train_loss_slope'])}`",
        f"- final_val_loss_slope: `{_format_optional(health['final_val_loss_slope'])}`",
        "",
        "## Speed",
        "",
        f"- avg_tokens_per_sec: `{_format_optional(speed['avg_tokens_per_sec'])}`",
        f"- avg_train_tokens_per_sec: `{_format_optional(speed['avg_train_tokens_per_sec'])}`",
        f"- final_elapsed_sec: `{_format_optional(speed['final_elapsed_sec'])}`",
        "",
        "## Performance",
        "",
        f"- final_mfu: `{_format_optional(performance.get('final_mfu'))}`",
        f"- avg_mfu: `{_format_optional(performance.get('avg_mfu'))}`",
        f"- flops_per_token: `{_format_optional(performance.get('flops_per_token'))}`",
        f"- avg_train_tokens_per_gpu_hour: `{_format_optional(performance.get('avg_train_tokens_per_gpu_hour'))}`",
        f"- peak_gpu_memory_bytes: `{_format_optional(performance.get('peak_gpu_memory_bytes'))}`",
        f"- avg_gpu_utilization_pct: `{_format_optional(performance.get('avg_gpu_utilization_pct'))}`",
        f"- avg_gpu_power_w: `{_format_optional(performance.get('avg_gpu_power_w'))}`",
        "",
        "## Epiplexity Proxy",
        "",
        f"- train_bpb_auc: `{_format_optional(epiplexity.get('train_bpb_auc'))}`",
        f"- train_bpb_auc_per_byte: `{_format_optional(epiplexity.get('train_bpb_auc_per_byte'))}`",
        f"- val_bpb_auc: `{_format_optional(epiplexity.get('val_bpb_auc'))}`",
        f"- val_bpb_auc_per_byte: `{_format_optional(epiplexity.get('val_bpb_auc_per_byte'))}`",
        ]
    )

    if evals["count"]:
        latest_eval = evals["latest"] or {}
        best_loss_eval = evals["best_loss"] or {}
        best_bpb_eval = evals["best_bpb"] or {}
        lines.extend(
            [
                "",
                "## Checkpoint Native Validation",
                "",
                f"- count: `{evals['count']}`",
                f"- latest_step: `{_format_optional(latest_eval.get('checkpoint_step'))}`",
                f"- latest_loss: `{_format_optional(latest_eval.get('loss'))}`",
                f"- latest_bpb: `{_format_optional(latest_eval.get('bpb'))}`",
                f"- best_step: `{_format_optional(best_loss_eval.get('checkpoint_step'))}`",
                f"- best_loss: `{_format_optional(best_loss_eval.get('loss'))}`",
                f"- best_bpb: `{_format_optional(best_bpb_eval.get('bpb'))}`",
            ]
        )

    training_domains = domains.get("training", {})
    if training_domains:
        lines.extend(
            [
                "",
                "## Training Domain Validation",
                "",
                f"{'domain':<12} {'first_loss':>12} {'final_loss':>12} {'best_loss':>12} {'delta':>12} {'final_bpb':>12} {'best_bpb':>12}",
                "-" * 90,
            ]
        )
        for name, domain in training_domains.items():
            lines.append(
                f"{name:<12} "
                f"{_format_optional(domain.get('first_loss')):>12} "
                f"{_format_optional(domain.get('final_loss')):>12} "
                f"{_format_optional(domain.get('best_loss')):>12} "
                f"{_format_optional(domain.get('delta_loss')):>12} "
                f"{_format_optional(domain.get('final_bpb')):>12} "
                f"{_format_optional(domain.get('best_bpb')):>12}"
            )

    checkpoint_domains = domains.get("checkpoint_evals", {})
    if checkpoint_domains:
        lines.extend(
            [
                "",
                "## Checkpoint Domain Validation",
                "",
                f"{'domain':<12} {'latest_step':>12} {'latest_loss':>12} {'best_loss':>12} {'latest_bpb':>12} {'best_bpb':>12}",
                "-" * 77,
            ]
        )
        for name, domain in checkpoint_domains.items():
            latest = domain.get("latest") or {}
            best_loss = domain.get("best_loss") or {}
            best_bpb = domain.get("best_bpb") or {}
            lines.append(
                f"{name:<12} "
                f"{_format_optional(latest.get('checkpoint_step')):>12} "
                f"{_format_optional(latest.get('loss')):>12} "
                f"{_format_optional(best_loss.get('loss')):>12} "
                f"{_format_optional(latest.get('bpb')):>12} "
                f"{_format_optional(best_bpb.get('bpb')):>12}"
            )

    if core_benchmark.get("count"):
        latest_core = core_benchmark.get("latest") or {}
        best_core = core_benchmark.get("best") or {}
        lines.extend(
            [
                "",
                "## Benchmark CORE",
                "",
                f"- count: `{core_benchmark['count']}`",
                f"- latest_step: `{_format_optional(latest_core.get('checkpoint_step'))}`",
                f"- latest_core: `{_format_optional(latest_core.get('core'))}`",
                f"- best_step: `{_format_optional(best_core.get('checkpoint_step'))}`",
                f"- best_core: `{_format_optional(best_core.get('core'))}`",
                "",
                f"{'task':<35} {'accuracy':>10} {'centered':>10} {'baseline':>10} {'examples':>10}",
                "-" * 81,
            ]
        )
        for task_name, task_metrics in (latest_core.get("tasks") or {}).items():
            lines.append(
                f"{task_name:<35} "
                f"{_format_optional(task_metrics.get('accuracy')):>10} "
                f"{_format_optional(task_metrics.get('centered')):>10} "
                f"{_format_optional(task_metrics.get('random_baseline')):>10} "
                f"{_format_optional(task_metrics.get('examples')):>10}"
            )

    if inference_benchmark.get("count"):
        latest_inference = inference_benchmark.get("latest") or {}
        lines.extend(
            [
                "",
                "## Inference Benchmark",
                "",
                f"- latest_step: `{_format_optional(latest_inference.get('checkpoint_step'))}`",
                f"- mode: `{_format_optional(latest_inference.get('mode'))}`",
                f"- decode_tokens_per_sec: `{_format_optional(latest_inference.get('decode_tokens_per_sec'))}`",
                f"- prefill_tokens_per_sec: `{_format_optional(latest_inference.get('prefill_tokens_per_sec'))}`",
                f"- ttft_sec: `{_format_optional(latest_inference.get('ttft_sec'))}`",
            ]
        )

    return "\n".join(lines) + "\n"


def write_summary_artifacts(run_dir: str | Path, summary: dict[str, Any], markdown: str) -> tuple[Path, Path]:
    summary_dir = Path(run_dir) / SUMMARY_DIR_NAME
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / SUMMARY_JSON_NAME
    md_path = summary_dir / SCORECARD_NAME
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def summarize_and_write(run_dir: str | Path) -> tuple[dict[str, Any], Path, Path, str]:
    summary = summarize_run(run_dir)
    markdown = format_scorecard(summary)
    json_path, md_path = write_summary_artifacts(run_dir, summary, markdown)
    return summary, json_path, md_path, markdown


def register_summary(summary: dict[str, Any], registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    records = _load_registry(registry_path)
    record = registry_record(summary)
    records = [
        existing
        for existing in records
        if not (existing.get("run_name") == record["run_name"] and existing.get("run_dir") == record["run_dir"])
    ]
    records.append(record)
    with registry_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    return registry_path


def registry_record(summary: dict[str, Any]) -> dict[str, Any]:
    run = summary["run"]
    model = summary["model"]
    training = summary["training"]
    quality = summary["quality"]
    speed = summary["speed"]
    performance = summary.get("performance", {})
    epiplexity = summary.get("epiplexity", {})
    health = summary["health"]
    score = summary.get("score", {})
    return {
        "run_name": run["name"],
        "run_dir": run["run_dir"],
        "created_at": run.get("created_at"),
        "model_params": model["params"],
        "tokens_seen": training["tokens_seen"],
        "final_step": training["final_step"],
        "final_val_loss": quality["final_val_loss"],
        "final_val_bpb": quality["final_val_bpb"],
        "best_val_loss": quality["best_val_loss"],
        "best_val_bpb": quality["best_val_bpb"],
        "avg_tokens_per_sec": speed["avg_tokens_per_sec"],
        "nan_count": health["nan_count"],
        "status": summary["status"],
        "decision_hint": summary["decision_hint"],
        "score": score.get("final_score"),
        "score_eligible": score.get("eligible"),
        "score_missing": score.get("missing"),
        "baseline_run_name": score.get("baseline_run_name"),
        "quality_score": _nested_metric(score, ["quality", "value"]),
        "training_efficiency_score": _nested_metric(score, ["training_efficiency", "value"]),
        "inference_efficiency_score": _nested_metric(score, ["inference_efficiency", "value"]),
        "health_score": _nested_metric(score, ["health", "value"]),
        "avg_mfu": performance.get("avg_mfu"),
        "final_mfu": performance.get("final_mfu"),
        "train_tokens_per_gpu_hour": performance.get("avg_train_tokens_per_gpu_hour"),
        "peak_gpu_memory_bytes": performance.get("peak_gpu_memory_bytes"),
        "checkpoint_eval_count": summary["checkpoint_evals"]["count"],
        "latest_checkpoint_step": _nested_metric(summary, ["checkpoint_evals", "latest", "checkpoint_step"]),
        "latest_checkpoint_loss": _nested_metric(summary, ["checkpoint_evals", "latest", "loss"]),
        "latest_checkpoint_bpb": _nested_metric(summary, ["checkpoint_evals", "latest", "bpb"]),
        "best_checkpoint_loss": _nested_metric(summary, ["checkpoint_evals", "best_loss", "loss"]),
        "best_checkpoint_bpb": _nested_metric(summary, ["checkpoint_evals", "best_bpb", "bpb"]),
        "latest_core": _nested_metric(summary, ["benchmark_core", "latest", "core"]),
        "best_core": _nested_metric(summary, ["benchmark_core", "best", "core"]),
        "latest_core_step": _nested_metric(summary, ["benchmark_core", "latest", "checkpoint_step"]),
        "latest_decode_tokens_per_sec": _nested_metric(
            summary, ["inference_benchmark", "latest", "decode_tokens_per_sec"]
        ),
        "latest_prefill_tokens_per_sec": _nested_metric(
            summary, ["inference_benchmark", "latest", "prefill_tokens_per_sec"]
        ),
        "latest_ttft_sec": _nested_metric(summary, ["inference_benchmark", "latest", "ttft_sec"]),
        "train_epiplexity_bpb_auc": epiplexity.get("train_bpb_auc"),
        "train_epiplexity_bpb_auc_per_byte": epiplexity.get("train_bpb_auc_per_byte"),
        "val_epiplexity_bpb_auc": epiplexity.get("val_bpb_auc"),
        "val_epiplexity_bpb_auc_per_byte": epiplexity.get("val_bpb_auc_per_byte"),
        **_latest_checkpoint_domain_rollup(summary),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Summarize a training run.")
    parser.add_argument("run_dir", help="Run directory containing config.toml and metrics.jsonl.")
    parser.add_argument("--register", action="store_true", help="Upsert the run into runs/registry.jsonl.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH), help="Registry JSONL path.")
    args = parser.parse_args(argv)

    summary, json_path, md_path, _ = summarize_and_write(args.run_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.register:
        registry_path = register_summary(summary, args.registry_path)
        print(f"wrote {registry_path}")


def _quality_summary(rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_row = rows[-1]
    return {
        "final_train_loss": _number_or_none(final_row.get("train/loss")),
        "final_train_ppl": _number_or_none(final_row.get("train/ppl")),
        "final_train_bpb": _number_or_none(final_row.get("train/bpb")),
        "best_train_loss": _min_metric(rows, "train/loss"),
        "best_train_bpb": _min_metric(rows, "train/bpb"),
        "final_val_loss": _last_metric(val_rows, "val/loss"),
        "final_val_ppl": _last_metric(val_rows, "val/ppl"),
        "final_val_bpb": _last_metric(val_rows, "val/bpb"),
        "best_val_loss": _min_metric(val_rows, "val/loss"),
        "best_val_bpb": _min_metric(val_rows, "val/bpb"),
        "first_val_loss": _first_metric(val_rows, "val/loss"),
    }


def _health_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "nan_count": int(_max_metric(rows, "health/nan_count") or 0),
        "loss_spike_count": int(_max_metric(rows, "health/loss_spike_count") or 0),
        "grad_norm_spike_count": int(_max_metric(rows, "health/grad_norm_spike_count") or 0),
        "spike_rate": _last_metric(rows, "health/spike_rate"),
        "final_train_val_gap": _last_metric(rows, "health/train_val_gap"),
        "final_train_loss_slope": _last_metric(rows, "health/train_loss_slope"),
        "final_val_loss_slope": _last_metric(rows, "health/val_loss_slope"),
    }


def _performance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "device_count": _last_metric(rows, "system/device_count"),
        "device_kind": _last_value(rows, "system/device_kind"),
        "flops_per_token": _last_metric(rows, "perf/flops_per_token"),
        "flops_per_step": _last_metric(rows, "perf/flops_per_step"),
        "peak_flops_per_device": _last_metric(rows, "perf/peak_flops_per_device"),
        "peak_flops_total": _last_metric(rows, "perf/peak_flops_total"),
        "final_mfu": _last_metric(rows, "perf/mfu"),
        "avg_mfu": _mean_metric(rows, "perf/mfu"),
        "avg_train_tokens_per_gpu_hour": _mean_metric(rows, "time/train_tokens_per_gpu_hour"),
        "final_train_tokens_per_gpu_hour": _last_metric(rows, "time/train_tokens_per_gpu_hour"),
        "peak_gpu_memory_bytes": _max_metric(rows, "system/gpu_memory_peak_bytes")
        or _max_metric(rows, "system/gpu_memory_used_bytes"),
        "avg_gpu_utilization_pct": _mean_metric(rows, "system/gpu_utilization_pct"),
        "avg_gpu_power_w": _mean_metric(rows, "system/gpu_power_w"),
    }


def _checkpoint_eval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_loss_rows = [row for row in rows if _is_finite(row.get("loss"))]
    finite_bpb_rows = [row for row in rows if _is_finite(row.get("bpb"))]
    return {
        "count": len(rows),
        "latest": rows[-1] if rows else None,
        "best_loss": min(finite_loss_rows, key=lambda row: float(row["loss"])) if finite_loss_rows else None,
        "best_bpb": min(finite_bpb_rows, key=lambda row: float(row["bpb"])) if finite_bpb_rows else None,
    }


def _core_benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_rows = [row for row in rows if _is_finite(row.get("core"))]
    return {
        "count": len(rows),
        "latest": rows[-1] if rows else None,
        "best": max(finite_rows, key=lambda row: float(row["core"])) if finite_rows else None,
    }


def _inference_benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "latest": rows[-1] if rows else None,
    }


def _epiplexity_summary(rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], train_config) -> dict[str, Any]:
    train = _train_bpb_auc(rows, train_config)
    val = _val_bpb_auc(val_rows, train_config)
    return {
        "train_bpb_auc": train["bpb_auc"],
        "train_bpb_auc_per_byte": train["bpb_auc_per_byte"],
        "train_bytes": train["bytes"],
        "train_points": train["points"],
        "val_bpb_auc": val["bpb_auc"],
        "val_bpb_auc_per_byte": val["bpb_auc_per_byte"],
        "val_bytes": val["bytes"],
        "val_points": val["points"],
    }


def _train_bpb_auc(rows: list[dict[str, Any]], train_config) -> dict[str, Any]:
    points = _points_from_cumulative_bytes(rows, "train/bpb", "train/bytes_seen")
    if points is None:
        points = _points_from_row_bytes(
            rows,
            "train/bpb",
            "train/bytes",
            "train/loss",
            token_delta_key="train/tokens_seen",
            target_token_ratio=(train_config.seq_len - 1) / train_config.seq_len,
        )
    return _auc_from_bpb_points(points)


def _val_bpb_auc(val_rows: list[dict[str, Any]], train_config) -> dict[str, Any]:
    default_target_tokens = train_config.eval_steps * train_config.batch_size * (train_config.seq_len - 1)
    points = _points_from_row_bytes(
        val_rows,
        "val/bpb",
        "val/bytes",
        "val/loss",
        default_target_tokens=default_target_tokens,
    )
    return _auc_from_bpb_points(points)


def _points_from_cumulative_bytes(rows: list[dict[str, Any]], bpb_key: str, bytes_seen_key: str) -> list[tuple[float, float]] | None:
    points = []
    for row in rows:
        bpb = _number_or_none(row.get(bpb_key))
        bytes_seen = _number_or_none(row.get(bytes_seen_key))
        if bpb is None:
            continue
        if bytes_seen is None:
            return None
        points.append((float(bpb), float(bytes_seen)))
    return points


def _points_from_row_bytes(
    rows: list[dict[str, Any]],
    bpb_key: str,
    bytes_key: str,
    loss_key: str,
    *,
    token_delta_key: str | None = None,
    target_token_ratio: float = 1.0,
    default_target_tokens: int | None = None,
) -> list[tuple[float, float]]:
    points = []
    cumulative_bytes = 0.0
    previous_tokens = 0.0
    for row in rows:
        bpb = _number_or_none(row.get(bpb_key))
        if bpb is None:
            continue

        row_bytes = _number_or_none(row.get(bytes_key))
        if row_bytes is None:
            target_tokens = default_target_tokens
            if target_tokens is None and token_delta_key is not None:
                tokens_seen = _number_or_none(row.get(token_delta_key))
                if tokens_seen is not None:
                    target_tokens = max(0.0, float(tokens_seen) - previous_tokens) * target_token_ratio
                    previous_tokens = float(tokens_seen)
            row_bytes = _estimate_row_bytes(row, loss_key, bpb_key, target_tokens)

        if row_bytes is None or row_bytes <= 0:
            continue
        cumulative_bytes += float(row_bytes)
        points.append((float(bpb), cumulative_bytes))
    return points


def _estimate_row_bytes(row: dict[str, Any], loss_key: str, bpb_key: str, target_tokens: float | int | None) -> float | None:
    loss = _number_or_none(row.get(loss_key))
    bpb = _number_or_none(row.get(bpb_key))
    if loss is None or bpb is None or bpb <= 0 or target_tokens is None or target_tokens <= 0:
        return None
    return float(loss) * float(target_tokens) / (math.log(2) * float(bpb))


def _auc_from_bpb_points(points: list[tuple[float, float]] | None) -> dict[str, Any]:
    if not points or len(points) < 2:
        return {"bpb_auc": None, "bpb_auc_per_byte": None, "bytes": None, "points": len(points or [])}

    final_bpb = points[-1][0]
    previous_bytes = 0.0
    total_bytes = 0.0
    auc = 0.0
    for bpb, cumulative_bytes in points:
        delta_bytes = cumulative_bytes - previous_bytes
        if delta_bytes <= 0:
            continue
        auc += max(0.0, bpb - final_bpb) * delta_bytes
        total_bytes += delta_bytes
        previous_bytes = cumulative_bytes

    if total_bytes <= 0:
        return {"bpb_auc": None, "bpb_auc_per_byte": None, "bytes": None, "points": len(points)}
    return {
        "bpb_auc": auc,
        "bpb_auc_per_byte": auc / total_bytes,
        "bytes": total_bytes,
        "points": len(points),
    }


def _domain_validation_summary(rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            parts[2]
            for row in rows
            for key in row
            if (parts := key.split("/")) and len(parts) == 4 and parts[:2] == ["val", "domain"]
        }
    )
    training = {}
    for name in names:
        prefix = f"val/domain/{name}"
        first_loss = _first_metric(rows, f"{prefix}/loss")
        final_loss = _last_metric(rows, f"{prefix}/loss")
        training[name] = {
            "first_loss": first_loss,
            "final_loss": final_loss,
            "best_loss": _min_metric(rows, f"{prefix}/loss"),
            "delta_loss": final_loss - first_loss if final_loss is not None and first_loss is not None else None,
            "final_ppl": _last_metric(rows, f"{prefix}/ppl"),
            "final_bpb": _last_metric(rows, f"{prefix}/bpb"),
            "best_bpb": _min_metric(rows, f"{prefix}/bpb"),
            "final_tokens": _last_metric(rows, f"{prefix}/tokens"),
        }

    checkpoint_names = sorted({name for row in eval_rows for name in row.get("domains", {})})
    checkpoint_evals = {}
    for name in checkpoint_names:
        domain_rows = [row["domains"][name] | {"checkpoint_step": row.get("checkpoint_step")} for row in eval_rows if name in row.get("domains", {})]
        finite_loss_rows = [row for row in domain_rows if _is_finite(row.get("loss"))]
        finite_bpb_rows = [row for row in domain_rows if _is_finite(row.get("bpb"))]
        checkpoint_evals[name] = {
            "latest": domain_rows[-1] if domain_rows else None,
            "best_loss": min(finite_loss_rows, key=lambda row: float(row["loss"])) if finite_loss_rows else None,
            "best_bpb": min(finite_bpb_rows, key=lambda row: float(row["bpb"])) if finite_bpb_rows else None,
        }

    return {
        "training": training,
        "checkpoint_evals": checkpoint_evals,
    }


def _verdict(config, rows: list[dict[str, Any]], final_row: dict[str, Any], val_rows: list[dict[str, Any]], health: dict[str, Any]) -> tuple[str, str]:
    final_train_loss = final_row.get("train/loss")
    final_val_raw = val_rows[-1].get("val/loss") if val_rows else None
    final_val_loss = _number_or_none(final_val_raw)
    final_step = int(final_row["step"])
    completed = final_step >= config.train.steps - max(config.train.log_every, 1)

    if health["nan_count"] > 0 or not _is_finite(final_train_loss) or (val_rows and not _is_finite(final_val_raw)):
        return "failed", "discard"
    if not completed:
        return "incomplete", "inspect"

    best_val_loss = _min_metric(val_rows, "val/loss")
    val_regressed = (
        final_val_loss is not None
        and best_val_loss is not None
        and final_val_loss > best_val_loss + VAL_REGRESSION_THRESHOLD
    )
    if health["loss_spike_count"] > 0 or health["grad_norm_spike_count"] > 0 or val_regressed:
        return "unstable", "inspect"

    first_val_loss = _first_metric(val_rows, "val/loss")
    if first_val_loss is not None and final_val_loss is not None and final_val_loss < first_val_loss:
        return "healthy", "scale"
    return "unstable", "retry"


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_metrics_sort_key(path: Path) -> tuple[int, str]:
    try:
        step = int(path.parent.name.removeprefix("step_"))
    except ValueError:
        step = -1
    return step, str(path)


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in rows:
        value = _number_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _last_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = _number_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _last_value(rows: list[dict[str, Any]], key: str) -> Any | None:
    for row in reversed(rows):
        if key in row and row[key] is not None:
            return row[key]
    return None


def _min_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _nested_metric(row: dict[str, Any], path: list[str]) -> float | int | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _number_or_none(value)


def _latest_checkpoint_domain_rollup(summary: dict[str, Any]) -> dict[str, Any]:
    latest = summary["checkpoint_evals"].get("latest") or {}
    domains = latest.get("domains") or {}
    domain_losses = [
        (name, _number_or_none(metrics.get("loss")))
        for name, metrics in domains.items()
        if isinstance(metrics, dict)
    ]
    domain_losses = [(name, loss) for name, loss in domain_losses if loss is not None]
    if not domain_losses:
        return {
            "latest_domain_mean_loss": None,
            "latest_domain_worst_name": None,
            "latest_domain_worst_loss": None,
        }

    worst_name, worst_loss = max(domain_losses, key=lambda item: item[1])
    return {
        "latest_domain_mean_loss": mean(loss for _, loss in domain_losses),
        "latest_domain_worst_name": worst_name,
        "latest_domain_worst_loss": worst_loss,
    }


def _number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
