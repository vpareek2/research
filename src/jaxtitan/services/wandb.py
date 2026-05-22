"""Optional W&B mirror for canonical local Jaxtitan artifacts."""

from collections.abc import Mapping
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any

from jaxtitan.errors import ContractError
from jaxtitan.services.artifacts import LocalArtifactWriter
from jaxtitan.specs.run import RunManifest, RunSpec

_SKIP_SCALAR_KEYS = {
    "optimizer_groups",
    "moe_router_layers",
    "final_optimizer_groups",
    "final_moe_router_layers",
}
_PERF_KEYS = {
    "tokens_per_sec",
    "train_tokens_per_sec",
    "examples_per_sec",
    "flops_per_token",
    "flops_per_step",
    "flops_per_sec",
    "mfu",
    "peak_flops_total",
    "device_memory_used_bytes",
    "device_memory_peak_bytes",
    "device_memory_limit_bytes",
    "gpu_memory_used_bytes",
    "gpu_memory_total_bytes",
    "gpu_utilization_pct",
    "gpu_memory_utilization_pct",
    "gpu_power_w",
    "gpu_temperature_c",
}
_DATA_KEYS = {
    "token_start",
    "token_end",
    "token_count",
    "tokens_seen",
    "examples",
    "target_tokens",
    "global_batch_size",
    "per_device_batch_size",
    "data_axis_size",
    "data_sec",
    "placement_sec",
    "data_worker_count",
    "data_worker_buffer_size",
    "data_prefetch",
}
_ROUTER_PREFIXES = ("router_", "smebu_")
_OPTIMIZER_PREFIXES = ("optimizer_",)


class WandbMirror:
    """Small W&B adapter that mirrors selected local artifact rows."""

    def __init__(self, *, spec: RunSpec, metadata: Mapping[str, Any], wandb_module: Any, run: Any) -> None:
        self.spec = spec
        self.metadata = dict(metadata)
        self._wandb = wandb_module
        self._run = run
        self._finished = False

    @classmethod
    def start_new(cls, *, spec: RunSpec, manifest: RunManifest) -> "WandbMirror":
        run_id = wandb_run_id_for_manifest(manifest)
        return cls._start(spec=spec, run_id=run_id, resume="allow")

    @classmethod
    def resume_existing(cls, *, spec: RunSpec, run_dir: Path) -> "WandbMirror":
        metadata = read_wandb_metadata(run_dir)
        run_id = _required_str(metadata, "wandb_run_id", "wandb metadata")
        return cls._start(spec=spec, run_id=run_id, resume="must")

    @classmethod
    def _start(cls, *, spec: RunSpec, run_id: str, resume: str) -> "WandbMirror":
        try:
            wandb_module = importlib.import_module("wandb")
            run = wandb_module.init(
                project=spec.artifacts.wandb_project,
                entity=spec.artifacts.wandb_entity,
                group=spec.artifacts.wandb_group,
                tags=list(spec.artifacts.wandb_tags),
                mode=spec.artifacts.wandb_mode,
                id=run_id,
                resume=resume,
                config={
                    "run_id": spec.run_id,
                    "model": spec.model.name,
                    "model_variant": spec.model.variant,
                    "optimizer": spec.optimizer.name,
                    "parallelism": spec.parallelism.mode,
                    "data_mode": spec.data.mode,
                },
            )
        except Exception as exc:
            raise ContractError(f"failed to initialize W&B mirror: {exc}") from exc
        metadata = wandb_metadata_from_run(spec=spec, run=run, run_id=run_id, resume=resume)
        return cls(spec=spec, metadata=metadata, wandb_module=wandb_module, run=run)

    def log_train_metrics(self, row: Mapping[str, Any]) -> None:
        payload = normalize_metrics_for_wandb(row, default_namespace="train")
        self._log(payload, step=_optional_step(row))

    def log_eval_metrics(self, row: Mapping[str, Any]) -> None:
        payload = normalize_metrics_for_wandb(row, default_namespace="eval")
        self._log(payload, step=_optional_step(row))

    def log_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in {
            "training_started",
            "checkpoint_saved",
            "training_completed",
            "training_failed",
        }:
            return
        payload = normalize_metrics_for_wandb(event, default_namespace=f"event/{event_type}")
        payload[f"event/{event_type}"] = 1
        self._log(payload, step=_optional_step(event))

    def log_runtime_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        summary = {
            "diagnostics/device_count": _nested_get(diagnostics, ("performance", "device_count")),
            "diagnostics/flops_per_token": _nested_get(diagnostics, ("performance", "flops_per_token")),
            "diagnostics/peak_flops_total": _nested_get(diagnostics, ("performance", "peak_flops_total")),
        }
        self._summary_update({key: value for key, value in summary.items() if _is_scalar(value)})

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        payload = normalize_metrics_for_wandb(summary, default_namespace="final")
        self._summary_update(payload)
        tables = final_tables_for_wandb(summary, self._wandb)
        if tables:
            self._log(tables, step=_optional_step(summary))

    def finish(self) -> None:
        if self._finished:
            return
        try:
            if hasattr(self._run, "finish"):
                self._run.finish()
            elif hasattr(self._wandb, "finish"):
                self._wandb.finish()
        finally:
            self._finished = True

    def _log(self, payload: Mapping[str, Any], *, step: int | None) -> None:
        if not payload:
            return
        if hasattr(self._run, "log"):
            self._run.log(dict(payload), step=step)
        else:
            self._wandb.log(dict(payload), step=step)

    def _summary_update(self, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        summary = getattr(self._run, "summary", None)
        if hasattr(summary, "update"):
            summary.update(dict(payload))
        elif isinstance(summary, dict):
            summary.update(dict(payload))


class MirroredArtifactWriter:
    """Artifact writer that writes local artifacts first, then mirrors to W&B."""

    def __init__(self, local: LocalArtifactWriter, mirror: WandbMirror) -> None:
        self.local = local
        self.mirror = mirror
        self.run_dir = local.run_dir
        self._failed = False

    @property
    def wandb_metadata(self) -> dict[str, Any]:
        return dict(self.mirror.metadata)

    def write_config(self, source_toml: str, resolved: RunSpec) -> None:
        self.local.write_config(source_toml, resolved)

    def append_event(self, event: Mapping[str, Any]) -> None:
        self.local.append_event(event)
        self._mirror("event", lambda: self.mirror.log_event(event))

    def append_train_metrics(self, row: Mapping[str, Any]) -> None:
        self.local.append_train_metrics(row)
        self._mirror("train_metrics", lambda: self.mirror.log_train_metrics(row))

    def append_eval_metrics(self, row: Mapping[str, Any]) -> None:
        self.local.append_eval_metrics(row)
        self._mirror("eval_metrics", lambda: self.mirror.log_eval_metrics(row))

    def write_runtime_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        self.local.write_runtime_diagnostics(diagnostics)
        self._mirror("runtime_diagnostics", lambda: self.mirror.log_runtime_diagnostics(diagnostics))

    def write_profiling_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        self.local.write_profiling_diagnostics(diagnostics)

    def write_wandb_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.local.write_wandb_metadata(metadata)

    def write_checkpoint_index(self, index: Mapping[str, Any]) -> None:
        self.local.write_checkpoint_index(index)

    def append_checkpoint_sample(self, step: int, row: Mapping[str, Any]) -> None:
        self.local.append_checkpoint_sample(step, row)

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self.local.write_summary(summary)
        self._mirror("summary", lambda: self.mirror.log_summary(summary))

    def close(self) -> None:
        self.mirror.finish()

    def _mirror(self, phase: str, callback: Any) -> None:
        if self._failed:
            return
        try:
            callback()
        except Exception as exc:
            self._failed = True
            self.local.append_event(
                {
                    "schema_version": 1,
                    "type": "wandb_failed",
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise ContractError(f"W&B mirror failed during {phase}: {exc}") from exc


def build_artifact_writer(
    *,
    spec: RunSpec,
    local: LocalArtifactWriter,
    manifest: RunManifest | None,
    resume: bool,
) -> LocalArtifactWriter | MirroredArtifactWriter:
    """Return the local writer or a W&B mirroring wrapper."""

    if not spec.artifacts.wandb_enabled:
        return local
    try:
        mirror = (
            WandbMirror.resume_existing(spec=spec, run_dir=local.run_dir)
            if resume
            else WandbMirror.start_new(spec=spec, manifest=_require_manifest(manifest))
        )
        local.write_wandb_metadata(mirror.metadata)
        return MirroredArtifactWriter(local, mirror)
    except Exception as exc:
        local.append_event(
            {
                "schema_version": 1,
                "type": "wandb_failed",
                "phase": "init",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"failed to initialize W&B mirror: {exc}") from exc


def wandb_run_id_for_manifest(manifest: RunManifest) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", manifest.run_id).strip("-")
    safe_run_id = safe_run_id or "run"
    return f"{safe_run_id}-{manifest.resolved_config_sha256[:16]}"


def wandb_metadata_from_run(*, spec: RunSpec, run: Any, run_id: str, resume: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "enabled": True,
        "wandb_run_id": str(getattr(run, "id", None) or run_id),
        "project": spec.artifacts.wandb_project,
        "entity": spec.artifacts.wandb_entity,
        "group": spec.artifacts.wandb_group,
        "tags": list(spec.artifacts.wandb_tags),
        "mode": spec.artifacts.wandb_mode,
        "url": getattr(run, "url", None),
        "name": getattr(run, "name", None),
        "resume": resume,
    }


def read_wandb_metadata(run_dir: str | Path) -> Mapping[str, Any]:
    path = Path(run_dir) / "diagnostics" / "wandb.json"
    if not path.is_file():
        raise ContractError("W&B resume requested but diagnostics/wandb.json is missing")
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise ContractError(f"failed to read W&B metadata {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ContractError("diagnostics/wandb.json must be a JSON object")
    return raw


def normalize_metrics_for_wandb(row: Mapping[str, Any], *, default_namespace: str) -> dict[str, int | float | bool]:
    payload: dict[str, int | float | bool] = {}
    for key, value in row.items():
        if key in _SKIP_SCALAR_KEYS or key == "step":
            continue
        if not _is_scalar(value):
            continue
        payload[_metric_key(default_namespace, key)] = value
    return payload


def final_tables_for_wandb(summary: Mapping[str, Any], wandb_module: Any) -> dict[str, Any]:
    tables = {}
    optimizer_groups = summary.get("final_optimizer_groups")
    if isinstance(optimizer_groups, list) and optimizer_groups:
        tables["final/optimizer_groups"] = _table(wandb_module, optimizer_groups)
    router_layers = summary.get("final_moe_router_layers")
    if isinstance(router_layers, list) and router_layers:
        tables["final/moe_router_layers"] = _table(wandb_module, router_layers)
    return tables


def _table(wandb_module: Any, rows: list[Any]) -> Any:
    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    columns = sorted({str(key) for row in normalized_rows for key in row})
    data = [[row.get(column) for column in columns] for row in normalized_rows]
    return wandb_module.Table(columns=columns, data=data)


def _metric_key(default_namespace: str, key: str) -> str:
    if key in _PERF_KEYS:
        return f"perf/{key}"
    if key in _DATA_KEYS:
        return f"data/{key}"
    if key.startswith(_ROUTER_PREFIXES):
        return f"router/{key}"
    if key.startswith(_OPTIMIZER_PREFIXES):
        return f"optimizer/{key}"
    return f"{default_namespace}/{key}"


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return False


def _optional_step(row: Mapping[str, Any]) -> int | None:
    value = row.get("step")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _nested_get(raw: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = raw
    for item in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(item)
    return value


def _require_manifest(manifest: RunManifest | None) -> RunManifest:
    if manifest is None:
        raise ContractError("W&B new run initialization requires a run manifest")
    return manifest


def _required_str(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}.{key} must be a non-empty string")
    return value
