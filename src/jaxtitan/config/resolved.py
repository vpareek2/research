"""Resolved RunSpec JSON loading."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from jaxtitan.config.validate import validate_run_spec
from jaxtitan.errors import ConfigError, ContractError
from jaxtitan.specs.data import DataSpec, HFStreamingSpec
from jaxtitan.specs.eval import EvalSpec
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ParamRouteRule, ScheduleSpec
from jaxtitan.specs.parallelism import ParallelismSpec
from jaxtitan.specs.run import ArtifactSpec, KernelSpec, ProfilingSpec, RunSpec, TrainingSpec


def load_resolved_config(path: str | Path) -> RunSpec:
    """Load a canonical resolved RunSpec JSON artifact."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text())
    except OSError as exc:
        raise ConfigError(f"failed to read resolved config {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"failed to parse resolved config {config_path}: {exc}") from exc
    return run_spec_from_resolved_mapping(_require_mapping(raw, "resolved config"))


def run_spec_from_resolved_mapping(raw: Mapping[str, Any]) -> RunSpec:
    """Convert resolved RunSpec JSON-compatible data back into a RunSpec."""

    try:
        optimizer_raw = _required_mapping(raw, "optimizer")
        optimizer = OptimizerSpec(
            name=_required_str(optimizer_raw, "name", "optimizer"),
            schedule=ScheduleSpec(**dict(_required_mapping(optimizer_raw, "schedule"))),
            weight_decay=float(optimizer_raw.get("weight_decay", 0.0)),
            grad_clip_norm=_optional_float(optimizer_raw, "grad_clip_norm", "optimizer"),
            adamw_fallback_schedule=None
            if optimizer_raw.get("adamw_fallback_schedule") is None
            else ScheduleSpec(**dict(_required_mapping(optimizer_raw, "adamw_fallback_schedule"))),
            route_rules=tuple(
                ParamRouteRule(**dict(_require_mapping(rule, "optimizer.route_rules[]")))
                for rule in _optional_list(optimizer_raw, "route_rules")
            ),
        )
        generation_raw = raw.get("generation")
        spec = RunSpec(
            run_id=_required_str(raw, "run_id", "resolved config"),
            seed=_required_int(raw, "seed", "resolved config"),
            output_dir=Path(_required_str(raw, "output_dir", "resolved config")),
            model=ModelSpec(**dict(_required_mapping(raw, "model"))),
            optimizer=optimizer,
            data=_data_spec(_required_mapping(raw, "data")),
            mesh=_mesh_spec(_required_mapping(raw, "mesh")),
            training=TrainingSpec(**dict(_required_mapping(raw, "training"))),
            parallelism=ParallelismSpec(**dict(raw.get("parallelism", {"mode": "ddp"}))),
            artifacts=ArtifactSpec(**dict(_required_mapping(raw, "artifacts"))),
            profiling=ProfilingSpec(**dict(raw.get("profiling", {}))),
            kernels=KernelSpec(**dict(raw.get("kernels", {}))),
            evals=tuple(EvalSpec(**dict(_require_mapping(item, "evals[]"))) for item in _optional_list(raw, "evals")),
            generation=None if generation_raw is None else GenerationSpec(**dict(_require_mapping(generation_raw, "generation"))),
        )
        validate_run_spec(spec)
    except (TypeError, ValueError, ContractError, ConfigError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc
    return spec


def _data_spec(raw: Mapping[str, Any]) -> DataSpec:
    validation_manifest = raw.get("validation_manifest")
    train_manifest = raw.get("train_manifest")
    streaming_raw = raw.get("hf_streaming")
    return DataSpec(
        mode=str(raw.get("mode", "prepared")),
        train_manifest=None if train_manifest is None else Path(_required_str(raw, "train_manifest", "data")),
        tokenizer_id=_optional_str(raw, "tokenizer_id", "data"),
        validation_manifest=None if validation_manifest is None else Path(_required_str(raw, "validation_manifest", "data")),
        hf_streaming=None
        if streaming_raw is None
        else HFStreamingSpec(**dict(_require_mapping(streaming_raw, "data.hf_streaming"))),
        order=str(raw.get("order", "sequential")),
        shuffle_seed=_optional_int(raw, "shuffle_seed", "data"),
        worker_count=_optional_int_with_default(raw, "worker_count", "data", default=0),
        worker_buffer_size=_optional_int_with_default(raw, "worker_buffer_size", "data", default=1),
        prefetch=_optional_bool(raw, "prefetch", "data", default=False),
        document_buffer_size=_optional_int(raw, "document_buffer_size", "data"),
        document_refill_size=_optional_int(raw, "document_refill_size", "data"),
    )


def _mesh_spec(raw: Mapping[str, Any]) -> MeshSpec:
    return MeshSpec(
        axis_names=tuple(_required_list(raw, "axis_names", "mesh")),
        axis_sizes=tuple(int(size) for size in _required_list(raw, "axis_sizes", "mesh")),
    )


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _require_mapping(raw.get(key), key)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a JSON object")
    return value


def _required_list(raw: Mapping[str, Any], key: str, name: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"{name}.{key} must be a JSON list")
    return value


def _optional_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a JSON list")
    return value


def _required_str(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name}.{key} must be a non-empty string")
    return value


def _optional_str(raw: Mapping[str, Any], key: str, name: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name}.{key} must be a non-empty string or null")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ConfigError(f"{name}.{key} must be an integer")
    return value


def _optional_float(raw: Mapping[str, Any], key: str, name: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ConfigError(f"{name}.{key} must be numeric or null")
    return float(value)


def _optional_int(raw: Mapping[str, Any], key: str, name: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name}.{key} must be an integer or null")
    return value


def _optional_int_with_default(raw: Mapping[str, Any], key: str, name: str, *, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name}.{key} must be an integer")
    return value


def _optional_bool(raw: Mapping[str, Any], key: str, name: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{name}.{key} must be a boolean")
    return value
