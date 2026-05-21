"""TOML config loading."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
import tomllib

from jaxtitan.config.schema import (
    TomlArtifactSection,
    TomlDataSection,
    TomlEvalSection,
    TomlGenerationSection,
    TomlMeshSection,
    TomlMoeBalanceSection,
    TomlModelSection,
    TomlOptimizerSection,
    TomlParallelismSection,
    TomlRunSection,
    TomlScheduleSection,
    TomlTrainingLossSection,
    TomlTrainingSection,
    TomlTrinityMoeSection,
    TomlTrinitySection,
)
from jaxtitan.config.validate import validate_run_spec
from jaxtitan.errors import ConfigError, ContractError
from jaxtitan.specs.data import DataSpec
from jaxtitan.specs.eval import EvalSpec
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.specs.parallelism import ParallelismSpec
from jaxtitan.specs.run import ArtifactSpec, RunSpec, TrainingSpec


def load_config(path: str | Path) -> RunSpec:
    """Load a Jaxtitan TOML config into a resolved RunSpec."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"failed to read config {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse config {config_path}: {exc}") from exc
    return run_spec_from_mapping(raw)


def run_spec_from_mapping(raw: Mapping[str, Any]) -> RunSpec:
    """Resolve a TOML-like mapping into a validated RunSpec."""

    try:
        run_section = _run_section(_required_mapping(raw, "run"))
        model_section = _model_section(_required_mapping(raw, "model"))
        optimizer_section = _optimizer_section(_required_mapping(raw, "optimizer"))
        data_section = _data_section(_required_mapping(raw, "data"))
        training_section = _training_section(_required_mapping(raw, "training"))
        mesh_section = _mesh_section(_optional_mapping(raw, "mesh"))
        parallelism_section = _parallelism_section(_optional_mapping(raw, "parallelism"))
        artifact_section = _artifact_section(_optional_mapping(raw, "artifacts"))
        eval_sections = tuple(_eval_section(item) for item in _optional_list(raw, "evals"))
        generation_raw = raw.get("generation")
        generation_section = None if generation_raw is None else _generation_section(_ensure_mapping(generation_raw, "generation"))

        spec = RunSpec(
            run_id=run_section.id,
            seed=run_section.seed,
            output_dir=run_section.output_dir,
            model=ModelSpec(**asdict(model_section)),
            optimizer=OptimizerSpec(
                name=optimizer_section.name,
                schedule=ScheduleSpec(**asdict(optimizer_section.schedule)),
                weight_decay=optimizer_section.weight_decay,
                grad_clip_norm=optimizer_section.grad_clip_norm,
                adamw_fallback_schedule=None
                if optimizer_section.adamw_fallback_schedule is None
                else ScheduleSpec(**asdict(optimizer_section.adamw_fallback_schedule)),
            ),
            data=DataSpec(**asdict(data_section)),
            mesh=MeshSpec(axis_names=mesh_section.axis_names, axis_sizes=mesh_section.axis_sizes),
            training=TrainingSpec(**asdict(training_section)),
            parallelism=ParallelismSpec(**asdict(parallelism_section)),
            artifacts=ArtifactSpec(root=run_section.output_dir, wandb_enabled=artifact_section.wandb_enabled),
            evals=tuple(EvalSpec(**asdict(section)) for section in eval_sections),
            generation=None if generation_section is None else GenerationSpec(**asdict(generation_section)),
        )
        validate_run_spec(spec)
    except ContractError as exc:
        raise ConfigError(str(exc)) from exc
    return spec


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in raw:
        raise ConfigError(f"missing required [{key}] section")
    return _ensure_mapping(raw[key], key)


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    return _ensure_mapping(value, key)


def _ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a TOML table")
    return value


def _optional_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be an array of TOML tables")
    return value


def _run_section(raw: Mapping[str, Any]) -> TomlRunSection:
    return TomlRunSection(
        id=_required_str(raw, "id", "run"),
        seed=int(raw.get("seed", 0)),
        output_dir=Path(_optional_str(raw, "output_dir", "run", default="runs")),
    )


def _model_section(raw: Mapping[str, Any]) -> TomlModelSection:
    rope_theta = _optional_float(raw, "rope_theta", "model")
    norm_epsilon = _optional_float(raw, "norm_epsilon", "model")
    trinity_raw = raw.get("trinity")
    return TomlModelSection(
        name=_required_str(raw, "name", "model"),
        variant=_required_str(raw, "variant", "model"),
        vocab_size=_required_int(raw, "vocab_size", "model"),
        hidden_size=_required_int(raw, "hidden_size", "model"),
        intermediate_size=_required_int(raw, "intermediate_size", "model"),
        num_layers=_required_int(raw, "num_layers", "model"),
        num_heads=_required_int(raw, "num_heads", "model"),
        max_seq_len=_required_int(raw, "max_seq_len", "model"),
        n_kv_heads=_optional_int(raw, "n_kv_heads", "model"),
        rope_theta=1_000_000.0 if rope_theta is None else rope_theta,
        norm_epsilon=1e-6 if norm_epsilon is None else norm_epsilon,
        tied_embeddings=_optional_bool(raw, "tied_embeddings", "model", default=False),
        param_dtype=_optional_str(raw, "param_dtype", "model", default="float32"),
        compute_dtype=_optional_str(raw, "compute_dtype", "model", default="bfloat16"),
        remat=_optional_str(raw, "remat", "model", default="none"),
        trinity=None if trinity_raw is None else _trinity_section(_ensure_mapping(trinity_raw, "model.trinity")),
    )


def _trinity_section(raw: Mapping[str, Any]) -> TomlTrinitySection:
    moe_raw = raw.get("moe")
    return TomlTrinitySection(
        initial_dense_layers=_required_int(raw, "initial_dense_layers", "model.trinity"),
        local_window=_required_int(raw, "local_window", "model.trinity"),
        local_layers_per_global=_required_int(raw, "local_layers_per_global", "model.trinity"),
        attention_gate=_optional_bool(raw, "attention_gate", "model.trinity", default=True),
        qk_norm=_optional_bool(raw, "qk_norm", "model.trinity", default=True),
        norm_policy=_optional_str(raw, "norm_policy", "model.trinity", default="depth_scaled_sandwich"),
        embedding_scale=_optional_str(raw, "embedding_scale", "model.trinity", default="sqrt_hidden"),
        init_std=_optional_float(raw, "init_std", "model.trinity"),
        moe=None if moe_raw is None else _trinity_moe_section(_ensure_mapping(moe_raw, "model.trinity.moe")),
    )


def _trinity_moe_section(raw: Mapping[str, Any]) -> TomlTrinityMoeSection:
    route_scale = raw.get("route_scale", 1.0)
    if not isinstance(route_scale, int | float) or isinstance(route_scale, bool):
        raise ConfigError("model.trinity.moe.route_scale must be numeric")
    balance_raw = raw.get("balance", {})
    return TomlTrinityMoeSection(
        num_experts=_required_int(raw, "num_experts", "model.trinity.moe"),
        top_k=_required_int(raw, "top_k", "model.trinity.moe"),
        expert_intermediate_size=_optional_int(raw, "expert_intermediate_size", "model.trinity.moe"),
        num_shared_experts=_optional_int_with_default(raw, "num_shared_experts", "model.trinity.moe", default=0),
        route_scale=float(route_scale),
        balance=_moe_balance_section(_ensure_mapping(balance_raw, "model.trinity.moe.balance")),
    )


def _moe_balance_section(raw: Mapping[str, Any]) -> TomlMoeBalanceSection:
    return TomlMoeBalanceSection(
        name=_optional_str(raw, "name", "model.trinity.moe.balance", default="none"),
        load_lr=_optional_float_with_default(raw, "load_lr", "model.trinity.moe.balance", default=5e-4),
        momentum=_optional_float_with_default(raw, "momentum", "model.trinity.moe.balance", default=0.5),
        clamp=_optional_float_with_default(raw, "clamp", "model.trinity.moe.balance", default=2.0),
        sequence_aux_loss_weight=_optional_float_with_default(
            raw,
            "sequence_aux_loss_weight",
            "model.trinity.moe.balance",
            default=1e-4,
        ),
    )


def _optimizer_section(raw: Mapping[str, Any]) -> TomlOptimizerSection:
    fallback_raw = raw.get("adamw_fallback_schedule")
    return TomlOptimizerSection(
        name=_required_str(raw, "name", "optimizer"),
        schedule=_schedule_section(_required_mapping(raw, "schedule")),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        grad_clip_norm=_optional_float(raw, "grad_clip_norm", "optimizer"),
        adamw_fallback_schedule=None
        if fallback_raw is None
        else _schedule_section(_ensure_mapping(fallback_raw, "optimizer.adamw_fallback_schedule")),
    )


def _schedule_section(raw: Mapping[str, Any]) -> TomlScheduleSection:
    return TomlScheduleSection(
        peak_lr=_required_float(raw, "peak_lr", "optimizer.schedule"),
        name=_optional_str(raw, "name", "optimizer.schedule", default="constant"),
        warmup_steps=int(raw.get("warmup_steps", 0)),
        total_steps=_optional_int(raw, "total_steps", "optimizer.schedule"),
        min_lr_ratio=float(raw.get("min_lr_ratio", 0.0)),
        stable_steps=_optional_int(raw, "stable_steps", "optimizer.schedule"),
    )


def _data_section(raw: Mapping[str, Any]) -> TomlDataSection:
    return TomlDataSection(
        train_manifest=Path(_required_str(raw, "train_manifest", "data")),
        tokenizer_id=_optional_str(raw, "tokenizer_id", "data", default=None),
        validation_manifest=_optional_path(raw, "validation_manifest", "data"),
        order=_optional_str(raw, "order", "data", default="sequential"),
        shuffle_seed=_optional_int(raw, "shuffle_seed", "data"),
        worker_count=_optional_int_with_default(raw, "worker_count", "data", default=0),
        worker_buffer_size=_optional_int_with_default(raw, "worker_buffer_size", "data", default=1),
        prefetch=_optional_bool(raw, "prefetch", "data", default=False),
        document_buffer_size=_optional_int(raw, "document_buffer_size", "data"),
        document_refill_size=_optional_int(raw, "document_refill_size", "data"),
    )


def _training_section(raw: Mapping[str, Any]) -> TomlTrainingSection:
    loss_raw = raw.get("loss", {})
    return TomlTrainingSection(
        seq_len=_required_int(raw, "seq_len", "training"),
        global_batch_size=_required_int(raw, "global_batch_size", "training"),
        target_tokens=_required_int(raw, "target_tokens", "training"),
        precision=_optional_str(raw, "precision", "training", default="bf16"),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        log_every_steps=int(raw.get("log_every_steps", 10)),
        checkpoint_every_steps=int(raw.get("checkpoint_every_steps", 1000)),
        eval_every_steps=_optional_int(raw, "eval_every_steps", "training"),
        grad_clip_norm=_optional_float(raw, "grad_clip_norm", "training"),
        loss=_training_loss_section(_ensure_mapping(loss_raw, "training.loss")),
    )


def _training_loss_section(raw: Mapping[str, Any]) -> TomlTrainingLossSection:
    return TomlTrainingLossSection(
        z_loss_weight=_optional_float_with_default(raw, "z_loss_weight", "training.loss", default=0.0),
    )


def _mesh_section(raw: Mapping[str, Any]) -> TomlMeshSection:
    return TomlMeshSection(
        axis_names=tuple(raw.get("axis_names", ("data",))),
        axis_sizes=tuple(int(size) for size in raw.get("axis_sizes", (1,))),
    )


def _parallelism_section(raw: Mapping[str, Any]) -> TomlParallelismSection:
    return TomlParallelismSection(mode=_optional_str(raw, "mode", "parallelism", default="ddp"))


def _artifact_section(raw: Mapping[str, Any]) -> TomlArtifactSection:
    return TomlArtifactSection(wandb_enabled=bool(raw.get("wandb_enabled", False)))


def _eval_section(value: Any) -> TomlEvalSection:
    raw = _ensure_mapping(value, "evals[]")
    return TomlEvalSection(
        name=_required_str(raw, "name", "evals[]"),
        every_steps=_required_int(raw, "every_steps", "evals[]"),
        num_batches=_required_int(raw, "num_batches", "evals[]"),
    )


def _generation_section(raw: Mapping[str, Any]) -> TomlGenerationSection:
    return TomlGenerationSection(
        max_new_tokens=_required_int(raw, "max_new_tokens", "generation"),
        temperature=float(raw.get("temperature", 1.0)),
        top_k=_optional_int(raw, "top_k", "generation"),
        top_p=_optional_float(raw, "top_p", "generation"),
    )


def _required_str(raw: Mapping[str, Any], key: str, section: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_str(raw: Mapping[str, Any], key: str, section: str, *, default: str | None) -> str | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_int(raw: Mapping[str, Any], key: str, section: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_int_with_default(raw: Mapping[str, Any], key: str, section: str, *, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float):
        raise ConfigError(f"{section}.{key} must be numeric")
    return float(value)


def _optional_float(raw: Mapping[str, Any], key: str, section: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be numeric")
    return float(value)


def _optional_float_with_default(raw: Mapping[str, Any], key: str, section: str, *, default: float) -> float:
    value = raw.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be numeric")
    return float(value)


def _optional_bool(raw: Mapping[str, Any], key: str, section: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be a boolean")
    return value


def _optional_path(raw: Mapping[str, Any], key: str, section: str) -> Path | None:
    value = _optional_str(raw, key, section, default=None)
    return None if value is None else Path(value)
