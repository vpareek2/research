"""Run-level specs."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jaxtitan.errors import ContractError
from jaxtitan.specs.data import DataSpec
from jaxtitan.specs.eval import EvalSpec
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec
from jaxtitan.specs.parallelism import ParallelismSpec

PrecisionName = Literal["fp32", "bf16", "mixed_bf16"]
_PRECISION_NAMES = {"fp32", "bf16", "mixed_bf16"}


@dataclass(frozen=True, slots=True)
class RunDirs:
    """Resolved local run directory contract."""

    root: Path
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.run_id:
            raise ContractError("run_id must be non-empty")

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def config_dir(self) -> Path:
        return self.run_dir / "config"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Static local artifact contract."""

    root: Path = Path("runs")
    wandb_enabled: bool = False
    wandb_project: str = "jaxtitan"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: str = "online"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "wandb_tags", tuple(self.wandb_tags))
        if not isinstance(self.wandb_enabled, bool):
            raise ContractError("artifacts.wandb_enabled must be a boolean")
        if not isinstance(self.wandb_project, str) or not self.wandb_project:
            raise ContractError("artifacts.wandb_project must be a non-empty string")
        if self.wandb_entity is not None and (not isinstance(self.wandb_entity, str) or not self.wandb_entity):
            raise ContractError("artifacts.wandb_entity must be non-empty when provided")
        if self.wandb_group is not None and (not isinstance(self.wandb_group, str) or not self.wandb_group):
            raise ContractError("artifacts.wandb_group must be non-empty when provided")
        if self.wandb_mode not in {"online", "offline"}:
            raise ContractError("artifacts.wandb_mode must be 'online' or 'offline'")
        if any(not isinstance(tag, str) or not tag for tag in self.wandb_tags):
            raise ContractError("artifacts.wandb_tags must contain only non-empty strings")


@dataclass(frozen=True, slots=True)
class ProfilingSpec:
    """Programmatic JAX profiler trace capture contract."""

    enabled: bool = False
    trace_start_step: int = 3
    trace_steps: int = 2
    create_perfetto_trace: bool = True
    create_perfetto_link: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ContractError("profiling.enabled must be a boolean")
        if not isinstance(self.trace_start_step, int) or isinstance(self.trace_start_step, bool):
            raise ContractError("profiling.trace_start_step must be an integer")
        if not isinstance(self.trace_steps, int) or isinstance(self.trace_steps, bool):
            raise ContractError("profiling.trace_steps must be an integer")
        if self.trace_start_step <= 0:
            raise ContractError(
                f"profiling.trace_start_step must be positive, got {self.trace_start_step}"
            )
        if self.trace_steps <= 0:
            raise ContractError(f"profiling.trace_steps must be positive, got {self.trace_steps}")
        if not isinstance(self.create_perfetto_trace, bool):
            raise ContractError("profiling.create_perfetto_trace must be a boolean")
        if not isinstance(self.create_perfetto_link, bool):
            raise ContractError("profiling.create_perfetto_link must be a boolean")

    @property
    def trace_end_step(self) -> int:
        return self.trace_start_step + self.trace_steps - 1


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """Kernel backend resolution contract."""

    enabled: bool = False
    strict: bool = False
    compile: str = "lazy"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ContractError("kernels.enabled must be a boolean")
        if not isinstance(self.strict, bool):
            raise ContractError("kernels.strict must be a boolean")
        if self.compile not in {"lazy", "ahead_of_time"}:
            raise ContractError("kernels.compile must be 'lazy' or 'ahead_of_time'")


@dataclass(frozen=True, slots=True)
class TrainingLossSpec:
    """Training objective loss controls."""

    z_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.z_loss_weight, int | float)
            or isinstance(self.z_loss_weight, bool)
            or self.z_loss_weight < 0.0
        ):
            raise ContractError(
                f"training.loss.z_loss_weight must be non-negative, got {self.z_loss_weight!r}"
            )


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Static training contract."""

    seq_len: int
    global_batch_size: int
    target_tokens: int
    precision: PrecisionName = "bf16"
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 10
    checkpoint_every_steps: int = 1000
    eval_every_steps: int | None = None
    grad_clip_norm: float | None = None
    loss: TrainingLossSpec = field(default_factory=TrainingLossSpec)

    def __post_init__(self) -> None:
        if isinstance(self.loss, Mapping):
            object.__setattr__(self, "loss", TrainingLossSpec(**self.loss))
        elif not isinstance(self.loss, TrainingLossSpec):
            raise ContractError("training.loss must be a TrainingLossSpec or mapping")
        if self.precision not in _PRECISION_NAMES:
            raise ContractError(f"training.precision must be one of {sorted(_PRECISION_NAMES)}, got {self.precision!r}")
        for field_name in (
            "seq_len",
            "global_batch_size",
            "target_tokens",
            "gradient_accumulation_steps",
            "log_every_steps",
            "checkpoint_every_steps",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise ContractError(f"training.{field_name} must be positive, got {value}")
        if self.eval_every_steps is not None and self.eval_every_steps <= 0:
            raise ContractError(f"training.eval_every_steps must be positive, got {self.eval_every_steps}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ContractError(f"training.grad_clip_norm must be positive, got {self.grad_clip_norm}")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Resolved static contract for a Jaxtitan run."""

    run_id: str
    seed: int
    output_dir: Path
    model: ModelSpec
    optimizer: OptimizerSpec
    data: DataSpec
    mesh: MeshSpec
    training: TrainingSpec
    parallelism: ParallelismSpec = ParallelismSpec()
    artifacts: ArtifactSpec = ArtifactSpec()
    profiling: ProfilingSpec = ProfilingSpec()
    kernels: KernelSpec = KernelSpec()
    evals: tuple[EvalSpec, ...] = ()
    generation: GenerationSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "evals", tuple(self.evals))
        if not self.run_id:
            raise ContractError("run.id must be non-empty")
        if self.seed < 0:
            raise ContractError(f"run.seed must be non-negative, got {self.seed}")

    @property
    def dirs(self) -> RunDirs:
        return RunDirs(root=self.output_dir, run_id=self.run_id)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Canonical local run initialization manifest."""

    schema_version: int
    artifact_layout_version: int
    run_id: str
    created_at: str
    source_config_path: Path
    source_config_sha256: str
    resolved_config_sha256: str
    package: dict[str, str]
    directories: dict[str, str]
    run_dir: Path
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_config_path", Path(self.source_config_path))
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        if self.schema_version <= 0:
            raise ContractError(f"manifest.schema_version must be positive, got {self.schema_version}")
        if self.artifact_layout_version <= 0:
            raise ContractError(
                f"manifest.artifact_layout_version must be positive, got {self.artifact_layout_version}"
            )
        if not self.run_id:
            raise ContractError("manifest.run_id must be non-empty")
        if not self.created_at:
            raise ContractError("manifest.created_at must be non-empty")
        if not self.source_config_sha256:
            raise ContractError("manifest.source_config_sha256 must be non-empty")
        if not self.resolved_config_sha256:
            raise ContractError("manifest.resolved_config_sha256 must be non-empty")
