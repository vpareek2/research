"""Run-level specs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jaxtitan.errors import ContractError
from jaxtitan.specs.data import DataSpec
from jaxtitan.specs.eval import EvalSpec
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Static training contract."""

    seq_len: int
    global_batch_size: int
    target_tokens: int
    precision: PrecisionName = "bf16"
    log_every_steps: int = 10
    checkpoint_every_steps: int = 1000
    eval_every_steps: int | None = None
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.precision not in _PRECISION_NAMES:
            raise ContractError(f"training.precision must be one of {sorted(_PRECISION_NAMES)}, got {self.precision!r}")
        for field_name in ("seq_len", "global_batch_size", "target_tokens", "log_every_steps", "checkpoint_every_steps"):
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
    artifacts: ArtifactSpec = ArtifactSpec()
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
