"""TOML-facing config schema dataclasses."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TomlRunSection:
    id: str
    seed: int = 0
    output_dir: Path = Path("runs")


@dataclass(frozen=True, slots=True)
class TomlModelSection:
    name: str
    variant: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    n_kv_heads: int | None = None
    rope_theta: float = 1_000_000.0
    norm_epsilon: float = 1e-6
    tied_embeddings: bool = False
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True, slots=True)
class TomlScheduleSection:
    peak_lr: float
    name: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.0
    stable_steps: int | None = None


@dataclass(frozen=True, slots=True)
class TomlOptimizerSection:
    name: str
    schedule: TomlScheduleSection
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None


@dataclass(frozen=True, slots=True)
class TomlDataSection:
    train_manifest: Path
    tokenizer_id: str | None = None
    validation_manifest: Path | None = None


@dataclass(frozen=True, slots=True)
class TomlTrainingSection:
    seq_len: int
    global_batch_size: int
    target_tokens: int
    precision: str = "bf16"
    log_every_steps: int = 10
    checkpoint_every_steps: int = 1000
    eval_every_steps: int | None = None
    grad_clip_norm: float | None = None


@dataclass(frozen=True, slots=True)
class TomlMeshSection:
    axis_names: tuple[str, ...] = ("data",)
    axis_sizes: tuple[int, ...] = (1,)


@dataclass(frozen=True, slots=True)
class TomlArtifactSection:
    wandb_enabled: bool = False


@dataclass(frozen=True, slots=True)
class TomlEvalSection:
    name: str
    every_steps: int
    num_batches: int


@dataclass(frozen=True, slots=True)
class TomlGenerationSection:
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
