"""
Configuration objects and TOML loading.
"""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

import jax.numpy as jnp


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    seq_len: int
    theta: float
    eps: float
    tied: bool

    def __post_init__(self):
        assert self.hidden_size % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0


@dataclass
class ExperimentConfig:
    name: str
    out_dir: str


@dataclass
class PrecisionConfig:
    compute_dtype: str = "fp32"
    param_dtype: str = "fp32"
    loss_dtype: str = "fp32"

    def __post_init__(self):
        for field_name in ("compute_dtype", "param_dtype", "loss_dtype"):
            value = getattr(self, field_name)
            if value not in {"fp32", "bf16"}:
                raise ValueError(f"{field_name} must be 'fp32' or 'bf16', got {value}")


def dtype_from_name(name: str):
    if name == "fp32":
        return jnp.float32
    if name == "bf16":
        return jnp.bfloat16
    raise ValueError(f"Unknown dtype name: {name}")


@dataclass
class LRScheduleConfig:
    type: str = "cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    stable_ratio: float = 0.80

    def __post_init__(self):
        if self.type not in {"cosine", "wsd"}:
            raise ValueError(f"lr_schedule.type must be 'cosine' or 'wsd', got {self.type}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if not 0.0 <= self.stable_ratio < 1.0:
            raise ValueError(f"stable_ratio must be in [0, 1), got {self.stable_ratio}")


@dataclass
class TrainConfig:
    seed: int
    batch_size: int
    seq_len: int
    steps: int
    lr: float
    decay: float
    log_every: int
    eval_every: int
    eval_steps: int
    checkpoint_every: int
    keep_last: int
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)


@dataclass
class DataConfig:
    path: str
    tokenizer: str
    val_fraction: float | None = None
    source: str = "text"

    def __post_init__(self):
        if self.source not in {"text", "tokens"}:
            raise ValueError(f"data source must be 'text' or 'tokens', got {self.source}")
        if self.source == "text":
            if self.val_fraction is None:
                raise ValueError("text data source requires val_fraction")
            if not 0.0 < self.val_fraction < 1.0:
                raise ValueError(f"val_fraction must be between 0 and 1, got {self.val_fraction}")


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "data-research"
    entity: str = ""
    tags: list[str] | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SamplingConfig:
    enabled: bool = False
    prompt: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50

    def __post_init__(self):
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or null")


@dataclass
class RunConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    sampling: SamplingConfig
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        positive_train_fields = {
            "batch_size": self.train.batch_size,
            "seq_len": self.train.seq_len,
            "steps": self.train.steps,
            "log_every": self.train.log_every,
            "eval_every": self.train.eval_every,
            "eval_steps": self.train.eval_steps,
            "keep_last": self.train.keep_last,
        }
        for name, value in positive_train_fields.items():
            if value <= 0:
                raise ValueError(f"train.{name} must be positive, got {value}")

        if self.model.seq_len != self.train.seq_len:
            raise ValueError(
                f"model.seq_len ({self.model.seq_len}) must equal "
                f"train.seq_len ({self.train.seq_len})"
            )


def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    train_data = data["train"].copy()
    train_data["lr_schedule"] = LRScheduleConfig(**train_data.get("lr_schedule", {}))

    return RunConfig(
        experiment=ExperimentConfig(**data["experiment"]),
        model=ModelConfig(**data["model"]),
        train=TrainConfig(**train_data),
        data=DataConfig(**data["data"]),
        sampling=SamplingConfig(**data.get("sampling", {})),
        precision=PrecisionConfig(**data.get("precision", {})),
        wandb=WandbConfig(**data.get("wandb", {})),
    )
