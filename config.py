"""
Configuration objects and TOML loading.
"""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


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


@dataclass
class DataConfig:
    path: str
    tokenizer: str
    val_fraction: float

    def __post_init__(self):
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
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    return RunConfig(
        experiment=ExperimentConfig(**data["experiment"]),
        model=ModelConfig(**data["model"]),
        train=TrainConfig(**data["train"]),
        data=DataConfig(**data["data"]),
        sampling=SamplingConfig(**data.get("sampling", {})),
        wandb=WandbConfig(**data.get("wandb", {})),
    )
