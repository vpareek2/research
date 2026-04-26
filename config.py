"""
Configuration objects and TOML loading.
"""

from dataclasses import dataclass
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
class TrainConfig:
    seed: int
    batch_size: int
    seq_len: int
    steps: int
    lr: float
    decay: float
    log_every: int

@dataclass
class DataConfig:
    path: str
    tokenizer: str

@dataclass
class RunConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    return RunConfig(
        model=ModelConfig(**data["model"]),
        train=TrainConfig(**data["train"]),
        data=DataConfig(**data["data"]),
    )
