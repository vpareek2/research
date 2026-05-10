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
class DistributedConfig:
    enabled: bool = True
    device_count: int | str = "auto"
    axis_name: str = "data"

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"distributed.enabled must be a bool, got {self.enabled!r}")
        if self.device_count != "auto":
            if not isinstance(self.device_count, int) or self.device_count <= 0:
                raise ValueError(f"distributed.device_count must be 'auto' or a positive integer, got {self.device_count!r}")
        if not isinstance(self.axis_name, str) or not self.axis_name:
            raise ValueError("distributed.axis_name must be a non-empty string")


@dataclass
class ProfilingConfig:
    enabled: bool = False
    profiler: str = "none"
    start_step: int = 100
    steps: int = 5
    output_dir: str = "profiles"

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"profiling.enabled must be a bool, got {self.enabled!r}")
        if self.profiler not in {"none", "jax", "nsys"}:
            raise ValueError(f"profiling.profiler must be 'none', 'jax', or 'nsys', got {self.profiler!r}")
        if not isinstance(self.start_step, int) or self.start_step < 0:
            raise ValueError(f"profiling.start_step must be a non-negative integer, got {self.start_step!r}")
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError(f"profiling.steps must be a positive integer, got {self.steps!r}")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("profiling.output_dir must be a non-empty string")


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
    log_every: int
    eval_every: int
    eval_steps: int
    checkpoint_every: int
    keep_last: int
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)


@dataclass
class AdamWOptimizerConfig:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    nesterov: bool = False

    def __post_init__(self):
        if not 0.0 <= self.b1 < 1.0:
            raise ValueError(f"optimizer.adamw.b1 must be in [0, 1), got {self.b1}")
        if not 0.0 <= self.b2 < 1.0:
            raise ValueError(f"optimizer.adamw.b2 must be in [0, 1), got {self.b2}")
        if self.eps <= 0.0:
            raise ValueError(f"optimizer.adamw.eps must be positive, got {self.eps}")
        if not isinstance(self.nesterov, bool):
            raise ValueError(f"optimizer.adamw.nesterov must be a bool, got {self.nesterov!r}")


@dataclass
class MuonOptimizerConfig:
    beta: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    eps: float = 1e-8

    def __post_init__(self):
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(f"optimizer.muon.beta must be in [0, 1), got {self.beta}")
        if not isinstance(self.nesterov, bool):
            raise ValueError(f"optimizer.muon.nesterov must be a bool, got {self.nesterov!r}")
        if not isinstance(self.ns_steps, int) or self.ns_steps <= 0:
            raise ValueError(f"optimizer.muon.ns_steps must be positive, got {self.ns_steps!r}")
        if len(self.ns_coeffs) != 3:
            raise ValueError("optimizer.muon.ns_coeffs must contain exactly three coefficients")
        if self.eps <= 0.0:
            raise ValueError(f"optimizer.muon.eps must be positive, got {self.eps}")


@dataclass
class AuroraOptimizerConfig:
    beta: float = 0.95
    nesterov: bool = True
    pp_iterations: int = 2
    pp_beta: float = 0.5
    eps: float = 1e-7

    def __post_init__(self):
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(f"optimizer.aurora.beta must be in [0, 1), got {self.beta}")
        if not isinstance(self.nesterov, bool):
            raise ValueError(f"optimizer.aurora.nesterov must be a bool, got {self.nesterov!r}")
        if not isinstance(self.pp_iterations, int) or self.pp_iterations <= 0:
            raise ValueError(f"optimizer.aurora.pp_iterations must be positive, got {self.pp_iterations!r}")
        if self.pp_beta <= 0.0:
            raise ValueError(f"optimizer.aurora.pp_beta must be positive, got {self.pp_beta}")
        if self.eps <= 0.0:
            raise ValueError(f"optimizer.aurora.eps must be positive, got {self.eps}")


@dataclass
class RiemannianAuroraOptimizerConfig:
    beta: float = 0.95
    nesterov: bool = True
    outer_steps: int = 3
    cg_steps: int = 20
    riemannian_eta: float = 0.1
    retraction_steps: int = 2
    eps: float = 1e-7

    def __post_init__(self):
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(f"optimizer.riemannian_aurora.beta must be in [0, 1), got {self.beta}")
        if not isinstance(self.nesterov, bool):
            raise ValueError(f"optimizer.riemannian_aurora.nesterov must be a bool, got {self.nesterov!r}")
        if not isinstance(self.outer_steps, int) or self.outer_steps <= 0:
            raise ValueError(f"optimizer.riemannian_aurora.outer_steps must be positive, got {self.outer_steps!r}")
        if not isinstance(self.cg_steps, int) or self.cg_steps <= 0:
            raise ValueError(f"optimizer.riemannian_aurora.cg_steps must be positive, got {self.cg_steps!r}")
        if self.riemannian_eta <= 0.0:
            raise ValueError(f"optimizer.riemannian_aurora.riemannian_eta must be positive, got {self.riemannian_eta}")
        if not isinstance(self.retraction_steps, int) or self.retraction_steps <= 0:
            raise ValueError(
                f"optimizer.riemannian_aurora.retraction_steps must be positive, got {self.retraction_steps!r}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"optimizer.riemannian_aurora.eps must be positive, got {self.eps}")


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    adamw: AdamWOptimizerConfig = field(default_factory=AdamWOptimizerConfig)
    muon: MuonOptimizerConfig = field(default_factory=MuonOptimizerConfig)
    aurora: AuroraOptimizerConfig = field(default_factory=AuroraOptimizerConfig)
    riemannian_aurora: RiemannianAuroraOptimizerConfig = field(default_factory=RiemannianAuroraOptimizerConfig)

    def __post_init__(self):
        if self.name not in {"adamw", "muon", "aurora", "riemannian_aurora"}:
            raise ValueError(
                "optimizer.name must be 'adamw', 'muon', 'aurora', or 'riemannian_aurora', "
                f"got {self.name!r}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"optimizer.lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(f"optimizer.weight_decay must be non-negative, got {self.weight_decay}")


@dataclass
class TargetConfig:
    tokens: int = 2_000_000_000

    def __post_init__(self):
        if not isinstance(self.tokens, int) or self.tokens <= 0:
            raise ValueError(f"target.tokens must be a positive integer, got {self.tokens!r}")


@dataclass
class DataConfig:
    path: str
    tokenizer: str
    val_fraction: float | None = None
    source: str = "text"
    prepare_config: str | None = None

    def __post_init__(self):
        if self.source not in {"text", "tokens"}:
            raise ValueError(f"data source must be 'text' or 'tokens', got {self.source}")
        if self.source == "text":
            if self.val_fraction is None:
                raise ValueError("text data source requires val_fraction")
            if not 0.0 < self.val_fraction < 1.0:
                raise ValueError(f"val_fraction must be between 0 and 1, got {self.val_fraction}")


@dataclass
class EvalConfig:
    domain_root: str | None = None
    domain_eval_steps: int | None = None
    prepare_config: str | None = "configs/data/eval_domains.toml"

    def __post_init__(self):
        if self.domain_eval_steps is not None and self.domain_eval_steps <= 0:
            raise ValueError(f"eval.domain_eval_steps must be positive, got {self.domain_eval_steps}")


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
    optimizer: OptimizerConfig
    data: DataConfig
    sampling: SamplingConfig
    target: TargetConfig = field(default_factory=TargetConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
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
    old_train_keys = sorted(set(train_data) & {"lr", "decay"})
    if old_train_keys:
        joined = ", ".join(f"train.{key}" for key in old_train_keys)
        raise ValueError(f"{joined} moved to the required [optimizer] section")
    train_data["lr_schedule"] = LRScheduleConfig(**train_data.get("lr_schedule", {}))
    optimizer_data = data.get("optimizer")
    if optimizer_data is None:
        raise ValueError("Missing required [optimizer] section")
    optimizer_data = optimizer_data.copy()
    optimizer_data["adamw"] = AdamWOptimizerConfig(**optimizer_data.get("adamw", {}))
    muon_data = optimizer_data.get("muon", {})
    if "ns_coeffs" in muon_data:
        muon_data = {**muon_data, "ns_coeffs": tuple(muon_data["ns_coeffs"])}
    optimizer_data["muon"] = MuonOptimizerConfig(**muon_data)
    optimizer_data["aurora"] = AuroraOptimizerConfig(**optimizer_data.get("aurora", {}))
    optimizer_data["riemannian_aurora"] = RiemannianAuroraOptimizerConfig(
        **optimizer_data.get("riemannian_aurora", {})
    )

    return RunConfig(
        experiment=ExperimentConfig(**data["experiment"]),
        model=ModelConfig(**data["model"]),
        train=TrainConfig(**train_data),
        optimizer=OptimizerConfig(**optimizer_data),
        data=DataConfig(**data["data"]),
        sampling=SamplingConfig(**data.get("sampling", {})),
        target=TargetConfig(**data.get("target", {})),
        distributed=DistributedConfig(**data.get("distributed", {})),
        eval=EvalConfig(**data.get("eval", {})),
        profiling=ProfilingConfig(**data.get("profiling", {})),
        precision=PrecisionConfig(**data.get("precision", {})),
        wandb=WandbConfig(**data.get("wandb", {})),
    )
