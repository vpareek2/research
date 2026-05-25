"""TOML-facing config schema dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TomlRunSection:
    id: str
    seed: int = 0
    output_dir: Path = Path("runs")


@dataclass(frozen=True, slots=True)
class TomlMoeBalanceSection:
    name: str = "none"
    load_lr: float = 5e-4
    momentum: float = 0.5
    clamp: float = 2.0
    sequence_aux_loss_weight: float = 1e-4


@dataclass(frozen=True, slots=True)
class TomlTrinityMoeSection:
    num_experts: int
    top_k: int
    expert_intermediate_size: int | None = None
    num_shared_experts: int = 0
    route_scale: float = 1.0
    balance: TomlMoeBalanceSection = field(default_factory=TomlMoeBalanceSection)


@dataclass(frozen=True, slots=True)
class TomlTrinitySection:
    initial_dense_layers: int
    local_window: int
    local_layers_per_global: int
    attention_gate: bool = True
    qk_norm: bool = True
    norm_policy: str = "depth_scaled_sandwich"
    embedding_scale: str = "sqrt_hidden"
    init_std: float | None = None
    moe: TomlTrinityMoeSection | None = None


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
    remat: str = "none"
    trinity: TomlTrinitySection | None = None


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
    adamw_fallback_schedule: TomlScheduleSection | None = None


@dataclass(frozen=True, slots=True)
class TomlHFStreamingSection:
    dataset: str
    split: str
    revision: str
    name: str | None = None
    data_dir: str | None = None
    text_column: str = "text"
    append_eot: bool = True


@dataclass(frozen=True, slots=True)
class TomlDataSection:
    mode: str = "prepared"
    train_manifest: Path | None = None
    tokenizer_id: str | None = None
    validation_manifest: Path | None = None
    hf_streaming: TomlHFStreamingSection | None = None
    order: str = "sequential"
    shuffle_seed: int | None = None
    worker_count: int = 0
    worker_buffer_size: int = 1
    prefetch: bool = False
    document_buffer_size: int | None = None
    document_refill_size: int | None = None


@dataclass(frozen=True, slots=True)
class TomlTrainingLossSection:
    z_loss_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class TomlTrainingSection:
    seq_len: int
    global_batch_size: int
    target_tokens: int
    precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 10
    checkpoint_every_steps: int = 1000
    eval_every_steps: int | None = None
    grad_clip_norm: float | None = None
    loss: TomlTrainingLossSection = field(default_factory=TomlTrainingLossSection)


@dataclass(frozen=True, slots=True)
class TomlMeshSection:
    axis_names: tuple[str, ...] = ("data",)
    axis_sizes: tuple[int, ...] = (1,)


@dataclass(frozen=True, slots=True)
class TomlParallelismSection:
    mode: str = "ddp"
    expert_parallel: bool = False


@dataclass(frozen=True, slots=True)
class TomlArtifactSection:
    wandb_enabled: bool = False
    wandb_project: str = "jaxtitan"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: str = "online"


@dataclass(frozen=True, slots=True)
class TomlProfilingSection:
    enabled: bool = False
    trace_start_step: int = 3
    trace_steps: int = 2
    create_perfetto_trace: bool = True
    create_perfetto_link: bool = False


@dataclass(frozen=True, slots=True)
class TomlKernelSection:
    enabled: bool = False
    strict: bool = False
    compile: str = "lazy"


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
