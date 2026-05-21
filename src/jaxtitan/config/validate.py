"""Cross-spec validation."""

from jaxtitan.errors import ConfigError
from jaxtitan.specs.run import RunSpec


def validate_run_spec(spec: RunSpec) -> None:
    """Validate relationships across a resolved run spec."""

    if spec.model.max_seq_len < spec.training.seq_len:
        raise ConfigError(
            f"model.max_seq_len ({spec.model.max_seq_len}) must be >= "
            f"training.seq_len ({spec.training.seq_len})"
        )
    effective_batch_tokens = (
        spec.training.global_batch_size
        * spec.training.seq_len
        * spec.training.gradient_accumulation_steps
    )
    if spec.training.target_tokens < effective_batch_tokens:
        raise ConfigError("training.target_tokens must cover at least one effective optimizer batch")
    if spec.optimizer.grad_clip_norm is not None and spec.training.grad_clip_norm is not None:
        raise ConfigError("set grad_clip_norm in either [optimizer] or [training], not both")
    data_axis_size = 1
    if "data" in spec.mesh.axis_names:
        data_axis_size = spec.mesh.axis_sizes[spec.mesh.axis_names.index("data")]
    if spec.training.global_batch_size % data_axis_size != 0:
        raise ConfigError(
            f"training.global_batch_size ({spec.training.global_batch_size}) must be divisible by "
            f"the data axis size ({data_axis_size})"
        )
    axis_sizes = dict(zip(spec.mesh.axis_names, spec.mesh.axis_sizes, strict=True))
    fsdp_axis_size = axis_sizes.get("fsdp", 1)
    if spec.parallelism.mode == "ddp" and fsdp_axis_size != 1:
        raise ConfigError("parallelism.mode='ddp' requires mesh fsdp axis size to be 1")
    if spec.parallelism.mode in {"zero2", "fsdp"} and "fsdp" not in axis_sizes:
        raise ConfigError(f"parallelism.mode='{spec.parallelism.mode}' requires a mesh fsdp axis")
