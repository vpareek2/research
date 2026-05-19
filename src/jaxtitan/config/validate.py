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
    if spec.training.target_tokens < spec.training.global_batch_size * spec.training.seq_len:
        raise ConfigError("training.target_tokens must cover at least one global training batch")
    if spec.optimizer.grad_clip_norm is not None and spec.training.grad_clip_norm is not None:
        raise ConfigError("set grad_clip_norm in either [optimizer] or [training], not both")
    mesh_size = 1
    for axis_size in spec.mesh.axis_sizes:
        mesh_size *= axis_size
    if spec.training.global_batch_size % mesh_size != 0:
        raise ConfigError(
            f"training.global_batch_size ({spec.training.global_batch_size}) must be divisible by "
            f"the mesh size ({mesh_size})"
        )
