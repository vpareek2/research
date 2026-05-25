"""Cross-spec validation."""

import math

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
    ep_axis_size = axis_sizes.get("ep", 1)
    if spec.parallelism.mode == "ddp" and fsdp_axis_size != 1:
        raise ConfigError("parallelism.mode='ddp' requires mesh fsdp axis size to be 1")
    if spec.parallelism.mode in {"zero2", "fsdp"} and "fsdp" not in axis_sizes:
        raise ConfigError(f"parallelism.mode='{spec.parallelism.mode}' requires a mesh fsdp axis")
    if "ep" in axis_sizes and ep_axis_size != 1 and not spec.parallelism.expert_parallel:
        raise ConfigError("mesh ep axis size greater than 1 requires parallelism.expert_parallel=true")
    if spec.parallelism.expert_parallel:
        if "ep" not in axis_sizes:
            raise ConfigError("parallelism.expert_parallel=true requires a mesh ep axis")
        if spec.model.name != "trinity" or spec.model.trinity is None or spec.model.trinity.moe is None:
            raise ConfigError("parallelism.expert_parallel=true requires a Trinity MoE model")
        num_experts = spec.model.trinity.moe.num_experts
        if num_experts % ep_axis_size != 0:
            raise ConfigError(
                f"model.trinity.moe.num_experts ({num_experts}) must be divisible by ep axis size ({ep_axis_size})"
            )
    if spec.data.mode == "hf_streaming" and spec.evals and spec.data.validation_manifest is None:
        raise ConfigError("data.validation_manifest is required for evals when data.mode='hf_streaming'")
    if spec.profiling.enabled:
        estimated_steps = math.ceil(spec.training.target_tokens / effective_batch_tokens)
        if spec.profiling.trace_start_step > estimated_steps:
            raise ConfigError(
                "profiling.trace_start_step must be reachable by training.target_tokens; "
                f"trace_start_step={spec.profiling.trace_start_step} estimated_steps={estimated_steps}"
            )
