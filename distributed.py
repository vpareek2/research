"""
Single-host replicated data parallel helpers.
"""

from dataclasses import dataclass

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from config import DistributedConfig, TrainConfig


@dataclass(frozen=True)
class DistributedContext:
    mesh: Mesh
    axis_name: str
    device_count: int
    global_batch_size: int
    per_device_batch_size: int
    tokens_per_step: int
    replicated_sharding: NamedSharding
    batch_sharding: NamedSharding


def create_distributed_context(config: DistributedConfig, train: TrainConfig) -> DistributedContext:
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError("JAX did not report any devices")

    device_count = 1 if not config.enabled else _resolve_device_count(config.device_count, len(devices))
    if train.batch_size % device_count != 0:
        raise ValueError(
            f"train.batch_size ({train.batch_size}) must divide evenly across "
            f"{device_count} distributed device(s)"
        )

    mesh = Mesh(np.asarray(devices[:device_count]), (config.axis_name,))
    replicated_sharding = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P(config.axis_name, None))
    return DistributedContext(
        mesh=mesh,
        axis_name=config.axis_name,
        device_count=device_count,
        global_batch_size=train.batch_size,
        per_device_batch_size=train.batch_size // device_count,
        tokens_per_step=train.batch_size * train.seq_len,
        replicated_sharding=replicated_sharding,
        batch_sharding=batch_sharding,
    )


def _resolve_device_count(requested: int | str, available: int) -> int:
    if requested == "auto":
        return available
    if requested > available:
        raise ValueError(f"distributed.device_count={requested} exceeds available JAX devices={available}")
    return requested


def place_replicated_model(model: nnx.Module, context: DistributedContext):
    nnx.update(model, jax.device_put(nnx.state(model), context.replicated_sharding))


def place_replicated_state(model: nnx.Module, optimizer: nnx.Optimizer, context: DistributedContext):
    place_replicated_model(model, context)
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), context.replicated_sharding))


def shard_batch(batch: dict, context: DistributedContext) -> dict:
    return {key: place_batch_array(value, context) for key, value in batch.items()}


def place_batch_array(value, context: DistributedContext):
    array = np.asarray(value)
    if array.ndim == 0:
        return jax.device_put(array)
    if array.shape[0] != context.global_batch_size:
        raise ValueError(
            f"Batch array leading dimension {array.shape[0]} does not match "
            f"global_batch_size={context.global_batch_size}"
        )
    sharding = NamedSharding(context.mesh, P(context.axis_name, *([None] * (array.ndim - 1))))
    return jax.device_put(array, sharding)
