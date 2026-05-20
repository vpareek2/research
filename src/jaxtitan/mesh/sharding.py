"""Single-host JAX mesh and placement contracts."""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.specs.mesh import MeshSpec

SUPPORTED_AXES = {"data", "fsdp", "tp"}


@dataclass(frozen=True, slots=True)
class MeshContext:
    """Runtime JAX mesh plus local device facts."""

    spec: MeshSpec
    mesh: Mesh
    devices: tuple[Any, ...]
    local_device_count: int
    global_device_count: int
    process_count: int
    process_index: int

    @property
    def data_axis_size(self) -> int:
        return self.spec.axis_sizes[self.spec.axis_names.index("data")]

    @property
    def selected_device_count(self) -> int:
        return len(self.devices)


@dataclass(frozen=True, slots=True)
class BatchSharding:
    """Named shardings for fixed-shape batch fields."""

    input_ids: NamedSharding
    target_ids: NamedSharding
    loss_mask: NamedSharding


@dataclass(frozen=True, slots=True)
class ShardingPlan:
    """Inspectable shardings for known runtime object classes."""

    mesh: MeshContext
    batch: BatchSharding
    replicated: NamedSharding
    metrics: NamedSharding
    kv_cache: Any | None = None


def build_mesh_context(spec: MeshSpec, devices: tuple[Any, ...] | list[Any] | None = None) -> MeshContext:
    """Build a local JAX mesh from a static MeshSpec."""

    validate_runtime_mesh_spec(spec)
    available_devices = tuple(jax.local_devices() if devices is None else devices)
    requested_devices = reduce(mul, spec.axis_sizes, 1)
    if requested_devices > len(available_devices):
        raise ContractError(
            f"mesh requires {requested_devices} device(s), but only {len(available_devices)} local device(s) are available"
        )
    selected = available_devices[:requested_devices]
    mesh_devices = np.asarray(selected, dtype=object).reshape(spec.axis_sizes)
    mesh = Mesh(mesh_devices, spec.axis_names)
    return MeshContext(
        spec=spec,
        mesh=mesh,
        devices=selected,
        local_device_count=len(available_devices),
        global_device_count=jax.device_count(),
        process_count=jax.process_count(),
        process_index=jax.process_index(),
    )


def build_sharding_plan(context: MeshContext) -> ShardingPlan:
    """Build v1 shardings for data-sharded batches and replicated state."""

    batch_matrix = NamedSharding(context.mesh, P("data", None))
    replicated = NamedSharding(context.mesh, P())
    return ShardingPlan(
        mesh=context,
        batch=BatchSharding(input_ids=batch_matrix, target_ids=batch_matrix, loss_mask=batch_matrix),
        replicated=replicated,
        metrics=replicated,
    )


def place_batch(batch: Batch, plan: ShardingPlan) -> Batch:
    """Place a Batch on device according to the sharding plan."""

    _validate_batch_leading_dims(batch)
    return Batch(
        input_ids=place_batch_array(batch.input_ids, plan.mesh),
        target_ids=place_batch_array(batch.target_ids, plan.mesh),
        loss_mask=place_batch_array(batch.loss_mask, plan.mesh),
        doc_ids=None if batch.doc_ids is None else place_batch_array(batch.doc_ids, plan.mesh),
    )


def place_batch_array(value: Any, context: MeshContext) -> jax.Array:
    """Place one batch array with its leading dimension sharded over data."""

    array = np.asarray(value)
    if array.ndim == 0:
        raise ContractError("batch arrays must have a leading batch dimension")
    if array.shape[0] % context.data_axis_size != 0:
        raise ContractError(
            f"batch leading dimension {array.shape[0]} must be divisible by data axis size {context.data_axis_size}"
        )
    sharding = NamedSharding(context.mesh, P("data", *([None] * (array.ndim - 1))))
    return jax.device_put(array, sharding)


def place_replicated(tree: Any, plan: ShardingPlan) -> Any:
    """Place every PyTree leaf with replicated sharding."""

    return jax.tree.map(lambda leaf: jax.device_put(leaf, plan.replicated), tree)


def replicated_shardings_like(tree: Any, plan: ShardingPlan) -> Any:
    """Build a replicated sharding PyTree matching a runtime state tree."""

    return jax.tree.map(lambda _leaf: plan.replicated, tree)


def require_single_process_runtime() -> None:
    """Fail until host-side data partitioning is implemented."""

    process_count = jax.process_count()
    if process_count != 1:
        raise ContractError(
            "distributed multi-process runtime is not supported yet; "
            f"jax.process_count() is {process_count}, but this slice supports exactly one process"
        )


def validate_runtime_mesh_spec(spec: MeshSpec) -> None:
    """Validate the mesh axes currently supported by the runtime."""

    _validate_supported_spec(spec)


def _validate_supported_spec(spec: MeshSpec) -> None:
    if "data" not in spec.axis_names:
        raise ContractError("mesh must include a data axis")
    for axis_name, axis_size in zip(spec.axis_names, spec.axis_sizes, strict=True):
        if axis_name not in SUPPORTED_AXES:
            raise ContractError(f"unsupported mesh axis {axis_name!r}; supported axes are {sorted(SUPPORTED_AXES)}")
        if axis_name != "data" and axis_size != 1:
            raise ContractError(f"mesh axis {axis_name!r} is reserved for later and must have size 1 for now")


def _validate_batch_leading_dims(batch: Batch) -> None:
    arrays = [batch.input_ids, batch.target_ids, batch.loss_mask]
    if batch.doc_ids is not None:
        arrays.append(batch.doc_ids)
    leading_dims = []
    for value in arrays:
        array = np.asarray(value)
        if array.ndim == 0:
            raise ContractError("batch arrays must have a leading batch dimension")
        leading_dims.append(array.shape[0])
    if len(set(leading_dims)) != 1:
        raise ContractError(f"all batch fields must share the same leading dimension, got {leading_dims}")
