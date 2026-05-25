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
from jaxtitan.models import ExpertLayout, ParamLayout
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.parallelism import ParallelismSpec

SUPPORTED_AXES = {"data", "fsdp", "tp", "ep"}


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
    def fsdp_axis_size(self) -> int:
        if "fsdp" not in self.spec.axis_names:
            return 1
        return self.spec.axis_sizes[self.spec.axis_names.index("fsdp")]

    @property
    def ep_axis_size(self) -> int:
        if "ep" not in self.spec.axis_names:
            return 1
        return self.spec.axis_sizes[self.spec.axis_names.index("ep")]

    @property
    def selected_device_count(self) -> int:
        return len(self.devices)


@dataclass(frozen=True, slots=True)
class BatchSharding:
    """Named shardings for fixed-shape batch fields."""

    input_ids: NamedSharding
    target_ids: NamedSharding
    loss_mask: NamedSharding
    accumulated_input_ids: NamedSharding
    accumulated_target_ids: NamedSharding
    accumulated_loss_mask: NamedSharding


@dataclass(frozen=True, slots=True)
class ShardingPlan:
    """Inspectable shardings for known runtime object classes."""

    mesh: MeshContext
    batch: BatchSharding
    replicated: NamedSharding
    metrics: NamedSharding
    parallelism: ParallelismSpec
    param_shardings: dict[tuple[str, ...], NamedSharding]
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


def build_sharding_plan(
    context: MeshContext,
    *,
    parallelism: ParallelismSpec | None = None,
    param_layouts: tuple[ParamLayout, ...] = (),
    expert_layouts: tuple[ExpertLayout, ...] = (),
) -> ShardingPlan:
    """Build shardings for data-sharded batches and policy-placed state."""

    parallelism = ParallelismSpec() if parallelism is None else parallelism
    if parallelism.mode in {"zero2", "fsdp"} and "fsdp" not in context.spec.axis_names:
        raise ContractError(f"parallelism.mode='{parallelism.mode}' requires a mesh fsdp axis")
    if parallelism.expert_parallel and "ep" not in context.spec.axis_names:
        raise ContractError("parallelism.expert_parallel=true requires a mesh ep axis")
    if parallelism.expert_parallel and not expert_layouts:
        raise ContractError("parallelism.expert_parallel=true requires routed expert layout metadata")
    batch_matrix = NamedSharding(context.mesh, P("data", None))
    accumulated_batch = NamedSharding(context.mesh, P(None, "data", None))
    replicated = NamedSharding(context.mesh, P())
    param_shardings = _param_shardings(context, parallelism, param_layouts, expert_layouts, replicated)
    return ShardingPlan(
        mesh=context,
        batch=BatchSharding(
            input_ids=batch_matrix,
            target_ids=batch_matrix,
            loss_mask=batch_matrix,
            accumulated_input_ids=accumulated_batch,
            accumulated_target_ids=accumulated_batch,
            accumulated_loss_mask=accumulated_batch,
        ),
        replicated=replicated,
        metrics=replicated,
        parallelism=parallelism,
        param_shardings=param_shardings,
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


def place_accumulated_batch(batch: Batch, plan: ShardingPlan) -> Batch:
    """Place an accumulated Batch with shape [accum, global_batch, seq]."""

    _validate_accumulated_batch_dims(batch)
    return Batch(
        input_ids=place_accumulated_batch_array(batch.input_ids, plan.mesh),
        target_ids=place_accumulated_batch_array(batch.target_ids, plan.mesh),
        loss_mask=place_accumulated_batch_array(batch.loss_mask, plan.mesh),
        doc_ids=None if batch.doc_ids is None else place_accumulated_batch_array(batch.doc_ids, plan.mesh),
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


def place_accumulated_batch_array(value: Any, context: MeshContext) -> jax.Array:
    """Place an accumulated batch array with its batch axis sharded over data."""

    array = np.asarray(value)
    if array.ndim < 2:
        raise ContractError("accumulated batch arrays must have accumulation and batch dimensions")
    if array.shape[1] % context.data_axis_size != 0:
        raise ContractError(
            f"accumulated batch dimension {array.shape[1]} must be divisible by data axis size {context.data_axis_size}"
        )
    sharding = NamedSharding(context.mesh, P(None, "data", *([None] * (array.ndim - 2))))
    return jax.device_put(array, sharding)


def place_replicated(tree: Any, plan: ShardingPlan) -> Any:
    """Place every PyTree leaf with replicated sharding."""

    return jax.tree.map(lambda leaf: jax.device_put(leaf, plan.replicated), tree)


def place_model_state(tree: Any, plan: ShardingPlan) -> Any:
    """Place model state leaves according to the sharding plan's parameter policy."""

    if plan.parallelism.mode in {"ddp", "zero2"} and not plan.parallelism.expert_parallel:
        return place_replicated(tree, plan)
    return _place_params_by_policy(tree, plan)


def place_optimizer_init_state(tree: Any, plan: ShardingPlan) -> Any:
    """Place a model-shaped tree used only to initialize optimizer state."""

    if plan.parallelism.mode == "ddp" and not plan.parallelism.expert_parallel:
        return place_replicated(tree, plan)
    return _place_params_by_policy(tree, plan)


def gradient_shardings_like(tree: Any, plan: ShardingPlan) -> Any:
    """Build a model-shaped sharding tree for gradients and optimizer updates."""

    if plan.parallelism.mode == "ddp" and not plan.parallelism.expert_parallel:
        return jax.tree.map(lambda _leaf: plan.replicated, tree)

    def leaf_sharding(path, _leaf):
        metadata_path = _metadata_path_from_jax_path(path)
        sharding = plan.param_shardings.get(metadata_path)
        if sharding is None:
            raise ContractError(f"missing sharding policy for model parameter {'.'.join(metadata_path)!r}")
        return sharding

    return jax.tree_util.tree_map_with_path(leaf_sharding, tree)


def optimizer_shardings_like(tree: Any, plan: ShardingPlan) -> Any:
    """Build a sharding PyTree matching an already placed optimizer state tree."""

    return replicated_shardings_like(tree, plan)


def _place_params_by_policy(tree: Any, plan: ShardingPlan) -> Any:
    def place_leaf(path, leaf):
        metadata_path = _metadata_path_from_jax_path(path)
        sharding = plan.param_shardings.get(metadata_path)
        if sharding is None:
            raise ContractError(f"missing sharding policy for model parameter {'.'.join(metadata_path)!r}")
        return jax.device_put(leaf, sharding)

    return jax.tree_util.tree_map_with_path(place_leaf, tree)


def replicated_shardings_like(tree: Any, plan: ShardingPlan) -> Any:
    """Build a sharding PyTree matching an already placed runtime state tree."""

    def leaf_sharding(leaf):
        sharding = getattr(leaf, "sharding", None)
        return sharding if isinstance(sharding, NamedSharding) else plan.replicated

    return jax.tree.map(leaf_sharding, tree)


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
        if axis_name == "tp" and axis_size != 1:
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


def _validate_accumulated_batch_dims(batch: Batch) -> None:
    arrays = [batch.input_ids, batch.target_ids, batch.loss_mask]
    if batch.doc_ids is not None:
        arrays.append(batch.doc_ids)
    prefix_dims = []
    for value in arrays:
        array = np.asarray(value)
        if array.ndim < 2:
            raise ContractError("accumulated batch arrays must have accumulation and batch dimensions")
        prefix_dims.append(array.shape[:2])
    if len(set(prefix_dims)) != 1:
        raise ContractError(f"all accumulated batch fields must share accumulation/batch dimensions, got {prefix_dims}")


def _param_shardings(
    context: MeshContext,
    parallelism: ParallelismSpec,
    param_layouts: tuple[ParamLayout, ...],
    expert_layouts: tuple[ExpertLayout, ...],
    replicated: NamedSharding,
) -> dict[tuple[str, ...], NamedSharding]:
    shardings = {}
    fsdp_axis_size = context.fsdp_axis_size
    expert_by_path = {layout.path: layout for layout in expert_layouts}
    for layout in param_layouts:
        expert_layout = expert_by_path.get(layout.path)
        if parallelism.expert_parallel and expert_layout is not None:
            shardings[layout.path] = _expert_sharding(context, expert_layout)
            continue
        if parallelism.mode == "ddp":
            shardings[layout.path] = replicated
            continue
        if layout.fsdp_axis is None:
            shardings[layout.path] = replicated
            continue
        if len(layout.shape) <= layout.fsdp_axis:
            raise ContractError(f"parameter {'.'.join(layout.path)!r} has invalid fsdp axis {layout.fsdp_axis}")
        dim_size = layout.shape[layout.fsdp_axis]
        if dim_size % fsdp_axis_size != 0:
            raise ContractError(
                f"parameter {'.'.join(layout.path)!r} dimension {layout.fsdp_axis} "
                f"with size {dim_size} must be divisible by fsdp axis size {fsdp_axis_size}"
            )
        spec = [None] * len(layout.shape)
        spec[layout.fsdp_axis] = "fsdp"
        shardings[layout.path] = NamedSharding(context.mesh, P(*spec))
    return shardings


def _expert_sharding(context: MeshContext, layout: ExpertLayout) -> NamedSharding:
    if len(layout.shape) <= layout.expert_axis:
        raise ContractError(f"expert parameter {'.'.join(layout.path)!r} has invalid expert axis {layout.expert_axis}")
    if len(layout.shape) != 3:
        raise ContractError(f"expert parameter {'.'.join(layout.path)!r} must be rank 3, got {layout.shape}")
    ep_axis_size = context.ep_axis_size
    dim_size = layout.shape[layout.expert_axis]
    if dim_size % ep_axis_size != 0:
        raise ContractError(
            f"expert parameter {'.'.join(layout.path)!r} dimension {layout.expert_axis} "
            f"with size {dim_size} must be divisible by ep axis size {ep_axis_size}"
        )
    spec = [None] * len(layout.shape)
    spec[layout.expert_axis] = "ep"
    return NamedSharding(context.mesh, P(*spec))


def _metadata_path_from_jax_path(path) -> tuple[str, ...]:
    parts = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name == "value":
            continue
        parts.append(str(name))
    return tuple(parts)
