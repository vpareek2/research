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
from jaxtitan.specs.parallelism import ParallelismSpec, resolve_expert_fsdp_axis, resolve_expert_parallel_axis
from jaxtitan.models.execution import expert_parallel_dispatcher_backend

SUPPORTED_AXES = {"data", "fsdp", "tp", "cp", "ep", "expert_fsdp"}
ROUTED_EXPERT_TAGS = frozenset({"moe_gate", "moe_up", "moe_down"})


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
    def tp_axis_size(self) -> int:
        if "tp" not in self.spec.axis_names:
            return 1
        return self.spec.axis_sizes[self.spec.axis_names.index("tp")]

    @property
    def cp_axis_size(self) -> int:
        if "cp" not in self.spec.axis_names:
            return 1
        return self.spec.axis_sizes[self.spec.axis_names.index("cp")]

    @property
    def expert_fsdp_axis_size(self) -> int:
        if "expert_fsdp" not in self.spec.axis_names:
            return 1
        return self.spec.axis_sizes[self.spec.axis_names.index("expert_fsdp")]

    def axis_size(self, axis_name: str) -> int:
        if axis_name not in self.spec.axis_names:
            raise ContractError(f"mesh does not define axis {axis_name!r}")
        return self.spec.axis_sizes[self.spec.axis_names.index(axis_name)]

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
    expert_parallel_axis: str | None = None
    expert_parallel_axis_size: int = 1
    expert_parallel_axis_sharing: str | None = None
    expert_parallel_dispatcher: str | None = None
    expert_fsdp_axis: str | None = None
    expert_fsdp_axis_size: int = 1
    expert_fsdp_axis_sharing: str | None = None
    tensor_parallel_axis: str | None = None
    tensor_parallel_axis_size: int = 1
    context_parallel_axis: str | None = None
    context_parallel_axis_size: int = 1
    expert_param_paths: tuple[tuple[str, ...], ...] = ()
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
    axis_sizes = dict(zip(context.spec.axis_names, context.spec.axis_sizes, strict=True))
    expert_axis_policy = resolve_expert_parallel_axis(parallelism, axis_sizes)
    expert_fsdp_policy = resolve_expert_fsdp_axis(parallelism, axis_sizes)
    if parallelism.mode in {"zero2", "fsdp"} and "fsdp" not in context.spec.axis_names:
        raise ContractError(f"parallelism.mode='{parallelism.mode}' requires a mesh fsdp axis")
    if context.tp_axis_size > 1 and not parallelism.tensor_parallel:
        raise ContractError("mesh tp axis size greater than 1 requires parallelism.tensor_parallel=true")
    if parallelism.tensor_parallel and "tp" not in context.spec.axis_names:
        raise ContractError("parallelism.tensor_parallel=true requires a mesh tp axis")
    if context.cp_axis_size > 1 and not parallelism.context_parallel:
        raise ContractError("mesh cp axis size greater than 1 requires parallelism.context_parallel=true")
    if parallelism.context_parallel and "cp" not in context.spec.axis_names:
        raise ContractError("parallelism.context_parallel=true requires a mesh cp axis")
    if parallelism.expert_parallel and not expert_layouts:
        raise ContractError("parallelism.expert_parallel=true requires routed expert layout metadata")
    seq_axis = "cp" if parallelism.context_parallel else None
    batch_matrix = NamedSharding(context.mesh, P("data", seq_axis))
    accumulated_batch = NamedSharding(context.mesh, P(None, "data", seq_axis))
    replicated = NamedSharding(context.mesh, P())
    param_shardings = _param_shardings(
        context,
        parallelism,
        param_layouts,
        expert_layouts,
        replicated,
        expert_axis_name=expert_axis_policy.axis,
        expert_fsdp_axis_name=expert_fsdp_policy.axis,
    )
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
        expert_parallel_axis=expert_axis_policy.axis,
        expert_parallel_axis_size=expert_axis_policy.axis_size,
        expert_parallel_axis_sharing=expert_axis_policy.axis_sharing,
        expert_parallel_dispatcher=expert_parallel_dispatcher_backend(expert_axis_policy.axis_sharing),
        expert_fsdp_axis=expert_fsdp_policy.axis,
        expert_fsdp_axis_size=expert_fsdp_policy.axis_size,
        expert_fsdp_axis_sharing=expert_fsdp_policy.axis_sharing,
        tensor_parallel_axis="tp" if parallelism.tensor_parallel else None,
        tensor_parallel_axis_size=context.tp_axis_size if parallelism.tensor_parallel else 1,
        context_parallel_axis="cp" if parallelism.context_parallel else None,
        context_parallel_axis_size=context.cp_axis_size if parallelism.context_parallel else 1,
        expert_param_paths=tuple(layout.path for layout in expert_layouts) if expert_axis_policy.enabled else (),
    )


def place_batch(batch: Batch, plan: ShardingPlan) -> Batch:
    """Place a Batch on device according to the sharding plan."""

    _validate_batch_leading_dims(batch)
    _validate_batch_partition_dims(batch.input_ids, plan)
    return Batch(
        input_ids=jax.device_put(np.asarray(batch.input_ids), plan.batch.input_ids),
        target_ids=jax.device_put(np.asarray(batch.target_ids), plan.batch.target_ids),
        loss_mask=jax.device_put(np.asarray(batch.loss_mask), plan.batch.loss_mask),
        doc_ids=None if batch.doc_ids is None else place_batch_array(batch.doc_ids, plan.mesh),
    )


def place_accumulated_batch(batch: Batch, plan: ShardingPlan) -> Batch:
    """Place an accumulated Batch with shape [accum, global_batch, seq]."""

    _validate_accumulated_batch_dims(batch)
    _validate_accumulated_batch_partition_dims(batch.input_ids, plan)
    return Batch(
        input_ids=jax.device_put(np.asarray(batch.input_ids), plan.batch.accumulated_input_ids),
        target_ids=jax.device_put(np.asarray(batch.target_ids), plan.batch.accumulated_target_ids),
        loss_mask=jax.device_put(np.asarray(batch.loss_mask), plan.batch.accumulated_loss_mask),
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
    tail = ["cp" if (context.cp_axis_size > 1 and index == 0) else None for index in range(array.ndim - 1)]
    sharding = NamedSharding(context.mesh, P("data", *tail))
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
    tail = ["cp" if (context.cp_axis_size > 1 and index == 0) else None for index in range(array.ndim - 2)]
    sharding = NamedSharding(context.mesh, P(None, "data", *tail))
    return jax.device_put(array, sharding)


def place_replicated(tree: Any, plan: ShardingPlan) -> Any:
    """Place every PyTree leaf with replicated sharding."""

    return jax.tree.map(lambda leaf: jax.device_put(leaf, plan.replicated), tree)


def place_model_state(tree: Any, plan: ShardingPlan) -> Any:
    """Place model state leaves according to the sharding plan's parameter policy."""

    if plan.parallelism.mode == "ddp" and not plan.parallelism.expert_parallel and not plan.parallelism.tensor_parallel:
        return place_replicated(tree, plan)
    return _place_params_by_policy(tree, plan, omit_fsdp=plan.parallelism.mode == "zero2")


def place_optimizer_init_state(tree: Any, plan: ShardingPlan) -> Any:
    """Place a model-shaped tree used only to initialize optimizer state."""

    if (
        plan.parallelism.mode == "ddp"
        and not plan.parallelism.expert_parallel
        and not plan.parallelism.tensor_parallel
    ):
        return place_replicated(tree, plan)
    return _place_params_by_policy(tree, plan)


def gradient_shardings_like(tree: Any, plan: ShardingPlan) -> Any:
    """Build a model-shaped sharding tree for gradients and optimizer updates."""

    if (
        plan.parallelism.mode == "ddp"
        and not plan.parallelism.expert_parallel
        and not plan.parallelism.tensor_parallel
    ):
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


def _place_params_by_policy(tree: Any, plan: ShardingPlan, *, omit_fsdp: bool = False) -> Any:
    def place_leaf(path, leaf):
        metadata_path = _metadata_path_from_jax_path(path)
        sharding = plan.param_shardings.get(metadata_path)
        if sharding is None:
            raise ContractError(f"missing sharding policy for model parameter {'.'.join(metadata_path)!r}")
        if omit_fsdp:
            sharding = _without_mesh_axis(sharding, "fsdp")
        return jax.device_put(leaf, sharding)

    return jax.tree_util.tree_map_with_path(place_leaf, tree)


def _without_mesh_axis(sharding: NamedSharding, axis_name: str) -> NamedSharding:
    def remove(axis: Any) -> Any:
        if axis == axis_name:
            return None
        if isinstance(axis, tuple):
            remaining = tuple(name for name in axis if name != axis_name)
            if not remaining:
                return None
            return remaining[0] if len(remaining) == 1 else remaining
        return axis

    spec = tuple(remove(axis) for axis in tuple(sharding.spec))
    return NamedSharding(sharding.mesh, P()) if all(axis is None for axis in spec) else NamedSharding(sharding.mesh, P(*spec))


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
    has_real_expert_fsdp = "expert_fsdp" in spec.axis_names and spec.axis_sizes[spec.axis_names.index("expert_fsdp")] > 1
    if has_real_expert_fsdp and "ep" not in spec.axis_names:
        raise ContractError("mesh expert_fsdp axis with size > 1 requires a mesh ep axis")
    for axis_name, axis_size in zip(spec.axis_names, spec.axis_sizes, strict=True):
        if axis_name not in SUPPORTED_AXES:
            raise ContractError(f"unsupported mesh axis {axis_name!r}; supported axes are {sorted(SUPPORTED_AXES)}")
        if axis_name == "tp" and axis_size < 1:
            raise ContractError("mesh tp axis size must be positive")
        if axis_name == "cp" and axis_size < 1:
            raise ContractError("mesh cp axis size must be positive")


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


def _validate_batch_partition_dims(value: Any, plan: ShardingPlan) -> None:
    array = np.asarray(value)
    if array.shape[0] % plan.mesh.data_axis_size != 0:
        raise ContractError(
            f"batch leading dimension {array.shape[0]} must be divisible by data axis size {plan.mesh.data_axis_size}"
        )
    if plan.parallelism.context_parallel:
        if array.ndim < 2:
            raise ContractError("context-parallel batch arrays must include a sequence dimension")
        if array.shape[1] % plan.context_parallel_axis_size != 0:
            raise ContractError(
                f"batch sequence dimension {array.shape[1]} must be divisible by cp axis size "
                f"{plan.context_parallel_axis_size}"
            )


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


def _validate_accumulated_batch_partition_dims(value: Any, plan: ShardingPlan) -> None:
    array = np.asarray(value)
    if array.shape[1] % plan.mesh.data_axis_size != 0:
        raise ContractError(
            f"accumulated batch dimension {array.shape[1]} must be divisible by data axis size {plan.mesh.data_axis_size}"
        )
    if plan.parallelism.context_parallel:
        if array.ndim < 3:
            raise ContractError("context-parallel accumulated batch arrays must include a sequence dimension")
        if array.shape[2] % plan.context_parallel_axis_size != 0:
            raise ContractError(
                f"accumulated sequence dimension {array.shape[2]} must be divisible by cp axis size "
                f"{plan.context_parallel_axis_size}"
            )


def _param_shardings(
    context: MeshContext,
    parallelism: ParallelismSpec,
    param_layouts: tuple[ParamLayout, ...],
    expert_layouts: tuple[ExpertLayout, ...],
    replicated: NamedSharding,
    *,
    expert_axis_name: str | None,
    expert_fsdp_axis_name: str | None,
) -> dict[tuple[str, ...], NamedSharding]:
    shardings = {}
    fsdp_axis_size = context.fsdp_axis_size
    expert_by_path = {layout.path: layout for layout in expert_layouts}
    for layout in param_layouts:
        if parallelism.tensor_parallel and layout.tag in ROUTED_EXPERT_TAGS and layout.tp_axis is not None:
            raise ContractError(
                f"routed expert parameter {'.'.join(layout.path)!r} cannot use tensor-parallel matrix-axis sharding yet"
            )
        expert_layout = expert_by_path.get(layout.path)
        if parallelism.expert_parallel and expert_layout is not None:
            if expert_axis_name is None:
                raise ContractError("parallelism.expert_parallel=true requires a resolved expert parallel axis")
            shardings[layout.path] = _expert_sharding(
                context,
                expert_layout,
                axis_name=expert_axis_name,
                expert_fsdp_axis_name=expert_fsdp_axis_name,
            )
            continue
        spec = [None] * len(layout.shape)
        if parallelism.tensor_parallel and layout.tp_axis is not None:
            if len(layout.shape) <= layout.tp_axis:
                raise ContractError(f"parameter {'.'.join(layout.path)!r} has invalid tp axis {layout.tp_axis}")
            tp_dim_size = layout.shape[layout.tp_axis]
            if tp_dim_size % context.tp_axis_size != 0:
                raise ContractError(
                    f"parameter {'.'.join(layout.path)!r} dimension {layout.tp_axis} "
                    f"with size {tp_dim_size} must be divisible by tp axis size {context.tp_axis_size}"
                )
            spec[layout.tp_axis] = "tp"
        if parallelism.mode == "ddp" or layout.fsdp_axis is None:
            shardings[layout.path] = NamedSharding(context.mesh, P(*spec)) if any(axis is not None for axis in spec) else replicated
            continue
        if parallelism.tensor_parallel and layout.fsdp_axis == layout.tp_axis:
            shardings[layout.path] = NamedSharding(context.mesh, P(*spec)) if any(axis is not None for axis in spec) else replicated
            continue
        if len(layout.shape) <= layout.fsdp_axis:
            raise ContractError(f"parameter {'.'.join(layout.path)!r} has invalid fsdp axis {layout.fsdp_axis}")
        dim_size = layout.shape[layout.fsdp_axis]
        if dim_size % fsdp_axis_size != 0:
            raise ContractError(
                f"parameter {'.'.join(layout.path)!r} dimension {layout.fsdp_axis} "
                f"with size {dim_size} must be divisible by fsdp axis size {fsdp_axis_size}"
            )
        spec[layout.fsdp_axis] = "fsdp"
        shardings[layout.path] = NamedSharding(context.mesh, P(*spec))
    return shardings


def _expert_sharding(
    context: MeshContext,
    layout: ExpertLayout,
    *,
    axis_name: str,
    expert_fsdp_axis_name: str | None,
) -> NamedSharding:
    if len(layout.shape) <= layout.expert_axis:
        raise ContractError(f"expert parameter {'.'.join(layout.path)!r} has invalid expert axis {layout.expert_axis}")
    if len(layout.shape) != 3:
        raise ContractError(f"expert parameter {'.'.join(layout.path)!r} must be rank 3, got {layout.shape}")
    axis_size = context.axis_size(axis_name)
    dim_size = layout.shape[layout.expert_axis]
    if dim_size % axis_size != 0:
        raise ContractError(
            f"expert parameter {'.'.join(layout.path)!r} dimension {layout.expert_axis} "
            f"with size {dim_size} must be divisible by {axis_name} axis size {axis_size}"
        )
    spec = [None] * len(layout.shape)
    spec[layout.expert_axis] = axis_name
    if expert_fsdp_axis_name is not None:
        if layout.fsdp_axis is None:
            raise ContractError(f"expert parameter {'.'.join(layout.path)!r} has no expert FSDP axis")
        if layout.fsdp_axis == layout.expert_axis:
            raise ContractError(f"expert parameter {'.'.join(layout.path)!r} cannot shard one axis by ep and expert_fsdp")
        if len(layout.shape) <= layout.fsdp_axis:
            raise ContractError(f"expert parameter {'.'.join(layout.path)!r} has invalid expert fsdp axis {layout.fsdp_axis}")
        fsdp_axis_size = context.axis_size(expert_fsdp_axis_name)
        matrix_dim_size = layout.shape[layout.fsdp_axis]
        if matrix_dim_size % fsdp_axis_size != 0:
            raise ContractError(
                f"expert parameter {'.'.join(layout.path)!r} dimension {layout.fsdp_axis} "
                f"with size {matrix_dim_size} must be divisible by {expert_fsdp_axis_name} axis size {fsdp_axis_size}"
            )
        spec[layout.fsdp_axis] = expert_fsdp_axis_name
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
