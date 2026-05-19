import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_batch, place_replicated
from jaxtitan.specs.mesh import MeshSpec

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_default_one_device_data_mesh() -> None:
    context = build_mesh_context(MeshSpec())

    assert context.spec.axis_names == ("data",)
    assert context.spec.axis_sizes == (1,)
    assert context.data_axis_size == 1
    assert context.mesh.axis_names == ("data",)
    assert len(context.devices) == 1


def test_four_device_data_mesh() -> None:
    require_fake_devices()

    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))

    assert context.mesh.devices.shape == (4,)
    assert len(context.devices) == 4
    assert context.local_device_count >= 4


def test_build_mesh_context_rejects_too_many_devices() -> None:
    devices = jax.local_devices()[:1]

    with pytest.raises(ContractError, match="requires 2 device"):
        build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(2,)), devices=devices)


def test_build_mesh_context_rejects_missing_data_axis() -> None:
    with pytest.raises(ContractError, match="data axis"):
        build_mesh_context(MeshSpec(axis_names=("tp",), axis_sizes=(1,)))


def test_build_mesh_context_rejects_unknown_axis() -> None:
    with pytest.raises(ContractError, match="unsupported mesh axis"):
        build_mesh_context(MeshSpec(axis_names=("data", "pipeline"), axis_sizes=(1, 1)))


def test_build_mesh_context_rejects_non_data_axis_size_greater_than_one() -> None:
    require_fake_devices()

    with pytest.raises(ContractError, match="reserved for later"):
        build_mesh_context(MeshSpec(axis_names=("data", "tp"), axis_sizes=(2, 2)))


def test_sharding_plan_contents() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))

    plan = build_sharding_plan(context)

    expected_batch = NamedSharding(context.mesh, P("data", None))
    expected_replicated = NamedSharding(context.mesh, P())
    assert plan.batch.input_ids == expected_batch
    assert plan.batch.target_ids == expected_batch
    assert plan.batch.loss_mask == expected_batch
    assert plan.replicated == expected_replicated
    assert plan.metrics == expected_replicated
    assert plan.kv_cache is None


def test_place_batch_shards_arrays_over_data_axis() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.arange(32, dtype=np.int32).reshape(8, 4),
        target_ids=np.arange(32, 64, dtype=np.int32).reshape(8, 4),
        loss_mask=np.ones((8, 4), dtype=np.bool_),
    )

    placed = place_batch(batch, plan)

    assert placed.input_ids.sharding == plan.batch.input_ids
    assert placed.target_ids.sharding == plan.batch.target_ids
    assert placed.loss_mask.sharding == plan.batch.loss_mask
    assert placed.doc_ids is None
    assert {shard.data.shape for shard in placed.input_ids.addressable_shards} == {(2, 4)}


def test_place_batch_places_doc_ids_when_present() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.arange(32, dtype=np.int32).reshape(8, 4),
        target_ids=np.arange(32, 64, dtype=np.int32).reshape(8, 4),
        loss_mask=np.ones((8, 4), dtype=np.bool_),
        doc_ids=np.arange(8, dtype=np.int32),
    )

    placed = place_batch(batch, plan)

    assert placed.doc_ids is not None
    assert placed.doc_ids.sharding == NamedSharding(context.mesh, P("data"))
    assert {shard.data.shape for shard in placed.doc_ids.addressable_shards} == {(2,)}


def test_place_batch_rejects_mismatched_leading_dims() -> None:
    context = build_mesh_context(MeshSpec())
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.zeros((4, 8), dtype=np.int32),
        target_ids=np.zeros((5, 8), dtype=np.int32),
        loss_mask=np.ones((4, 8), dtype=np.bool_),
    )

    with pytest.raises(ContractError, match="same leading dimension"):
        place_batch(batch, plan)


def test_place_batch_rejects_non_divisible_batch_size() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.zeros((6, 8), dtype=np.int32),
        target_ids=np.zeros((6, 8), dtype=np.int32),
        loss_mask=np.ones((6, 8), dtype=np.bool_),
    )

    with pytest.raises(ContractError, match="divisible by data axis size"):
        place_batch(batch, plan)


def test_place_replicated_keeps_full_leaf_shape_on_each_device() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    tree = {
        "weight": jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        "bias": jnp.arange(4, dtype=jnp.float32),
    }

    placed = place_replicated(tree, plan)

    assert placed["weight"].sharding == plan.replicated
    assert {shard.data.shape for shard in placed["weight"].addressable_shards} == {(4, 4)}
    assert placed["bias"].sharding == plan.replicated
    assert {shard.data.shape for shard in placed["bias"].addressable_shards} == {(4,)}
