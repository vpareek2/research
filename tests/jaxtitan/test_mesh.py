import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.mesh import (
    build_mesh_context,
    build_sharding_plan,
    gradient_shardings_like,
    optimizer_shardings_like,
    place_model_state,
    place_optimizer_init_state,
    place_accumulated_batch,
    place_batch,
    place_replicated,
    replicated_shardings_like,
    require_single_process_runtime,
)
from jaxtitan.models import build_model
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.parallelism import ParallelismSpec

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
    assert context.selected_device_count == 4
    assert context.local_device_count >= 4
    assert context.global_device_count >= 4
    assert context.process_count == 1
    assert context.process_index == 0


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


def test_build_mesh_context_accepts_fsdp_axis() -> None:
    require_fake_devices()

    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))

    assert context.mesh.devices.shape == (1, 4)
    assert context.data_axis_size == 1
    assert context.fsdp_axis_size == 4


def test_sharding_plan_contents() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))

    plan = build_sharding_plan(context)

    expected_batch = NamedSharding(context.mesh, P("data", None))
    expected_accumulated_batch = NamedSharding(context.mesh, P(None, "data", None))
    expected_replicated = NamedSharding(context.mesh, P())
    assert plan.batch.input_ids == expected_batch
    assert plan.batch.target_ids == expected_batch
    assert plan.batch.loss_mask == expected_batch
    assert plan.batch.accumulated_input_ids == expected_accumulated_batch
    assert plan.batch.accumulated_target_ids == expected_accumulated_batch
    assert plan.batch.accumulated_loss_mask == expected_accumulated_batch
    assert plan.replicated == expected_replicated
    assert plan.metrics == expected_replicated
    assert plan.kv_cache is None


def test_fsdp_sharding_plan_maps_decoder_layouts_to_partition_specs() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))

    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="fsdp"), param_layouts=built.param_layouts)

    by_tag = {layout.tag: plan.param_shardings[layout.path] for layout in built.param_layouts}
    assert by_tag["embedding"].spec == P()
    assert by_tag["attention_q"].spec == P(None, "fsdp")
    assert by_tag["attention_k"].spec == P(None, "fsdp")
    assert by_tag["attention_v"].spec == P(None, "fsdp")
    assert by_tag["attention_o"].spec == P("fsdp", None)
    assert by_tag["mlp_gate"].spec == P(None, "fsdp")
    assert by_tag["mlp_up"].spec == P(None, "fsdp")
    assert by_tag["mlp_down"].spec == P("fsdp", None)
    assert by_tag["lm_head"].spec == P("fsdp", None)
    assert plan.metrics.spec == P()
    assert plan.batch.accumulated_input_ids.spec == P(None, "data", None)


def test_fsdp_sharding_plan_rejects_non_divisible_parameter_axis() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 3)))

    with pytest.raises(ContractError, match="divisible by fsdp axis size"):
        build_sharding_plan(
            context,
            parallelism=ParallelismSpec(mode="fsdp"),
            param_layouts=built.param_layouts,
        )


def test_place_model_state_uses_fsdp_param_shardings() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="fsdp"), param_layouts=built.param_layouts)

    placed = place_model_state(built.state, plan)
    shardings = replicated_shardings_like(placed, plan)
    placed_by_path = {_metadata_path(path): value for path, value in jax.tree_util.tree_flatten_with_path(placed)[0]}
    layout_by_tag = {layout.tag: layout for layout in built.param_layouts}

    q = placed_by_path[layout_by_tag["attention_q"].path]
    o = placed_by_path[layout_by_tag["attention_o"].path]
    embed = placed_by_path[layout_by_tag["embedding"].path]
    assert q.sharding.spec == P(None, "fsdp")
    assert {shard.data.shape for shard in q.addressable_shards} == {(16, 4)}
    assert o.sharding.spec == P("fsdp", None)
    assert {shard.data.shape for shard in o.addressable_shards} == {(4, 16)}
    assert embed.sharding.spec == P()
    assert shardings == jax.tree.map(lambda leaf: leaf.sharding, placed)


def test_zero2_sharding_plan_shards_optimizer_not_model_state() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="zero2"), param_layouts=built.param_layouts)

    model_state = place_model_state(built.state, plan)
    optimizer_init_state = place_optimizer_init_state(built.state, plan)
    grad_shardings = gradient_shardings_like(model_state, plan)
    opt_shardings = optimizer_shardings_like(optimizer_init_state, plan)
    model_by_path = {_metadata_path(path): value for path, value in jax.tree_util.tree_flatten_with_path(model_state)[0]}
    opt_by_path = {_metadata_path(path): value for path, value in jax.tree_util.tree_flatten_with_path(optimizer_init_state)[0]}
    grad_by_path = {_metadata_path(path): value for path, value in jax.tree_util.tree_flatten_with_path(grad_shardings)[0]}
    layout_by_tag = {layout.tag: layout for layout in built.param_layouts}

    q_path = layout_by_tag["attention_q"].path
    embed_path = layout_by_tag["embedding"].path
    assert plan.param_shardings[q_path].spec == P(None, "fsdp")
    assert model_by_path[q_path].sharding == plan.replicated
    assert model_by_path[embed_path].sharding == plan.replicated
    assert opt_by_path[q_path].sharding.spec == P(None, "fsdp")
    assert opt_by_path[embed_path].sharding == plan.replicated
    assert grad_by_path[q_path].spec == P(None, "fsdp")
    assert grad_by_path[embed_path] == plan.replicated
    assert opt_shardings == jax.tree.map(lambda leaf: leaf.sharding, optimizer_init_state)


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


def test_place_accumulated_batch_shards_batch_axis_over_data_axis() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.arange(64, dtype=np.int32).reshape(2, 8, 4),
        target_ids=np.arange(64, 128, dtype=np.int32).reshape(2, 8, 4),
        loss_mask=np.ones((2, 8, 4), dtype=np.bool_),
    )

    placed = place_accumulated_batch(batch, plan)

    assert placed.input_ids.sharding == plan.batch.accumulated_input_ids
    assert placed.target_ids.sharding == plan.batch.accumulated_target_ids
    assert placed.loss_mask.sharding == plan.batch.accumulated_loss_mask
    assert placed.doc_ids is None
    assert {shard.data.shape for shard in placed.input_ids.addressable_shards} == {(2, 2, 4)}


def test_place_accumulated_batch_rejects_non_divisible_batch_axis() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    batch = Batch(
        input_ids=np.zeros((2, 6, 4), dtype=np.int32),
        target_ids=np.zeros((2, 6, 4), dtype=np.int32),
        loss_mask=np.ones((2, 6, 4), dtype=np.bool_),
    )

    with pytest.raises(ContractError, match="divisible by data axis size"):
        place_accumulated_batch(batch, plan)


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


def test_replicated_shardings_like_matches_tree_structure() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    tree = {"weight": jnp.ones((2, 2)), "nested": (jnp.ones((1,)), None)}

    shardings = replicated_shardings_like(tree, plan)

    assert shardings["weight"] == plan.replicated
    assert shardings["nested"][0] == plan.replicated
    assert shardings["nested"][1] is None


def test_require_single_process_runtime_rejects_multi_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jaxtitan.mesh.sharding.jax.process_count", lambda: 2)

    with pytest.raises(ContractError, match="exactly one process"):
        require_single_process_runtime()


def _tiny_spec(**overrides) -> ModelSpec:
    values = {
        "name": "decoder",
        "variant": "tiny",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_layers": 1,
        "num_heads": 4,
        "n_kv_heads": 4,
        "max_seq_len": 8,
        "compute_dtype": "float32",
    }
    values.update(overrides)
    return ModelSpec(**values)


def _metadata_path(path) -> tuple[str, ...]:
    parts = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name == "value":
            continue
        parts.append(str(name))
    return tuple(parts)
