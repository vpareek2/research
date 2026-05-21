import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.mesh import (
    build_mesh_context,
    build_sharding_plan,
    place_batch,
    place_model_state,
    place_optimizer_init_state,
)
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.services import LocalOrbaxCheckpointService
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec, TrinitySpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.specs.parallelism import ParallelismSpec
from jaxtitan.state import DataPipelineState, HostState
from jaxtitan.steps import initialize_train_state, make_train_step

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_orbax_checkpoint_restores_train_dataset_host_and_metadata(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    train_state = _advanced_state(built, optimizer)
    dataset_state = _dataset_state(token_offset=8, next_record_index=2)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=1, wallclock_start_ns=123, run_id="smoke")
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)

    service.save(
        1,
        train_state,
        dataset_state,
        host_state,
        {"run_id": "smoke", "note": "checkpoint"},
    )
    restored = service.restore_latest(initialize_train_state(built.state, optimizer.transform, seed=9))
    metadata = service.restore_latest_metadata()

    assert service.latest_step() == 1
    assert service.latest_path() == tmp_path / "run" / "checkpoints" / "000001"
    assert restored.step == 1
    assert restored.path == tmp_path / "run" / "checkpoints" / "000001"
    assert restored.dataset_state == dataset_state
    assert restored.host_state == host_state
    assert restored.metadata == {"run_id": "smoke", "note": "checkpoint"}
    assert metadata == {"run_id": "smoke", "note": "checkpoint"}
    assert _trees_equal(restored.train_state.model, train_state.model)
    assert _trees_equal(restored.train_state.opt_state, train_state.opt_state)
    assert jnp.array_equal(restored.train_state.step, train_state.step)
    assert jnp.array_equal(restored.train_state.tokens_seen, train_state.tokens_seen)
    assert _trees_equal(restored.train_state.rng, train_state.rng)
    service.close()


def test_multiple_saves_keep_only_max_to_keep_checkpoints(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    train_state = initialize_train_state(built.state, optimizer.transform, seed=1)
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)

    for step in (1, 2, 3):
        dataset_state = _dataset_state(token_offset=step * 8, next_record_index=step * 2)
        host_state = HostState(dataset=dataset_state, last_checkpoint_step=step, wallclock_start_ns=123, run_id="smoke")
        train_state = train_state.replace(step=jnp.asarray(step, dtype=jnp.int32))
        service.save(step, train_state, dataset_state, host_state, {"step": step})

    assert service.latest_step() == 3
    assert sorted(path.name for path in (tmp_path / "run" / "checkpoints").iterdir()) == ["000002", "000003"]
    service.close()


def test_protected_checkpoint_survives_max_to_keep_cleanup(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    train_state = initialize_train_state(built.state, optimizer.transform, seed=1)
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)

    for step in (1, 2, 3):
        if step == 3:
            service.set_protected_steps({1})
        dataset_state = _dataset_state(token_offset=step * 8, next_record_index=step * 2)
        host_state = HostState(dataset=dataset_state, last_checkpoint_step=step, wallclock_start_ns=123, run_id="smoke")
        train_state = train_state.replace(step=jnp.asarray(step, dtype=jnp.int32))
        service.save(step, train_state, dataset_state, host_state, {"step": step})

    assert service.latest_step() == 3
    assert "000001" in {path.name for path in (tmp_path / "run" / "checkpoints").iterdir()}
    assert "000003" in {path.name for path in (tmp_path / "run" / "checkpoints").iterdir()}
    service.close()


def test_restore_latest_fails_when_no_checkpoint_exists(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    template = initialize_train_state(built.state, optimizer.transform, seed=1)
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)

    with pytest.raises(ContractError, match="no checkpoints"):
        service.restore_latest(template)

    service.close()


def test_checkpoint_save_requires_matching_host_dataset(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    train_state = initialize_train_state(built.state, optimizer.transform, seed=1)
    dataset_state = _dataset_state(token_offset=8, next_record_index=2)
    host_state = HostState(
        dataset=_dataset_state(token_offset=0, next_record_index=0),
        last_checkpoint_step=1,
        wallclock_start_ns=123,
        run_id="smoke",
    )
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)

    with pytest.raises(ContractError, match="host_state.dataset"):
        service.save(1, train_state, dataset_state, host_state, {})

    service.close()


def test_restored_train_state_can_continue_one_train_step(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    train_state = _advanced_state(built, optimizer)
    dataset_state = _dataset_state(token_offset=8, next_record_index=2)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=1, wallclock_start_ns=123, run_id="smoke")
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)
    service.save(1, train_state, dataset_state, host_state, {"step": 1})

    restored = service.restore_latest(initialize_train_state(built.state, optimizer.transform, seed=2))
    next_state, metrics = make_train_step(built.graph, optimizer)(restored.train_state, _batch(offset=8))
    service.close()

    assert restored.dataset_state.token_offset == 8
    assert next_state.step == 2
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 8


def test_muon_optimizer_state_round_trips_and_can_continue(tmp_path) -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata, optimizer_name="muon")
    train_state = _advanced_state(built, optimizer)
    dataset_state = _dataset_state(token_offset=8, next_record_index=2)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=1, wallclock_start_ns=123, run_id="smoke")
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)
    service.save(1, train_state, dataset_state, host_state, {"step": 1, "optimizer": "muon"})

    restored = service.restore_latest(initialize_train_state(built.state, optimizer.transform, seed=2))
    next_state, metrics = make_train_step(built.graph, optimizer)(restored.train_state, _batch(offset=8))
    service.close()

    assert restored.metadata["optimizer"] == "muon"
    assert _trees_equal(restored.train_state.opt_state, train_state.opt_state)
    assert next_state.step == 2
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 8


def test_trinity_moe_expert_bias_round_trips_and_stays_fixed(tmp_path) -> None:
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )
    optimizer = _optimizer(built.state, built.metadata, optimizer_name="muon")
    train_state = _advanced_state(built, optimizer)
    dataset_state = _dataset_state(token_offset=8, next_record_index=2)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=1, wallclock_start_ns=123, run_id="smoke")
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)
    service.save(1, train_state, dataset_state, host_state, {"step": 1, "model": "trinity-moe"})

    template = initialize_train_state(built.state, optimizer.transform, seed=2)
    restored = service.restore_latest(template)
    service.close()

    bias_path = next(item.path for item in built.metadata if item.tag == "moe_expert_bias")
    assert jnp.array_equal(_state_value_by_path(train_state.model, bias_path), jnp.zeros((3,), dtype=jnp.float32))
    assert jnp.array_equal(_state_value_by_path(restored.train_state.model, bias_path), jnp.zeros((3,), dtype=jnp.float32))


def test_auto_dion2_optimizer_state_round_trips_and_can_continue(tmp_path) -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="zero2"), param_layouts=built.param_layouts)
    model_state = place_model_state(built.state, plan)
    optimizer_init_state = place_optimizer_init_state(built.state, plan)
    optimizer = _optimizer(optimizer_init_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(
        model_state,
        optimizer.transform,
        seed=1,
        optimizer_init_model_state=optimizer_init_state,
    )
    step = make_train_step(built.graph, optimizer, sharding=plan, state_template=state, expected_batch_shape=(1, 4, 4))
    train_state, _metrics = step(state, place_batch(_batch(batch_size=4), plan))
    dataset_state = _dataset_state(token_offset=16, next_record_index=4)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=1, wallclock_start_ns=123, run_id="smoke")
    service = LocalOrbaxCheckpointService(tmp_path / "run", max_to_keep=2)
    service.save(1, train_state, dataset_state, host_state, {"step": 1, "optimizer": "muon"})

    template = initialize_train_state(
        model_state,
        optimizer.transform,
        seed=2,
        optimizer_init_model_state=optimizer_init_state,
    )
    restored = service.restore_latest(template)
    next_state, metrics = step(restored.train_state, place_batch(_batch(offset=16, batch_size=4), plan))
    service.close()

    assert {assignment.backend for assignment in optimizer.route_assignments} == {"adamw", "dion2"}
    assert restored.metadata["optimizer"] == "muon"
    assert _trees_equal(restored.train_state.opt_state, train_state.opt_state)
    assert next_state.step == 2
    assert next_state.tokens_seen == 32
    assert metrics.token_count == 16


def _advanced_state(built, optimizer):
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    next_state, _ = make_train_step(built.graph, optimizer)(state, _batch())
    return next_state


def _optimizer(model_state, metadata, *, optimizer_name: str = "adamw"):
    return build_optimizer(
        OptimizerSpec(name=optimizer_name, schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.01),
        model_state,
        metadata,
    )


def _tiny_spec(
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_heads: int = 2,
    n_kv_heads: int = 1,
) -> ModelSpec:
    return ModelSpec(
        name="decoder",
        variant="tiny",
        vocab_size=16,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=1,
        num_heads=num_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=4,
        compute_dtype="float32",
    )


def _tiny_trinity_spec(**overrides) -> ModelSpec:
    trinity_values = {
        "initial_dense_layers": 1,
        "local_window": 4,
        "local_layers_per_global": 1,
        "norm_policy": "depth_scaled_sandwich",
        "moe": None,
    }
    for key in tuple(overrides):
        if key in trinity_values:
            trinity_values[key] = overrides.pop(key)
    values = {
        "name": "trinity",
        "variant": "tiny",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_layers": 2,
        "num_heads": 2,
        "n_kv_heads": 1,
        "max_seq_len": 4,
        "compute_dtype": "float32",
        "trinity": TrinitySpec(**trinity_values),
    }
    values.update(overrides)
    return ModelSpec(**values)


def _batch(*, offset: int = 0, batch_size: int = 2) -> Batch:
    input_ids = (jnp.arange(batch_size * 4, dtype=jnp.int32).reshape(batch_size, 4) + offset) % 16
    target_ids = (input_ids + 1) % 16
    return Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=jnp.ones((batch_size, 4), dtype=jnp.bool_))


def _dataset_state(*, token_offset: int, next_record_index: int) -> DataPipelineState:
    return DataPipelineState(
        schema_version=2,
        backend="grain",
        backend_version="0.2.16",
        split="train",
        order="sequential",
        shuffle_seed=None,
        worker_count=0,
        worker_buffer_size=1,
        prefetch=False,
        manifest_path="data/train/manifest.json",
        manifest_sha256="hash",
        tokenizer_id="toy-tokenizer",
        seq_len=4,
        batch_size=2,
        num_records=100,
        next_record_index=next_record_index,
        token_offset=token_offset,
        epoch=0,
        sampler_summary="sampler",
        source_summary="source",
        grain_state={"version": 2, "last_seen_indices": {"0": next_record_index - 1}},
    )


def _trees_equal(left, right) -> bool:
    return all(
        np.array_equal(_leaf_array(left_leaf), _leaf_array(right_leaf))
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _state_value_by_path(state, target_path: tuple[str, ...]):
    for path, variable in nnx.to_flat_state(state):
        if tuple(str(part) for part in path) == target_path:
            return variable.get_value()
    raise AssertionError(f"state path {'.'.join(target_path)} not found")


def _leaf_array(value) -> np.ndarray:
    value = jax.device_get(value)
    try:
        return np.asarray(value)
    except TypeError:
        return np.asarray(jax.random.key_data(value))
