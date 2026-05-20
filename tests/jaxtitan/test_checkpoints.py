import jax
import jax.numpy as jnp
import pytest

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.services import LocalOrbaxCheckpointService
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.state import DataPipelineState, HostState
from jaxtitan.steps import initialize_train_state, make_train_step


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


def _advanced_state(built, optimizer):
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    next_state, _ = make_train_step(built.graph, optimizer)(state, _batch())
    return next_state


def _optimizer(model_state, metadata):
    return build_optimizer(
        OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.01),
        model_state,
        metadata,
    )


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        name="decoder",
        variant="tiny",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        n_kv_heads=1,
        max_seq_len=4,
        compute_dtype="float32",
    )


def _batch(*, offset: int = 0) -> Batch:
    input_ids = (jnp.arange(8, dtype=jnp.int32).reshape(2, 4) + offset) % 16
    target_ids = (input_ids + 1) % 16
    return Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=jnp.ones((2, 4), dtype=jnp.bool_))


def _dataset_state(*, token_offset: int, next_record_index: int) -> DataPipelineState:
    return DataPipelineState(
        schema_version=1,
        backend="grain",
        backend_version="0.2.16",
        split="train",
        order="sequential",
        worker_count=0,
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
        jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )
