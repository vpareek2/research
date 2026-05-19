from dataclasses import FrozenInstanceError
from pathlib import Path

import jax
import pytest

from jaxtitan.errors import ContractError
from jaxtitan.metrics import EvalMetrics, StepMetrics
from jaxtitan.specs import (
    ArtifactSpec,
    DataSpec,
    MeshSpec,
    ModelSpec,
    OptimizerSpec,
    RunSpec,
    ScheduleSpec,
    TrainingSpec,
)
from jaxtitan.state import DatasetState, HostState, RngState, TrainState


def test_run_spec_is_constructible_and_frozen() -> None:
    spec = RunSpec(
        run_id="smoke",
        seed=7,
        output_dir=Path("runs"),
        model=ModelSpec(
            name="decoder",
            variant="tiny",
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=512,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        ),
        optimizer=OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1e-3)),
        data=DataSpec(train_manifest=Path("data/train/manifest.json"), tokenizer_id="tok"),
        mesh=MeshSpec(axis_names=("data",), axis_sizes=(1,)),
        training=TrainingSpec(seq_len=64, global_batch_size=2, target_tokens=128),
        artifacts=ArtifactSpec(root=Path("runs"), wandb_enabled=False),
    )

    assert spec.model.n_kv_heads == 4
    assert spec.dirs.run_dir == Path("runs/smoke")
    with pytest.raises(FrozenInstanceError):
        spec.run_id = "changed"  # type: ignore[misc]


def test_contracts_reject_invalid_shapes() -> None:
    with pytest.raises(ContractError, match="hidden_size"):
        ModelSpec(
            name="decoder",
            variant="bad",
            vocab_size=100,
            hidden_size=130,
            intermediate_size=512,
            num_layers=2,
            num_heads=8,
            max_seq_len=64,
        )

    with pytest.raises(ContractError, match="unique"):
        MeshSpec(axis_names=("data", "data"), axis_sizes=(1, 1))


def test_state_and_metrics_contracts_are_constructible() -> None:
    rng = RngState(train="train-key", data="data-key", eval="eval-key", sample="sample-key")
    train_state = TrainState(step=0, tokens_seen=0, model={"params": {}}, opt_state={}, rng=rng)
    dataset_state = DatasetState(shard_index=0, token_offset=0, epoch=0)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=0, wallclock_start_ns=123, run_id="smoke")

    assert train_state.rng.train == "train-key"
    assert host_state.dataset == dataset_state
    assert StepMetrics(loss_sum=2.0, token_count=4, lr=1e-3).token_count == 4
    assert EvalMetrics(loss_sum=3.0, token_count=6, num_batches=2).num_batches == 2


def test_device_state_is_pytree_and_host_state_is_not() -> None:
    rng = RngState(train=1, data=2, eval=3, sample=4)
    train_state = TrainState(step=0, tokens_seen=0, model={"weight": 5}, opt_state={"momentum": 6}, rng=rng)
    dataset_state = DatasetState(shard_index=0, token_offset=0, epoch=0)
    host_state = HostState(dataset=dataset_state, last_checkpoint_step=0, wallclock_start_ns=123, run_id="smoke")

    assert jax.tree.leaves(rng) == [1, 2, 3, 4]
    assert jax.tree.leaves(train_state) == [0, 0, 5, 6, 1, 2, 3, 4]
    assert jax.tree.leaves(dataset_state) == [dataset_state]
    assert jax.tree.leaves(host_state) == [host_state]
