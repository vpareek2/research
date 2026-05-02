"""
Checkpoint save/restore helpers.
"""

from pathlib import Path
from typing import Iterator

from flax import nnx
import grain.python as grain_py
import orbax.checkpoint as ocp

from research.data import Batch
from research.model import Model


def create_checkpoint_manager(run_dir: Path, keep_last: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        (run_dir / "checkpoints").resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep_last,
            step_format_fixed_length=6,
        ),
    )


def save_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    next_step: int,
    model: Model,
    optimizer: nnx.Optimizer,
    train_iter: Iterator[Batch],
):
    manager.save(
        next_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            metadata=ocp.args.JsonSave({"next_step": next_step}),
            train_iter=grain_py.PyGrainCheckpointSave(train_iter),
        ),
    )


def restore_latest_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    model: Model,
    optimizer: nnx.Optimizer,
    train_iter: Iterator[Batch],
) -> int:
    latest_step = manager.latest_step()
    if latest_step is None:
        raise FileNotFoundError("No checkpoint found to resume from.")

    restored = manager.restore(
        latest_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model)),
            optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
            metadata=ocp.args.JsonRestore(),
            train_iter=grain_py.PyGrainCheckpointRestore(train_iter),
        ),
    )
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    return int(restored["metadata"]["next_step"])


def restore_model_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    model: Model,
    step: int | None = None,
) -> int:
    checkpoint_step = manager.latest_step() if step is None else step
    if checkpoint_step is None:
        raise FileNotFoundError("No checkpoint found to evaluate.")

    restored = manager.restore(
        checkpoint_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model)),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    nnx.update(model, restored["model"])
    return int(restored["metadata"].get("next_step", checkpoint_step))
