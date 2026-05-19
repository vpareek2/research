"""Orbax-backed local checkpoint service."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from jaxtitan.errors import ContractError
from jaxtitan.state import DatasetState, HostState, TrainState


@dataclass(frozen=True, slots=True)
class CheckpointRestore:
    """Restored checkpoint payload."""

    train_state: TrainState
    dataset_state: DatasetState
    host_state: HostState
    metadata: dict[str, Any]
    step: int
    path: Path


class CheckpointService(Protocol):
    """Host-side checkpoint service protocol."""

    def save(
        self,
        step: int,
        train_state: TrainState,
        dataset_state: DatasetState,
        host_state: HostState,
        metadata: Mapping[str, Any],
    ) -> None: ...

    def restore_latest(self, template_train_state: TrainState) -> CheckpointRestore: ...

    def restore_latest_metadata(self) -> dict[str, Any]: ...

    def set_protected_steps(self, steps: set[int]) -> None: ...

    def latest_step(self) -> int | None: ...

    def latest_path(self) -> Path | None: ...

    def close(self) -> None: ...


class LocalOrbaxCheckpointService:
    """Concrete local checkpoint service using Orbax CheckpointManager."""

    def __init__(self, run_dir: str | Path, *, max_to_keep: int = 2, protected_steps: set[int] | None = None) -> None:
        if max_to_keep <= 0:
            raise ContractError(f"max_to_keep must be positive, got {max_to_keep}")
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self._protected_steps = set() if protected_steps is None else set(protected_steps)
        import orbax.checkpoint as ocp

        self._ocp = ocp
        self._manager = ocp.CheckpointManager(
            self.checkpoints_dir.resolve(),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                step_format_fixed_length=6,
                create=True,
                should_keep_fn=self._should_keep,
            ),
        )

    def save(
        self,
        step: int,
        train_state: TrainState,
        dataset_state: DatasetState,
        host_state: HostState,
        metadata: Mapping[str, Any],
    ) -> None:
        """Save one complete Jaxtitan checkpoint."""

        if step < 0:
            raise ContractError(f"checkpoint step must be non-negative, got {step}")
        if host_state.dataset != dataset_state:
            raise ContractError("host_state.dataset must match dataset_state when saving a checkpoint")
        self._manager.save(
            step,
            args=self._ocp.args.Composite(
                train=self._ocp.args.StandardSave(train_state),
                dataset=self._ocp.args.JsonSave(_dataset_to_dict(dataset_state)),
                host=self._ocp.args.JsonSave(_host_to_dict(host_state)),
                metadata=self._ocp.args.JsonSave(dict(metadata)),
            ),
        )
        self._manager.wait_until_finished()

    def restore_latest(self, template_train_state: TrainState) -> CheckpointRestore:
        """Restore the newest checkpoint using a template TrainState tree."""

        step = self.latest_step()
        if step is None:
            raise ContractError(f"no checkpoints found in {self.checkpoints_dir}")
        restored = self._manager.restore(
            step,
            args=self._ocp.args.Composite(
                train=self._ocp.args.StandardRestore(template_train_state),
                dataset=self._ocp.args.JsonRestore(),
                host=self._ocp.args.JsonRestore(),
                metadata=self._ocp.args.JsonRestore(),
            ),
        )
        dataset_state = _dataset_from_mapping(_require_mapping(restored["dataset"], "dataset"))
        host_state = _host_from_mapping(_require_mapping(restored["host"], "host"))
        if host_state.dataset != dataset_state:
            raise ContractError("restored host_state.dataset does not match restored dataset_state")
        return CheckpointRestore(
            train_state=restored["train"],
            dataset_state=dataset_state,
            host_state=host_state,
            metadata=dict(_require_mapping(restored["metadata"], "metadata")),
            step=step,
            path=self.latest_path() or self.checkpoints_dir / f"{step:06d}",
        )

    def restore_latest_metadata(self) -> dict[str, Any]:
        """Restore only metadata from the newest checkpoint."""

        step = self.latest_step()
        if step is None:
            raise ContractError(f"no checkpoints found in {self.checkpoints_dir}")
        restored = self._manager.restore(
            step,
            args=self._ocp.args.Composite(metadata=self._ocp.args.JsonRestore()),
        )
        return dict(_require_mapping(restored["metadata"], "metadata"))

    def latest_step(self) -> int | None:
        """Return the latest checkpoint step if one exists."""

        step = self._manager.latest_step()
        return None if step is None else int(step)

    def latest_path(self) -> Path | None:
        """Return the latest checkpoint path if one exists."""

        step = self.latest_step()
        return None if step is None else self.checkpoints_dir / f"{step:06d}"

    def set_protected_steps(self, steps: set[int]) -> None:
        """Protect specific checkpoint steps from max_to_keep deletion."""

        self._protected_steps = set(steps)

    def close(self) -> None:
        """Close the underlying Orbax manager."""

        self._manager.close()

    def _should_keep(self, step: int) -> bool:
        return int(step) in self._protected_steps


def _dataset_to_dict(state: DatasetState) -> dict[str, Any]:
    return asdict(state)


def _host_to_dict(state: HostState) -> dict[str, Any]:
    return asdict(state)


def _dataset_from_mapping(raw: Mapping[str, Any]) -> DatasetState:
    return DatasetState(
        shard_index=_required_int(raw, "shard_index", "dataset"),
        token_offset=_required_int(raw, "token_offset", "dataset"),
        epoch=_required_int(raw, "epoch", "dataset"),
        shuffle_state=_optional_int(raw, "shuffle_state", "dataset"),
    )


def _host_from_mapping(raw: Mapping[str, Any]) -> HostState:
    dataset_raw = _require_mapping(raw.get("dataset"), "host.dataset")
    return HostState(
        dataset=_dataset_from_mapping(dataset_raw),
        last_checkpoint_step=_required_int(raw, "last_checkpoint_step", "host"),
        wallclock_start_ns=_required_int(raw, "wallclock_start_ns", "host"),
        run_id=_required_str(raw, "run_id", "host"),
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"checkpoint {name} must be a JSON object")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ContractError(f"checkpoint {name}.{key} must be an integer")
    return value


def _optional_int(raw: Mapping[str, Any], key: str, name: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ContractError(f"checkpoint {name}.{key} must be an integer or null")
    return value


def _required_str(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"checkpoint {name}.{key} must be a non-empty string")
    return value
