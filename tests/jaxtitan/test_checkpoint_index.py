import json
from pathlib import Path

from jaxtitan.runtime.checkpoint_index import (
    CheckpointIndex,
    CheckpointRecord,
    checkpoint_index_to_json,
    load_checkpoint_index,
    record_checkpoint,
)


def test_checkpoint_index_tracks_latest_and_best_eval(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _checkpoint_dir(run_dir, 1)
    _checkpoint_dir(run_dir, 2)
    index = CheckpointIndex()

    index = record_checkpoint(
        index,
        run_dir,
        step=1,
        tokens_seen=8,
        checkpoint_path=run_dir / "checkpoints" / "000001",
        reason="interval",
        train_loss=3.0,
        eval_loss=2.0,
    )
    index = record_checkpoint(
        index,
        run_dir,
        step=2,
        tokens_seen=16,
        checkpoint_path=run_dir / "checkpoints" / "000002",
        reason="final",
        train_loss=2.5,
        eval_loss=1.5,
    )

    assert index.latest_record.step == 2
    assert index.best_record.step == 2
    assert index.to_dict()["latest_checkpoint_path"] == "checkpoints/000002"
    assert index.to_dict()["best_checkpoint_path"] == "checkpoints/000002"


def test_checkpoint_index_ties_keep_earlier_best_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _checkpoint_dir(run_dir, 1)
    _checkpoint_dir(run_dir, 2)
    index = CheckpointIndex(
        records=(
            CheckpointRecord(1, 8, Path("checkpoints/000001"), "interval", train_loss=3.0, eval_loss=1.5),
            CheckpointRecord(2, 16, Path("checkpoints/000002"), "interval", train_loss=2.5, eval_loss=1.5),
        )
    )

    assert index.best_record.step == 1


def test_checkpoint_index_marks_missing_paths_not_retained(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _checkpoint_dir(run_dir, 2)
    raw = {
        "schema_version": 1,
        "latest_step": 2,
        "latest_checkpoint_path": "checkpoints/000002",
        "best_eval_step": 1,
        "best_eval_loss": 1.0,
        "best_checkpoint_path": "checkpoints/000001",
        "records": [
            {
                "step": 1,
                "tokens_seen": 8,
                "checkpoint_path": "checkpoints/000001",
                "reason": "interval",
                "train_loss": 2.0,
                "eval_loss": 1.0,
                "retained": True,
            },
            {
                "step": 2,
                "tokens_seen": 16,
                "checkpoint_path": "checkpoints/000002",
                "reason": "final",
                "train_loss": 1.5,
                "eval_loss": 1.2,
                "retained": True,
            },
        ],
    }
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "index.json").write_text(json.dumps(raw))

    index = load_checkpoint_index(run_dir)

    assert [record.retained for record in index.records] == [False, True]
    assert index.latest_record.step == 2
    assert index.best_record.step == 2


def test_checkpoint_index_json_round_trips_stably(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _checkpoint_dir(run_dir, 1)
    index = record_checkpoint(
        CheckpointIndex(),
        run_dir,
        step=1,
        tokens_seen=8,
        checkpoint_path=run_dir / "checkpoints" / "000001",
        reason="final",
        train_loss=2.0,
        eval_loss=None,
    )
    (run_dir / "checkpoints" / "index.json").write_text(checkpoint_index_to_json(index) + "\n")

    restored = load_checkpoint_index(run_dir)

    assert checkpoint_index_to_json(restored) == checkpoint_index_to_json(index)


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    path = run_dir / "checkpoints" / f"{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    return path
