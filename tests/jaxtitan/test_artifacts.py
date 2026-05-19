from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from jaxtitan.config import load_config, resolved_config_sha256, run_spec_to_json
from jaxtitan.errors import ContractError
from jaxtitan.services import initialize_run

MINIMAL_CONFIG = """
[run]
id = "smoke"
seed = 11
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 32000
hidden_size = 128
num_layers = 2
num_heads = 4
max_seq_len = 64

[optimizer]
name = "adamw"
weight_decay = 0.1

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
train_manifest = "data/train/manifest.json"
tokenizer_id = "toy-tokenizer"

[training]
seq_len = 64
global_batch_size = 2
target_tokens = 128
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]
"""


def test_initialize_run_creates_canonical_artifact_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    manifest = initialize_run(config_path)
    run_dir = tmp_path / "runs" / "smoke"

    assert manifest.run_dir == Path("runs/smoke")
    assert run_dir.is_dir()
    for relative in (
        "config",
        "metrics",
        "checkpoints",
        "evals",
        "samples",
        "summaries",
    ):
        assert (run_dir / relative).is_dir()

    assert (run_dir / "config" / "source.toml").read_text() == MINIMAL_CONFIG
    resolved = json.loads((run_dir / "config" / "resolved.json").read_text())
    assert resolved["run_id"] == "smoke"
    assert resolved["model"]["hidden_size"] == 128


def test_initialize_run_writes_manifest_and_initial_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    spec = load_config(config_path)

    initialize_run(config_path)
    run_dir = tmp_path / "runs" / "smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]

    assert manifest["schema_version"] == 1
    assert manifest["artifact_layout_version"] == 1
    assert manifest["run_id"] == "smoke"
    assert "run_dir" not in manifest
    assert manifest["source_config_path"] == config_path.as_posix()
    assert manifest["source_config_sha256"] == sha256(MINIMAL_CONFIG.encode("utf-8")).hexdigest()
    assert manifest["resolved_config_sha256"] == resolved_config_sha256(spec)
    assert manifest["package"] == {"name": "jaxtitan", "version": "0.1.0"}
    assert manifest["directories"]["checkpoints"] == "checkpoints"

    assert len(events) == 1
    assert events[0]["type"] == "run_initialized"
    assert events[0]["run_id"] == "smoke"
    assert events[0]["source_config_sha256"] == manifest["source_config_sha256"]
    assert events[0]["resolved_config_sha256"] == manifest["resolved_config_sha256"]


def test_initialize_run_refuses_to_overwrite_existing_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    (tmp_path / "runs" / "smoke").mkdir(parents=True)

    with pytest.raises(ContractError, match="already exists"):
        initialize_run(config_path)

    assert not any(path.name.startswith(".smoke.tmp-") for path in (tmp_path / "runs").iterdir())


def test_initialize_run_does_not_leave_final_dir_for_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG.replace("max_seq_len = 64", "max_seq_len = 32"))

    with pytest.raises(Exception):
        initialize_run(config_path)

    assert not (tmp_path / "runs" / "smoke").exists()
    assert not (tmp_path / "runs").exists()


def test_resolved_config_artifact_matches_canonical_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    spec = load_config(config_path)

    initialize_run(config_path)

    assert (tmp_path / "runs" / "smoke" / "config" / "resolved.json").read_text() == run_spec_to_json(spec) + "\n"
