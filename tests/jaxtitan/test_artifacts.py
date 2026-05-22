from hashlib import sha256
import json
from pathlib import Path

import pytest

from jaxtitan.config import load_config, resolved_config_sha256, run_spec_to_json
from jaxtitan.errors import ContractError
from jaxtitan.services import initialize_run
from jaxtitan.services.wandb import final_tables_for_wandb, normalize_metrics_for_wandb, wandb_run_id_for_manifest


def test_initialize_run_creates_canonical_artifact_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config)

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

    assert (run_dir / "config" / "source.toml").read_text() == minimal_config
    resolved = json.loads((run_dir / "config" / "resolved.json").read_text())
    assert resolved["run_id"] == "smoke"
    assert resolved["model"]["hidden_size"] == 128


def test_initialize_run_writes_manifest_and_initial_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config: str, prepared_dataset: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config)
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
    assert manifest["source_config_sha256"] == sha256(minimal_config.encode("utf-8")).hexdigest()
    assert manifest["resolved_config_sha256"] == resolved_config_sha256(spec)
    assert manifest["package"] == {"name": "jaxtitan", "version": "0.1.0"}
    assert manifest["directories"]["checkpoints"] == "checkpoints"
    assert manifest["data"]["manifest_path"] == prepared_dataset.as_posix()
    assert manifest["data"]["tokenizer_id"] == "toy-tokenizer"
    assert manifest["data"]["total_tokens"] == 8
    assert manifest["data"]["train_tokens"] == 6
    assert manifest["data"]["val_tokens"] == 2
    assert manifest["data"]["shard_count"] == 2
    assert manifest["data"]["token_bytes_path"] == "token_bytes.bin"

    assert len(events) == 1
    assert events[0]["type"] == "run_initialized"
    assert events[0]["run_id"] == "smoke"
    assert events[0]["source_config_sha256"] == manifest["source_config_sha256"]
    assert events[0]["resolved_config_sha256"] == manifest["resolved_config_sha256"]


def test_initialize_run_refuses_to_overwrite_existing_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config)
    (tmp_path / "runs" / "smoke").mkdir(parents=True)

    with pytest.raises(ContractError, match="already exists"):
        initialize_run(config_path)

    assert not any(path.name.startswith(".smoke.tmp-") for path in (tmp_path / "runs").iterdir())


def test_initialize_run_does_not_leave_final_dir_for_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "bad.toml"
    config_path.write_text(minimal_config.replace("max_seq_len = 64", "max_seq_len = 32"))

    with pytest.raises(Exception):
        initialize_run(config_path)

    assert not (tmp_path / "runs" / "smoke").exists()
    assert not (tmp_path / "runs").exists()


def test_initialize_run_does_not_leave_final_dir_for_invalid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config_builder
) -> None:
    monkeypatch.chdir(tmp_path)
    missing_manifest = tmp_path / "data" / "train" / "manifest.json"
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config_builder(missing_manifest))

    with pytest.raises(ContractError, match="manifest does not exist"):
        initialize_run(config_path)

    assert not (tmp_path / "runs" / "smoke").exists()
    assert not (tmp_path / "runs").exists()


def test_resolved_config_artifact_matches_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config)
    spec = load_config(config_path)

    initialize_run(config_path)

    assert (tmp_path / "runs" / "smoke" / "config" / "resolved.json").read_text() == run_spec_to_json(spec) + "\n"


def test_wandb_metric_normalization_namespaces_scalars_and_skips_nested_fields() -> None:
    row = {
        "step": 3,
        "loss": 1.25,
        "tokens_seen": 128,
        "tokens_per_sec": 1000.0,
        "router_mean_load_cv": 0.5,
        "optimizer_groups_with_zero_grad": 1,
        "optimizer_groups": [{"group": "attention_q:muon"}],
        "moe_router_layers": [{"layer_index": 0}],
        "status": "completed",
        "nan": float("nan"),
    }

    payload = normalize_metrics_for_wandb(row, default_namespace="train")

    assert payload == {
        "train/loss": 1.25,
        "data/tokens_seen": 128,
        "perf/tokens_per_sec": 1000.0,
        "router/router_mean_load_cv": 0.5,
        "optimizer/optimizer_groups_with_zero_grad": 1,
    }


def test_wandb_final_tables_use_compact_nested_summaries() -> None:
    fake_wandb = _FakeWandbModule()
    summary = {
        "final_optimizer_groups": [{"group": "lm_head:adamw", "grad_norm": 1.0}],
        "final_moe_router_layers": [{"layer_index": 0, "load_cv": 2.0}],
    }

    tables = final_tables_for_wandb(summary, fake_wandb)

    assert set(tables) == {"final/optimizer_groups", "final/moe_router_layers"}
    assert tables["final/optimizer_groups"].columns == ["grad_norm", "group"]
    assert tables["final/optimizer_groups"].data == [[1.0, "lm_head:adamw"]]
    assert tables["final/moe_router_layers"].columns == ["layer_index", "load_cv"]
    assert tables["final/moe_router_layers"].data == [[0, 2.0]]


def test_wandb_run_id_is_deterministic_and_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_config: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config.replace('id = "smoke"', 'id = "smoke run"'))

    manifest = initialize_run(config_path)

    assert wandb_run_id_for_manifest(manifest) == wandb_run_id_for_manifest(manifest)
    assert wandb_run_id_for_manifest(manifest).startswith("smoke-run-")


class _FakeWandbModule:
    class Table:
        def __init__(self, *, columns, data) -> None:
            self.columns = columns
            self.data = data
