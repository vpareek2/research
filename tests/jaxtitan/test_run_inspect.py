import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training
from jaxtitan.runtime.inspect import format_run_inspection, inspect_run, run_inspection_to_json


def test_inspect_run_reports_summary_checkpoints_and_recent_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("inspect", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)

    inspection = inspect_run(tmp_path / "runs" / "loop")
    payload = inspection.payload
    text = format_run_inspection(inspection)

    assert payload["run_id"] == "loop"
    assert payload["status"] == "completed"
    assert payload["final"]["step"] == 2
    assert payload["latest_checkpoint"]["checkpoint_path"] == "checkpoints/000002"
    assert payload["best_checkpoint"]["eval_loss"] is not None
    assert payload["checkpoints"][0]["retained"] is True
    assert payload["diagnostics"]["path"] == "diagnostics/runtime.json"
    assert payload["diagnostics"]["jax_backend"]
    assert payload["diagnostics"]["flops_per_token"] > 0
    assert payload["diagnostics"]["parallelism"]["execution_mode"] == "replicated_data_parallel"
    assert payload["diagnostics"]["parallelism"]["metrics_scope"] == "global"
    assert payload["diagnostics"]["parallelism"]["artifact_writer"] == "single_host"
    assert payload["diagnostics"]["parallelism"]["mesh"]["data_axis_size"] == 1
    assert payload["diagnostics"]["parallelism"]["batch"]["global_batch_size"] == 2
    assert payload["diagnostics"]["parallelism"]["batch"]["per_device_batch_size"] == 2
    assert payload["diagnostics"]["sharding"]["batch"]["input_ids"]["partition_spec"] == "PartitionSpec('data', None)"
    assert payload["diagnostics"]["data_pipeline"]["backend"] == "grain"
    assert payload["diagnostics"]["data_pipeline"]["order"] == "sequential"
    assert payload["diagnostics"]["data_pipeline"]["worker_buffer_size"] == 1
    assert payload["diagnostics"]["data_pipeline"]["document_aware"] is False
    assert payload["diagnostics"]["data_pipeline"]["document_count"] is None
    assert payload["profiling"]["enabled"] is False
    assert payload["profiling"]["status"] == "disabled"
    assert payload["profiling"]["trace_dir"] == "profiles"
    assert payload["kernels"]["enabled"] is False
    assert payload["kernels"]["active_count"] == 0
    assert payload["kernels"]["fallback"]["rmsnorm"] == "kernels_disabled"
    assert payload["router_health"] is None
    assert payload["optimizer_health"]["group_count"] > 0
    optimizer_leaf_count = sum(group["leaf_count"] for group in payload["recent_train_metrics"][-1]["optimizer_groups"])
    assert payload["optimizer_health"]["route_backend_counts"] == {"adamw": optimizer_leaf_count}
    assert payload["recent_train_metrics"][-1]["step"] == 2
    assert payload["recent_eval_metrics"][-1]["eval_name"] == "validation"
    assert "run: loop" in text
    assert "runtime:" in text
    assert "parallelism:" in text
    assert "data pipeline:" in text
    assert "documents=False" in text
    assert "mode=replicated_data_parallel" in text
    assert "artifacts=single_host" in text
    assert "optimizer health:" in text
    assert "kernels: enabled=False strict=False active=0" in text
    assert "profiling:" not in text
    assert "best checkpoint:" in text
    assert json.loads(run_inspection_to_json(inspection))["run_id"] == "loop"


def test_inspect_run_reports_document_pipeline_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "inspect-documents",
        shard_token_groups=(tuple(range(0, 50)),),
        train_tokens=25,
        document_offsets=(0, 6, 12, 25, 50),
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            data_order="document_buffer",
            shuffle_seed=123,
            document_buffer_size=3,
            document_refill_size=2,
            target_tokens=12,
        )
    )
    run_training(config_path)

    inspection = inspect_run(tmp_path / "runs" / "loop")
    payload = inspection.payload
    text = format_run_inspection(inspection)

    assert payload["diagnostics"]["data_pipeline"]["document_aware"] is True
    assert payload["diagnostics"]["data_pipeline"]["document_count"] == 4
    assert payload["diagnostics"]["data_pipeline"]["document_offsets_path"] == "document_offsets.u64"
    assert payload["diagnostics"]["data_pipeline"]["order"] == "document_buffer"
    assert payload["diagnostics"]["data_pipeline"]["document_buffer_size"] == 3
    assert payload["recent_train_metrics"][-1]["document_aware"] is True
    assert "documents=True count=4" in text


def test_inspect_run_reports_moe_router_and_optimizer_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("inspect-moe", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            model_name="trinity",
            num_layers=2,
            trinity_moe_balance_name="none",
        )
    )
    run_training(config_path)

    inspection = inspect_run(tmp_path / "runs" / "loop")
    payload = inspection.payload
    text = format_run_inspection(inspection)

    assert payload["router_health"]["layer_count"] == 1
    assert payload["router_health"]["worst_layer"]["layer_index"] == 0
    assert payload["optimizer_health"]["group_count"] > 0
    assert any(group["tag"] == "moe_router" for group in payload["recent_train_metrics"][-1]["optimizer_groups"])
    assert "router health:" in text
    assert "optimizer health:" in text


def test_inspect_run_reports_wandb_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("inspect-wandb", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)
    wandb_metadata = {
        "schema_version": 1,
        "enabled": True,
        "wandb_run_id": "loop-abc",
        "project": "jaxtitan-test",
        "entity": "team",
        "group": "unit",
        "tags": ["inspect"],
        "mode": "offline",
        "url": "https://wandb.test/loop-abc",
        "name": "run-loop-abc",
        "resume": "allow",
    }
    run_dir = tmp_path / "runs" / "loop"
    (run_dir / "diagnostics" / "wandb.json").write_text(json.dumps(wandb_metadata))
    runtime_path = run_dir / "diagnostics" / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["wandb"] = wandb_metadata
    runtime_path.write_text(json.dumps(runtime))

    inspection = inspect_run(run_dir)
    payload = inspection.payload
    text = format_run_inspection(inspection)

    assert payload["wandb"]["wandb_run_id"] == "loop-abc"
    assert payload["diagnostics"]["wandb"]["project"] == "jaxtitan-test"
    assert "wandb: project=jaxtitan-test entity=team mode=offline id=loop-abc" in text
    assert json.loads(run_inspection_to_json(inspection))["wandb"]["mode"] == "offline"


def test_inspect_run_reports_profiling_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("inspect-profiling", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)
    profiling = {
        "schema_version": 1,
        "enabled": True,
        "status": "completed",
        "trace_dir": "profiles",
        "trace_start_step": 3,
        "trace_steps": 2,
        "trace_end_step": 4,
        "traced_step_range": {"start": 3, "end": 4},
        "create_perfetto_trace": True,
        "create_perfetto_link": False,
        "trace_files": ["profiles/trace.trace.json.gz"],
        "started_at": "2026-05-22T00:00:00Z",
        "stopped_at": "2026-05-22T00:00:01Z",
        "error": None,
    }
    run_dir = tmp_path / "runs" / "loop"
    (run_dir / "diagnostics" / "profiling.json").write_text(json.dumps(profiling))

    inspection = inspect_run(run_dir)
    payload = inspection.payload
    text = format_run_inspection(inspection)

    assert payload["profiling"]["status"] == "completed"
    assert payload["profiling"]["trace_step_range"] == {"start": 3, "end": 4}
    assert payload["profiling"]["perfetto_trace_available"] is True
    assert "profiling: status=completed range={'start': 3, 'end': 4} dir=profiles perfetto=yes" in text
    assert json.loads(run_inspection_to_json(inspection))["profiling"]["status"] == "completed"


def test_inspect_run_missing_required_artifact_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("missing-index", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)
    (tmp_path / "runs" / "loop" / "checkpoints" / "index.json").unlink()

    with pytest.raises(ContractError, match="missing checkpoint index"):
        inspect_run(tmp_path / "runs" / "loop")


def test_inspect_run_missing_retained_checkpoint_path_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("stale-index", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)
    checkpoint_dir = tmp_path / "runs" / "loop" / "checkpoints" / "000002"
    for path in sorted(checkpoint_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    checkpoint_dir.rmdir()

    with pytest.raises(ContractError, match="retained path does not exist"):
        inspect_run(tmp_path / "runs" / "loop")


def test_cli_run_inspect_json_does_not_import_jax(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fake"
    (run_dir / "summaries").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "fake"}))
    (run_dir / "summaries" / "final.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "steps": 0,
                "tokens_seen": 0,
                "target_tokens": 0,
                "final_loss": None,
                "final_eval_loss": None,
                "final_eval_token_count": None,
                "final_eval_num_batches": None,
            }
        )
    )
    (run_dir / "checkpoints" / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "latest_step": None,
                "latest_checkpoint_path": None,
                "best_eval_step": None,
                "best_eval_loss": None,
                "best_checkpoint_path": None,
                "records": [],
            }
        )
    )
    script = (
        "import sys\n"
        "from jaxtitan.cli import main\n"
        f"code = main(['run', 'inspect', {str(run_dir)!r}, '--json'])\n"
        "print('JAX_LOADED', 'jax' in sys.modules)\n"
        "raise SystemExit(code)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0
    assert '"run_id":"fake"' in result.stdout
    assert '"diagnostics":null' in result.stdout
    assert "JAX_LOADED False" in result.stdout


def _training_config(
    train_manifest: Path,
    *,
    data_order: str = "sequential",
    shuffle_seed: int | None = None,
    document_buffer_size: int | None = None,
    document_refill_size: int | None = None,
    target_tokens: int = 16,
    model_name: str = "decoder",
    num_layers: int = 1,
    trinity_moe_balance_name: str | None = None,
) -> str:
    shuffle_seed_line = "" if shuffle_seed is None else f"shuffle_seed = {shuffle_seed}\n"
    document_buffer_size_line = "" if document_buffer_size is None else f"document_buffer_size = {document_buffer_size}\n"
    document_refill_size_line = "" if document_refill_size is None else f"document_refill_size = {document_refill_size}\n"
    trinity_block = ""
    if model_name == "trinity":
        balance_name = "none" if trinity_moe_balance_name is None else trinity_moe_balance_name
        trinity_block = f"""
[model.trinity]
initial_dense_layers = 1
local_window = 4
local_layers_per_global = 1

[model.trinity.moe]
num_experts = 3
top_k = 2

[model.trinity.moe.balance]
name = "{balance_name}"
"""
    return f"""
[run]
id = "loop"
seed = 7
output_dir = "runs"

[model]
name = "{model_name}"
variant = "tiny"
vocab_size = 64
hidden_size = 8
intermediate_size = 16
num_layers = {num_layers}
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"
{trinity_block}

[optimizer]
name = "adamw"
weight_decay = 0.0

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "toy-tokenizer"
order = "{data_order}"
{shuffle_seed_line}{document_buffer_size_line}{document_refill_size_line}

[training]
seq_len = 4
global_batch_size = 2
target_tokens = {target_tokens}
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]

[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""
