import json
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime.preflight import format_preflight_report, preflight_report_to_json, run_preflight


def test_run_preflight_validates_full_runtime_path_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("preflight", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, eval_every_steps=1, eval_num_batches=2))

    report = run_preflight(config_path)
    payload = report.payload
    text = format_preflight_report(report)
    decoded = json.loads(preflight_report_to_json(report))

    assert payload["status"] == "passed"
    assert payload["run_id"] == "loop"
    assert payload["run_dir"] == "runs/loop"
    assert payload["data"]["train_split_tokens"] == 25
    assert payload["data"]["first_batch"]["target_tokens"] == 8
    assert payload["devices"]["selected_device_count"] == 1
    assert payload["mesh"]["axis_names"] == ["data"]
    assert payload["model"]["parameters"] > 0
    assert payload["optimizer"]["name"] == "adamw"
    assert payload["training"]["estimated_steps"] == 2
    assert payload["training"]["compile"] == "passed"
    assert payload["eval"]["name"] == "validation"
    assert payload["eval"]["num_batches"] == 2
    assert payload["eval"]["compile"] == "passed"
    assert decoded == payload
    assert "preflight: passed" in text
    assert "mesh:" in text
    assert "devices:" in text
    assert "compile=passed" in text
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_skips_eval_when_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("no-eval", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))

    report = run_preflight(config_path)

    assert report.payload["eval"] is None
    assert "eval: skipped" in format_preflight_report(report)
    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_resolves_auto_schedule_total_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("cosine", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, schedule_name="cosine", target_tokens=16))

    report = run_preflight(config_path)

    assert report.payload["optimizer"]["schedule"] == "cosine"
    assert report.payload["optimizer"]["total_steps"] == 2
    assert report.payload["training"]["estimated_steps"] == 2


def test_run_preflight_rejects_existing_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("existing", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))
    (tmp_path / "runs" / "loop").mkdir(parents=True)

    with pytest.raises(ContractError, match="run directory already exists"):
        run_preflight(config_path)


@pytest.mark.parametrize(
    ("config_kwargs", "match"),
    [
        ({"tokenizer_id": "wrong-tokenizer"}, "does not match config tokenizer"),
        ({"eval_name": "perplexity", "eval_every_steps": 1}, "validation"),
        ({"second_eval": True, "eval_every_steps": 1}, "exactly one eval"),
        ({"optimizer_name": "muon"}, "no Jaxtitan runtime adapter"),
    ],
)
def test_run_preflight_rejects_invalid_runtime_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
    config_kwargs,
    match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("invalid", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, **config_kwargs))

    with pytest.raises(ContractError, match=match):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_insufficient_local_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("devices", shard_token_groups=(tuple(range(0, 80)),), train_tokens=50)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, axis_sizes=(8,), global_batch_size=8, target_tokens=32))

    with pytest.raises(ContractError, match="only 4 local device"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_missing_train_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(tmp_path / "missing" / "manifest.json"))

    with pytest.raises(ContractError, match="manifest does not exist"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_too_small_train_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("small-train", shard_token_groups=(tuple(range(0, 20)),), train_tokens=8)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, target_tokens=8))

    with pytest.raises(ContractError, match="train split has"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def test_run_preflight_rejects_too_small_validation_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("small-val", shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest, eval_every_steps=1, eval_num_batches=1))

    with pytest.raises(ContractError, match="val split has"):
        run_preflight(config_path)

    assert not (tmp_path / "runs" / "loop").exists()


def _preflight_config(
    train_manifest: Path,
    *,
    target_tokens: int = 16,
    tokenizer_id: str = "toy-tokenizer",
    schedule_name: str = "constant",
    optimizer_name: str = "adamw",
    axis_sizes: tuple[int, ...] = (1,),
    global_batch_size: int = 2,
    eval_every_steps: int | None = None,
    eval_num_batches: int = 1,
    eval_name: str = "validation",
    second_eval: bool = False,
) -> str:
    eval_block = ""
    if eval_every_steps is not None:
        eval_block = f"""
[[evals]]
name = "{eval_name}"
every_steps = {eval_every_steps}
num_batches = {eval_num_batches}
"""
        if second_eval:
            eval_block += """
[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""
    return f"""
[run]
id = "loop"
seed = 7
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 64
hidden_size = 8
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"

[optimizer]
name = "{optimizer_name}"
weight_decay = 0.0

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = 0.001

[data]
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "{tokenizer_id}"

[training]
seq_len = 4
global_batch_size = {global_batch_size}
target_tokens = {target_tokens}
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]
{eval_block}
"""
