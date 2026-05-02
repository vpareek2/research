from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.preflight import PreflightError, data_artifact_status, eval_artifact_status, prepare_missing_artifacts, run_preflight


def write_run_config(
    path: Path,
    data_dir: Path,
    eval_root: Path,
    data_prepare: Path,
    eval_prepare: Path,
    *,
    eval_steps: int = 1,
    domain_eval_steps: int = 1,
):
    path.write_text(
        f"""
[experiment]
name = "unit"
out_dir = "{path.parent / 'runs'}"

[target]
tokens = 32

[model]
vocab_size = 128
hidden_size = 32
intermediate_size = 64
n_layers = 1
n_heads = 4
n_kv_heads = 1
seq_len = 8
theta = 10000.0
eps = 0.000001
tied = false

[distributed]
enabled = false
device_count = "auto"
axis_name = "data"

[train]
seed = 0
batch_size = 2
seq_len = 8
steps = 2
lr = 0.001
decay = 0.1
log_every = 1
eval_every = 1
eval_steps = {eval_steps}
checkpoint_every = 2
keep_last = 2

[data]
source = "tokens"
path = "{data_dir}"
tokenizer = "gpt2"
prepare_config = "{data_prepare}"

[eval]
domain_root = "{eval_root}"
domain_eval_steps = {domain_eval_steps}
prepare_config = "{eval_prepare}"

[wandb]
enabled = false
""",
        encoding="utf-8",
    )


def write_data_prepare_config(path: Path, output: Path, *, source_type: str = "text", max_tokens: int | None = None):
    source_path = path.parent / "source.txt"
    source_path.write_text("hello\n", encoding="utf-8")
    if source_type == "hf":
        source_block = """
[source]
type = "hf"
dataset = "fake/dataset"
split = "train"
text_column = "text"
streaming = true
"""
    else:
        source_block = f"""
[source]
type = "text"
path = "{source_path}"
"""
    max_tokens_line = f"max_tokens = {max_tokens}\n" if max_tokens is not None else ""
    path.write_text(
        f"""
{source_block}
[tokenizer]
name = "gpt2"
append_eot = true

[output]
path = "{output}"
dtype = "uint32"
val_fraction = 0.1
{max_tokens_line}
""",
        encoding="utf-8",
    )


def write_eval_prepare_config(path: Path, output: Path, *, tokens_per_domain: int = 16):
    lines = [
        'kind = "eval_domains"',
        "",
        "[tokenizer]",
        'name = "gpt2"',
        "append_eot = true",
        "",
        "[output]",
        f'path = "{output}"',
        'dtype = "uint32"',
        f"tokens_per_domain = {tokens_per_domain}",
        "",
    ]
    for name in ("web", "knowledge", "books", "news", "code", "math", "reasoning", "docs", "dialogue"):
        source_path = path.parent / f"{name}.txt"
        source_path.write_text(f"{name}\n", encoding="utf-8")
        lines.extend(
            [
                "[[domain]]",
                f'name = "{name}"',
                'source.type = "text"',
                f'source.path = "{source_path}"',
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def completed(returncode=0):
    return SimpleNamespace(returncode=returncode)


def fake_runner(*args, **kwargs):
    return completed(0)


def make_configs(tmp_path):
    data_dir = tmp_path / "prepared"
    eval_root = tmp_path / "eval_domains"
    data_prepare = tmp_path / "data_prep.toml"
    eval_prepare = tmp_path / "eval_prep.toml"
    config_path = tmp_path / "run.toml"
    write_data_prepare_config(data_prepare, data_dir)
    write_eval_prepare_config(eval_prepare, eval_root)
    write_run_config(config_path, data_dir, eval_root, data_prepare, eval_prepare)
    return config_path, data_dir, eval_root


def test_preflight_rejects_uncapped_hf_data_prepare(tmp_path, monkeypatch):
    data_dir = tmp_path / "prepared"
    eval_root = tmp_path / "eval_domains"
    data_prepare = tmp_path / "data_prep.toml"
    eval_prepare = tmp_path / "eval_prep.toml"
    config_path = tmp_path / "run.toml"
    write_data_prepare_config(data_prepare, data_dir, source_type="hf")
    write_eval_prepare_config(eval_prepare, eval_root)
    write_run_config(config_path, data_dir, eval_root, data_prepare, eval_prepare)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="output.max_tokens"):
        run_preflight(config_path, interactive=False, runner=fake_runner)


def test_preflight_rejects_hf_data_cap_below_target_train_tokens(tmp_path, monkeypatch):
    data_dir = tmp_path / "prepared"
    eval_root = tmp_path / "eval_domains"
    data_prepare = tmp_path / "data_prep.toml"
    eval_prepare = tmp_path / "eval_prep.toml"
    config_path = tmp_path / "run.toml"
    write_data_prepare_config(data_prepare, data_dir, source_type="hf", max_tokens=16)
    write_eval_prepare_config(eval_prepare, eval_root)
    write_run_config(config_path, data_dir, eval_root, data_prepare, eval_prepare)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="below target.tokens"):
        run_preflight(config_path, interactive=False, runner=fake_runner)


def test_preflight_accepts_hf_data_cap_above_target_train_tokens(tmp_path, monkeypatch):
    data_dir = tmp_path / "prepared"
    eval_root = tmp_path / "eval_domains"
    data_prepare = tmp_path / "data_prep.toml"
    eval_prepare = tmp_path / "eval_prep.toml"
    config_path = tmp_path / "run.toml"
    write_data_prepare_config(data_prepare, data_dir, source_type="hf", max_tokens=640)
    write_eval_prepare_config(eval_prepare, eval_root)
    write_run_config(config_path, data_dir, eval_root, data_prepare, eval_prepare)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    result = run_preflight(config_path, interactive=False, runner=fake_runner)

    assert result.failures == []


def test_preflight_rejects_train_eval_capacity_below_eval_workload(tmp_path, monkeypatch):
    data_dir = tmp_path / "prepared"
    eval_root = tmp_path / "eval_domains"
    data_prepare = tmp_path / "data_prep.toml"
    eval_prepare = tmp_path / "eval_prep.toml"
    config_path = tmp_path / "run.toml"
    write_data_prepare_config(data_prepare, data_dir, source_type="hf", max_tokens=64)
    write_eval_prepare_config(eval_prepare, eval_root)
    write_run_config(config_path, data_dir, eval_root, data_prepare, eval_prepare, eval_steps=3)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="train eval capacity"):
        run_preflight(config_path, interactive=False, runner=fake_runner)


def test_preflight_rejects_domain_eval_capacity_below_eval_workload(tmp_path, monkeypatch):
    data_dir = tmp_path / "prepared"
    eval_root = tmp_path / "eval_domains"
    data_prepare = tmp_path / "data_prep.toml"
    eval_prepare = tmp_path / "eval_prep.toml"
    config_path = tmp_path / "run.toml"
    write_data_prepare_config(data_prepare, data_dir, source_type="hf", max_tokens=640)
    write_eval_prepare_config(eval_prepare, eval_root, tokens_per_domain=15)
    write_run_config(config_path, data_dir, eval_root, data_prepare, eval_prepare)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="domain eval capacity"):
        run_preflight(config_path, interactive=False, runner=fake_runner)


def test_preflight_allows_missing_prepared_artifacts(tmp_path, monkeypatch):
    config_path, data_dir, eval_root = make_configs(tmp_path)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    result = run_preflight(config_path, interactive=False, runner=fake_runner)

    assert result.failures == []
    assert data_artifact_status(result.config).missing
    assert eval_artifact_status(result.config).missing
    assert [artifact.status for artifact in result.artifacts] == ["MISSING", "MISSING"]
    assert result.artifacts[0].path == data_dir
    assert result.artifacts[1].path == eval_root


def test_preflight_fails_invalid_existing_train_data(tmp_path, monkeypatch):
    config_path, data_dir, _ = make_configs(tmp_path)
    data_dir.mkdir()
    np.arange(16, dtype=np.uint32).tofile(data_dir / "tokens.bin")
    (data_dir / "manifest.json").write_text(
        """
{"dtype":"uint32","num_tokens":16,"tokenizer":{"name":"not-gpt2"},"files":{"tokens":{"path":"tokens.bin"}},"splits":{"train":{"start":0,"end":8,"tokens":8},"val":{"start":8,"end":16,"tokens":8}}}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="Prepared token manifest tokenizer"):
        run_preflight(config_path, interactive=False, runner=fake_runner)


def test_preflight_require_ready_fails_missing_artifacts(tmp_path, monkeypatch):
    config_path, _, _ = make_configs(tmp_path)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="required artifact is missing"):
        run_preflight(config_path, interactive=False, require_ready=True, runner=fake_runner)


def test_prepare_missing_artifacts_runs_eval_then_train(tmp_path, monkeypatch):
    config_path, _, _ = make_configs(tmp_path)
    calls = []
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)
    monkeypatch.setattr("research.preflight.prepare_dataset", lambda config: calls.append(config.kind))

    result = run_preflight(config_path, interactive=False, runner=fake_runner)
    prepare_missing_artifacts(result)

    assert calls == ["eval_domains", "dataset"]


def test_preflight_rejects_prepare_output_mismatch(tmp_path, monkeypatch):
    config_path, _, _ = make_configs(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace('path = "' + str(tmp_path / "prepared") + '"', 'path = "' + str(tmp_path / "other") + '"'), encoding="utf-8")
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: name)

    with pytest.raises(PreflightError, match="data.prepare_config"):
        run_preflight(config_path, interactive=False, runner=fake_runner)


def test_preflight_installs_missing_gh_when_interactive(tmp_path, monkeypatch):
    config_path, _, _ = make_configs(tmp_path)
    calls = []
    installed = {"gh": False}

    def fake_which(name):
        if name == "gh":
            return "/usr/bin/gh" if installed["gh"] else None
        if name in {"apt-get", "sudo"}:
            return f"/usr/bin/{name}"
        return name

    def runner(command, **kwargs):
        calls.append(command)
        if command[-3:] == ["install", "-y", "gh"]:
            installed["gh"] = True
        return completed(0)

    monkeypatch.setattr("research.preflight.shutil.which", fake_which)
    monkeypatch.setattr("research.preflight.os.geteuid", lambda: 0)

    result = run_preflight(config_path, interactive=True, runner=runner)

    assert result.failures == []
    assert ["apt-get", "update"] in calls
    assert ["apt-get", "install", "-y", "gh"] in calls
    assert ["gh", "auth", "status"] in calls


def test_preflight_fails_missing_gh_when_noninteractive(tmp_path, monkeypatch):
    config_path, _, _ = make_configs(tmp_path)
    monkeypatch.setattr("research.preflight.shutil.which", lambda name: None if name == "gh" else name)

    with pytest.raises(PreflightError, match="gh CLI is not installed"):
        run_preflight(config_path, interactive=False, runner=fake_runner)
