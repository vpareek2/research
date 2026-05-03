import json
import hashlib
import sys
import types

import numpy as np
import pytest
import tiktoken

from research.data import REQUIRED_EVAL_DOMAINS
from research.prepare_data import (
    DomainConfig,
    HfConfig,
    OutputConfig,
    PrepareConfig,
    SourceConfig,
    TokenizationConfig,
    TokenizerConfig,
    load_texts,
    load_hf_texts,
    load_prepare_config,
    prepare_dataset,
    prepare_eval_domains,
    prepare_texts,
    resolve_hf_token,
)


def test_prepare_texts_writes_bins_manifest_and_eot(tmp_path):
    output_dir = tmp_path / "prepared"
    config = PrepareConfig(
        source=SourceConfig(type="hf", dataset="fake/dataset", split="train", text_column="text"),
        tokenizer=TokenizerConfig(name="gpt2", append_eot=True),
        output=OutputConfig(path=str(output_dir), dtype="uint32", val_fraction=0.25, max_tokens=100, shard_tokens=4),
        tokenization=TokenizationConfig(workers=1, batch_docs=2),
    )
    texts = ["hello", "world", "again"]
    tokenizer = tiktoken.get_encoding("gpt2")

    manifest = prepare_texts(texts, config, hf_auth="prompt")

    tokens = np.concatenate([
        np.memmap(output_dir / shard["path"], dtype=np.uint32, mode="r")
        for shard in manifest["shards"]
    ])
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
        all_tokens.append(tokenizer.eot_token)
    split_idx = int(len(all_tokens) * 0.75)

    np.testing.assert_array_equal(np.asarray(tokens), np.asarray(all_tokens, dtype=np.uint32))
    assert manifest["num_tokens"] == len(all_tokens)
    assert manifest["schema_version"] == 2
    assert len(manifest["shards"]) > 1
    for shard in manifest["shards"]:
        assert hashlib.sha256((output_dir / shard["path"]).read_bytes()).hexdigest() == shard["sha256"]
    assert manifest["train_tokens"] == split_idx
    assert manifest["val_tokens"] == len(all_tokens) - split_idx
    assert manifest["splits"]["train"] == {"start": 0, "end": split_idx, "tokens": split_idx}
    assert manifest["splits"]["val"] == {"start": split_idx, "end": len(all_tokens), "tokens": len(all_tokens) - split_idx}
    assert manifest["tokenizer"]["append_eot"] is True
    assert manifest["hf_auth"] == "prompt"
    assert manifest["files"]["token_bytes"]["path"] == "token_bytes.bin"
    assert manifest["files"]["token_bytes"]["sha256"]
    token_bytes = np.fromfile(output_dir / "token_bytes.bin", dtype=np.uint16)
    assert token_bytes[tokenizer.eot_token] == 0
    hello_token = tokenizer.encode("hello")[0]
    assert token_bytes[hello_token] == len(tokenizer.decode_single_token_bytes(hello_token))

    disk_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert disk_manifest["source"]["dataset"] == "fake/dataset"
    assert disk_manifest["dtype"] == "uint32"
    manifest_text = json.dumps(disk_manifest)
    assert "prompt-token" not in manifest_text
    assert "env-token" not in manifest_text


def test_prepare_texts_streams_generator(tmp_path):
    output_dir = tmp_path / "prepared"
    config = PrepareConfig(
        source=SourceConfig(type="text", path="input.txt"),
        tokenizer=TokenizerConfig(name="gpt2", append_eot=True),
        output=OutputConfig(path=str(output_dir), dtype="uint32", val_fraction=0.25, max_tokens=100),
        tokenization=TokenizationConfig(workers=1),
    )
    consumed = []

    def texts():
        for text in ["hello", "world", "again"]:
            consumed.append(text)
            yield text

    manifest = prepare_texts(texts(), config)

    assert consumed == ["hello", "world", "again"]
    assert manifest["num_tokens"] > 0
    assert (output_dir / manifest["shards"][0]["path"]).exists()
    assert (output_dir / "token_bytes.bin").exists()


def test_prepare_texts_respects_max_tokens(tmp_path):
    output_dir = tmp_path / "prepared"
    config = PrepareConfig(
        source=SourceConfig(type="text", path="input.txt"),
        tokenizer=TokenizerConfig(name="gpt2", append_eot=True),
        output=OutputConfig(path=str(output_dir), dtype="uint32", val_fraction=0.2, max_tokens=5, shard_tokens=3),
        tokenization=TokenizationConfig(workers=1),
    )

    manifest = prepare_texts(["hello world", "this should be truncated"], config)

    tokens = np.concatenate([
        np.memmap(output_dir / shard["path"], dtype=np.uint32, mode="r")
        for shard in manifest["shards"]
    ])
    assert len(tokens) == 5
    assert [shard["tokens"] for shard in manifest["shards"]] == [3, 2]
    assert manifest["max_tokens"] == 5
    assert manifest["num_tokens"] == 5
    assert manifest["train_tokens"] == 4


def test_load_texts_streams_local_file(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")
    texts = load_texts(SourceConfig(type="text", path=str(path)))

    assert iter(texts) is texts
    assert list(texts) == ["hello", "world"]


def test_load_prepare_config_parses_hf_section(tmp_path):
    path = tmp_path / "prep.toml"
    path.write_text(
        f'''
[source]
type = "hf"
dataset = "fake/dataset"
split = "train"
text_column = "text"

[hf]
prompt_for_token = false
token_env = "MY_HF_TOKEN"

[tokenizer]
name = "gpt2"
append_eot = true

[output]
path = "{tmp_path / 'out'}"
dtype = "uint32"
val_fraction = 0.2
max_tokens = 1234
'''.strip()
    )

    config = load_prepare_config(path)

    assert config.hf.prompt_for_token is False
    assert config.hf.token_env == "MY_HF_TOKEN"
    assert config.output.max_tokens == 1234


def test_resolve_hf_token_prefers_env(monkeypatch):
    monkeypatch.setenv("MY_HF_TOKEN", "env-token")

    token, source = resolve_hf_token(HfConfig(token_env="MY_HF_TOKEN"))

    assert token == "env-token"
    assert source == "env"


def test_resolve_hf_token_prompts_when_env_missing(monkeypatch):
    fake_hub = types.SimpleNamespace(get_token=lambda: None, login=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "prompt-token")

    token, source = resolve_hf_token(HfConfig(prompt_for_token=True))

    assert token == "prompt-token"
    assert source == "prompt"


def test_resolve_hf_token_allows_blank_anonymous(monkeypatch):
    fake_hub = types.SimpleNamespace(get_token=lambda: None, login=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "")

    token, source = resolve_hf_token(HfConfig(prompt_for_token=True))

    assert token is None
    assert source == "anonymous"


def test_resolve_hf_token_uses_saved_hub_token(monkeypatch):
    fake_hub = types.SimpleNamespace(get_token=lambda: "saved-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token, source = resolve_hf_token(HfConfig(prompt_for_token=True))

    assert token == "saved-token"
    assert source == "saved"


def test_resolve_hf_token_saves_prompted_token(monkeypatch):
    login_calls = []
    fake_hub = types.SimpleNamespace(
        get_token=lambda: None,
        login=lambda **kwargs: login_calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "prompt-token")

    token, source = resolve_hf_token(HfConfig(prompt_for_token=True))

    assert token == "prompt-token"
    assert source == "prompt"
    assert login_calls == [{"token": "prompt-token", "skip_if_logged_in": True}]


def test_prepare_eval_domains_writes_pack_manifest_and_domain_artifacts(tmp_path):
    source_path = tmp_path / "domain.txt"
    source_path.write_text("hello world\n" * 50, encoding="utf-8")
    output_dir = tmp_path / "eval_domains"
    config = PrepareConfig(
        kind="eval_domains",
        tokenizer=TokenizerConfig(name="gpt2", append_eot=True),
        output=OutputConfig(path=str(output_dir), tokens_per_domain=32),
        domains=[
            DomainConfig(name=name, source=SourceConfig(type="text", path=str(source_path)))
            for name in REQUIRED_EVAL_DOMAINS
        ],
    )

    manifest = prepare_eval_domains(config)

    assert manifest["kind"] == "eval_domains"
    assert manifest["required_domains"] == list(REQUIRED_EVAL_DOMAINS)
    assert manifest["files"]["token_bytes"]["path"] == "token_bytes.bin"
    assert set(manifest["domains"]) == set(REQUIRED_EVAL_DOMAINS)
    assert (output_dir / "token_bytes.bin").exists()
    for name in REQUIRED_EVAL_DOMAINS:
        domain_dir = output_dir / name
        domain_manifest = json.loads((domain_dir / "manifest.json").read_text(encoding="utf-8"))
        tokens = np.memmap(domain_dir / "tokens.bin", dtype=np.uint32, mode="r")
        assert len(tokens) == 32
        assert domain_manifest["kind"] == "eval_domain"
        assert domain_manifest["domain"] == name
        assert domain_manifest["splits"]["train"] == {"start": 0, "end": 0, "tokens": 0}
        assert domain_manifest["splits"]["val"] == {"start": 0, "end": 32, "tokens": 32}


def test_prepare_eval_domains_requires_exact_domain_panel(tmp_path):
    source_path = tmp_path / "domain.txt"
    source_path.write_text("hello world\n", encoding="utf-8")

    domains = [
        DomainConfig(name=name, source=SourceConfig(type="text", path=str(source_path)))
        for name in REQUIRED_EVAL_DOMAINS[:-1]
    ]

    with pytest.raises(ValueError, match="missing"):
        PrepareConfig(
            kind="eval_domains",
            tokenizer=TokenizerConfig(name="gpt2"),
            output=OutputConfig(path=str(tmp_path / "out"), tokens_per_domain=8),
            domains=domains,
        )


def test_load_prepare_config_parses_eval_domain_config_and_hf_subset(tmp_path):
    path = tmp_path / "eval_domains.toml"
    domain_sections = []
    for name in REQUIRED_EVAL_DOMAINS:
        domain_sections.append(
            f"""
[[domain]]
name = "{name}"
source.type = "hf"
source.dataset = "fake/dataset"
source.subset = "subset-name"
source.data_dir = "data/python"
source.split = "train"
source.text_column = "text"
"""
        )
    path.write_text(
        f"""
kind = "eval_domains"

[tokenizer]
name = "gpt2"

[output]
path = "{tmp_path / 'out'}"
tokens_per_domain = 8

{''.join(domain_sections)}
""",
        encoding="utf-8",
    )

    config = load_prepare_config(path)

    assert config.kind == "eval_domains"
    assert config.domains[0].source.subset == "subset-name"
    assert config.domains[0].source.data_dir == "data/python"


def test_hf_subset_is_passed_to_load_dataset(monkeypatch):
    calls = []

    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"text": "hello"}

    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: calls.append((args, kwargs)) or FakeDataset()
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    texts = load_hf_texts(SourceConfig(type="hf", dataset="fake/dataset", subset="subset-name", split="train"), token=None)

    assert list(texts) == ["hello"]
    assert calls == [(("fake/dataset", "subset-name"), {"split": "train", "token": None, "streaming": False})]


def test_hf_data_dir_is_passed_to_load_dataset(monkeypatch):
    calls = []

    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"content": "print('hello')"}

    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: calls.append((args, kwargs)) or FakeDataset()
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    texts = load_hf_texts(
        SourceConfig(
            type="hf",
            dataset="bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            text_column="content",
        ),
        token="token",
    )

    assert list(texts) == ["print('hello')"]
    assert calls == [(("bigcode/the-stack-dedup",), {"split": "train", "token": "token", "streaming": False, "data_dir": "data/python"})]
