import json
import sys
import types

import numpy as np
import tiktoken

from prepare_data import (
    HfConfig,
    OutputConfig,
    PrepareConfig,
    SourceConfig,
    TokenizerConfig,
    load_texts,
    load_prepare_config,
    prepare_texts,
    resolve_hf_token,
)


def test_prepare_texts_writes_bins_manifest_and_eot(tmp_path):
    output_dir = tmp_path / "prepared"
    config = PrepareConfig(
        source=SourceConfig(type="hf", dataset="fake/dataset", split="train", text_column="text"),
        tokenizer=TokenizerConfig(name="gpt2", append_eot=True),
        output=OutputConfig(path=str(output_dir), dtype="uint32", val_fraction=0.25),
    )
    texts = ["hello", "world", "again"]
    tokenizer = tiktoken.get_encoding("gpt2")

    manifest = prepare_texts(texts, config, hf_auth="prompt")

    tokens = np.memmap(output_dir / "tokens.bin", dtype=np.uint32, mode="r")
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
        all_tokens.append(tokenizer.eot_token)
    split_idx = int(len(all_tokens) * 0.75)

    np.testing.assert_array_equal(np.asarray(tokens), np.asarray(all_tokens, dtype=np.uint32))
    assert manifest["num_tokens"] == len(all_tokens)
    assert manifest["train_tokens"] == split_idx
    assert manifest["val_tokens"] == len(all_tokens) - split_idx
    assert manifest["splits"]["train"] == {"start": 0, "end": split_idx, "tokens": split_idx}
    assert manifest["splits"]["val"] == {"start": split_idx, "end": len(all_tokens), "tokens": len(all_tokens) - split_idx}
    assert manifest["tokenizer"]["append_eot"] is True
    assert manifest["hf_auth"] == "prompt"
    assert manifest["files"]["tokens"]["sha256"]
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
        output=OutputConfig(path=str(output_dir), dtype="uint32", val_fraction=0.25),
    )
    consumed = []

    def texts():
        for text in ["hello", "world", "again"]:
            consumed.append(text)
            yield text

    manifest = prepare_texts(texts(), config)

    assert consumed == ["hello", "world", "again"]
    assert manifest["num_tokens"] > 0
    assert (output_dir / "tokens.bin").exists()
    assert (output_dir / "token_bytes.bin").exists()


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
'''.strip()
    )

    config = load_prepare_config(path)

    assert config.hf.prompt_for_token is False
    assert config.hf.token_env == "MY_HF_TOKEN"


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
