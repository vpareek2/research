from pathlib import Path

import jax
import tiktoken
from flax import nnx

from config import ModelConfig, SamplingConfig
from model import Model
from sample import generate, write_sample


def tiny_model_config():
    return ModelConfig(
        vocab_size=50257,
        hidden_size=32,
        intermediate_size=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        seq_len=8,
        theta=10000.0,
        eps=1e-6,
        tied=False,
    )


def test_generate_is_deterministic_for_same_key():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    tokenizer = tiktoken.get_encoding("gpt2")
    sampling = SamplingConfig(
        enabled=True,
        prompt="ROMEO:",
        max_new_tokens=2,
        temperature=0.0,
        top_k=10,
    )
    key = jax.random.key(0)

    first = generate(model, cfg, sampling, tokenizer, key)
    second = generate(model, cfg, sampling, tokenizer, key)

    assert first == second
    assert first.startswith("ROMEO:")


def test_write_sample(tmp_path):
    path = write_sample(tmp_path, 7, "ROMEO:", "ROMEO:\nhello")

    assert path == Path(tmp_path) / "samples" / "sample_step_000007.txt"
    text = path.read_text()
    assert "step: 7" in text
    assert "prompt:" in text
    assert "sample:" in text
    assert "hello" in text
