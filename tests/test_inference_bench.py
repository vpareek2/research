import json

from flax import nnx
import pytest
import tiktoken

from research.config import ModelConfig
from research.model import Model
from research.utils import inference_bench


def tiny_model_config():
    return ModelConfig(
        vocab_size=50257,
        hidden_size=16,
        intermediate_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        seq_len=8,
        theta=10000.0,
        eps=1e-6,
        tied=False,
    )


def test_make_input_ids_uses_fixed_seq_len():
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = inference_bench._make_input_ids("hello", tokenizer, prompt_tokens=4)

    assert input_ids.shape == (1, 4)
    assert input_ids.dtype.name == "int32"


def test_benchmark_inference_returns_positive_metrics(tmp_path):
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    tokenizer = tiktoken.get_encoding("gpt2")
    bench_config = inference_bench.InferenceBenchConfig(prompt_tokens=4, decode_tokens=2)

    metrics = inference_bench.benchmark_inference(
        model,
        tokenizer,
        cfg,
        run_dir=tmp_path,
        checkpoint_step=3,
        prompts=["hello world"],
        bench_config=bench_config,
    )

    assert metrics["mode"] == "kv_cache_decode_loop_prefill"
    assert metrics["checkpoint_step"] == 3
    assert metrics["prompt_tokens"] == 4
    assert metrics["decode_tokens"] == 2
    assert metrics["prefill_tokens_per_sec"] > 0.0
    assert metrics["decode_tokens_per_sec"] > 0.0
    assert metrics["first_decode_sec"] > 0.0
    assert metrics["ttft_sec"] > 0.0
    assert metrics["memory_used_bytes"] is None


def test_write_inference_artifacts(tmp_path):
    metrics = {
        "run_dir": str(tmp_path),
        "checkpoint_step": 7,
        "mode": "kv_cache_decode_loop_prefill",
        "batch_size": 1,
        "prompt_count": 3,
        "prompt_tokens": 4,
        "decode_tokens": 4,
        "seq_len": 8,
        "prefill_elapsed_sec": 0.1,
        "decode_elapsed_sec": 0.2,
        "first_decode_sec": 0.01,
        "ttft_sec": 0.01,
        "prefill_tokens_per_sec": 120.0,
        "decode_tokens_per_sec": 60.0,
        "memory_used_bytes": None,
        "memory_peak_bytes": None,
    }

    metrics_path, summary_path = inference_bench.write_inference_artifacts(tmp_path, 7, metrics)

    assert metrics_path == tmp_path / "evals" / "step_7" / "inference_metrics.json"
    assert summary_path == tmp_path / "evals" / "step_7" / "inference_summary.md"
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["decode_tokens_per_sec"] == pytest.approx(60.0)
    assert "kv_cache_decode_loop_prefill" in summary_path.read_text(encoding="utf-8")
