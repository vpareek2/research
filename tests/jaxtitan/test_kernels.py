from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import os
import pytest
import shutil

from jaxtitan.config import load_config
from jaxtitan.errors import ContractError
from jaxtitan.kernels import (
    compile_kernel_plan,
    enrich_kernel_plan_with_cache,
    format_kernel_plan,
    kernel_plan,
    kernel_registry_payload,
    load_cache_manifest,
    require_kernel_plan_supported,
)
from jaxtitan.kernels import _ffi
from jaxtitan.kernels.bench import benchmark_rmsnorm, benchmark_to_json, parse_rows
from jaxtitan.kernels.rmsnorm import rmsnorm_reference, rmsnorm_tk_forward


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
intermediate_size = 512
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


def test_kernel_registry_lists_known_internal_candidates() -> None:
    payload = kernel_registry_payload()

    ops = {item["op"] for item in payload["kernels"]}
    assert "rmsnorm" in ops
    assert "moe_dispatch_combine" in ops
    assert all(item["implementation"] == "thunderkittens" for item in payload["kernels"])


def test_kernel_plan_defaults_to_xla_when_backend_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    plan = kernel_plan(load_config(config_path), device_kind="cpu")

    assert plan["enabled"] is False
    assert plan["mode"] == "xla"
    assert plan["target_ops"] == ["rmsnorm", "attention", "swiglu"]
    assert plan["active_count"] == 0
    assert plan["fallback"] == {
        "rmsnorm": "kernels_disabled",
        "attention": "kernels_disabled",
        "swiglu": "kernels_disabled",
    }
    assert "rmsnorm: backend=xla reason=kernels_disabled" in format_kernel_plan(plan)
    enriched = enrich_kernel_plan_with_cache(plan, root=tmp_path / "kernel-cache")
    assert enriched["missing_cache"] == {"rmsnorm": "not_compiled"}
    assert enriched["decisions"][0]["cache_status"] == "missing"


def test_kernel_plan_reports_unavailable_candidates_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[kernels]
enabled = true
"""
    )

    plan = kernel_plan(load_config(config_path))

    assert plan["enabled"] is True
    assert plan["mode"] == "auto"
    assert plan["active_count"] == 0
    assert plan["fallback_count"] == 3
    assert set(plan["unavailable"]) == {"rmsnorm", "attention", "swiglu"}
    assert set(plan["fallback"].values()) == {"no_jax_ffi_implementation"}


def test_kernel_strict_mode_rejects_unavailable_candidates(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[kernels]
enabled = true
strict = true
"""
    )

    plan = kernel_plan(load_config(config_path))

    with pytest.raises(ContractError, match="kernels.strict=true"):
        require_kernel_plan_supported(plan)


def test_compile_kernel_plan_writes_cache_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    cache_dir = tmp_path / "cache"

    def fake_run(command, *, cwd, check, capture_output, text):
        output = Path(next(part.removeprefix("OUT=") for part in command if part.startswith("OUT=")))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"compiled-{output.name}".encode())
        return subprocess.CompletedProcess(command, 0, stdout=f"built in {cwd}", stderr="")

    monkeypatch.setattr("jaxtitan.kernels.build.subprocess.run", fake_run)
    monkeypatch.setattr("jaxtitan.kernels.build._nvcc_version", lambda: "nvcc fake")

    plan = compile_kernel_plan(load_config(config_path), arch="SM90", root=cache_dir)
    manifest = load_cache_manifest(cache_dir)

    assert manifest is not None
    assert manifest["arch"] == "SM90"
    assert manifest["artifacts"]["rmsnorm"]["standalone_test"]["path"] == "rmsnorm_test.out"
    assert manifest["artifacts"]["rmsnorm"]["ffi"]["path"] == "rmsnorm_ffi.so"
    assert plan["cached"] == {"rmsnorm": "rmsnorm_ffi.so"}
    assert plan["missing_cache_count"] == 0
    assert plan["decisions"][0]["cache_status"] == "ffi_cached"


def test_compile_kernel_plan_rejects_bad_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    with pytest.raises(ContractError, match="kernel ARCH"):
        compile_kernel_plan(load_config(config_path), arch="SM70", root=tmp_path / "cache")


def test_parse_benchmark_rows() -> None:
    assert parse_rows("1, 4,17") == (1, 4, 17)
    with pytest.raises(ContractError, match="rows"):
        parse_rows("1,0")
    with pytest.raises(ContractError, match="rows"):
        parse_rows("nope")


def test_rmsnorm_benchmark_missing_ffi_artifact_fails_cleanly(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    with pytest.raises(ContractError, match="kernel cache manifest is missing"):
        benchmark_rmsnorm(load_config(config_path), cache_root=tmp_path / "cache", rows=(1,), iters=1)


def test_benchmark_to_json_is_stable() -> None:
    payload = {"schema_version": 1, "op": "rmsnorm", "rows": [{"rows": 1}]}
    assert benchmark_to_json(payload) == '{"op":"rmsnorm","rows":[{"rows":1}],"schema_version":1}'


def test_rmsnorm_reference_shape_dtype_and_value() -> None:
    x = jnp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.bfloat16)
    weight = jnp.array([0.5, 1.5], dtype=jnp.bfloat16)

    out = rmsnorm_reference(x, weight, eps=0.0)

    expected = x.astype(jnp.float32)
    expected = expected * jax.lax.rsqrt(jnp.mean(jnp.square(expected), axis=-1, keepdims=True))
    expected = expected * weight.astype(jnp.float32)
    assert out.shape == x.shape
    assert out.dtype == jnp.bfloat16
    assert jnp.allclose(out.astype(jnp.float32), expected.astype(jnp.bfloat16).astype(jnp.float32))


def test_rmsnorm_tk_wrapper_rejects_unsupported_inputs() -> None:
    x = jnp.ones((2, 1024), dtype=jnp.float32)
    weight = jnp.ones((1024,), dtype=jnp.bfloat16)
    with pytest.raises(ContractError, match="bfloat16 input"):
        rmsnorm_tk_forward(x, weight)

    x = jnp.ones((2, 512), dtype=jnp.bfloat16)
    with pytest.raises(ContractError, match="hidden size"):
        rmsnorm_tk_forward(x, weight)

    x = jnp.ones((2, 1024), dtype=jnp.bfloat16)
    weight = jnp.ones((512,), dtype=jnp.bfloat16)
    with pytest.raises(ContractError, match="weight shape"):
        rmsnorm_tk_forward(x, weight)


def test_rmsnorm_ffi_missing_shared_object_fails_cleanly(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        """
{
  "artifacts": {
    "rmsnorm": {
      "ffi": {
        "path": "rmsnorm_ffi.so",
        "sha256": "abc"
      }
    }
  }
}
"""
    )

    with pytest.raises(ContractError, match="shared object is missing"):
        _ffi.rmsnorm_shared_object_path(tmp_path)


def test_rmsnorm_ffi_registration_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_object = tmp_path / "rmsnorm_ffi.so"
    shared_object.write_bytes(b"ffi")
    import hashlib

    (tmp_path / "manifest.json").write_text(
        f"""
{{
  "artifacts": {{
    "rmsnorm": {{
      "ffi": {{
        "path": "rmsnorm_ffi.so",
        "sha256": "{hashlib.sha256(b"ffi").hexdigest()}"
      }}
    }}
  }}
}}
"""
    )

    class FakeLibrary:
        JaxtitanRmsNormBf16H1024 = object()

    calls = []
    monkeypatch.setattr(_ffi.ctypes, "CDLL", lambda path: FakeLibrary())
    monkeypatch.setattr(_ffi.jax.ffi, "pycapsule", lambda symbol: ("capsule", symbol))
    monkeypatch.setattr(
        _ffi.jax.ffi,
        "register_ffi_target",
        lambda name, fn, platform, api_version: calls.append((name, fn, platform, api_version)),
    )
    _ffi.reset_registrations_for_tests()
    try:
        assert _ffi.register_rmsnorm(tmp_path) == _ffi.RMSNORM_TARGET_NAME
        assert _ffi.register_rmsnorm(tmp_path) == _ffi.RMSNORM_TARGET_NAME
    finally:
        _ffi.reset_registrations_for_tests()

    assert len(calls) == 1
    assert calls[0][0] == _ffi.RMSNORM_TARGET_NAME
    assert calls[0][2] == "CUDA"


@pytest.mark.skipif(
    not bool(os.environ.get("JAXTITAN_RUN_REAL_KERNEL_TESTS")),
    reason="real CUDA FFI test is opt-in",
)
def test_rmsnorm_ffi_matches_reference_when_compiled(tmp_path: Path) -> None:
    try:
        backend = jax.default_backend()
    except RuntimeError as exc:
        pytest.skip(f"JAX gpu backend unavailable: {exc}")
    if backend != "gpu":
        pytest.skip("JAX gpu backend unavailable")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc unavailable")

    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    compile_kernel_plan(load_config(config_path), arch="SM121", root=tmp_path / "cache")
    _ffi.reset_registrations_for_tests()

    x = jnp.arange(2 * 1024, dtype=jnp.float32).reshape(2, 1024)
    x = ((x % 251.0) / 31.0).astype(jnp.bfloat16)
    weight = (1.0 + (jnp.arange(1024, dtype=jnp.float32) % 97.0) / 257.0).astype(jnp.bfloat16)

    got = jax.jit(lambda a, b: rmsnorm_tk_forward(a, b, cache_root=tmp_path / "cache"))(x, weight)
    expected = rmsnorm_reference(x, weight)
    assert jnp.max(jnp.abs(got.astype(jnp.float32) - expected.astype(jnp.float32))) <= 1.0e-2
