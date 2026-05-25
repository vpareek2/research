"""JAX FFI registration helpers for Jaxtitan-owned CUDA kernels."""

import ctypes
from pathlib import Path
from typing import Any

import jax

from jaxtitan.errors import ContractError
from jaxtitan.kernels.build import cache_dir, load_cache_manifest

RMSNORM_TARGET_NAME = "jaxtitan_rmsnorm_bf16_h1024"
RMSNORM_SYMBOL_NAME = "JaxtitanRmsNormBf16H1024"

_REGISTERED: set[str] = set()
_LOADED_LIBS: dict[str, ctypes.CDLL] = {}


def register_rmsnorm(cache_root: str | Path | None = None) -> str:
    """Register the compiled RMSNorm FFI target and return its JAX target name."""

    if RMSNORM_TARGET_NAME in _REGISTERED:
        return RMSNORM_TARGET_NAME

    shared_object = rmsnorm_shared_object_path(cache_root)
    try:
        library = ctypes.CDLL(shared_object.as_posix())
    except OSError as exc:
        raise ContractError(f"failed to load RMSNorm FFI shared object {shared_object}: {exc}") from exc
    try:
        symbol = getattr(library, RMSNORM_SYMBOL_NAME)
    except AttributeError as exc:
        raise ContractError(
            f"RMSNorm FFI shared object {shared_object} is missing symbol {RMSNORM_SYMBOL_NAME}"
        ) from exc

    jax.ffi.register_ffi_target(
        RMSNORM_TARGET_NAME,
        jax.ffi.pycapsule(symbol),
        platform="CUDA",
        api_version=1,
    )
    _LOADED_LIBS[RMSNORM_TARGET_NAME] = library
    _REGISTERED.add(RMSNORM_TARGET_NAME)
    return RMSNORM_TARGET_NAME


def rmsnorm_shared_object_path(cache_root: str | Path | None = None) -> Path:
    """Return the cached RMSNorm FFI shared object path after checksum validation."""

    manifest = load_cache_manifest(cache_root)
    if manifest is None:
        raise ContractError(
            "kernel cache manifest is missing; run `uv run jaxtitan kernels compile <config>` first"
        )
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ContractError("kernel cache manifest has invalid artifacts section")
    rmsnorm = artifacts.get("rmsnorm")
    if not isinstance(rmsnorm, dict):
        raise ContractError("kernel cache manifest does not contain an RMSNorm artifact")
    ffi_artifact = _ffi_artifact(rmsnorm)
    rel_path = ffi_artifact.get("path")
    expected_sha = ffi_artifact.get("sha256")
    if not isinstance(rel_path, str) or not isinstance(expected_sha, str):
        raise ContractError("RMSNorm FFI artifact must record path and sha256")
    path = cache_dir(cache_root) / rel_path
    if not path.exists():
        raise ContractError(f"RMSNorm FFI shared object is missing: {path}")

    from jaxtitan.kernels.build import _sha256

    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ContractError(f"RMSNorm FFI shared object checksum mismatch: {path}")
    return path


def reset_registrations_for_tests() -> None:
    """Clear local registration bookkeeping for focused unit tests."""

    _REGISTERED.clear()
    _LOADED_LIBS.clear()


def _ffi_artifact(rmsnorm: dict[str, Any]) -> dict[str, Any]:
    artifact = rmsnorm.get("ffi")
    if isinstance(artifact, dict):
        return artifact
    if rmsnorm.get("kind") == "ffi":
        return rmsnorm
    raise ContractError(
        "kernel cache manifest does not contain an RMSNorm FFI artifact; "
        "rerun `uv run jaxtitan kernels compile <config>`"
    )
