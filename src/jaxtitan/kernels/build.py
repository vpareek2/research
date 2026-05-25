"""Compile and inspect Jaxtitan-owned CUDA kernel artifacts."""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from jaxtitan.errors import ContractError
from jaxtitan.kernels.registry import kernel_plan
from jaxtitan.specs.run import RunSpec

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path(".jaxtitan") / "kernels"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class BuildTarget:
    """One buildable local CUDA target."""

    op: str
    source_dir: Path
    make_target: str
    output_name: str
    artifact_kind: str
    source_files: tuple[str, ...]
    extra_nvccflags: tuple[str, ...] = ()
    needs_jax_ffi_include: bool = False


def repo_root() -> Path:
    """Return the source checkout root for local kernel builds."""

    return Path(__file__).resolve().parents[3]


def cache_dir(root: str | Path | None = None) -> Path:
    """Resolve the generated kernel cache directory."""

    base = DEFAULT_CACHE_DIR if root is None else Path(root)
    return base if base.is_absolute() else repo_root() / base


def build_targets() -> tuple[BuildTarget, ...]:
    """Return buildable kernel targets in this checkout."""

    root = repo_root()
    return (
        BuildTarget(
            op="rmsnorm",
            source_dir=root / "src" / "jaxtitan" / "kernels" / "cuda" / "rmsnorm",
            make_target="all",
            output_name="rmsnorm_test.out",
            artifact_kind="standalone_test",
            source_files=("rmsnorm_test.cu", "rmsnorm.cu"),
        ),
        BuildTarget(
            op="rmsnorm",
            source_dir=root / "src" / "jaxtitan" / "kernels" / "cuda" / "rmsnorm",
            make_target="all",
            output_name="rmsnorm_ffi.so",
            artifact_kind="ffi",
            source_files=("rmsnorm_ffi.cu", "rmsnorm.cu"),
            extra_nvccflags=(
                "-shared",
                "-Xcompiler=-fPIC",
            ),
            needs_jax_ffi_include=True,
        ),
    )


def load_cache_manifest(root: str | Path | None = None) -> dict[str, Any] | None:
    """Load a kernel cache manifest if one exists."""

    path = cache_dir(root) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ContractError(f"failed to parse kernel cache manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"kernel cache manifest must be a JSON object: {path}")
    return payload


def enrich_kernel_plan_with_cache(plan: dict[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    """Attach cache status to a resolved kernel plan."""

    manifest = load_cache_manifest(root)
    artifacts = {} if manifest is None else manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    cache_root = cache_dir(root)
    decisions = []
    cached: dict[str, str] = {}
    missing: dict[str, str] = {}
    stale: dict[str, str] = {}
    for decision in plan["decisions"]:
        item = dict(decision)
        op = str(item["op"])
        artifact_group = artifacts.get(op)
        status = "not_buildable"
        if artifact_group is not None and isinstance(artifact_group, dict):
            normalized = _normalize_artifact_group(artifact_group)
            checks = {
                kind: _artifact_cache_status(cache_root, artifact)
                for kind, artifact in normalized.items()
            }
            if checks.get("ffi", (None, None))[0] == "cached":
                status = "ffi_cached"
                cached[op] = str(checks["ffi"][1])
            elif checks.get("standalone_test", (None, None))[0] == "cached":
                status = "standalone_cached"
                cached[op] = str(checks["standalone_test"][1])
            elif any(value[0] == "stale" for value in checks.values()):
                status = "stale"
                stale[op] = _first_artifact_path(checks)
            else:
                status = "missing"
                missing[op] = _first_artifact_path(checks)
        elif op in {target.op for target in build_targets()}:
            status = "missing"
            missing[op] = "not_compiled"
        item["cache_status"] = status
        decisions.append(item)
    enriched = dict(plan)
    enriched["cache_dir"] = cache_root.as_posix()
    enriched["cache_manifest"] = None if manifest is None else (cache_root / MANIFEST_NAME).as_posix()
    enriched["cached"] = cached
    enriched["missing_cache"] = missing
    enriched["stale_cache"] = stale
    enriched["cached_count"] = len(cached)
    enriched["missing_cache_count"] = len(missing)
    enriched["stale_cache_count"] = len(stale)
    enriched["decisions"] = decisions
    return enriched


def compile_kernel_plan(
    spec: RunSpec,
    *,
    arch: str | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Compile buildable kernel targets for a run config and write a manifest."""

    resolved_arch = _resolve_arch(arch)
    out_dir = cache_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = kernel_plan(spec, device_kind=None)
    target_ops = set(plan["target_ops"])
    artifacts: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    logs: dict[str, str] = {}
    for target in build_targets():
        if target.op not in target_ops:
            skipped[target.op] = "not_required_by_config"
            continue
        output_path = out_dir / target.output_name
        extra_nvccflags = list(target.extra_nvccflags)
        if target.needs_jax_ffi_include:
            extra_nvccflags.append(f"-I{_jax_ffi_include_dir()}")
        extra_flags = " ".join(extra_nvccflags)
        command = [
            "make",
            f"ARCH={resolved_arch}",
            f"SRC={' '.join(target.source_files)}",
            f"OUT={output_path.as_posix()}",
            f"EXTRA_NVCCFLAGS={extra_flags}",
            target.make_target,
        ]
        command_key = f"{target.op}:{target.artifact_kind}"
        commands[command_key] = command
        result = subprocess.run(
            command,
            cwd=target.source_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        logs[command_key] = (result.stdout + result.stderr)[-12000:]
        if result.returncode != 0:
            raise ContractError(
                f"kernel compile failed for {target.op}:{target.artifact_kind} "
                f"with exit code {result.returncode}: {logs[command_key]}"
            )
        if not output_path.exists():
            raise ContractError(
                f"kernel compile for {target.op}:{target.artifact_kind} did not produce {output_path}"
            )
        artifacts.setdefault(target.op, {})[target.artifact_kind] = {
            "path": output_path.relative_to(out_dir).as_posix(),
            "sha256": _sha256(output_path),
            "bytes": output_path.stat().st_size,
            "kind": target.artifact_kind,
            "source_dir": target.source_dir.relative_to(repo_root()).as_posix(),
        }
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cache_dir": out_dir.as_posix(),
        "arch": resolved_arch,
        "nvcc": _nvcc_version(),
        "thunderkittens": _thunderkittens_metadata(),
        "target_ops": plan["target_ops"],
        "artifacts": artifacts,
        "skipped": skipped,
        "commands": commands,
        "logs": logs,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return enrich_kernel_plan_with_cache(plan, root=out_dir)


def format_compile_result(plan: dict[str, Any]) -> str:
    """Format a compiled kernel cache summary."""

    lines = [
        (
            "kernels compiled: "
            f"cache={plan['cache_dir']} cached={plan['cached_count']} "
            f"missing={plan['missing_cache_count']} stale={plan['stale_cache_count']}"
        )
    ]
    for decision in plan["decisions"]:
        lines.append(
            "  "
            f"{decision['op']}: backend={decision['backend']} "
            f"cache={decision.get('cache_status')} reason={decision['reason']}"
        )
    return "\n".join(lines)


def _resolve_arch(arch: str | None) -> str:
    value = arch or os.environ.get("JAXTITAN_KERNEL_ARCH")
    if value is None:
        value = "SM90"
    value = value.upper()
    if value not in {"SM80", "SM90", "SM100", "SM103", "SM120", "SM121"}:
        raise ContractError(
            "kernel ARCH must be one of SM80, SM90, SM100, SM103, SM120, or SM121"
        )
    return value


def _nvcc_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    result = subprocess.run([nvcc, "--version"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _thunderkittens_metadata() -> dict[str, Any]:
    root = repo_root()
    version_path = root / "third_party" / "ThunderKittens.VERSION"
    patches_path = root / "third_party" / "ThunderKittens.PATCHES.md"
    return {
        "version_path": "third_party/ThunderKittens.VERSION",
        "version_sha256": _sha256(version_path) if version_path.exists() else None,
        "patches_path": "third_party/ThunderKittens.PATCHES.md",
        "patches_sha256": _sha256(patches_path) if patches_path.exists() else None,
    }


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _jax_ffi_include_dir() -> str:
    import jax

    return str(jax.ffi.include_dir())


def _normalize_artifact_group(group: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "path" in group:
        kind = str(group.get("kind", "standalone_test"))
        return {kind: group}
    normalized: dict[str, dict[str, Any]] = {}
    for kind, artifact in group.items():
        if isinstance(artifact, dict):
            normalized[str(kind)] = artifact
    return normalized


def _artifact_cache_status(cache_root: Path, artifact: dict[str, Any]) -> tuple[str, str]:
    rel_path = artifact.get("path")
    expected_sha = artifact.get("sha256")
    artifact_path = cache_root / rel_path if isinstance(rel_path, str) else None
    if artifact_path is not None and artifact_path.exists() and isinstance(expected_sha, str):
        actual_sha = _sha256(artifact_path)
        if actual_sha == expected_sha:
            return "cached", rel_path
        return "stale", rel_path
    return "missing", str(rel_path)


def _first_artifact_path(checks: dict[str, tuple[str, str]]) -> str:
    if not checks:
        return "not_compiled"
    for kind in ("ffi", "standalone_test"):
        if kind in checks:
            return checks[kind][1]
    return next(iter(checks.values()))[1]
