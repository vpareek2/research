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
        artifact = artifacts.get(op)
        status = "not_buildable"
        if artifact is not None and isinstance(artifact, dict):
            rel_path = artifact.get("path")
            expected_sha = artifact.get("sha256")
            artifact_path = cache_root / rel_path if isinstance(rel_path, str) else None
            if artifact_path is not None and artifact_path.exists() and isinstance(expected_sha, str):
                actual_sha = _sha256(artifact_path)
                if actual_sha == expected_sha:
                    status = "cached"
                    cached[op] = rel_path
                else:
                    status = "stale"
                    stale[op] = rel_path
            else:
                status = "missing"
                missing[op] = str(rel_path)
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
        command = [
            "make",
            f"ARCH={resolved_arch}",
            f"OUT={output_path.as_posix()}",
            target.make_target,
        ]
        commands[target.op] = command
        result = subprocess.run(
            command,
            cwd=target.source_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        logs[target.op] = (result.stdout + result.stderr)[-12000:]
        if result.returncode != 0:
            raise ContractError(
                f"kernel compile failed for {target.op} with exit code {result.returncode}: "
                f"{logs[target.op]}"
            )
        if not output_path.exists():
            raise ContractError(f"kernel compile for {target.op} did not produce {output_path}")
        artifacts[target.op] = {
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
