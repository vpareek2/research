"""
Preflight checks for experiment launch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import jax
import tiktoken

from research.config import RunConfig, load_config
from research.data import REQUIRED_EVAL_DOMAINS, domain_eval_steps, eval_domain_root, load_token_manifest, load_validated_token_manifest, validate_eval_domain_pack
from research.distributed import create_distributed_context
from research.logs import _has_wandb_netrc_key
from research.prepare_data import PrepareConfig, load_prepare_config, prepare_dataset, resolve_hf_token


class PreflightError(RuntimeError):
    def __init__(self, message: str, result: "PreflightResult | None" = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    status: str
    path: Path
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    @property
    def missing(self) -> bool:
        return self.status == "MISSING"

    @property
    def invalid(self) -> bool:
        return self.status == "INVALID"


@dataclass
class PreflightResult:
    config_path: Path
    config: RunConfig
    data_prepare_config: PrepareConfig | None
    eval_prepare_config: PrepareConfig | None
    checks: list[tuple[str, str, str]] = field(default_factory=list)
    artifacts: list[ArtifactStatus] = field(default_factory=list)

    def ok(self, name: str, detail: str = ""):
        self.checks.append(("OK", name, detail))

    def warn(self, name: str, detail: str = ""):
        self.checks.append(("WARN", name, detail))

    def fail(self, name: str, detail: str = ""):
        self.checks.append(("FAIL", name, detail))

    @property
    def failures(self) -> list[tuple[str, str, str]]:
        return [check for check in self.checks if check[0] == "FAIL"] + [
            ("FAIL", artifact.name, artifact.detail)
            for artifact in self.artifacts
            if artifact.invalid
        ]


def run_preflight(
    config_path: str | Path,
    *,
    interactive: bool = True,
    require_ready: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> PreflightResult:
    config_path = Path(config_path)
    config = load_config(config_path)
    data_prepare = _load_data_prepare_config(config)
    eval_prepare = _load_eval_prepare_config(config)
    result = PreflightResult(config_path, config, data_prepare, eval_prepare)

    _check_imports(result)
    _check_jax(result)
    _check_tokenizer(result)
    _check_distributed(result)
    _check_training_memory(result)
    _check_run_dir(result)
    _check_budget(result)
    _check_prepare_configs(result)
    _check_artifact_capacity(result)
    _check_auth(result, interactive=interactive, runner=runner)

    result.artifacts.append(data_artifact_status(config))
    result.artifacts.append(eval_artifact_status(config))
    if require_ready:
        for artifact in result.artifacts:
            if artifact.missing:
                result.fail(artifact.name, f"required artifact is missing: {artifact.path}")

    _raise_if_failed(result)
    return result


def prepare_missing_artifacts(result: PreflightResult) -> None:
    data_status = next(artifact for artifact in result.artifacts if artifact.name == "train data")
    eval_status = next(artifact for artifact in result.artifacts if artifact.name == "eval domains")

    if eval_status.missing:
        if result.eval_prepare_config is None:
            raise PreflightError(f"Eval domains are missing and eval.prepare_config is not set: {eval_status.path}")
        prepare_dataset(result.eval_prepare_config)

    if data_status.missing:
        if result.data_prepare_config is None:
            raise PreflightError(f"Train data is missing and data.prepare_config is not set: {data_status.path}")
        prepare_dataset(result.data_prepare_config)


def data_artifact_status(config: RunConfig) -> ArtifactStatus:
    path = Path(config.data.path)
    if config.data.source == "text":
        if path.exists():
            return ArtifactStatus("train data", "READY", path, "text file exists")
        return ArtifactStatus("train data", "MISSING", path, "text file does not exist")

    if not path.exists():
        return ArtifactStatus("train data", "MISSING", path, "prepared token manifest is missing")
    if not (path / "manifest.json").exists():
        return ArtifactStatus("train data", "INVALID", path, "prepared token directory exists without manifest.json")
    try:
        load_validated_token_manifest(config.data)
    except Exception as exc:
        return ArtifactStatus("train data", "INVALID", path, str(exc))
    return ArtifactStatus("train data", "READY", path, "prepared token manifest valid")


def eval_artifact_status(config: RunConfig) -> ArtifactStatus:
    path = eval_domain_root(config.eval, config.data)
    if not path.exists():
        return ArtifactStatus("eval domains", "MISSING", path, "eval domain manifest is missing")
    if not (path / "manifest.json").exists():
        return ArtifactStatus("eval domains", "INVALID", path, "eval domain directory exists without manifest.json")
    try:
        validate_eval_domain_pack(config.eval, config.data)
    except Exception as exc:
        return ArtifactStatus("eval domains", "INVALID", path, str(exc))
    return ArtifactStatus("eval domains", "READY", path, "eval domain pack valid")


def _load_data_prepare_config(config: RunConfig) -> PrepareConfig | None:
    if config.data.prepare_config is None:
        return None
    path = Path(config.data.prepare_config)
    if not path.exists():
        raise PreflightError(f"data.prepare_config does not exist: {path}")
    return load_prepare_config(path)


def _load_eval_prepare_config(config: RunConfig) -> PrepareConfig | None:
    if config.eval.prepare_config is None:
        return None
    path = Path(config.eval.prepare_config)
    if not path.exists():
        raise PreflightError(f"eval.prepare_config does not exist: {path}")
    return load_prepare_config(path)


def _check_imports(result: PreflightResult):
    for name in ("research", "research.pretrain", "research.prepare_data"):
        importlib.import_module(name)
    result.ok("package imports", "research package imports cleanly")


def _check_jax(result: PreflightResult):
    devices = jax.local_devices()
    if not devices:
        result.fail("jax devices", "JAX reported no local devices")
        return
    platform = getattr(devices[0], "platform", "unknown")
    detail = f"{len(devices)} device(s), platform={platform}"
    if platform == "cpu":
        result.warn("jax devices", detail)
    else:
        result.ok("jax devices", detail)


def _check_tokenizer(result: PreflightResult):
    try:
        tokenizer = tiktoken.get_encoding(result.config.data.tokenizer)
    except Exception as exc:
        result.fail("tokenizer", str(exc))
        return
    result.ok("tokenizer", f"{result.config.data.tokenizer} vocab={tokenizer.n_vocab}")


def _check_distributed(result: PreflightResult):
    try:
        context = create_distributed_context(result.config.distributed, result.config.train)
    except Exception as exc:
        result.fail("distributed", str(exc))
        return
    result.ok("distributed", f"devices={context.device_count} global_batch={context.global_batch_size}")


def _check_training_memory(result: PreflightResult):
    logits_bytes = result.config.train.batch_size * result.config.train.seq_len * result.config.model.vocab_size * _dtype_bytes(result.config.precision.compute_dtype)
    loss_logits_bytes = result.config.train.batch_size * (result.config.train.seq_len - 1) * result.config.model.vocab_size * _dtype_bytes(result.config.precision.loss_dtype)
    detail = f"logits={_format_gib(logits_bytes)} loss_logits={_format_gib(loss_logits_bytes)}"

    memory_limit = _first_gpu_memory_limit()
    if memory_limit is None:
        result.warn("training memory", detail)
        return

    detail = f"{detail} gpu_memory={_format_gib(memory_limit)}"
    if logits_bytes + loss_logits_bytes > 0.65 * memory_limit:
        result.warn("training memory", detail)
    else:
        result.ok("training memory", detail)


def _dtype_bytes(dtype: str) -> int:
    if dtype == "bf16":
        return 2
    if dtype == "fp32":
        return 4
    raise ValueError(f"Unknown dtype name: {dtype}")


def _format_gib(value: int) -> str:
    return f"{value / 1024**3:.2f}GiB"


def _first_gpu_memory_limit() -> int | None:
    try:
        devices = [device for device in jax.local_devices() if getattr(device, "platform", "") == "gpu"]
        if not devices:
            return None
        stats = devices[0].memory_stats() or {}
    except Exception:
        return None
    for key in ("bytes_limit", "bytes_limit_total", "memory_limit"):
        value = stats.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _check_run_dir(result: PreflightResult):
    run_dir = Path(result.config.experiment.out_dir) / result.config.experiment.name
    if run_dir.exists():
        result.fail("run dir", f"already exists: {run_dir}")
    else:
        result.ok("run dir", f"available: {run_dir}")


def _check_budget(result: PreflightResult):
    tokens_per_step = result.config.train.batch_size * result.config.train.seq_len
    configured_tokens = result.config.train.steps * tokens_per_step
    target_steps = math.ceil(result.config.target.tokens / tokens_per_step)
    detail = (
        f"tokens_per_step={tokens_per_step:,} configured_tokens={configured_tokens:,} "
        f"target_tokens={result.config.target.tokens:,} target_steps={target_steps:,}"
    )
    if configured_tokens != result.config.target.tokens:
        result.warn("budget", detail)
    else:
        result.ok("budget", detail)


def _check_prepare_configs(result: PreflightResult):
    config = result.config
    data_prepare = result.data_prepare_config
    if config.data.source == "tokens" and data_prepare is None:
        result.warn("data.prepare_config", "not set; missing train data cannot be auto-prepared")
    elif data_prepare is not None:
        _check_dataset_prepare_config(result, data_prepare)

    eval_prepare = result.eval_prepare_config
    if eval_prepare is None:
        result.warn("eval.prepare_config", "not set; missing eval domains cannot be auto-prepared")
    else:
        _check_eval_prepare_config(result, eval_prepare)


def _check_dataset_prepare_config(result: PreflightResult, prepare_config: PrepareConfig):
    if prepare_config.kind != "dataset":
        result.fail("data.prepare_config", f"kind must be dataset, got {prepare_config.kind}")
    elif prepare_config.output.path != result.config.data.path:
        result.fail("data.prepare_config", f"output.path {prepare_config.output.path!r} != data.path {result.config.data.path!r}")
    elif prepare_config.tokenizer.name != result.config.data.tokenizer:
        result.fail("data.prepare_config", f"tokenizer {prepare_config.tokenizer.name!r} != data.tokenizer {result.config.data.tokenizer!r}")
    elif prepare_config.source is not None and prepare_config.source.type == "hf" and prepare_config.output.max_tokens is None:
        result.fail("data.prepare_config", "HF dataset preparation requires output.max_tokens to avoid uncapped dataset tokenization")
    elif prepare_config.output.max_tokens is None:
        result.fail("data.prepare_config", "training dataset preparation requires output.max_tokens")
    elif prepare_config.output.shard_tokens <= 0:
        result.fail("data.prepare_config", f"output.shard_tokens must be positive, got {prepare_config.output.shard_tokens}")
    elif prepare_config.tokenization.workers != "auto" and prepare_config.tokenization.workers <= 0:
        result.fail("data.prepare_config", f"tokenization.workers must be 'auto' or positive, got {prepare_config.tokenization.workers}")
    elif prepare_config.output.max_tokens is not None and _prepared_train_token_cap(prepare_config) < result.config.target.tokens:
        result.fail(
            "data.prepare_config",
            f"output.max_tokens leaves {_prepared_train_token_cap(prepare_config):,} train tokens after val split, "
            f"below target.tokens={result.config.target.tokens:,}",
        )
    else:
        shards = math.ceil(prepare_config.output.max_tokens / prepare_config.output.shard_tokens)
        result.ok("data.prepare_config", f"{result.config.data.prepare_config or ''} expected_shards={shards:,}")


def _prepared_train_token_cap(prepare_config: PrepareConfig) -> int:
    if prepare_config.output.max_tokens is None:
        return 0
    return int(prepare_config.output.max_tokens * (1.0 - prepare_config.output.val_fraction))


def _prepared_val_token_cap(prepare_config: PrepareConfig) -> int:
    if prepare_config.output.max_tokens is None:
        return 0
    return prepare_config.output.max_tokens - _prepared_train_token_cap(prepare_config)


def _check_eval_prepare_config(result: PreflightResult, prepare_config: PrepareConfig):
    expected_path = str(eval_domain_root(result.config.eval, result.config.data))
    if prepare_config.kind != "eval_domains":
        result.fail("eval.prepare_config", f"kind must be eval_domains, got {prepare_config.kind}")
    elif prepare_config.output.path != expected_path:
        result.fail("eval.prepare_config", f"output.path {prepare_config.output.path!r} != eval root {expected_path!r}")
    elif prepare_config.tokenizer.name != result.config.data.tokenizer:
        result.fail("eval.prepare_config", f"tokenizer {prepare_config.tokenizer.name!r} != data.tokenizer {result.config.data.tokenizer!r}")
    else:
        result.ok("eval.prepare_config", result.config.eval.prepare_config or "")


def _check_artifact_capacity(result: PreflightResult):
    if result.config.data.source == "tokens":
        _check_train_val_capacity(result)
    _check_eval_domain_capacity(result)


def _check_train_val_capacity(result: PreflightResult):
    required_val_tokens = result.config.train.eval_steps * result.config.train.batch_size * result.config.train.seq_len
    if result.data_prepare_config is not None and result.data_prepare_config.output.max_tokens is not None:
        val_cap = _prepared_val_token_cap(result.data_prepare_config)
        if val_cap < required_val_tokens:
            result.fail(
                "train eval capacity",
                f"data.prepare_config val split leaves at most {val_cap:,} tokens, "
                f"but eval requires {required_val_tokens:,} tokens "
                f"(eval_steps={result.config.train.eval_steps} * batch_size={result.config.train.batch_size} * seq_len={result.config.train.seq_len})",
            )
            return

    data_path = Path(result.config.data.path)
    if not (data_path / "manifest.json").exists():
        return
    try:
        manifest = load_validated_token_manifest(result.config.data)
    except Exception:
        return
    val_tokens = int(manifest["splits"]["val"]["tokens"])
    if val_tokens < required_val_tokens:
        result.fail(
            "train eval capacity",
            f"prepared val split has {val_tokens:,} tokens, but eval requires {required_val_tokens:,} tokens "
            f"(eval_steps={result.config.train.eval_steps} * batch_size={result.config.train.batch_size} * seq_len={result.config.train.seq_len})",
        )


def _check_eval_domain_capacity(result: PreflightResult):
    steps = domain_eval_steps(result.config.eval, result.config.train)
    required_tokens = steps * result.config.train.batch_size * result.config.train.seq_len
    if result.eval_prepare_config is not None and result.eval_prepare_config.output.tokens_per_domain is not None:
        if result.eval_prepare_config.output.tokens_per_domain < required_tokens:
            result.fail(
                "domain eval capacity",
                f"eval.prepare_config tokens_per_domain={result.eval_prepare_config.output.tokens_per_domain:,}, "
                f"but domain eval requires {required_tokens:,} tokens "
                f"(domain_eval_steps={steps} * batch_size={result.config.train.batch_size} * seq_len={result.config.train.seq_len})",
            )
            return

    eval_root = eval_domain_root(result.config.eval, result.config.data)
    if not (eval_root / "manifest.json").exists():
        return
    try:
        validate_eval_domain_pack(result.config.eval, result.config.data)
    except Exception:
        return
    for name in REQUIRED_EVAL_DOMAINS:
        domain_manifest = load_token_manifest(eval_root / name)
        domain_tokens = int(domain_manifest["num_tokens"])
        if domain_tokens < required_tokens:
            result.fail(
                "domain eval capacity",
                f"eval domain {name!r} has {domain_tokens:,} tokens, but domain eval requires {required_tokens:,} tokens "
                f"(domain_eval_steps={steps} * batch_size={result.config.train.batch_size} * seq_len={result.config.train.seq_len})",
            )
            return


def _check_auth(
    result: PreflightResult,
    *,
    interactive: bool,
    runner: Callable[..., subprocess.CompletedProcess],
):
    _check_wandb_auth(result, interactive=interactive, runner=runner)
    _check_hf_auth(result, interactive=interactive)
    _check_gh_auth(result, interactive=interactive, runner=runner)


def _check_wandb_auth(result: PreflightResult, *, interactive: bool, runner):
    if not result.config.wandb.enabled:
        result.ok("wandb auth", "disabled")
        return
    if os.environ.get("WANDB_MODE") == "offline":
        result.ok("wandb auth", "WANDB_MODE=offline")
        return
    if os.environ.get("WANDB_API_KEY") or _has_wandb_netrc_key():
        result.ok("wandb auth", "credentials available")
        return
    if interactive and shutil.which("wandb"):
        runner(["wandb", "login"], check=False)
        if os.environ.get("WANDB_API_KEY") or _has_wandb_netrc_key():
            result.ok("wandb auth", "credentials available after login")
            return
    result.fail("wandb auth", "wandb enabled but credentials are unavailable")


def _check_hf_auth(result: PreflightResult, *, interactive: bool):
    configs = [cfg for cfg in (result.data_prepare_config, result.eval_prepare_config) if cfg is not None]
    needs_hf = any(_prepare_uses_hf(cfg) for cfg in configs)
    if not needs_hf:
        result.ok("hf auth", "not required")
        return
    prompt_config = next((cfg.hf for cfg in configs if _prepare_uses_hf(cfg)), None)
    assert prompt_config is not None
    token, source = resolve_hf_token(prompt_config if interactive else type(prompt_config)(prompt_for_token=False))
    if token is not None:
        result.ok("hf auth", source)
    else:
        result.warn("hf auth", "no token found; anonymous downloads will be used")


def _prepare_uses_hf(config: PrepareConfig) -> bool:
    if config.source is not None and config.source.type == "hf":
        return True
    return any(domain.source.type == "hf" for domain in config.domains)


def _check_gh_auth(result: PreflightResult, *, interactive: bool, runner):
    if shutil.which("gh") is None:
        if interactive and _install_gh_cli(runner):
            result.ok("github cli", "installed gh")
        else:
            result.fail("github auth", "gh CLI is not installed and automatic install failed")
            return
    status = runner(["gh", "auth", "status"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if status.returncode == 0:
        result.ok("github auth", "gh auth status passed")
        return
    if interactive:
        runner(["gh", "auth", "login"], check=False)
        status = runner(["gh", "auth", "status"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if status.returncode == 0:
            result.ok("github auth", "gh auth status passed after login")
            return
    result.fail("github auth", "gh auth status failed")


def _install_gh_cli(runner: Callable[..., subprocess.CompletedProcess]) -> bool:
    if shutil.which("apt-get") is not None:
        prefix = [] if os.name != "posix" or os.geteuid() == 0 else ["sudo"]
        update = runner([*prefix, "apt-get", "update"], check=False)
        if update.returncode != 0:
            return False
        install = runner([*prefix, "apt-get", "install", "-y", "gh"], check=False)
        return install.returncode == 0 and shutil.which("gh") is not None

    if shutil.which("brew") is not None:
        install = runner(["brew", "install", "gh"], check=False)
        return install.returncode == 0 and shutil.which("gh") is not None

    return False


def _raise_if_failed(result: PreflightResult):
    failures = result.failures
    if not failures:
        return
    details = "\n".join(f"{name}: {detail}" for _, name, detail in failures)
    raise PreflightError(f"Preflight failed:\n{details}", result=result)


def format_preflight_result(result: PreflightResult) -> str:
    lines = [f"Preflight: {result.config_path}"]
    for status, name, detail in result.checks:
        suffix = f" - {detail}" if detail else ""
        lines.append(f"{status:<7} {name}{suffix}")
    for artifact in result.artifacts:
        suffix = f" - {artifact.detail}" if artifact.detail else ""
        lines.append(f"{artifact.status:<7} {artifact.name}: {artifact.path}{suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Check environment, auth, configs, and prepared-data readiness.")
    parser.add_argument("config", help="Training config TOML.")
    parser.add_argument("--no-interactive", action="store_true", help="Do not prompt or launch auth setup commands.")
    args = parser.parse_args(argv)

    try:
        result = run_preflight(args.config, interactive=not args.no_interactive)
    except PreflightError as exc:
        if exc.result is not None:
            print(format_preflight_result(exc.result))
        print(str(exc))
        raise SystemExit(1) from None
    else:
        print(format_preflight_result(result))


if __name__ == "__main__":
    main()
