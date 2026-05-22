"""Programmatic JAX profiling helpers for training runs."""

from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from jaxtitan.errors import ContractError
from jaxtitan.services.artifacts import ArtifactWriter
from jaxtitan.specs.run import RunSpec


class ProfilingManager:
    """Owns one configured JAX trace window for a training run."""

    def __init__(self, *, spec: RunSpec, writer: ArtifactWriter) -> None:
        self.spec = spec
        self.writer = writer
        self.profile_dir = spec.dirs.run_dir / "profiles"
        self._active = False
        self._status = "disabled" if not spec.profiling.enabled else "pending"
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._start_step: int | None = None
        self._stop_step: int | None = None
        self._error: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.spec.profiling.enabled

    def write_initial_diagnostics(self) -> None:
        self.writer.write_profiling_diagnostics(self.diagnostics())

    def validate_resume_step(self, current_step: int) -> None:
        if not self.enabled:
            return
        if current_step >= self.spec.profiling.trace_start_step:
            message = (
                "profiling window is already past for this resume checkpoint; "
                "move profiling.trace_start_step forward or disable profiling"
            )
            self._fail("resume", message, step=current_step)

    @contextmanager
    def step(self, step: int):
        if not self.enabled:
            yield
            return
        if step == self.spec.profiling.trace_start_step:
            self._start(step)
        with self.annotation("train_loop_step"):
            try:
                yield
            finally:
                if self._active and step >= self.spec.profiling.trace_end_step:
                    self._stop(step)

    def annotation(self, name: str):
        if not self.enabled or not self._active:
            return nullcontext()
        return jax.profiler.TraceAnnotation(name)

    def close(self) -> None:
        if self._active:
            self._stop(self._stop_step or self._start_step or self.spec.profiling.trace_start_step)

    def diagnostics(self) -> dict[str, Any]:
        trace_start = self.spec.profiling.trace_start_step
        trace_end = self.spec.profiling.trace_end_step
        return {
            "schema_version": 1,
            "enabled": self.spec.profiling.enabled,
            "status": self._status,
            "trace_dir": "profiles",
            "trace_start_step": trace_start,
            "trace_steps": self.spec.profiling.trace_steps,
            "trace_end_step": trace_end,
            "traced_step_range": None
            if self._start_step is None
            else {
                "start": self._start_step,
                "end": self._stop_step if self._stop_step is not None else trace_end,
            },
            "create_perfetto_trace": self.spec.profiling.create_perfetto_trace,
            "create_perfetto_link": self.spec.profiling.create_perfetto_link,
            "trace_files": self._trace_files(),
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "error": self._error,
        }

    def _start(self, step: int) -> None:
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(
                str(self.profile_dir),
                create_perfetto_link=self.spec.profiling.create_perfetto_link,
                create_perfetto_trace=self.spec.profiling.create_perfetto_trace,
            )
        except Exception as exc:
            self._fail("start", str(exc), step=step, error_type=type(exc).__name__)
        self._active = True
        self._status = "active"
        self._start_step = step
        self._started_at = _utc_now()
        self.writer.append_event(
            {
                **_event("profiling_trace_started", self.spec),
                "step": step,
                "trace_dir": "profiles",
                "trace_end_step": self.spec.profiling.trace_end_step,
            }
        )
        self.writer.write_profiling_diagnostics(self.diagnostics())

    def _stop(self, step: int) -> None:
        try:
            jax.profiler.stop_trace()
        except Exception as exc:
            self._fail("stop", str(exc), step=step, error_type=type(exc).__name__)
        self._active = False
        self._status = "completed"
        self._stop_step = step
        self._stopped_at = _utc_now()
        diagnostics = self.diagnostics()
        self.writer.write_profiling_diagnostics(diagnostics)
        self.writer.append_event(
            {
                **_event("profiling_trace_completed", self.spec),
                "step": step,
                "trace_dir": "profiles",
                "trace_files": diagnostics["trace_files"],
                "trace_start_step": self._start_step,
            }
        )

    def _fail(self, phase: str, message: str, *, step: int, error_type: str = "ContractError") -> None:
        self._active = False
        self._status = "failed"
        self._stopped_at = _utc_now()
        self._error = {
            "phase": phase,
            "step": step,
            "error_type": error_type,
            "error": message,
        }
        self.writer.write_profiling_diagnostics(self.diagnostics())
        self.writer.append_event(
            {
                **_event("profiling_failed", self.spec),
                "phase": phase,
                "step": step,
                "error_type": error_type,
                "error": message,
            }
        )
        raise ContractError(f"JAX profiling failed during {phase}: {message}")

    def _trace_files(self) -> list[str]:
        if not self.profile_dir.exists():
            return []
        files = [path for path in self.profile_dir.rglob("*") if path.is_file()]
        return sorted(path.relative_to(self.spec.dirs.run_dir).as_posix() for path in files)


def profiling_runtime_summary(spec: RunSpec) -> dict[str, Any]:
    """Return config-only profiling diagnostics for runtime/preflight artifacts."""

    return {
        "schema_version": 1,
        "enabled": spec.profiling.enabled,
        "trace_start_step": spec.profiling.trace_start_step,
        "trace_steps": spec.profiling.trace_steps,
        "trace_end_step": spec.profiling.trace_end_step,
        "create_perfetto_trace": spec.profiling.create_perfetto_trace,
        "create_perfetto_link": spec.profiling.create_perfetto_link,
        "trace_dir": "profiles",
    }


def _event(event_type: str, spec: RunSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "type": event_type,
        "run_id": spec.run_id,
        "created_at": _utc_now(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
