"""
Lightweight training-loop timing helpers.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path
import time

import jax
import nvtx

from config import ProfilingConfig


class StepTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self.durations: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - start)

    def add(self, name: str, seconds: float):
        self.durations[name] = self.durations.get(name, 0.0) + seconds

    def get(self, name: str, default: float = 0.0) -> float:
        return self.durations.get(name, default)

    def metrics(self) -> dict[str, float]:
        return {f"time/{name}_sec": seconds for name, seconds in sorted(self.durations.items())}


class TraceProfiler:
    def __init__(self, config: ProfilingConfig, run_dir: str | Path):
        self.config = config
        self.trace_dir = Path(run_dir) / config.output_dir / "jax_trace"
        self.end_step = config.start_step + config.steps
        self.active = False
        self.started = False
        self.stopped = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.profiler in {"jax", "nsys"}

    @property
    def starts_jax_trace(self) -> bool:
        return self.config.enabled and self.config.profiler == "jax"

    @property
    def emits_nvtx(self) -> bool:
        return self.config.enabled and self.config.profiler in {"jax", "nsys"}

    def begin_step(self, step: int):
        if self.enabled and not self.started and self.config.start_step <= step < self.end_step:
            if self.starts_jax_trace:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(str(self.trace_dir))
            self.started = True
            self.active = True
        elif self.started and not self.stopped and self.config.start_step <= step < self.end_step:
            self.active = True
        else:
            self.active = False

    def end_current_step(self, step: int):
        if self.active and step + 1 >= self.end_step:
            self.stop()
        else:
            self.active = False

    def stop(self):
        if self.started and not self.stopped:
            if self.starts_jax_trace:
                jax.profiler.stop_trace()
            self.stopped = True
            self.active = False

    @contextmanager
    def annotate(self, name: str, *, step: int | None = None) -> Iterator[None]:
        if not self.active:
            yield
            return

        with ExitStack() as stack:
            if self.starts_jax_trace:
                if step is None:
                    stack.enter_context(jax.profiler.TraceAnnotation(name))
                else:
                    stack.enter_context(jax.profiler.StepTraceAnnotation(name, step_num=step))
            if self.emits_nvtx:
                stack.enter_context(nvtx.annotate(name))
            yield
