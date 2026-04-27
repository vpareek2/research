"""
Lightweight training-loop timing helpers.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import time


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
