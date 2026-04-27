import pytest

from config import ProfilingConfig
from profiling import StepTimer, TraceProfiler


class FakeContext:
    def __init__(self, calls, enter_name, exit_name):
        self.calls = calls
        self.enter_name = enter_name
        self.exit_name = exit_name

    def __enter__(self):
        self.calls.append(self.enter_name)

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(self.exit_name)


def test_step_timer_records_positive_duration():
    values = iter([1.0, 1.25])
    timer = StepTimer(clock=lambda: next(values))

    with timer.phase("data"):
        pass

    assert timer.get("data") == pytest.approx(0.25)


def test_step_timer_accumulates_repeated_phase_names():
    values = iter([1.0, 1.1, 2.0, 2.3])
    timer = StepTimer(clock=lambda: next(values))

    with timer.phase("eval"):
        pass
    with timer.phase("eval"):
        pass

    assert timer.get("eval") == pytest.approx(0.4)


def test_step_timer_metrics_use_time_prefix_and_sec_suffix():
    values = iter([1.0, 1.2])
    timer = StepTimer(clock=lambda: next(values))

    with timer.phase("train_step"):
        pass

    assert timer.metrics() == {"time/train_step_sec": pytest.approx(0.2)}


def test_disabled_trace_profiler_is_noop(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("profiling.jax.profiler.start_trace", lambda path: calls.append(("start", path)))
    monkeypatch.setattr("profiling.jax.profiler.stop_trace", lambda: calls.append(("stop",)))

    profiler = TraceProfiler(ProfilingConfig(enabled=False, profiler="none"), tmp_path)
    profiler.begin_step(100)
    with profiler.annotate("train_step"):
        calls.append(("body",))
    profiler.end_current_step(100)
    profiler.stop()

    assert calls == [("body",)]
    assert profiler.started is False
    assert profiler.stopped is False


def test_trace_profiler_starts_and_stops_exact_window(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("profiling.jax.profiler.start_trace", lambda path: calls.append(("start", path)))
    monkeypatch.setattr("profiling.jax.profiler.stop_trace", lambda: calls.append(("stop",)))

    profiler = TraceProfiler(ProfilingConfig(enabled=True, profiler="jax", start_step=2, steps=2), tmp_path)

    profiler.begin_step(1)
    assert profiler.active is False

    profiler.begin_step(2)
    assert profiler.active is True
    assert calls == [("start", str(tmp_path / "profiles" / "jax_trace"))]
    profiler.end_current_step(2)
    assert profiler.active is False
    assert calls == [("start", str(tmp_path / "profiles" / "jax_trace"))]

    profiler.begin_step(3)
    assert profiler.active is True
    profiler.end_current_step(3)
    assert profiler.active is False
    assert profiler.stopped is True
    assert calls == [("start", str(tmp_path / "profiles" / "jax_trace")), ("stop",)]

    profiler.stop()
    assert calls == [("start", str(tmp_path / "profiles" / "jax_trace")), ("stop",)]


def test_trace_profiler_annotates_with_jax_and_nvtx(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr("profiling.jax.profiler.start_trace", lambda path: calls.append(("start", path)))
    monkeypatch.setattr("profiling.jax.profiler.stop_trace", lambda: calls.append(("stop",)))
    monkeypatch.setattr(
        "profiling.jax.profiler.StepTraceAnnotation",
        lambda name, **kwargs: FakeContext(calls, ("jax_step_enter", name, kwargs), ("jax_step_exit", name)),
    )
    monkeypatch.setattr(
        "profiling.jax.profiler.TraceAnnotation",
        lambda name: FakeContext(calls, ("jax_enter", name), ("jax_exit", name)),
    )
    monkeypatch.setattr(
        "profiling.nvtx.annotate",
        lambda name: FakeContext(calls, ("nvtx_enter", name), ("nvtx_exit", name)),
    )

    profiler = TraceProfiler(ProfilingConfig(enabled=True, profiler="jax", start_step=0, steps=1), tmp_path)
    profiler.begin_step(0)

    with profiler.annotate("step", step=0):
        calls.append(("step_body",))
    with profiler.annotate("train_step"):
        calls.append(("train_body",))

    assert ("jax_step_enter", "step", {"step_num": 0}) in calls
    assert ("nvtx_enter", "step") in calls
    assert ("jax_enter", "train_step") in calls
    assert ("nvtx_enter", "train_step") in calls
    assert ("step_body",) in calls
    assert ("train_body",) in calls


def test_nsys_trace_profiler_uses_nvtx_without_jax_trace(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr("profiling.jax.profiler.start_trace", lambda path: calls.append(("start", path)))
    monkeypatch.setattr("profiling.jax.profiler.stop_trace", lambda: calls.append(("stop",)))
    monkeypatch.setattr(
        "profiling.jax.profiler.StepTraceAnnotation",
        lambda name, **kwargs: FakeContext(calls, ("jax_step_enter", name, kwargs), ("jax_step_exit", name)),
    )
    monkeypatch.setattr(
        "profiling.jax.profiler.TraceAnnotation",
        lambda name: FakeContext(calls, ("jax_enter", name), ("jax_exit", name)),
    )
    monkeypatch.setattr(
        "profiling.nvtx.annotate",
        lambda name: FakeContext(calls, ("nvtx_enter", name), ("nvtx_exit", name)),
    )

    profiler = TraceProfiler(ProfilingConfig(enabled=True, profiler="nsys", start_step=2, steps=2), tmp_path)

    profiler.begin_step(1)
    assert profiler.active is False

    profiler.begin_step(2)
    assert profiler.active is True
    assert ("start", str(tmp_path / "profiles" / "jax_trace")) not in calls

    with profiler.annotate("step", step=2):
        calls.append(("body",))

    assert ("nvtx_enter", "step") in calls
    assert ("body",) in calls
    assert not any(call[0].startswith("jax") for call in calls if isinstance(call, tuple))

    profiler.end_current_step(2)
    profiler.begin_step(3)
    assert profiler.active is True
    profiler.end_current_step(3)

    assert profiler.stopped is True
    assert ("stop",) not in calls
