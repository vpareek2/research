import pytest

from profiling import StepTimer


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
