import pytest

from research.config import LRScheduleConfig, TrainConfig
from research.lr_schedule import build_lr_schedule, describe_lr_schedule


def train_config(**overrides):
    values = dict(
        seed=0,
        batch_size=2,
        seq_len=8,
        steps=10,
        log_every=1,
        eval_every=1,
        eval_steps=1,
        checkpoint_every=2,
        keep_last=2,
        lr_schedule=LRScheduleConfig(warmup_ratio=0.2, min_lr_ratio=0.1),
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_cosine_schedule_warms_up_and_decays_to_floor():
    schedule = build_lr_schedule(train_config(), peak_lr=1.0)

    assert float(schedule(0)) == pytest.approx(0.5)
    assert float(schedule(1)) == pytest.approx(1.0)
    assert float(schedule(2)) == pytest.approx(1.0)
    assert 0.1 < float(schedule(5)) < 1.0
    assert float(schedule(9)) == pytest.approx(0.1)


def test_wsd_schedule_warms_up_holds_peak_then_decays():
    schedule = build_lr_schedule(
        train_config(
            lr_schedule=LRScheduleConfig(
                type="wsd",
                warmup_ratio=0.2,
                stable_ratio=0.5,
                min_lr_ratio=0.1,
            )
        ),
        peak_lr=1.0,
    )

    assert float(schedule(0)) == pytest.approx(0.5)
    assert float(schedule(1)) == pytest.approx(1.0)
    assert float(schedule(6)) == pytest.approx(1.0)
    assert 0.1 < float(schedule(8)) < 1.0
    assert float(schedule(9)) == pytest.approx(0.1)


def test_default_cosine_schedule_is_valid_for_small_step_counts():
    schedule = build_lr_schedule(
        train_config(
            steps=2,
            lr_schedule=LRScheduleConfig(),
        ),
        peak_lr=0.001,
    )

    assert float(schedule(0)) == pytest.approx(0.001)
    assert float(schedule(1)) == pytest.approx(0.0001)


def test_wsd_schedule_requires_decay_step():
    with pytest.raises(ValueError, match="at least one decay step"):
        build_lr_schedule(
            train_config(
                lr_schedule=LRScheduleConfig(
                    type="wsd",
                    warmup_ratio=0.6,
                    stable_ratio=0.4,
                )
            ),
            peak_lr=1.0,
        )


def test_describe_lr_schedule_includes_phase_counts():
    config = train_config(
        lr_schedule=LRScheduleConfig(
            type="wsd",
            warmup_ratio=0.2,
            stable_ratio=0.5,
            min_lr_ratio=0.1,
        )
    )

    description = describe_lr_schedule(config)

    assert "wsd" in description
    assert "warmup_steps=2" in description
    assert "stable_steps=5" in description
    assert "decay_steps=3" in description
