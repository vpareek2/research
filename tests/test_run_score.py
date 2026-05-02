from research.utils.run_score import BASE_SCORE, attach_score, score_summary


def _summary(**overrides):
    summary = {
        "run": {"name": "baseline", "run_dir": "runs/baseline"},
        "status": "healthy",
        "quality": {"final_val_bpb": 2.0, "best_val_bpb": 2.0},
        "health": {"nan_count": 0, "loss_spike_count": 0, "grad_norm_spike_count": 0},
        "speed": {"avg_train_tokens_per_sec": 1000.0},
        "performance": {"avg_mfu": 10.0, "flops_per_token": 100.0, "peak_flops_total": 10000.0},
        "training": {"tokens_seen": 1000},
        "checkpoint_evals": {
            "latest": {
                "bpb": 2.0,
                "domains": {
                    "web": {"bpb": 2.0},
                    "code": {"bpb": 4.0},
                },
            }
        },
        "benchmark_core": {"latest": {"core": 0.1, "max_per_task": -1}},
        "inference_benchmark": {
            "latest": {
                "decode_tokens_per_sec": 100.0,
                "prefill_tokens_per_sec": 1000.0,
                "ttft_sec": 0.1,
            }
        },
        "epiplexity": {"train_bpb_auc_per_byte": 0.3},
    }
    _deep_update(summary, overrides)
    return summary


def test_baseline_scores_to_base_score():
    summary = _summary()
    score = score_summary(summary, summary)

    assert score["eligible"] is True
    assert score["final_score"] == BASE_SCORE
    assert score["quality"]["value"] == 1.0
    assert score["training_efficiency"]["value"] == 1.0
    assert score["inference_efficiency"]["value"] == 1.0
    assert score["health"]["value"] == 1.0


def test_core_gain_moves_score_up():
    baseline = _summary()
    run = _summary(run={"name": "better", "run_dir": "runs/better"}, benchmark_core={"latest": {"core": 0.3, "max_per_task": -1}})

    assert score_summary(run, baseline)["final_score"] > BASE_SCORE


def test_partial_core_is_not_score_eligible():
    summary = _summary(benchmark_core={"latest": {"core": 0.3, "max_per_task": 5}})
    score = score_summary(summary, _summary())

    assert score["eligible"] is False
    assert "full_core" in score["missing"]


def test_attach_score_preserves_summary_fields():
    summary = _summary()
    scored = attach_score(summary, summary)

    assert scored["run"]["name"] == "baseline"
    assert scored["score"]["eligible"] is True


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
