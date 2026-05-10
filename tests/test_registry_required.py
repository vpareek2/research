import json

from research.utils.registry_required import validate_registry_requirement


def registry_row(**overrides):
    row = {
        "run_name": "1_mha",
        "run_dir": "runs/1_mha",
        "score": 26.0,
        "score_eligible": True,
        "status": "stable",
        "final_step": 30518,
        "final_val_loss": 3.2,
        "best_core": 0.12,
        "avg_mfu": 6.0,
    }
    row.update(overrides)
    return row


def write_event(path, labels):
    path.write_text(
        json.dumps({"pull_request": {"labels": [{"name": label} for label in labels]}}),
        encoding="utf-8",
    )


def validate(monkeypatch, tmp_path, *, base_rows, head_rows, event_labels=None):
    registry = tmp_path / "runs" / "registry.jsonl"
    base_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in base_rows)
    head_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in head_rows)
    event_path = None
    if event_labels is not None:
        event_path = tmp_path / "event.json"
        write_event(event_path, event_labels)

    def fake_git_show(revision, path):
        assert path == registry
        return {"base": base_text, "head": head_text}.get(revision)

    monkeypatch.setattr("research.utils.registry_required._git_show_text", fake_git_show)
    return validate_registry_requirement(
        base_ref="master",
        base_revision="base",
        head_revision="head",
        registry_path=registry,
        event_path=event_path,
    )


def test_registry_required_accepts_new_scored_row(monkeypatch, tmp_path):
    result = validate(monkeypatch, tmp_path, base_rows=[], head_rows=[registry_row()])

    assert result.ok
    assert "1_mha" in result.message


def test_registry_required_fails_when_registry_is_unchanged(monkeypatch, tmp_path):
    row = registry_row()
    result = validate(monkeypatch, tmp_path, base_rows=[row], head_rows=[row])

    assert not result.ok
    assert "must add or update" in result.message


def test_registry_required_fails_when_changed_row_is_not_score_eligible(monkeypatch, tmp_path):
    result = validate(
        monkeypatch,
        tmp_path,
        base_rows=[],
        head_rows=[registry_row(score_eligible=False)],
    )

    assert not result.ok
    assert "score_eligible must be true" in result.message


def test_registry_required_allows_no_run_required_label(monkeypatch, tmp_path):
    result = validate(
        monkeypatch,
        tmp_path,
        base_rows=[],
        head_rows=[],
        event_labels=["no-run-required"],
    )

    assert result.ok
    assert "no-run-required" in result.message


def test_registry_required_skips_non_master_base(monkeypatch, tmp_path):
    result = validate_registry_requirement(
        base_ref="dev",
        base_revision="base",
        head_revision="head",
        registry_path=tmp_path / "runs" / "registry.jsonl",
    )

    assert result.ok
    assert "skipped" in result.message
