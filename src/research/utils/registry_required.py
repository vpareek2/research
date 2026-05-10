"""
Validate that a PR into master records a scored run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_REGISTRY_PATH = Path("runs") / "registry.jsonl"
NO_RUN_REQUIRED_LABEL = "no-run-required"
REQUIRED_SCORED_FIELDS = (
    "run_name",
    "run_dir",
    "score",
    "score_eligible",
    "status",
    "final_step",
    "final_val_loss",
    "best_core",
    "avg_mfu",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Require a new scored registry row for PRs into master.")
    parser.add_argument("--base-ref", default=os.environ.get("GITHUB_BASE_REF") or "master")
    parser.add_argument("--base-revision", default=os.environ.get("GITHUB_BASE_SHA") or "origin/master")
    parser.add_argument("--head-revision", default=os.environ.get("GITHUB_SHA") or "HEAD")
    parser.add_argument("--event-path", default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    args = parser.parse_args(argv)

    result = validate_registry_requirement(
        base_ref=args.base_ref,
        base_revision=args.base_revision,
        head_revision=args.head_revision,
        registry_path=Path(args.registry_path),
        event_path=Path(args.event_path) if args.event_path else None,
    )
    if result.ok:
        print(result.message)
        return 0
    print(result.message, file=sys.stderr)
    return 1


class ValidationResult:
    def __init__(self, ok: bool, message: str):
        self.ok = ok
        self.message = message


def validate_registry_requirement(
    *,
    base_ref: str,
    base_revision: str,
    head_revision: str,
    registry_path: Path,
    event_path: Path | None = None,
) -> ValidationResult:
    if base_ref != "master":
        return ValidationResult(True, f"registry check skipped for base branch {base_ref!r}")
    if _has_no_run_required_label(event_path):
        return ValidationResult(True, f"registry check skipped because label {NO_RUN_REQUIRED_LABEL!r} is present")

    base_text = _git_show_text(base_revision, registry_path) or ""
    head_text = _git_show_text(head_revision, registry_path)
    if head_text is None:
        return ValidationResult(False, f"{registry_path} is missing; run `uv run experiment ...` before merging")

    try:
        base_rows = _parse_registry(base_text, source=f"{base_revision}:{registry_path}")
        head_rows = _parse_registry(head_text, source=f"{head_revision}:{registry_path}")
    except ValueError as exc:
        return ValidationResult(False, str(exc))

    changed_rows = _changed_registry_rows(base_rows, head_rows)
    if not changed_rows:
        return ValidationResult(
            False,
            f"{registry_path} must add or update a scored run row, or apply the {NO_RUN_REQUIRED_LABEL!r} label",
        )

    valid_rows = []
    errors = []
    for row in changed_rows:
        row_errors = _scored_row_errors(row)
        if row_errors:
            run_name = row.get("run_name", "<unknown>")
            errors.append(f"{run_name}: {', '.join(row_errors)}")
        else:
            valid_rows.append(row)

    if not valid_rows:
        return ValidationResult(False, "changed registry rows are not valid scored runs: " + "; ".join(errors))

    names = ", ".join(str(row["run_name"]) for row in valid_rows)
    return ValidationResult(True, f"registry check passed for scored run(s): {names}")


def _has_no_run_required_label(event_path: Path | None) -> bool:
    if event_path is None or not event_path.exists():
        return False
    event = json.loads(event_path.read_text(encoding="utf-8"))
    labels = event.get("pull_request", {}).get("labels", [])
    return any(label.get("name") == NO_RUN_REQUIRED_LABEL for label in labels if isinstance(label, dict))


def _git_show_text(revision: str, path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _parse_registry(text: str, *, source: str) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{lineno} is not valid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{lineno} must be a JSON object")
        rows.append(row)
    return rows


def _changed_registry_rows(base_rows: list[dict[str, Any]], head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_key = {_registry_key(row): row for row in base_rows}
    changed = []
    for row in head_rows:
        key = _registry_key(row)
        if base_by_key.get(key) != row:
            changed.append(row)
    return changed


def _registry_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("run_name"), row.get("run_dir")


def _scored_row_errors(row: dict[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_SCORED_FIELDS if row.get(field) is None]
    if row.get("score_eligible") is not True:
        errors.append("score_eligible must be true")
    if not isinstance(row.get("score"), int | float):
        errors.append("score must be numeric")
    if not isinstance(row.get("final_step"), int | float):
        errors.append("final_step must be numeric")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
