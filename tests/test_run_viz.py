import json

from research.utils.run_viz import write_readme_chart, write_registry_charts


def test_registry_chart_contains_all_and_new_best_sections(tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"run_name": "a", "score": 25.0, "latest_core": 0.1},
                {"run_name": "b", "score": 24.0, "latest_core": 0.0},
                {"run_name": "c", "score": 27.0, "latest_core": 0.2},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    path = write_registry_charts(registry)
    text = path.read_text(encoding="utf-8")

    assert "New Best Scores" in text
    assert "All Scores" in text
    assert "a" in text
    assert "b" in text
    assert "c" in text


def test_registry_chart_handles_empty_registry(tmp_path):
    registry = tmp_path / "registry.jsonl"

    path = write_registry_charts(registry)

    assert "No scored runs yet" in path.read_text(encoding="utf-8")


def test_readme_chart_writes_svg(tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({"run_name": "a", "score": 25.0}) + "\n", encoding="utf-8")

    path = write_readme_chart(registry, tmp_path / "chart.svg")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("<svg")
    assert "New best scores" in text
    assert "All scored runs" in text
