"""
Static registry visualization.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from utils.run_summary import DEFAULT_REGISTRY_PATH


def write_registry_charts(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    output_path: str | Path | None = None,
) -> Path:
    registry_path = Path(registry_path)
    output_path = Path(output_path) if output_path is not None else registry_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _scored_rows(_load_registry(registry_path))
    best_rows = _new_best_rows(rows)
    output_path.write_text(_render_page(rows, best_rows), encoding="utf-8")
    return output_path


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scored_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for idx, row in enumerate(rows):
        score = row.get("score")
        if isinstance(score, (int, float)):
            scored.append({**row, "_idx": idx, "_score": float(score)})
    return scored


def _new_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = float("-inf")
    out = []
    for row in rows:
        if row["_score"] > best:
            out.append(row)
            best = row["_score"]
    return out


def _render_page(rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run Score Progression</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    svg {{ max-width: 100%; height: auto; border: 1px solid #d8dee9; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .empty {{ color: #687586; }}
  </style>
</head>
<body>
  <h1>Run Score Progression</h1>
  <section>
    <h2>New Best Scores</h2>
    {_render_chart(best_rows, "best")}
  </section>
  <section>
    <h2>All Scores</h2>
    {_render_chart(rows, "all")}
  </section>
  <section>
    <h2>Runs</h2>
    {_render_table(rows)}
  </section>
</body>
</html>
"""


def _render_chart(rows: list[dict[str, Any]], chart_id: str) -> str:
    if not rows:
        return '<p class="empty">No scored runs yet.</p>'

    width, height = 920, 320
    left, right, top, bottom = 60, 24, 24, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    scores = [row["_score"] for row in rows]
    y_min, y_max = min(scores), max(scores)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = (y_max - y_min) * 0.1
        y_min -= pad
        y_max += pad

    def xy(i: int, score: float) -> tuple[float, float]:
        x = left + (plot_w * i / max(1, len(rows) - 1))
        y = top + plot_h * (1.0 - (score - y_min) / (y_max - y_min))
        return x, y

    points = [xy(i, row["_score"]) for i, row in enumerate(rows)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    y_ticks = [y_min + (y_max - y_min) * frac / 4 for frac in range(5)]
    parts = [f'<svg id="{chart_id}" viewBox="0 0 {width} {height}" role="img">']
    for tick in y_ticks:
        _, y = xy(0, tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#edf1f5"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#52606d">{tick:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9aa5b1"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9aa5b1"/>')
    parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{polyline}"/>')
    for i, row in enumerate(rows):
        x, y = points[i]
        name = html.escape(str(row.get("run_name", "")))
        score = row["_score"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#1d4ed8"><title>{name}: {score:.3f}</title></circle>')
        if len(rows) <= 12:
            parts.append(f'<text x="{x:.1f}" y="{height-18}" text-anchor="middle" font-size="11" fill="#52606d">{_short(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No scored runs yet.</p>'

    lines = [
        "<table>",
        "<thead><tr><th>run</th><th>score</th><th>core</th><th>val BPB</th><th>decode/s</th><th>MFU</th><th>baseline</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('run_name', '')))}</td>"
            f"<td>{_fmt(row.get('score'))}</td>"
            f"<td>{_fmt(row.get('latest_core'))}</td>"
            f"<td>{_fmt(row.get('best_val_bpb'))}</td>"
            f"<td>{_fmt(row.get('latest_decode_tokens_per_sec'))}</td>"
            f"<td>{_fmt(row.get('avg_mfu'))}</td>"
            f"<td>{html.escape(str(row.get('baseline_run_name') or ''))}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if abs(float(value)) >= 1000:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def _short(name: str) -> str:
    return name if len(name) <= 16 else name[:13] + "..."
