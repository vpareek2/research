from __future__ import annotations

import ast
from pathlib import Path


def test_jaxtitan_does_not_import_research() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "jaxtitan"
    offenders: list[str] = []

    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "research" or alias.name.startswith("research."):
                        offenders.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "research" or module.startswith("research."):
                    offenders.append(f"{path}: from {module} import ...")

    assert offenders == []
