from __future__ import annotations

import importlib.metadata

import jaxtitan


def test_installed_package_imports_as_jaxtitan() -> None:
    assert jaxtitan.__version__ == "0.1.0"
    assert importlib.metadata.version("jaxtitan") == jaxtitan.__version__
