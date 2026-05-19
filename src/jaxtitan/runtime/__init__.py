"""Runtime orchestration entrypoints."""

__all__ = ["RunSummary", "run_training"]


def __getattr__(name: str):
    if name in __all__:
        from jaxtitan.runtime.training import RunSummary, run_training

        exports = {"RunSummary": RunSummary, "run_training": run_training}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
