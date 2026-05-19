"""Runtime orchestration entrypoints."""

__all__ = ["PreflightReport", "RunSummary", "run_preflight", "run_training"]


def __getattr__(name: str):
    if name in {"RunSummary", "run_training"}:
        from jaxtitan.runtime.training import RunSummary, run_training

        exports = {"RunSummary": RunSummary, "run_training": run_training}
        return exports[name]
    if name in {"PreflightReport", "run_preflight"}:
        from jaxtitan.runtime.preflight import PreflightReport, run_preflight

        exports = {"PreflightReport": PreflightReport, "run_preflight": run_preflight}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
