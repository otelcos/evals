"""Open Telco benchmarks package.

Benchmark modules are imported lazily to avoid slow startup.
Use: from evals.telelogs import telelogs
"""

__all__ = [
    "telelogs",
    "telemath",
    "teleqna",
    "teletables",
    "three_gpp",
]


def __getattr__(name: str):
    """Lazy import benchmark modules."""
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'evals' has no attribute {name!r}")
