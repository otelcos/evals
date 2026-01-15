"""Open Telco benchmarks package.

Benchmark modules are imported lazily to avoid slow startup.
Use: from evals.telelogs import telelogs
"""

__all__ = [
    "telelogs",
    "telemath",
    "teleqna",
    "three_gpp",
]


def __getattr__(name: str):
    """Lazy import benchmark modules."""
    if name == "telelogs":
        from evals import telelogs as mod

        return mod
    if name == "telemath":
        from evals import telemath as mod

        return mod
    if name == "teleqna":
        from evals import teleqna as mod

        return mod
    if name == "three_gpp":
        from evals import three_gpp as mod

        return mod
    raise AttributeError(f"module 'evals' has no attribute {name!r}")
