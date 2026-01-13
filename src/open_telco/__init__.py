"""Open Telco benchmarks package.

Benchmark modules are imported lazily to avoid slow startup.
Use: from open_telco.telelogs import telelogs
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
        from open_telco import telelogs as mod

        return mod
    if name == "telemath":
        from open_telco import telemath as mod

        return mod
    if name == "teleqna":
        from open_telco import teleqna as mod

        return mod
    if name == "three_gpp":
        from open_telco import three_gpp as mod

        return mod
    raise AttributeError(f"module 'open_telco' has no attribute {name!r}")
