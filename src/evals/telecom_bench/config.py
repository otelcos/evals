"""telecom_bench paths and dataset locations. Read this to understand the eval."""

from pathlib import Path

# repo_root/src/evals/telecom_bench/config.py -> parents[3] == repo root
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "telecom_bench" / "datasets"
UPSTREAM_REF = (
    Path(__file__).resolve().parents[3] / "data" / "telecom_bench" / "upstream_ref"
)

KC = DATA_DIR / "Knowledge_Comprehension"
KA = DATA_DIR / "Knowledge_Application"

# Default judge panel: three calls to the active model. Override per-run.
DEFAULT_JUDGES: list[str | None] = [None, None, None]
