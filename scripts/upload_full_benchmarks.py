#!/usr/bin/env python3
"""Download full benchmark datasets, transform schemas, validate, and upload.

Populates GSMA/ot-full-benchmarks with five configs:
  teletables, teleqna, telemath, telelogs, 3gpp_tsg

Usage:
    uv run python scripts/upload_full_benchmarks.py --dry-run   # validate only
    uv run python scripts/upload_full_benchmarks.py              # upload to HF
"""

from __future__ import annotations

import argparse
import logging
import sys

from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET_REPO = "GSMA/ot-full-benchmarks"


# ── TeleLogs handling ───────────────────────────────────────────────────────


def load_telelogs() -> Dataset:
    """Load TeleLogs, falling back to open_telco if netop format differs."""
    try:
        ds = load_dataset("netop/TeleLogs", "troubleshooting", split="test")
        cols = set(ds.column_names)
        if "question" in cols and "answer" in cols:
            log.info("Using netop/TeleLogs troubleshooting (%d samples)", len(ds))
            return ds
        log.warning(
            "netop/TeleLogs missing question/answer columns (got %s). "
            "Falling back to GSMA/open_telco telelogs.",
            cols,
        )
    except Exception:
        log.warning(
            "Could not load netop/TeleLogs troubleshooting. "
            "Falling back to GSMA/open_telco telelogs.",
            exc_info=True,
        )
    ds = load_dataset("GSMA/open_telco", "telelogs", split="test")
    log.info("Using GSMA/open_telco telelogs fallback (%d samples)", len(ds))
    return ds


# ── Validation ──────────────────────────────────────────────────────────────

EXPECTED_COLUMNS: dict[str, set[str]] = {
    "teletables": {"question", "choices", "answer"},
    "teleqna": {"question", "choices", "answer", "subject"},
    "telemath": {"question", "answer"},
    "telelogs": {"question", "answer"},
    "3gpp_tsg": {"question", "answer"},
}


def validate(name: str, ds: Dataset) -> None:
    """Assert expected columns and log sample count."""
    cols = set(ds.column_names)
    expected = EXPECTED_COLUMNS[name]
    missing = expected - cols
    if missing:
        msg = f"{name}: missing columns {missing} (have {cols})"
        raise ValueError(msg)
    log.info("  %-12s %5d samples, columns: %s", name, len(ds), sorted(cols))

    # Extra validation for teleqna: answer indices in range
    if name == "teleqna":
        for i, row in enumerate(ds):
            n_choices = len(row["choices"])
            if not (0 <= row["answer"] < n_choices):
                msg = f"teleqna row {i}: answer {row['answer']} out of range [0, {n_choices})"
                raise ValueError(msg)


# ── Main ────────────────────────────────────────────────────────────────────


def build_all() -> dict[str, Dataset]:
    """Download and transform all configs."""
    configs: dict[str, Dataset] = {}

    log.info("Loading teletables from netop/TeleTables...")
    configs["teletables"] = load_dataset("netop/TeleTables", split="test")

    log.info("Loading teleqna from netop/TeleQnA...")
    teleqna_ds = load_dataset("netop/TeleQnA", split="test")
    # Keep only the columns the eval code needs (drop 'explaination')
    keep_cols = {"question", "choices", "answer", "subject"}
    drop_cols = [c for c in teleqna_ds.column_names if c not in keep_cols]
    if drop_cols:
        teleqna_ds = teleqna_ds.remove_columns(drop_cols)
    configs["teleqna"] = teleqna_ds

    log.info("Loading telemath from netop/TeleMath...")
    configs["telemath"] = load_dataset("netop/TeleMath", split="test")

    log.info("Loading telelogs...")
    configs["telelogs"] = load_telelogs()

    log.info("Loading 3gpp_tsg from GSMA/open_telco...")
    configs["3gpp_tsg"] = load_dataset("GSMA/open_telco", "3gpp_tsg", split="test")

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate only, do not upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var).",
    )
    args = parser.parse_args()

    configs = build_all()

    log.info("Validation:")
    for name, ds in configs.items():
        validate(name, ds)

    if args.dry_run:
        log.info("Dry-run complete — no upload performed.")
        return

    log.info("Uploading to %s ...", TARGET_REPO)
    for name, ds in configs.items():
        log.info("  Pushing config=%s ...", name)
        ds.push_to_hub(
            TARGET_REPO,
            config_name=name,
            token=args.token,
        )
    log.info("Upload complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
