"""Dataset paths and trip metadata for the API."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from src.parser import discover_trips, parse_trip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(os.getenv("PATHSNAP_RAW_ROOT", PROJECT_ROOT / "dataset_raw" / "Student folder"))
TEST_DIR = RAW_ROOT / "student_test"


def get_student_test_trips() -> list[str]:
    """Return trip IDs available to the student-side evaluation interface."""
    return [path.name for path in discover_trips(TEST_DIR)]


def get_trip(trip_id: str) -> Any:
    """Load a validated student-test trip, rejecting path traversal IDs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
        raise ValueError("Invalid trip ID")
    folder = (TEST_DIR / trip_id).resolve()
    if folder.parent != TEST_DIR.resolve():
        raise ValueError("Invalid trip ID")
    if not folder.is_dir():
        raise FileNotFoundError(f"Trip {trip_id} not found")
    return parse_trip(folder, require_route=False)
