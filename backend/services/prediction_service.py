"""Lazy model loading and map-matching inference service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate import predict_trip

from .data_service import PROJECT_ROOT, get_trip

MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL_PATH", PROJECT_ROOT / "models" / "pathsnap_model.joblib"))
OUTPUT_DIR = Path(os.getenv("PATHSNAP_OUTPUT_DIR", PROJECT_ROOT / "submissions" / "api"))
_artifact: dict[str, Any] | None = None


def model_is_available() -> bool:
    """Return whether the trained model checkpoint exists."""
    return MODEL_PATH.is_file()


def load_model() -> dict[str, Any]:
    """Load the trained model once per API process."""
    global _artifact
    if _artifact is None:
        if not model_is_available():
            raise FileNotFoundError(
                f"Model checkpoint not found at {MODEL_PATH}. Run `python main.py train` first."
            )
        import joblib

        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def match_trip(trip_id: str) -> dict[str, Any]:
    """Run candidate scoring and return point matches plus full route geometry."""
    trip = get_trip(trip_id)
    artifact = load_model()
    result = predict_trip(trip.folder, artifact, OUTPUT_DIR)
    records = result.to_dict(orient="records")
    matches = [
        {key: value.item() if isinstance(value, np.generic) else value for key, value in row.items()}
        for row in records
    ]
    route_arc_ids = result.attrs.get("route_arc_ids", [])
    endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    coordinates = trip.nodes[["longitude", "latitude"]].to_numpy()
    route = [
        {
            "sequence_index": index,
            "arc_id": int(arc_id),
            "from_longitude": float(coordinates[endpoints[arc_id, 0], 0]),
            "from_latitude": float(coordinates[endpoints[arc_id, 0], 1]),
            "to_longitude": float(coordinates[endpoints[arc_id, 1], 0]),
            "to_latitude": float(coordinates[endpoints[arc_id, 1], 1]),
        }
        for index, arc_id in enumerate(route_arc_ids)
    ]
    return {"matches": matches, "route": route}
