"""Robust readers for PathSnap's whitespace-delimited trip files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class TripData:
    """Parsed files belonging to one trip."""

    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray
    folder: Path


def _single_file(folder: Path, suffix: str) -> Path:
    """Return the only file with ``suffix`` in a trip folder."""
    matches = list(folder.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{folder}: expected one *{suffix} file, found {len(matches)}"
        )
    return matches[0]


def _read_numeric(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a headerless whitespace-delimited numeric table."""
    try:
        frame = pd.read_csv(
            path, sep=r"\s+", header=None, comment="#", on_bad_lines="skip"
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    if frame.empty or frame.shape[1] < len(columns):
        raise ValueError(
            f"{path}: expected at least {len(columns)} columns, got {frame.shape[1]}"
        )
    frame = frame.iloc[:, : len(columns)].copy()
    frame.columns = columns
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def parse_trip(folder: str | Path, require_route: bool = True) -> TripData:
    """Parse and validate one sample directory; route may be omitted at inference."""
    trip_dir = Path(folder)
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {trip_dir}")
    track = _read_numeric(_single_file(trip_dir, ".track"), ["longitude", "latitude", "timestamp"])
    nodes = _read_numeric(_single_file(trip_dir, ".nodes"), ["longitude", "latitude"])
    arcs = _read_numeric(_single_file(trip_dir, ".arcs"), ["from_node_id", "to_node_id"])
    route_path = next(iter(trip_dir.glob("*.route")), None) if require_route else None
    route = np.array([], dtype=np.int64)
    if route_path is None:
        if require_route:
            raise FileNotFoundError(f"Missing .route file in {trip_dir}")
    else:
        route_frame = _read_numeric(route_path, ["arc_id"])
        route = route_frame["arc_id"].to_numpy(dtype=np.int64)
    if len(track) < 2 or nodes.empty or arcs.empty:
        raise ValueError(f"{trip_dir.name}: track, nodes, or arcs are empty/too small")
    if not track["longitude"].between(-180, 180).all() or not track["latitude"].between(-90, 90).all():
        raise ValueError(f"{trip_dir.name}: invalid longitude or latitude")
    if not np.allclose(arcs.to_numpy(), np.round(arcs.to_numpy())):
        raise ValueError(f"{trip_dir.name}: arc endpoints must be integer node IDs")
    arcs = arcs.astype("int64")
    endpoints = arcs.to_numpy()
    if endpoints.min() < 0 or endpoints.max() >= len(nodes):
        raise ValueError(f"{trip_dir.name}: an arc references a missing node")
    if len(route) and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"{trip_dir.name}: route references a missing arc")
    track = track.sort_values("timestamp", kind="stable").drop_duplicates().reset_index(drop=True)
    return TripData(trip_dir.name, track, nodes, arcs, route, trip_dir)


def discover_trips(split_dir: str | Path) -> list[Path]:
    """Return sorted child directories containing trip samples."""
    root = Path(split_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {root}")
    trips = sorted(path for path in root.iterdir() if path.is_dir())
    if not trips:
        raise ValueError(f"No trip directories found under {root}")
    return trips


def parse_many(paths: Iterable[str | Path], require_route: bool = True) -> list[TripData]:
    """Parse several trips, preserving clear errors for malformed samples."""
    return [parse_trip(path, require_route=require_route) for path in paths]
