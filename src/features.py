"""Spatial candidate generation and route-level feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .parser import TripData, parse_trip

EARTH_RADIUS_M = 6_371_000.0
FEATURE_COLUMNS = [
    "distance_m", "heading_diff_deg", "speed_mps", "road_length_m",
    "time_gap_seconds", "gps_step_m", "candidate_bearing_deg",
    "endpoint_distance_m", "local_density",
]


def local_xy(longitude: np.ndarray, latitude: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Project nearby longitude/latitude values into local metric coordinates."""
    x = np.radians(np.asarray(longitude, dtype=float) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(latitude, dtype=float) - lat0) * EARTH_RADIUS_M
    return x, y


def point_segment_distance(px: float, py: float, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Compute vectorized Euclidean distances from a point to road segments."""
    vx, vy = bx - ax, by - ay
    denom = np.maximum(vx * vx + vy * vy, 1e-12)
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / denom, 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def bearing_degrees(x1: np.ndarray | float, y1: np.ndarray | float, x2: np.ndarray | float, y2: np.ndarray | float) -> np.ndarray:
    """Return compass bearings in degrees, clockwise from north."""
    return (np.degrees(np.arctan2(np.asarray(x2) - y1, np.asarray(x2) - x1)) + 360.0) % 360.0


def angular_difference(a: float, b: np.ndarray) -> np.ndarray:
    """Return the smallest absolute angular difference in degrees."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def build_candidate_features(
    trip: TripData,
    candidates_per_observation: int = 15,
    midpoint_neighbors: int = 50,
    max_label_distance_m: float = 75.0,
    include_labels: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Build candidate-arc rows for each GPS point.

    Training labels use the nearest supplied route arc as a *weak spatial
    reference*. They are not asserted to be independently verified labels.
    """
    if include_labels and not len(trip.route):
        raise ValueError(f"{trip.trip_id}: labels requested but no route was supplied")
    track, nodes = trip.track, trip.nodes
    lon0, lat0 = float(track.longitude.mean()), float(track.latitude.mean())
    gx, gy = local_xy(track.longitude.to_numpy(), track.latitude.to_numpy(), lon0, lat0)
    nx, ny = local_xy(nodes.longitude.to_numpy(), nodes.latitude.to_numpy(), lon0, lat0)
    ends = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    ax, ay, bx, by = nx[ends[:, 0]], ny[ends[:, 0]], nx[ends[:, 1]], ny[ends[:, 1]]
    lengths = np.hypot(bx - ax, by - ay)
    bearings = bearing_degrees(ax, ay, bx, by)
    tree_points = np.column_stack(((ax + bx) / 2, (ay + by) / 2))
    nn = NearestNeighbors(n_neighbors=min(midpoint_neighbors, len(ends))).fit(tree_points)
    nearby = nn.kneighbors(np.column_stack((gx, gy)), return_distance=False)
    timestamps = track.timestamp.to_numpy(dtype=float)
    dt = np.diff(timestamps, prepend=timestamps[0])
    dx = np.diff(gx, prepend=gx[0])
    dy = np.diff(gy, prepend=gy[0])
    step = np.hypot(dx, dy)
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    gps_heading = bearing_degrees(np.roll(gx, 1), np.roll(gy, 1), gx, gy)
    if len(gx) > 1:
        gps_heading[0] = gps_heading[1]
    route_arc_ids = np.unique(trip.route.astype(int)) if len(trip.route) else np.array([], dtype=int)
    rows: list[dict[str, float | int | str]] = []
    misses = 0
    skipped = 0
    for obs_id, (px, py) in enumerate(zip(gx, gy)):
        ref_arc, ref_dist = -1, np.nan
        if len(route_arc_ids):
            route_distances = point_segment_distance(px, py, ax[route_arc_ids], ay[route_arc_ids], bx[route_arc_ids], by[route_arc_ids])
            ref_pos = int(np.argmin(route_distances))
            ref_arc, ref_dist = int(route_arc_ids[ref_pos]), float(route_distances[ref_pos])
            if include_labels and ref_dist > max_label_distance_m:
                skipped += 1
                continue
        candidate_ids = nearby[obs_id]
        distances = point_segment_distance(px, py, ax[candidate_ids], ay[candidate_ids], bx[candidate_ids], by[candidate_ids])
        order = np.argsort(distances)[:min(candidates_per_observation, len(candidate_ids))]
        candidate_ids, distances = candidate_ids[order], distances[order]
        if include_labels and ref_arc not in candidate_ids:
            misses += 1
        for arc_id, distance in zip(candidate_ids, distances):
            vx, vy = bx[arc_id] - ax[arc_id], by[arc_id] - ay[arc_id]
            denominator = max(vx * vx + vy * vy, 1e-12)
            segment_fraction = float(np.clip(((px - ax[arc_id]) * vx + (py - ay[arc_id]) * vy) / denominator, 0.0, 1.0))
            rows.append({
                "trip_id": trip.trip_id, "observation_id": obs_id,
                "candidate_arc_id": int(arc_id), "distance_m": float(distance),
                "heading_diff_deg": float(angular_difference(gps_heading[obs_id], np.asarray([bearings[arc_id]]))[0]),
                "speed_mps": float(speed[obs_id]), "road_length_m": float(lengths[arc_id]),
                "time_gap_seconds": float(dt[obs_id]), "gps_step_m": float(step[obs_id]),
                "candidate_bearing_deg": float(bearings[arc_id]),
                "endpoint_distance_m": float(min(np.hypot(px-ax[arc_id], py-ay[arc_id]), np.hypot(px-bx[arc_id], py-by[arc_id]))),
                "segment_fraction": segment_fraction,
                "local_density": float(len(candidate_ids)),
                **({"label": int(int(arc_id) == ref_arc)} if include_labels else {}),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"{trip.trip_id}: no candidate features produced")
    summary: dict[str, int | float | str] = {
        "trip_id": trip.trip_id, "observations": len(track),
        "candidate_observations": int(result.observation_id.nunique()),
        "candidate_rows": len(result), "candidate_reference_misses": misses,
        "observations_beyond_label_distance": skipped,
        "candidate_miss_rate": misses / max(result.observation_id.nunique(), 1),
    }
    return result, summary


def trip_feature_summary(trip: TripData) -> dict[str, float | int | str]:
    """Create compact graph, route, and trajectory statistics for EDA."""
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    intervals = track.timestamp.diff().dropna()
    ends = arcs.to_numpy(dtype=int)
    degree = np.bincount(ends.ravel(), minlength=len(nodes))
    out: dict[str, float | int | str] = {
        "trip_id": trip.trip_id, "gps_observations": len(track), "node_count": len(nodes),
        "arc_count": len(arcs), "route_arc_count": len(trip.route),
        "mean_sampling_interval_s": float(intervals.mean()) if len(intervals) else 0.0,
        "irregular_interval_fraction": float((np.abs(intervals - 1.0) > 0.1).mean()) if len(intervals) else 0.0,
        "isolated_node_count": int((degree == 0).sum()),
        "edge_density": float(len(arcs) / max(len(nodes) * (len(nodes) - 1), 1)),
        "longitude_span": float(track.longitude.max() - track.longitude.min()),
        "latitude_span": float(track.latitude.max() - track.latitude.min()),
    }
    return out


def build_split_features(split_dir: str | Path, include_labels: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all sample folders in a split and concatenate features/reports."""
    from .parser import discover_trips

    all_rows, summaries = [], []
    for folder in discover_trips(split_dir):
        trip = parse_trip(folder, require_route=include_labels)
        rows, _ = build_candidate_features(trip, include_labels=include_labels)
        all_rows.append(rows)
        summaries.append(trip_feature_summary(trip))
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summaries)
