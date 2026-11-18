
"""
PathSnap candidate generation using Shapely 2 STRtree.

Inputs:
  data/processed/train_<trip_id>.csv
  data/processed/student_test_<trip_id>.csv
  dataset_raw/Student folder/<split>/<trip_id>/*.nodes, *.arcs, *.route

Outputs:
  experiments/train_features.csv
  experiments/student_test_features.csv
  experiments/candidate_generation_summary.csv

Reference labels remain provisional spatial weak labels.
Evaluator-testing data is never read.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


EARTH_RADIUS_M = 6_371_000.0
MAX_REFERENCE_DISTANCE_M = 75.0
CANDIDATES_PER_OBSERVATION = 15

# Search actual road geometries, expanding the search area if needed.
SEARCH_RADII_M = [50, 100, 200, 400, 800, 1600]


def read_whitespace(path, columns):
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def find_single_file(folder, suffix):
    files = list(folder.glob("*" + suffix))

    if len(files) != 1:
        raise ValueError(
            f"{folder}: expected one *{suffix} file, found {len(files)}"
        )

    return files[0]


def local_xy(longitude, latitude, lon0, lat0):
    """Convert longitude/latitude to approximate local metric coordinates."""

    x = (
        np.radians(np.asarray(longitude, dtype=float) - lon0)
        * EARTH_RADIUS_M
        * np.cos(np.radians(lat0))
    )

    y = (
        np.radians(np.asarray(latitude, dtype=float) - lat0)
        * EARTH_RADIUS_M
    )

    return x, y


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Calculate exact Euclidean point-to-segment distances."""

    vx = bx - ax
    vy = by - ay

    denominator = vx * vx + vy * vy
    denominator = np.where(denominator == 0, 1.0, denominator)

    t = ((px - ax) * vx + (py - ay) * vy) / denominator
    t = np.clip(t, 0.0, 1.0)

    closest_x = ax + t * vx
    closest_y = ay + t * vy

    return np.hypot(px - closest_x, py - closest_y)


def bearing_degrees(x1, y1, x2, y2):
    """Bearing clockwise from north."""

    return (
        np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360.0
    ) % 360.0


def angular_difference(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def load_trip_network(trip_dir, cleaned):
    nodes = read_whitespace(
        find_single_file(trip_dir, ".nodes"),
        ["longitude", "latitude"]
    )

    arcs = read_whitespace(
        find_single_file(trip_dir, ".arcs"),
        ["from_node_id", "to_node_id"]
    )

    route = read_whitespace(
        find_single_file(trip_dir, ".route"),
        ["arc_id"]
    )["arc_id"].to_numpy()

    nodes = nodes.apply(pd.to_numeric, errors="coerce")
    arcs = arcs.apply(pd.to_numeric, errors="coerce")

    route = pd.to_numeric(
        pd.Series(route), errors="coerce"
    ).dropna().to_numpy()

    if nodes.isna().any().any() or arcs.isna().any().any():
        raise ValueError(
            f"{trip_dir.name}: missing/non-numeric node or arc data"
        )

    arcs = arcs.astype(int)
    route = route.astype(int)

    if len(nodes) == 0 or len(arcs) == 0 or len(route) == 0:
        raise ValueError(
            f"{trip_dir.name}: empty nodes, arcs, or route"
        )

    if (arcs.to_numpy() < 0).any():
        raise ValueError(f"{trip_dir.name}: negative node ID in arcs")

    if arcs.to_numpy().max() >= len(nodes):
        raise ValueError(f"{trip_dir.name}: arc references missing node")

    if route.min() < 0 or route.max() >= len(arcs):
        raise ValueError(f"{trip_dir.name}: invalid route arc ID")

    lon0 = float(cleaned["longitude"].mean())
    lat0 = float(cleaned["latitude"].mean())

    node_x, node_y = local_xy(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy(),
        lon0,
        lat0
    )

    arc_nodes = arcs[
        ["from_node_id", "to_node_id"]
    ].to_numpy()

    ax = node_x[arc_nodes[:, 0]]
    ay = node_y[arc_nodes[:, 0]]
    bx = node_x[arc_nodes[:, 1]]
    by = node_y[arc_nodes[:, 1]]

    return ax, ay, bx, by, route, lon0, lat0


def build_spatial_index(ax, ay, bx, by):
    """Build an STRtree over actual road-segment geometries."""

    lines = [
        LineString([(x1, y1), (x2, y2)])
        for x1, y1, x2, y2 in zip(ax, ay, bx, by)
    ]

    tree = STRtree(lines)

    return lines, tree


def retrieve_candidates(
    point,
    tree,
    ax,
    ay,
    bx,
    by
):
    """
    Retrieve nearby road segments using STRtree.

    Search radius expands until at least 15 segments are found.
    Exact point-to-segment distances determine the final ranking.
    """

    candidate_ids = np.array([], dtype=int)

    for radius in SEARCH_RADII_M:
        nearby = tree.query(point.buffer(radius))

        candidate_ids = np.asarray(nearby, dtype=int)

        if len(candidate_ids) >= CANDIDATES_PER_OBSERVATION:
            break

    if len(candidate_ids) == 0:
        # Fallback: retrieve the nearest geometry.
        nearest = tree.query_nearest(point)
        candidate_ids = np.atleast_1d(nearest).astype(int)

    distances = point_to_segment_distance(
        point.x,
        point.y,
        ax[candidate_ids],
        ay[candidate_ids],
        bx[candidate_ids],
        by[candidate_ids]
    )

    order = np.argsort(distances)

    order = order[:CANDIDATES_PER_OBSERVATION]

    return candidate_ids[order], distances[order]


def process_trip(trip_dir, split, processed_dir):
    trip_id = trip_dir.name

    cleaned_path = processed_dir / f"{split}_{trip_id}.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned GPS file: {cleaned_path}"
        )

    gps = pd.read_csv(cleaned_path)

    required = {
        "observation_id",
        "longitude",
        "latitude",
        "timestamp",
        "speed_mps",
        "time_gap_seconds",
        "is_gps_gap",
        "irregular_interval_flag",
        "high_speed_flag"
    }

    missing = required.difference(gps.columns)

    if missing:
        raise ValueError(
            f"{cleaned_path}: missing columns {sorted(missing)}"
        )

    gps = gps.sort_values("timestamp").reset_index(drop=True)

    if len(gps) < 2:
        raise ValueError(
            f"{trip_id}: fewer than two GPS observations"
        )

    ax, ay, bx, by, route, lon0, lat0 = load_trip_network(
        trip_dir, gps
    )

    gps_x, gps_y = local_xy(
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy(),
        lon0,
        lat0
    )

    road_length = np.hypot(bx - ax, by - ay)

    road_bearing = bearing_degrees(ax, ay, bx, by)

    # Build spatial index over actual road segments.
    lines, tree = build_spatial_index(ax, ay, bx, by)

    # Weak spatial reference labels from route-listed arcs.
    route_arc_ids = np.unique(route)

    ref_distances = np.empty(len(gps), dtype=float)
    reference_arc_ids = np.empty(len(gps), dtype=int)

    for i, (px, py) in enumerate(zip(gps_x, gps_y)):
        distances = point_to_segment_distance(
            px,
            py,
            ax[route_arc_ids],
            ay[route_arc_ids],
            bx[route_arc_ids],
            by[route_arc_ids]
        )

        best = int(np.argmin(distances))

        reference_arc_ids[i] = int(route_arc_ids[best])
        ref_distances[i] = float(distances[best])

    # Estimate GPS movement heading.
    gps_heading = np.zeros(len(gps), dtype=float)

    for i in range(len(gps)):
        previous_i = max(0, i - 1)
        next_i = min(len(gps) - 1, i + 1)

        gps_heading[i] = bearing_degrees(
            gps_x[previous_i],
            gps_y[previous_i],
            gps_x[next_i],
            gps_y[next_i]
        )

    rows = []

    eligible_observations = 0
    too_far_from_route = 0
    reference_candidate_misses = 0

    for i, (px, py) in enumerate(zip(gps_x, gps_y)):

        if ref_distances[i] > MAX_REFERENCE_DISTANCE_M:
            too_far_from_route += 1
            continue

        eligible_observations += 1

        point = Point(px, py)

        candidate_ids, candidate_distances = retrieve_candidates(
            point,
            tree,
            ax,
            ay,
            bx,
            by
        )

        reference_arc = int(reference_arc_ids[i])

        if reference_arc not in candidate_ids:
            reference_candidate_misses += 1

        for arc_id, distance_m in zip(
            candidate_ids,
            candidate_distances
        ):

            rows.append({
                "trip_id": trip_id,
                "observation_id": int(
                    gps.loc[i, "observation_id"]
                ),
                "candidate_arc_id": int(arc_id),
                "distance_m": float(distance_m),

                "heading_diff_deg": float(
                    angular_difference(
                        gps_heading[i],
                        road_bearing[arc_id]
                    )
                ),

                "speed_mps": float(
                    gps.loc[i, "speed_mps"]
                ),

                "road_length_m": float(
                    road_length[arc_id]
                ),

                "time_gap_seconds": float(
                    gps.loc[i, "time_gap_seconds"]
                ),

                "is_gps_gap": int(
                    bool(gps.loc[i, "is_gps_gap"])
                ),

                "irregular_interval_flag": int(
                    bool(gps.loc[i, "irregular_interval_flag"])
                ),

                "high_speed_flag": int(
                    bool(gps.loc[i, "high_speed_flag"])
                ),

                "label": int(arc_id == reference_arc)
            })

    features = pd.DataFrame(rows)

    if features.empty:
        raise ValueError(
            f"{trip_id}: no candidate rows; inspect GPS/reference alignment"
        )

    summary = {
        "trip_id": trip_id,
        "split": split,
        "gps_observations": len(gps),

        "observations_with_reference_within_threshold":
            eligible_observations,

        "observations_too_far_from_reference_route":
            too_far_from_route,

        "reference_arc_missing_from_candidates":
            reference_candidate_misses,

        "candidate_miss_rate": (
            reference_candidate_misses / eligible_observations
            if eligible_observations else np.nan
        ),

        "candidate_rows": len(features),

        "positive_candidate_rows":
            int(features["label"].sum()),

        "positive_label_rate":
            float(features["label"].mean())
    }

    return features, summary


def process_split(raw_root, processed_dir, split):
    split_dir = raw_root / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing raw trip folder: {split_dir}"
        )

    trip_dirs = sorted(
        path for path in split_dir.iterdir()
        if path.is_dir()
    )

    if not trip_dirs:
        raise ValueError(
            f"No trip directories found in {split_dir}"
        )

    all_features = []
    all_summaries = []

    for index, trip_dir in enumerate(trip_dirs, start=1):

        print(
            f"[{split}] {index}/{len(trip_dirs)} - {trip_dir.name}"
        )

        feature_df, summary = process_trip(
            trip_dir,
            split,
            processed_dir
        )

        all_features.append(feature_df)
        all_summaries.append(summary)

    return (
        pd.concat(all_features, ignore_index=True),
        pd.DataFrame(all_summaries)
    )


def main():
    raw_root = Path("dataset_raw/Student folder")

    processed_dir = Path("data/processed")

    output_dir = Path("experiments")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_summary = process_split(
        raw_root,
        processed_dir,
        "train"
    )

    test_features, test_summary = process_split(
        raw_root,
        processed_dir,
        "student_test"
    )

    train_features.to_csv(
        output_dir / "train_features.csv",
        index=False
    )

    test_features.to_csv(
        output_dir / "student_test_features.csv",
        index=False
    )

    summary = pd.concat(
        [train_summary, test_summary],
        ignore_index=True
    )

    summary.to_csv(
        output_dir / "candidate_generation_summary.csv",
        index=False
    )

    print("\nCANDIDATE GENERATION COMPLETE")

    print(
        "Training candidate rows:",
        len(train_features)
    )

    print(
        "Student-test candidate rows:",
        len(test_features)
    )

    print(
        "Training positive labels:",
        int(train_features["label"].sum())
    )

    print(
        "Student-test positive labels:",
        int(test_features["label"].sum())
    )

    print("\nCandidate miss-rate summary by split:")

    print(
        summary.groupby("split")["candidate_miss_rate"].describe()
    )

    print(
        "\nSaved files in:",
        output_dir.resolve()
    )

    print(
        "\nIMPORTANT: Labels are provisional spatial weak labels. "
        "Validate route/GPS alignment before training or reporting "
        "model performance."
    )


if __name__ == "__main__":
    main()
