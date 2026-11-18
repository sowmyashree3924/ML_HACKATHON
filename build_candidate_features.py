"""
PathSnap baseline candidate-feature generator.

Reads trip folders containing .track, .nodes, .arcs and .route files.
Creates candidate-level feature tables for the 80 training trips and
10 student-testing trips. It deliberately does NOT read evaluator-testing data.

Labeling assumption:
For each GPS observation, the closest segment among that trip's reference
route arcs (within MAX_LABEL_DISTANCE_M) is treated as the reference arc.
This is a practical baseline alignment, NOT proof of exact GPS-to-route labels.
Inspect the generated miss-rate/label coverage before using the CSVs.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EARTH_M_PER_DEG = 111320.0
MAX_LABEL_DISTANCE_M = 75.0
MIDPOINT_NEIGHBORS = 50
CANDIDATES_PER_OBSERVATION = 15


def read_table(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, comment="#")


def point_segment_distances(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    denom = np.where(denom == 0, 1.0, denom)
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / denom, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy)


def bearing_deg(x1, y1, x2, y2):
    # Planar bearing, clockwise from north.
    return (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


def angle_difference(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def find_one(folder, suffix):
    matches = list(folder.glob("*" + suffix))
    if len(matches) != 1:
        raise ValueError(
            f"{folder}: expected exactly one *{suffix} file, found {len(matches)}"
        )
    return matches[0]


def process_trip(trip_dir, split):
    trip_id = trip_dir.name
    track = read_table(find_one(trip_dir, ".track"),
                       ["longitude", "latitude", "timestamp"])
    nodes = read_table(find_one(trip_dir, ".nodes"), ["longitude", "latitude"])
    arcs = read_table(find_one(trip_dir, ".arcs"), ["from_node_id", "to_node_id"])
    route = read_table(find_one(trip_dir, ".route"), ["arc_id"])["arc_id"].to_numpy()

    track = track.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nodes = nodes.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    arcs = arcs.dropna().astype(int).reset_index(drop=True)
    route = route[np.isfinite(route)].astype(int)

    if len(track) < 2 or len(nodes) == 0 or len(arcs) == 0 or len(route) == 0:
        raise ValueError(f"{trip_id}: one or more input files are empty/too small")
    if (arcs[["from_node_id", "to_node_id"]].to_numpy() < 0).any():
        raise ValueError(f"{trip_id}: negative node ID in arcs")
    if arcs[["from_node_id", "to_node_id"]].to_numpy().max() >= len(nodes):
        raise ValueError(f"{trip_id}: arc references a node outside .nodes")
    if route.min() < 0 or route.max() >= len(arcs):
        raise ValueError(f"{trip_id}: route references an arc outside .arcs")

    # Local metric projection for this trip (suitable for a city-scale trip).
    lon0 = float(track["longitude"].mean())
    lat0 = float(track["latitude"].mean())
    sx = EARTH_M_PER_DEG * np.cos(np.radians(lat0))
    sy = EARTH_M_PER_DEG
    gps_x = (track["longitude"].to_numpy() - lon0) * sx
    gps_y = (track["latitude"].to_numpy() - lat0) * sy
    node_x = (nodes["longitude"].to_numpy() - lon0) * sx
    node_y = (nodes["latitude"].to_numpy() - lat0) * sy

    arc_arr = arcs[["from_node_id", "to_node_id"]].to_numpy()
    ax, ay = node_x[arc_arr[:, 0]], node_y[arc_arr[:, 0]]
    bx, by = node_x[arc_arr[:, 1]], node_y[arc_arr[:, 1]]
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    road_length = np.hypot(bx - ax, by - ay)
    road_bearing = bearing_deg(ax, ay, bx, by)

    # Candidate lookup by nearby arc midpoints, then rank by point-to-segment distance.
    midpoint_xy = np.column_stack([mx, my])
    nn = NearestNeighbors(
        n_neighbors=min(MIDPOINT_NEIGHBORS, len(arcs)), algorithm="auto"
    ).fit(midpoint_xy)
    _, nearby_indices = nn.kneighbors(np.column_stack([gps_x, gps_y]))

    route_ids = np.unique(route)
    route_ax, route_ay = ax[route_ids], ay[route_ids]
    route_bx, route_by = bx[route_ids], by[route_ids]

    timestamps = track["timestamp"].to_numpy(dtype=float)
    speed = np.zeros(len(track), dtype=float)
    gps_heading = np.zeros(len(track), dtype=float)
    for i in range(len(track)):
        j = min(i + 1, len(track) - 1)
        k = max(i - 1, 0)
        dt = timestamps[j] - timestamps[k]
        d = np.hypot(gps_x[j] - gps_x[k], gps_y[j] - gps_y[k])
        if dt > 0:
            speed[i] = d / dt
        gps_heading[i] = bearing_deg(gps_x[k], gps_y[k], gps_x[j], gps_y[j])

    rows = []
    miss_count = 0
    skipped_far = 0
    for obs_id, (px, py) in enumerate(zip(gps_x, gps_y)):
        # Weak reference alignment: closest route arc to this GPS observation.
        route_dist = point_segment_distances(
            px, py, route_ax, route_ay, route_bx, route_by
        )
        best_route_pos = int(np.argmin(route_dist))
        reference_arc = int(route_ids[best_route_pos])
        reference_distance = float(route_dist[best_route_pos])
        if reference_distance > MAX_LABEL_DISTANCE_M:
            skipped_far += 1
            continue

        cand = nearby_indices[obs_id]
        dists = point_segment_distances(px, py, ax[cand], ay[cand], bx[cand], by[cand])
        order = np.argsort(dists)[:min(CANDIDATES_PER_OBSERVATION, len(cand))]
        cand = cand[order]
        dists = dists[order]

        if reference_arc not in cand:
            miss_count += 1

        for arc_id, dist in zip(cand, dists):
            rows.append({
                "trip_id": trip_id,
                "split": split,
                "observation_id": obs_id,
                "candidate_arc_id": int(arc_id),
                "distance_m": float(dist),
                "heading_diff_deg": float(angle_difference(
                    gps_heading[obs_id], road_bearing[arc_id]
                )),
                "speed_mps": float(speed[obs_id]),
                "road_length_m": float(road_length[arc_id]),
                "label": int(int(arc_id) == reference_arc),
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"{trip_id}: no candidate rows generated")
    summary = {
        "trip_id": trip_id,
        "split": split,
        "gps_observations": len(track),
        "candidate_observations_labeled": len(track) - skipped_far,
        "candidate_rows": len(frame),
        "positive_rows": int(frame["label"].sum()),
        "observations_reference_arc_missed_by_candidates": miss_count,
        "observations_too_far_from_reference_route": skipped_far,
    }
    return frame, summary


def process_split(root, folder_name, split):
    base = root / folder_name
    if not base.exists():
        raise FileNotFoundError(f"Missing folder: {base}")
    trip_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if not trip_dirs:
        raise ValueError(f"No trip folders found under {base}")
    frames, summaries = [], []
    for i, trip_dir in enumerate(trip_dirs, 1):
        print(f"[{split}] {i}/{len(trip_dirs)}: {trip_dir.name}")
        frame, summary = process_trip(trip_dir, split)
        frames.append(frame)
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"dataset_raw\Student folder",
                        help="Folder containing train and student_test")
    parser.add_argument("--out", default="experiments")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, train_summary = process_split(root, "train", "train")
    test, test_summary = process_split(root, "student_test", "student_test")

    train.to_csv(out / "train_features.csv", index=False)
    test.to_csv(out / "student_test_features.csv", index=False)
    summary = pd.concat([train_summary, test_summary], ignore_index=True)
    summary.to_csv(out / "candidate_generation_summary.csv", index=False)

    for name, frame in [("train", train), ("student_test", test)]:
        print(f"\n{name}: {len(frame):,} candidate rows")
        print(f"positive labels: {int(frame['label'].sum()):,}")
        print(f"positive rate: {frame['label'].mean():.4f}")
    print("\nSaved:")
    print(out / "train_features.csv")
    print(out / "student_test_features.csv")
    print(out / "candidate_generation_summary.csv")
    print("\nIMPORTANT: review candidate_generation_summary.csv and validate route/GPS alignment.")

if __name__ == "__main__":
    main()
