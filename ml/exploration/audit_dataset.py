from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dataset_raw" / "Student folder"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "student_test"


def read_file(path, columns):
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns
    )


def inspect_trip(folder):
    trip_id = folder.name

    track_path = folder / f"{trip_id}.track"
    nodes_path = folder / f"{trip_id}.nodes"
    arcs_path = folder / f"{trip_id}.arcs"
    route_path = folder / f"{trip_id}.route"

    result = {"trip_id": trip_id}

    try:
        track = read_file(
            track_path,
            ["longitude", "latitude", "timestamp"]
        )

        nodes = read_file(
            nodes_path,
            ["longitude", "latitude"]
        )

        arcs = read_file(
            arcs_path,
            ["from_node", "to_node"]
        )

        route = read_file(route_path, ["arc_id"])

        timestamps = track["timestamp"]

        time_diffs = timestamps.diff().dropna()

        valid_arc_ids = (
            route["arc_id"].between(0, len(arcs) - 1)
        )

        result.update({
            "gps_observations": len(track),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "reference_route_entries": len(route),
            "missing_track_values": int(track.isna().sum().sum()),
            "missing_node_values": int(nodes.isna().sum().sum()),
            "missing_arc_values": int(arcs.isna().sum().sum()),
            "missing_route_values": int(route.isna().sum().sum()),
            "timestamp_non_increasing": int((time_diffs <= 0).sum()),
            "timestamp_gap_count_gt_1s": int((time_diffs > 1).sum()),
            "route_arc_ids_valid": bool(valid_arc_ids.all()),
            "route_arc_id_min": int(route["arc_id"].min()),
            "route_arc_id_max": int(route["arc_id"].max()),
            "status": "OK"
        })

    except Exception as error:
        result["status"] = f"ERROR: {error}"

    return result


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(TRAIN_DIR)

    folders = sorted(
        [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    )

    results = [inspect_trip(folder) for folder in folders]

    output_dir = ROOT / "experiments"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "dataset_audit.csv"

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print("\nPATHSNAP DATASET AUDIT")
    print("=" * 50)
    print("Training trips:", len(folders))
    print("Successful inspections:", (df["status"] == "OK").sum())
    print("Failed inspections:", (df["status"] != "OK").sum())

    print("\nAVERAGE DATASET STATISTICS")
    for col in [
        "gps_observations",
        "road_nodes",
        "road_arcs",
        "reference_route_entries"
    ]:
        if col in df:
            print(f"{col}: {df[col].mean():.2f}")

    print("\nAUDIT SAVED TO:", output_file)


if __name__ == "__main__":
    main()
