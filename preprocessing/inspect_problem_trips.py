
from pathlib import Path
import pandas as pd

summary = pd.read_csv(
    "experiments/candidate_generation_summary.csv"
)

for trip_id in ["00000033", "00000054"]:
    print("\n" + "=" * 50)
    print("TRIP:", trip_id)

    row = summary[
        (summary["trip_id"].astype(str).str.zfill(8) == trip_id)
        & (summary["split"] == "train")
    ]

    print("\nGeneration summary:")
    print(row.to_string(index=False))

    gps = pd.read_csv(
        f"data/processed/train_{trip_id}.csv"
    )

    trip_dir = Path(
        f"dataset_raw/Student folder/train/{trip_id}"
    )

    nodes_file = list(trip_dir.glob("*.nodes"))[0]

    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    print("\nGPS coordinate range:")
    print(gps[["longitude", "latitude"]].agg(["min", "max"]))

    print("\nRoad network coordinate range:")
    print(nodes[["longitude", "latitude"]].agg(["min", "max"]))

    print("\nGPS observations:", len(gps))
    print("Road network nodes:", len(nodes))
