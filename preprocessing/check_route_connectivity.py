
from pathlib import Path
import numpy as np
import pandas as pd

EARTH_RADIUS = 6371000


def inspect_trip(trip_id):
    folder = Path(
        f"dataset_raw/Student folder/train/{trip_id}"
    )

    nodes = pd.read_csv(
        list(folder.glob("*.nodes"))[0],
        sep=r"\s+", header=None,
        names=["lon", "lat"]
    )

    arcs = pd.read_csv(
        list(folder.glob("*.arcs"))[0],
        sep=r"\s+", header=None,
        names=["from_node", "to_node"]
    )

    route = pd.read_csv(
        list(folder.glob("*.route"))[0],
        sep=r"\s+", header=None
    ).iloc[:, 0].astype(int).to_numpy()

    # Convert coordinates to approximate local metres
    lon0 = nodes["lon"].mean()
    lat0 = nodes["lat"].mean()

    x = (
        np.radians(nodes["lon"] - lon0)
        * EARTH_RADIUS * np.cos(np.radians(lat0))
    )
    y = np.radians(nodes["lat"] - lat0) * EARTH_RADIUS

    route_arcs = arcs.iloc[route].reset_index(drop=True)

    start = route_arcs["from_node"].to_numpy()
    end = route_arcs["to_node"].to_numpy()

    # Check if the end node of one arc is the
    # start node of the next arc
    discontinuities = np.sum(end[:-1] != start[1:])

    # Calculate reference route arc lengths
    lengths = np.hypot(
        x.iloc[end].to_numpy() - x.iloc[start].to_numpy(),
        y.iloc[end].to_numpy() - y.iloc[start].to_numpy()
    )

    print("\n" + "=" * 50)
    print("TRIP:", trip_id)
    print("Route entries:", len(route))
    print("Unique route arcs:", len(np.unique(route)))
    print("Route discontinuities:", discontinuities)

    print("\nRoute arc length statistics (metres):")
    print(pd.Series(lengths).describe())

    print("\nFirst 10 route arc lengths:")
    print(np.round(lengths[:10], 2))


for trip_id in ["00000001", "00000033", "00000054"]:
    inspect_trip(trip_id)
