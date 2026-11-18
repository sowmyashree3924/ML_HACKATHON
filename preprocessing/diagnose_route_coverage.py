
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS = 6371000


def local_xy(lon, lat, lon0, lat0):
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS * np.cos(
        np.radians(lat0)
    )
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS
    return x, y


def segment_distance(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    denom = np.where(denom == 0, 1, denom)
    t = np.clip(
        ((px - ax) * vx + (py - ay) * vy) / denom, 0, 1
    )
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def inspect_trip(trip_id):
    base = Path(
        f"dataset_raw/Student folder/train/{trip_id}"
    )

    gps = pd.read_csv(
        f"data/processed/train_{trip_id}.csv"
    )

    nodes = pd.read_csv(
        list(base.glob("*.nodes"))[0],
        sep=r"\s+", header=None,
        names=["lon", "lat"]
    )

    arcs = pd.read_csv(
        list(base.glob("*.arcs"))[0],
        sep=r"\s+", header=None,
        names=["start", "end"]
    )

    route = pd.read_csv(
        list(base.glob("*.route"))[0],
        sep=r"\s+", header=None
    ).iloc[:, 0].astype(int).to_numpy()

    lon0 = gps.longitude.mean()
    lat0 = gps.latitude.mean()

    gx, gy = local_xy(
        gps.longitude, gps.latitude, lon0, lat0
    )
    nx, ny = local_xy(
        nodes.lon, nodes.lat, lon0, lat0
    )

    arc_array = arcs.to_numpy(dtype=int)

    ax, ay = nx[arc_array[:, 0]], ny[arc_array[:, 0]]
    bx, by = nx[arc_array[:, 1]], ny[arc_array[:, 1]]

    route_ids = np.unique(route)

    print("\nTRIP:", trip_id)
    print("GPS observations:", len(gps))
    print("Total network arcs:", len(arcs))
    print("Route entries:", len(route))
    print("Unique route arcs:", len(route_ids))

    for name, ids in [
        ("ALL NETWORK ARCS", np.arange(len(arcs))),
        ("REFERENCE ROUTE ARCS", route_ids)
    ]:
        midpoint = np.column_stack([
            (ax[ids] + bx[ids]) / 2,
            (ay[ids] + by[ids]) / 2
        ])

        model = NearestNeighbors(
            n_neighbors=min(100, len(ids))
        ).fit(midpoint)

        _, indexes = model.kneighbors(
            np.column_stack([gx, gy])
        )

        distances = []

        for i, (px, py) in enumerate(zip(gx, gy)):
            candidates = ids[indexes[i]]

            d = segment_distance(
                px, py,
                ax[candidates], ay[candidates],
                bx[candidates], by[candidates]
            )
            distances.append(d.min())

        distances = np.asarray(distances)

        print("\n", name)
        print("Distance percentiles (metres):")
        print(np.percentile(distances, [0, 25, 50, 75, 90, 95, 99, 100]))
        print("Percentage beyond 75m:", round(
            100 * np.mean(distances > 75), 2
        ))


for trip in ["00000033", "00000054"]:
    inspect_trip(trip)
