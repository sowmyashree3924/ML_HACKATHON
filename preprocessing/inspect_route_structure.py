
from pathlib import Path
import pandas as pd
import numpy as np

for trip_id in ["00000033", "00000054", "00000001"]:
    folder = Path(
        f"dataset_raw/Student folder/train/{trip_id}"
    )

    route_file = list(folder.glob("*.route"))[0]
    arcs_file = list(folder.glob("*.arcs"))[0]
    nodes_file = list(folder.glob("*.nodes"))[0]

    route = pd.read_csv(
        route_file, sep=r"\s+", header=None
    ).iloc[:, 0].astype(int)

    arcs = pd.read_csv(
        arcs_file, sep=r"\s+", header=None,
        names=["from_node", "to_node"]
    )

    nodes = pd.read_csv(
        nodes_file, sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )

    gps = pd.read_csv(
        f"data/processed/train_{trip_id}.csv"
    )

    print("\n" + "=" * 55)
    print("TRIP:", trip_id)

    print("GPS observations:", len(gps))
    print("Network nodes:", len(nodes))
    print("Network arcs:", len(arcs))
    print("Route entries:", len(route))
    print("Unique route arcs:", route.nunique())
    print("Repeated route entries:", route.duplicated().sum())

    route_arcs = arcs.iloc[route.to_numpy()]

    route_node_ids = np.unique(
        route_arcs.to_numpy().flatten()
    )

    route_nodes = nodes.iloc[route_node_ids]

    print("\nGPS bounds:")
    print(gps[["longitude", "latitude"]].agg(["min", "max"]))

    print("\nRoute-node bounds:")
    print(route_nodes.agg(["min", "max"]))

    print("\nFirst 15 route arc IDs:")
    print(route.head(15).tolist())

    print("\nFirst 15 route arcs:")
    print(route_arcs.head(15).to_string(index=False))
