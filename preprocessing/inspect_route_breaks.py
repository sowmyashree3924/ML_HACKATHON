
import pandas as pd
from pathlib import Path

for trip_id in ["00000001", "00000033", "00000054"]:

    folder = Path(
        f"dataset_raw/Student folder/train/{trip_id}"
    )

    arcs = pd.read_csv(
        list(folder.glob("*.arcs"))[0],
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    route = pd.read_csv(
        list(folder.glob("*.route"))[0],
        sep=r"\s+",
        header=None
    ).iloc[:, 0].astype(int).to_numpy()

    route_arcs = arcs.iloc[route].reset_index(drop=True)

    end_nodes = route_arcs["to_node"].to_numpy()
    start_nodes = route_arcs["from_node"].to_numpy()

    breaks = [
        i for i in range(len(route) - 1)
        if end_nodes[i] != start_nodes[i + 1]
    ]

    print("\nTRIP:", trip_id)
    print("Number of breaks:", len(breaks))

    for i in breaks:
        print("\nBreak between route positions:", i, "and", i + 1)
        print("First arc ID:", route[i])
        print("First arc:", route_arcs.iloc[i].to_dict())
        print("Next arc ID:", route[i + 1])
        print("Next arc:", route_arcs.iloc[i + 1].to_dict())
