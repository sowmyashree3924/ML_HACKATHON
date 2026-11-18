
import pandas as pd
import numpy as np
from pathlib import Path

trip_id = "00000033"

folder = Path(
    f"dataset_raw/Student folder/train/{trip_id}"
)

nodes = pd.read_csv(
    list(folder.glob("*.nodes"))[0],
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
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

position = 164

first_arc_id = route[position]
second_arc_id = route[position + 1]

first_arc = arcs.iloc[first_arc_id]
second_arc = arcs.iloc[second_arc_id]

end_node = int(first_arc["to_node"])
next_start_node = int(second_arc["from_node"])

lat1 = nodes.iloc[end_node]["latitude"]
lon1 = nodes.iloc[end_node]["longitude"]

lat2 = nodes.iloc[next_start_node]["latitude"]
lon2 = nodes.iloc[next_start_node]["longitude"]

R = 6371000

phi1, phi2 = np.radians([lat1, lat2])
dphi = phi2 - phi1
dlambda = np.radians(lon2 - lon1)

a = (
    np.sin(dphi / 2) ** 2
    + np.cos(phi1) * np.cos(phi2)
    * np.sin(dlambda / 2) ** 2
)

distance = 2 * R * np.arcsin(np.sqrt(a))

print("First arc:", first_arc_id)
print("First arc ending node:", end_node)
print("Second arc:", second_arc_id)
print("Second arc starting node:", next_start_node)
print("Distance between disconnected nodes:", distance, "metres")
