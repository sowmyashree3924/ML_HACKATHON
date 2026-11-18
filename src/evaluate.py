"""Model inference, graph-aware HMM decoding, and route output."""

from __future__ import annotations

import heapq
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_candidate_features, local_xy
from .parser import TripData, discover_trips, parse_trip


class RoadNetworkRouter:
    """Bounded directed shortest-path queries over a trip's road network."""

    def __init__(self, trip: TripData, cache_size: int = 128) -> None:
        """Build weighted adjacency from directed arc endpoints and coordinates."""
        lon0 = float(trip.track.longitude.mean())
        lat0 = float(trip.track.latitude.mean())
        x, y = local_xy(trip.nodes.longitude.to_numpy(), trip.nodes.latitude.to_numpy(), lon0, lat0)
        self.endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
        self.lengths = np.hypot(x[self.endpoints[:, 1]] - x[self.endpoints[:, 0]], y[self.endpoints[:, 1]] - y[self.endpoints[:, 0]])
        self.adjacency: list[list[tuple[int, float, int]]] = [[] for _ in range(len(trip.nodes))]
        for arc_id, ((source, target), length) in enumerate(zip(self.endpoints, self.lengths)):
            self.adjacency[int(source)].append((int(target), float(length), arc_id))
        self.cache_size = cache_size
        self._trees: OrderedDict[int, tuple[float, dict[int, float], dict[int, tuple[int, int]]]] = OrderedDict()

    def _tree(self, source: int, cutoff_m: float) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
        """Return shortest distances and predecessor arcs from a source node."""
        cached = self._trees.get(source)
        if cached is not None and cached[0] >= cutoff_m:
            self._trees.move_to_end(source)
            return cached[1], cached[2]
        distances = {source: 0.0}
        parents: dict[int, tuple[int, int]] = {}
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node) or distance > cutoff_m:
                continue
            for next_node, weight, arc_id in self.adjacency[node]:
                candidate = distance + weight
                if candidate < distances.get(next_node, float("inf")) and candidate <= cutoff_m:
                    distances[next_node] = candidate
                    parents[next_node] = (node, arc_id)
                    heapq.heappush(queue, (candidate, next_node))
        self._trees[source] = (cutoff_m, distances, parents)
        self._trees.move_to_end(source)
        while len(self._trees) > self.cache_size:
            self._trees.popitem(last=False)
        return distances, parents

    def shortest_path(self, source: int, target: int, cutoff_m: float) -> tuple[float, list[int]] | None:
        """Return directed network distance and connecting arc IDs when reachable."""
        if source == target:
            return 0.0, []
        distances, parents = self._tree(source, cutoff_m)
        if target not in distances:
            return None
        route: list[int] = []
        node = target
        while node != source:
            previous, arc_id = parents[node]
            route.append(arc_id)
            node = previous
        route.reverse()
        return distances[target], route

    def transition(self, previous: pd.Series, current: pd.Series, gps_step_m: float) -> tuple[float, list[int]]:
        """Score road distance against GPS displacement and return path arcs."""
        previous_arc = int(previous.candidate_arc_id)
        current_arc = int(current.candidate_arc_id)
        previous_fraction = float(previous.get("segment_fraction", 0.5))
        current_fraction = float(current.get("segment_fraction", 0.5))
        source = int(self.endpoints[previous_arc, 1])
        target = int(self.endpoints[current_arc, 0])
        expected_distance = max(float(gps_step_m), 0.0)
        cutoff = min(5000.0, max(250.0, expected_distance * 3.0 + 150.0))

        if previous_arc == current_arc and current_fraction >= previous_fraction:
            distance = (current_fraction - previous_fraction) * float(self.lengths[previous_arc])
            connector: list[int] = []
            path_arcs = [previous_arc]
        else:
            path = self.shortest_path(source, target, cutoff)
            if path is None:
                # A high fixed penalty still lets strong emissions recover when
                # the road graph is incomplete or one-way constraints disagree.
                return -12.0, [previous_arc, current_arc]
            connector_distance, connector = path
            distance = (
                (1.0 - previous_fraction) * float(self.lengths[previous_arc])
                + connector_distance
                + current_fraction * float(self.lengths[current_arc])
            )
            path_arcs = [previous_arc, *connector, current_arc]

        scale = max(25.0, expected_distance * 0.5)
        score = -abs(distance - expected_distance * 1.15) / scale
        return float(score), path_arcs


def viterbi_decode(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    trip: TripData,
    beam_width: int = 8,
) -> tuple[pd.DataFrame, list[int]]:
    """Run HMM/Viterbi with classifier emissions and directed shortest paths."""
    work = candidates.copy()
    work["probability"] = np.clip(probabilities, 1e-8, 1.0)
    groups = [
        group.nlargest(beam_width, "probability").reset_index(drop=True)
        for _, group in work.groupby("observation_id", sort=True)
    ]
    if not groups:
        return work.iloc[:0].copy(), []
    router = RoadNetworkRouter(trip)
    scores: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    scores.append(np.log(groups[0].probability.to_numpy()))
    backpointers.append(np.full(len(groups[0]), -1, dtype=int))
    for previous, current in zip(groups[:-1], groups[1:]):
        transition = np.zeros((len(previous), len(current)), dtype=float)
        gps_step = float(current.gps_step_m.iloc[0]) if "gps_step_m" in current else 0.0
        for i, (_, previous_state) in enumerate(previous.iterrows()):
            for j, (_, current_state) in enumerate(current.iterrows()):
                transition[i, j] = router.transition(previous_state, current_state, gps_step)[0]
        next_scores = scores[-1][:, None] + transition
        backpointers.append(np.argmax(next_scores, axis=0))
        scores.append(np.max(next_scores, axis=0) + np.log(current.probability.to_numpy()))
    path = np.zeros(len(groups), dtype=int)
    path[-1] = int(np.argmax(scores[-1]))
    for idx in range(len(groups) - 1, 0, -1):
        path[idx - 1] = backpointers[idx][path[idx]]
    selected = [group.iloc[state] for group, state in zip(groups, path)]
    selected_frame = pd.DataFrame(selected).reset_index(drop=True)
    route_arc_ids: list[int] = []
    if len(selected_frame):
        route_arc_ids.append(int(selected_frame.candidate_arc_id.iloc[0]))
    for index in range(1, len(selected_frame)):
        previous, current = selected_frame.iloc[index - 1], selected_frame.iloc[index]
        gps_step = float(current.gps_step_m) if "gps_step_m" in current else 0.0
        _, segment_path = router.transition(previous, current, gps_step)
        for arc_id in segment_path:
            if not route_arc_ids or route_arc_ids[-1] != int(arc_id):
                route_arc_ids.append(int(arc_id))
    return selected_frame, route_arc_ids


def predict_trip(trip_dir: str | Path, artifact: dict[str, Any], output_dir: str | Path) -> pd.DataFrame:
    """Predict one trip and write observation-level matched road segments."""
    trip = parse_trip(trip_dir, require_route=False)
    candidates, _ = build_candidate_features(trip, include_labels=False)
    features = candidates.reindex(columns=artifact.get("feature_columns", FEATURE_COLUMNS))
    probabilities = artifact["model"].predict_proba(features)
    classes = artifact["model"].named_steps["classifier"].classes_
    positive_index = list(classes).index(1) if 1 in classes else 0
    probabilities = probabilities[:, positive_index]
    selected, route_arc_ids = viterbi_decode(candidates, probabilities, trip)
    ends = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    selected = selected.sort_values("observation_id").reset_index(drop=True)
    track = trip.track.reset_index(drop=True)
    node_coordinates = trip.nodes[["longitude", "latitude"]].to_numpy()
    result = pd.DataFrame({
        "trip_id": trip.trip_id,
        "observation_id": selected.observation_id.astype(int),
        "timestamp": track.loc[selected.observation_id.astype(int), "timestamp"].to_numpy(),
        "gps_longitude": track.loc[selected.observation_id.astype(int), "longitude"].to_numpy(),
        "gps_latitude": track.loc[selected.observation_id.astype(int), "latitude"].to_numpy(),
        "matched_arc_id": selected.candidate_arc_id.astype(int).to_numpy(),
        "confidence": selected.probability.to_numpy(dtype=float),
        "matched_from_longitude": node_coordinates[ends[selected.candidate_arc_id.astype(int), 0], 0],
        "matched_from_latitude": node_coordinates[ends[selected.candidate_arc_id.astype(int), 0], 1],
        "matched_to_longitude": node_coordinates[ends[selected.candidate_arc_id.astype(int), 1], 0],
        "matched_to_latitude": node_coordinates[ends[selected.candidate_arc_id.astype(int), 1], 1],
    })
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path / f"{trip.trip_id}.csv", index=False)
    pd.DataFrame({"sequence_index": np.arange(len(route_arc_ids)), "arc_id": route_arc_ids}).to_csv(
        output_path / f"{trip.trip_id}_route.csv", index=False
    )
    result.attrs["route_arc_ids"] = route_arc_ids
    return result


def evaluate_split(split_dir: str | Path, model_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Generate predictions for all trip folders in an unlabeled split."""
    artifact = joblib.load(model_path)
    frames = [predict_trip(path, artifact, output_dir) for path in discover_trips(split_dir)]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(Path(output_dir) / "submission.csv", index=False)
    return combined
