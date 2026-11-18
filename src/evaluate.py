"""Model inference, lightweight sequence decoding, metrics, and CSV output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_candidate_features
from .parser import discover_trips, parse_trip


def viterbi_decode(candidates: pd.DataFrame, probabilities: np.ndarray, arcs: pd.DataFrame, continuity_weight: float = 2.0) -> pd.DataFrame:
    """Select a coherent candidate sequence using an endpoint continuity HMM.

    Emissions are model probabilities. Transitions favor consecutive road arcs
    sharing an endpoint, with a smooth distance penalty when observations skip
    one or more arcs. This is a compact decoder, not a full routing engine.
    """
    work = candidates.copy()
    work["probability"] = np.clip(probabilities, 1e-8, 1.0)
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    groups = [group for _, group in work.groupby("observation_id", sort=True)]
    if not groups:
        return work.iloc[:0].copy()
    scores: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    first_ids = groups[0].candidate_arc_id.to_numpy(dtype=int)
    scores.append(np.log(groups[0].probability.to_numpy()))
    backpointers.append(np.full(len(first_ids), -1, dtype=int))
    for previous, current in zip(groups[:-1], groups[1:]):
        prev_ids = previous.candidate_arc_id.to_numpy(dtype=int)
        curr_ids = current.candidate_arc_id.to_numpy(dtype=int)
        transition = np.full((len(prev_ids), len(curr_ids)), -continuity_weight, dtype=float)
        for i, prev_id in enumerate(prev_ids):
            prev_end = endpoints[prev_id, 1]
            for j, curr_id in enumerate(curr_ids):
                curr_start = endpoints[curr_id, 0]
                if prev_end == curr_start:
                    transition[i, j] = 0.0
                elif endpoints[prev_id, 0] == endpoints[curr_id, 1]:
                    transition[i, j] = -continuity_weight * 1.5
        next_scores = scores[-1][:, None] + transition
        backpointers.append(np.argmax(next_scores, axis=0))
        scores.append(np.max(next_scores, axis=0) + np.log(current.probability.to_numpy()))
    path = np.zeros(len(groups), dtype=int)
    path[-1] = int(np.argmax(scores[-1]))
    for idx in range(len(groups) - 1, 0, -1):
        path[idx - 1] = backpointers[idx][path[idx]]
    selected = [group.iloc[state] for group, state in zip(groups, path)]
    return pd.DataFrame(selected).reset_index(drop=True)


def predict_trip(trip_dir: str | Path, artifact: dict[str, Any], output_dir: str | Path) -> pd.DataFrame:
    """Predict one trip and write observation-level matched road segments."""
    trip = parse_trip(trip_dir, require_route=False)
    candidates, _ = build_candidate_features(trip, include_labels=False)
    features = candidates.reindex(columns=artifact.get("feature_columns", FEATURE_COLUMNS))
    probabilities = artifact["model"].predict_proba(features)
    classes = artifact["model"].named_steps["classifier"].classes_
    positive_index = list(classes).index(1) if 1 in classes else 0
    probabilities = probabilities[:, positive_index]
    selected = viterbi_decode(candidates, probabilities, trip.arcs)
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
    return result


def evaluate_split(split_dir: str | Path, model_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Generate predictions for all trip folders in an unlabeled split."""
    artifact = joblib.load(model_path)
    frames = [predict_trip(path, artifact, output_dir) for path in discover_trips(split_dir)]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(Path(output_dir) / "submission.csv", index=False)
    return combined
