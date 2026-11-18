"""Trip-grouped cross-validation, student-set comparison, and model checkpointing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from .features import FEATURE_COLUMNS, build_split_features
from .model import build_models, positive_probabilities

METADATA_COLUMNS = {"trip_id", "observation_id", "candidate_arc_id", "label"}


def _top1_accuracy(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Score the highest-probability candidate for each observation."""
    scored = frame[["trip_id", "observation_id", "label"]].copy()
    scored["score"] = probabilities
    winners = scored.loc[scored.groupby(["trip_id", "observation_id"]).score.idxmax()]
    return float(winners.label.mean()) if len(winners) else float("nan")


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate candidate and observation-level classification metrics."""
    predicted = probabilities >= 0.5
    labels = frame.label.to_numpy(dtype=int)
    return {
        "candidate_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "candidate_log_loss": float(log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "observation_top1_accuracy": _top1_accuracy(frame, probabilities),
    }


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select a stable engineered feature set, inserting absent columns."""
    return frame.reindex(columns=FEATURE_COLUMNS)


def _read_fast_training_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Normalize the supplied compact candidate CSV to pipeline column names."""
    frame = pd.read_csv(path)
    mapping = {
        "gps_index": "observation_id",
        "arc_id": "candidate_arc_id",
        "segment_length_m": "road_length_m",
        "heading_difference": "heading_diff_deg",
    }
    frame = frame.rename(columns=mapping)
    required = {"trip_id", "observation_id", "candidate_arc_id", "distance_m", "road_length_m", "heading_diff_deg", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fast training CSV is missing required columns: {sorted(missing)}")
    frame["trip_id"] = frame["trip_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    for column in ("observation_id", "candidate_arc_id", "distance_m", "road_length_m", "heading_diff_deg", "label"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).reset_index(drop=True)
    frame["observation_id"] = frame.observation_id.astype(int)
    frame["candidate_arc_id"] = frame.candidate_arc_id.astype(int)
    frame["label"] = frame.label.astype(int)
    if not set(frame.label.unique()).issubset({0, 1}) or frame.label.nunique() != 2:
        raise ValueError("Fast training CSV labels must contain both binary classes 0 and 1")
    # The supplied fast table stores the angular difference in radians.
    frame["heading_diff_deg"] = np.degrees(frame.heading_diff_deg)
    features = ["distance_m", "road_length_m", "heading_diff_deg"]
    return frame, features


def train_pipeline(
    raw_root: str | Path,
    processed_dir: str | Path = "data/processed",
    model_dir: str | Path = "models",
    experiments_dir: str | Path = "experiments",
    folds: int = 5,
    random_state: int = 42,
    training_csv: str | Path | None = None,
    selected_model: str | None = None,
) -> dict[str, Any]:
    """Run group-aware CV, compare on student-test trips, and save the winner.

    The student-test split is development data and has reference routes. It is
    evaluated for model comparison only; evaluator-test data is not discovered.
    """
    root, processed = Path(raw_root), Path(processed_dir)
    model_path, exp_path = Path(model_dir), Path(experiments_dir)
    processed.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    exp_path.mkdir(parents=True, exist_ok=True)
    train_path, student_path = processed / "train_candidates.csv", processed / "student_test_candidates.csv"
    fast_features: list[str] | None = None
    if training_csv is not None:
        train, fast_features = _read_fast_training_csv(training_csv)
    elif train_path.exists():
        train = pd.read_csv(train_path)
    else:
        train, _ = build_split_features(root / "train", include_labels=True)
        train.to_csv(train_path, index=False)
    if student_path.exists():
        student = pd.read_csv(student_path)
    else:
        student, _ = build_split_features(root / "student_test", include_labels=True)
        student.to_csv(student_path, index=False)
    if train.label.nunique() < 2:
        raise ValueError("Training candidates must contain both positive and negative labels")
    model_features = fast_features or FEATURE_COLUMNS
    if selected_model is not None:
        models = build_models(random_state)
        if selected_model not in models:
            raise ValueError(f"Unknown model '{selected_model}'. Choose from {sorted(models)}")
        estimator = models[selected_model]
        print(f"Fitting selected model {selected_model} on {len(train):,} candidate rows using {len(model_features)} features", flush=True)
        estimator.fit(train.reindex(columns=model_features), train.label.to_numpy(dtype=int))
        started = time.perf_counter()
        student_probabilities = positive_probabilities(estimator, student.reindex(columns=model_features))
        inference_seconds = time.perf_counter() - started
        student_metrics = _metrics(student, student_probabilities)
        row = {
            "model": selected_model,
            "cv_macro_f1": np.nan,
            "cv_top1_accuracy": np.nan,
            "cv_log_loss": np.nan,
            "student_macro_f1": student_metrics["candidate_macro_f1"],
            "student_top1_accuracy": student_metrics["observation_top1_accuracy"],
            "student_log_loss": student_metrics["candidate_log_loss"],
            "student_candidate_miss_rate": float(student.groupby(["trip_id", "observation_id"]).label.sum().eq(0).mean()),
            "student_inference_seconds": inference_seconds,
            "folds_completed": 0,
            "selection_basis": "user-reported model comparison; final fit only",
        }
        results_frame = pd.DataFrame([row])
        results_frame.to_csv(exp_path / "model_comparison.csv", index=False)
        best_name, best_model = selected_model, estimator
        selection_metric = "User selected Random Forest based on reported highest F1 score" if selected_model == "random_forest" else "Explicit CLI model selection"
    else:
        groups = train.trip_id.astype(str).to_numpy()
        n_splits = min(folds, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError("At least two training trips are required for cross-validation")
        cv = GroupKFold(n_splits=n_splits)
        results: list[dict[str, Any]] = []
        fitted: dict[str, Any] = {}
        for name, estimator in build_models(random_state).items():
            fold_metrics: list[dict[str, float]] = []
            for train_idx, valid_idx in cv.split(train, train.label, groups):
                x_fit, x_valid = train.iloc[train_idx].reindex(columns=model_features), train.iloc[valid_idx].reindex(columns=model_features)
                y_fit = train.label.iloc[train_idx].to_numpy(dtype=int)
                if len(np.unique(y_fit)) < 2:
                    continue
                estimator.fit(x_fit, y_fit)
                p_valid = positive_probabilities(estimator, x_valid)
                fold_frame = train.iloc[valid_idx].reset_index(drop=True)
                fold_metrics.append(_metrics(fold_frame, p_valid))
            if not fold_metrics:
                raise ValueError(f"{name}: no valid cross-validation fold had both classes")
            estimator.fit(train.reindex(columns=model_features), train.label.to_numpy(dtype=int))
            started = time.perf_counter()
            student_probabilities = positive_probabilities(estimator, student.reindex(columns=model_features))
            inference_seconds = time.perf_counter() - started
            row: dict[str, Any] = {
                "model": name,
                "cv_macro_f1": float(np.mean([metric["candidate_macro_f1"] for metric in fold_metrics])),
                "cv_top1_accuracy": float(np.mean([metric["observation_top1_accuracy"] for metric in fold_metrics])),
                "cv_log_loss": float(np.mean([metric["candidate_log_loss"] for metric in fold_metrics])),
                "student_macro_f1": _metrics(student, student_probabilities)["candidate_macro_f1"],
                "student_top1_accuracy": _top1_accuracy(student, student_probabilities),
                "student_log_loss": _metrics(student, student_probabilities)["candidate_log_loss"],
                "student_candidate_miss_rate": float(student.groupby(["trip_id", "observation_id"]).label.sum().eq(0).mean()),
                "student_inference_seconds": inference_seconds,
                "folds_completed": len(fold_metrics),
            }
            results.append(row)
            fitted[name] = estimator
            print(f"{name}: CV macro-F1={row['cv_macro_f1']:.4f}, student macro-F1={row['student_macro_f1']:.4f}", flush=True)
        results_frame = pd.DataFrame(results).sort_values("cv_macro_f1", ascending=False)
        results_frame.to_csv(exp_path / "model_comparison.csv", index=False)
        best_name = str(results_frame.iloc[0].model)
        best_model = fitted[best_name]
        selection_metric = "trip-grouped CV candidate macro F1"
    label_definition = (
        "Reference-route weak labels provided in the compact training CSV"
        if training_csv is not None
        else "Nearest supplied route arc within 75m; weak spatial supervision"
    )
    artifact = {"model": best_model, "model_name": best_name, "feature_columns": model_features,
                "label_definition": label_definition,
                "random_state": random_state}
    joblib.dump(artifact, model_path / "pathsnap_model.joblib", compress=3)
    (model_path / "model_metadata.json").write_text(json.dumps({
        "selected_model": best_name, "selection_metric": selection_metric,
        "feature_columns": model_features, "label_definition": artifact["label_definition"],
        "student_test_used_for_development_comparison": True,
    }, indent=2), encoding="utf-8")
    return {"selected_model": best_name, "results": results_frame, "model_path": model_path / "pathsnap_model.joblib"}
