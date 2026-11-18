"""Command-line entry point for PathSnap feature, training, and inference flows."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_split
from src.features import build_split_features
from src.parser import discover_trips, parse_trip
from src.train import train_pipeline


def _write_eda(raw_root: Path, output_dir: Path) -> None:
    """Write per-trip data quality and graph inventory CSVs."""
    import pandas as pd

    from src.features import trip_feature_summary

    records = []
    for split in ("train", "student_test"):
        for folder in discover_trips(raw_root / split):
            trip = parse_trip(folder, require_route=True)
            row = trip_feature_summary(trip)
            row["split"] = split
            row["route_available"] = bool(len(trip.route))
            records.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "dataset_inventory.csv", index=False)


def main() -> None:
    """Parse CLI options and dispatch the selected pipeline stage."""
    parser = argparse.ArgumentParser(description="PathSnap map-matching pipeline")
    parser.add_argument("command", choices=["eda", "features", "train", "evaluate", "all"])
    parser.add_argument("--raw-root", default="dataset_raw/Student folder", help="Directory containing train/ and student_test/")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--model-path", default="models/pathsnap_model.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--training-csv", help="Optional compact labeled candidate CSV")
    parser.add_argument("--selected-model", choices=["logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting", "gradient_boosting"], default="random_forest", help="Fit this model after your comparison; defaults to the user-selected Random Forest")
    parser.add_argument("--compare-models", action="store_true", help="Run the full five-model grouped cross-validation comparison")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    if args.command == "eda":
        _write_eda(raw_root, Path(args.output_dir))
    elif args.command == "features":
        processed = Path(args.processed_dir)
        processed.mkdir(parents=True, exist_ok=True)
        for split in ("train", "student_test"):
            rows, inventory = build_split_features(raw_root / split, include_labels=True)
            rows.to_csv(processed / f"{split}_candidates.csv", index=False)
            inventory.to_csv(processed / f"{split}_inventory.csv", index=False)
    elif args.command == "train":
        result = train_pipeline(
            raw_root, args.processed_dir, args.model_dir, args.experiments_dir, args.folds,
            training_csv=args.training_csv,
            selected_model=None if args.compare_models else args.selected_model,
        )
        print(f"Selected {result['selected_model']}; checkpoint: {result['model_path']}")
    elif args.command == "evaluate":
        split_dir = raw_root / "student_test"
        result = evaluate_split(split_dir, args.model_path, args.output_dir)
        print(f"Wrote {len(result)} matched observations to {Path(args.output_dir) / 'submission.csv'}")
    else:
        _write_eda(raw_root, Path(args.output_dir))
        train_pipeline(
            raw_root, args.processed_dir, args.model_dir, args.experiments_dir, args.folds,
            training_csv=args.training_csv,
            selected_model=None if args.compare_models else args.selected_model,
        )
        result = evaluate_split(raw_root / "student_test", args.model_path, args.output_dir)
        print(f"Wrote {len(result)} development predictions to {Path(args.output_dir) / 'submission.csv'}")


if __name__ == "__main__":
    main()
