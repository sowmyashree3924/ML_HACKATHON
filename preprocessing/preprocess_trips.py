

from pathlib import Path
import pandas as pd
from clean_gps import clean_gps

ROOT = Path("dataset_raw/Student folder")
OUTPUT = Path("data/processed")

OUTPUT.mkdir(parents=True, exist_ok=True)

all_summaries = []

for split in ["train", "student_test"]:
    trip_folder = ROOT / split

    for trip_dir in sorted(trip_folder.iterdir()):
        if not trip_dir.is_dir():
            continue

        track_files = list(trip_dir.glob("*.track"))

        if not track_files:
            print(f"Skipping {trip_dir.name}: no .track file")
            continue

        cleaned, summary = clean_gps(track_files[0])

        # Add trip ID without inserting observation_id again
        cleaned.insert(0, "trip_id", trip_dir.name)

        output_file = OUTPUT / f"{split}_{trip_dir.name}.csv"
        cleaned.to_csv(output_file, index=False)

        summary["trip_id"] = trip_dir.name
        summary["split"] = split

        all_summaries.append(summary)

        print(f"Processed {split}: {trip_dir.name}")

# Save preprocessing report
summary_df = pd.DataFrame(all_summaries)

summary_df.to_csv(
    OUTPUT / "preprocessing_summary.csv",
    index=False
)

print("\nPREPROCESSING COMPLETE")
print("Total trips:", len(all_summaries))
print("Training trips:", len(summary_df[summary_df["split"] == "train"]))
print("Student testing trips:", len(summary_df[summary_df["split"] == "student_test"]))
print("Output folder:", OUTPUT)
