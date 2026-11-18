
from pathlib import Path
import pandas as pd

folder = Path("data/processed")
results = []

for file in folder.glob("train_*.csv"):
    df = pd.read_csv(file)

    results.append({
        "trip_id": file.stem.replace("train_", ""),
        "observations": len(df),
        "mean_speed_mps": df["speed_mps"].mean(),
        "median_speed_mps": df["speed_mps"].median(),
        "max_speed_mps": df["speed_mps"].max(),
        "gps_gaps": int(df["is_gps_gap"].sum())
    })

result = pd.DataFrame(results)
result.to_csv(folder / "speed_quality_report.csv", index=False)

print(result.describe())
print("\nFirst 15 trips:")
print(result.head(15).to_string(index=False))
