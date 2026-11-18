
import pandas as pd

path = "data/processed/train_00000001.csv"
df = pd.read_csv(path)

print("\nTOTAL OBSERVATIONS:", len(df))
print("\nSPEED STATISTICS (m/s):")
print(df["speed_mps"].describe())

print("\nTOP 15 HIGHEST SPEED OBSERVATIONS:")
print(
    df.nlargest(15, "speed_mps")[
        ["observation_id", "longitude", "latitude",
         "timestamp", "time_gap_seconds", "speed_mps"]
    ].to_string(index=False)
)

print("\nTIMESTAMP GAPS:")
print(
    df[df["time_gap_seconds"] > 1][
        ["observation_id", "timestamp",
         "time_gap_seconds", "speed_mps"]
    ].to_string(index=False)
)
