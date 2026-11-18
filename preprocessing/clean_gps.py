
import pandas as pd
import numpy as np


def clean_gps(track_path):
    """
    Preprocess GPS observations for PathSnap.

    Steps:
    1. Load raw GPS data.
    2. Convert columns to numeric.
    3. Remove invalid/missing coordinates and timestamps.
    4. Sort observations chronologically.
    5. Remove exact duplicate observations.
    6. Calculate timestamp gaps.
    7. Calculate GPS distance and speed.
    8. Generate GPS quality flags.

    Returns:
        cleaned_df: Preprocessed GPS DataFrame
        summary: Data quality statistics
    """

    # -------------------------------------------------
    # 1. LOAD RAW GPS DATA
    # -------------------------------------------------

    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    original_count = len(df)

    # -------------------------------------------------
    # 2. CONVERT DATA TYPES
    # -------------------------------------------------

    for column in ["longitude", "latitude", "timestamp"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # -------------------------------------------------
    # 3. HANDLE MISSING VALUES
    # -------------------------------------------------

    missing_rows = int(
        df[["longitude", "latitude", "timestamp"]]
        .isna()
        .any(axis=1)
        .sum()
    )

    df = df.dropna(
        subset=["longitude", "latitude", "timestamp"]
    )

    # -------------------------------------------------
    # 4. REMOVE INVALID COORDINATES
    # -------------------------------------------------

    invalid_coordinates = (
        ~df["longitude"].between(-180, 180)
        | ~df["latitude"].between(-90, 90)
    )

    invalid_count = int(invalid_coordinates.sum())

    df = df.loc[~invalid_coordinates].copy()

    # -------------------------------------------------
    # 5. SORT CHRONOLOGICALLY
    # -------------------------------------------------

    df = df.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    # -------------------------------------------------
    # 6. REMOVE EXACT DUPLICATE OBSERVATIONS
    # -------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=["longitude", "latitude", "timestamp"]
        ).sum()
    )

    df = df.drop_duplicates(
        subset=["longitude", "latitude", "timestamp"]
    ).reset_index(drop=True)

    # Preserve a stable observation identifier
    df.insert(
        0,
        "observation_id",
        range(len(df))
    )

    # -------------------------------------------------
    # 7. CALCULATE TIMESTAMP GAPS
    # -------------------------------------------------

    df["time_gap_seconds"] = (
        df["timestamp"].diff()
    )

    # First observation has no previous timestamp
    df["time_gap_seconds"] = (
        df["time_gap_seconds"].fillna(0)
    )

    # Flag gaps greater than 5 seconds
    df["is_gps_gap"] = (
        df["time_gap_seconds"] > 5
    )

    # Flag irregular sampling intervals
    df["irregular_interval_flag"] = (
        (df["time_gap_seconds"] > 1)
        & (df["time_gap_seconds"] > 0)
    )

    # -------------------------------------------------
    # 8. CALCULATE GEOGRAPHICAL DISTANCE
    # -------------------------------------------------

    EARTH_RADIUS_M = 6371000

    lat_rad = np.radians(df["latitude"])
    lon_rad = np.radians(df["longitude"])

    previous_lat = lat_rad.shift(1)
    previous_lon = lon_rad.shift(1)

    dlat = lat_rad - previous_lat
    dlon = lon_rad - previous_lon

    haversine_a = (
        np.sin(dlat / 2) ** 2
        + np.cos(previous_lat)
        * np.cos(lat_rad)
        * np.sin(dlon / 2) ** 2
    )

    haversine_a = haversine_a.clip(0, 1)

    df["distance_m"] = (
        EARTH_RADIUS_M
        * 2
        * np.arcsin(np.sqrt(haversine_a))
    )

    df["distance_m"] = (
        df["distance_m"].fillna(0)
    )

    # -------------------------------------------------
    # 9. CALCULATE SPEED
    # -------------------------------------------------

    valid_time = df["time_gap_seconds"] > 0

    df["speed_mps"] = np.where(
        valid_time,
        df["distance_m"] / df["time_gap_seconds"].where(
            valid_time, np.nan
        ),
        np.nan
    )

    df["speed_mps"] = (
        pd.Series(df["speed_mps"], index=df.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # -------------------------------------------------
    # 10. GENERATE SPEED QUALITY FLAGS
    # -------------------------------------------------

    # Flag unusually high speeds using a physical
    # screening threshold. Do not delete these rows.
    df["high_speed_flag"] = (
        df["speed_mps"] > 60
    )

    # Flag extremely high speeds for further inspection
    df["extreme_speed_flag"] = (
        df["speed_mps"] > 100
    )

    # -------------------------------------------------
    # 11. CALCULATE SUMMARY STATISTICS
    # -------------------------------------------------

    summary = {
        "original_rows": original_count,
        "cleaned_rows": len(df),
        "missing_rows_removed": missing_rows,
        "invalid_coordinates_removed": invalid_count,
        "duplicate_rows_removed": duplicate_count,
        "total_removed_rows": original_count - len(df),
        "gps_gaps": int(df["is_gps_gap"].sum()),
        "irregular_intervals": int(
            df["irregular_interval_flag"].sum()
        ),
        "high_speed_observations": int(
            df["high_speed_flag"].sum()
        ),
        "extreme_speed_observations": int(
            df["extreme_speed_flag"].sum()
        ),
        "average_speed_mps": float(
            df["speed_mps"].mean()
        ),
        "maximum_speed_mps": float(
            df["speed_mps"].max()
        )
    }

    return df, summary

