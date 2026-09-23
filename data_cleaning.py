"""
data_cleaning.py — Weather Dataset Cleaning Pipeline
- Load raw CSV
- Report missing values
- Impute missing values (median for numerics)
- Remove duplicates
- Validate data ranges
- Save cleaned CSV
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "weather_ml_project/data/weather_data.csv"
CLEAN_PATH = "weather_ml_project/data/weather_data_clean.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"\n[Load] Shape: {df.shape}")
    return df


def report_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0]
    print("\n[Missing Values Report]")
    print(report.to_string())


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed '{col}' with median={median_val:.2f}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n[Duplicates] Removed {before - after} duplicate rows.")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    constraints = {
        "temperature_c": (-50, 60),
        "humidity_pct": (0, 100),
        "wind_speed_kmh": (0, 200),
        "precipitation_mm": (0, 500),
        "cloud_cover_pct": (0, 100),
    }
    for col, (lo, hi) in constraints.items():
        invalid = df[(df[col] < lo) | (df[col] > hi)]
        if len(invalid):
            print(f"  [Warning] {len(invalid)} out-of-range rows in '{col}' — clipping.")
            df[col] = df[col].clip(lo, hi)
    print("[Validation] Range check complete.")
    return df


def save_clean(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[Save] Clean data saved to: {path}  ({df.shape[0]} rows x {df.shape[1]} cols)")


def run():
    df = load_data(RAW_PATH)
    report_missing(df)
    df = impute_missing(df)
    df = remove_duplicates(df)
    df = validate_ranges(df)
    save_clean(df, CLEAN_PATH)
    print("\n[Summary]")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    run()
