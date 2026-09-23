"""
exploratory_analysis.py — Exploratory Data Analysis (EDA)
- Descriptive statistics
- Seasonal statistics
- Correlation matrix
- Outlier detection (IQR)
- Save summary reports to outputs/
"""

import pandas as pd
import numpy as np
import os

CLEAN_PATH = "weather_ml_project/data/weather_data_clean.csv"
OUTPUT_DIR = "weather_ml_project/outputs"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def descriptive_stats(df: pd.DataFrame) -> None:
    print("\n=== Descriptive Statistics ===")
    stats = df.describe().round(3)
    print(stats.to_string())
    stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))


def seasonal_stats(df: pd.DataFrame) -> None:
    print("\n=== Seasonal Averages ===")
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    seasonal = df.groupby("season")[numeric_cols].mean().round(2)
    seasonal = seasonal.reindex(season_order)
    print(seasonal.to_string())
    seasonal.to_csv(os.path.join(OUTPUT_DIR, "seasonal_stats.csv"))


def correlation_matrix(df: pd.DataFrame) -> None:
    print("\n=== Correlation Matrix ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr().round(3)
    print(corr.to_string())
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))


def detect_outliers(df: pd.DataFrame) -> None:
    print("\n=== Outlier Detection (IQR Method) ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = []
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        summary.append({"column": col, "outlier_count": len(outliers), "pct": round(len(outliers) / len(df) * 100, 2)})
        print(f"  {col}: {len(outliers)} outliers ({summary[-1]['pct']}%)")
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "outlier_report.csv"), index=False)


def monthly_trends(df: pd.DataFrame) -> None:
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")[["temperature_c", "precipitation_mm", "humidity_pct"]].mean().round(2)
    monthly.to_csv(os.path.join(OUTPUT_DIR, "monthly_trends.csv"))
    print("\n=== Monthly Trends (saved) ===")
    print(monthly.to_string())


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(CLEAN_PATH)
    descriptive_stats(df)
    seasonal_stats(df)
    correlation_matrix(df)
    detect_outliers(df)
    monthly_trends(df)
    print("\n[EDA] All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
