"""
Generate a synthetic weather dataset with 1000 rows and 7 columns.
Columns: date, temperature_c, humidity_pct, wind_speed_kmh, precipitation_mm, cloud_cover_pct, season
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000
start_date = pd.Timestamp("2020-01-01")
dates = pd.date_range(start_date, periods=n, freq="D")

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

seasons = [get_season(d.month) for d in dates]

# Temperature: seasonal sinusoidal pattern + noise
day_of_year = np.array([d.day_of_year for d in dates])
base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
temperature = base_temp + np.random.normal(0, 3, n)

# Humidity: inversely related to temperature + noise
humidity = 80 - 0.8 * (temperature - temperature.min()) + np.random.normal(0, 5, n)
humidity = np.clip(humidity, 20, 100)

# Wind speed: random with seasonal variation
wind_speed = np.abs(np.random.normal(15, 7, n)) + (np.array([1.5 if s == "Winter" else 0 for s in seasons]))
wind_speed = np.clip(wind_speed, 0, 80)

# Precipitation: higher in spring/autumn
precip_base = np.array([
    5 if s == "Summer" else (12 if s == "Spring" else (10 if s == "Autumn" else 8))
    for s in seasons
])
precipitation = np.random.exponential(precip_base)
precipitation = np.clip(precipitation, 0, 80)

# Cloud cover: correlated with precipitation + noise
cloud_cover = 20 + 0.9 * precipitation + np.random.normal(0, 10, n)
cloud_cover = np.clip(cloud_cover, 0, 100)

# Introduce ~5% missing values in temperature, humidity, precipitation
for col_arr in [temperature, humidity, precipitation]:
    miss_idx = np.random.choice(n, size=int(0.05 * n), replace=False)
    col_arr[miss_idx] = np.nan

df = pd.DataFrame({
    "date": dates,
    "temperature_c": np.round(temperature, 2),
    "humidity_pct": np.round(humidity, 2),
    "wind_speed_kmh": np.round(wind_speed, 2),
    "precipitation_mm": np.round(precipitation, 2),
    "cloud_cover_pct": np.round(cloud_cover, 2),
    "season": seasons,
})

df.to_csv("weather_ml_project/data/weather_data.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())
