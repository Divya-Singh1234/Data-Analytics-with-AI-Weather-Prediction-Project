"""
visualization.py — Seasonal Weather Trend Visualizations
Generates and saves the following plots to outputs/plots/:
  1. Seasonal temperature box plots
  2. Monthly average temperature line chart
  3. Correlation heatmap
  4. Precipitation distribution by season
  5. Wind speed vs temperature scatter
  6. Humidity vs temperature scatter
  7. Monthly precipitation bar chart
  8. Cloud cover vs precipitation scatter
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

CLEAN_PATH = "weather_ml_project/data/weather_data_clean.csv"
PLOT_DIR = "weather_ml_project/outputs/plots"
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {"Winter": "#4e91d2", "Spring": "#5ab96b", "Summer": "#e8a838", "Autumn": "#d2693c"}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    return df


# ── 1. Seasonal Temperature Box Plot ─────────────────────────────────────────
def plot_seasonal_temperature(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_season = [df[df["season"] == s]["temperature_c"].values for s in SEASON_ORDER]
    bp = ax.boxplot(data_by_season, patch_artist=True, notch=True,
                    medianprops=dict(color="black", linewidth=2))
    ax.set_xticklabels(SEASON_ORDER)
    for patch, season in zip(bp["boxes"], SEASON_ORDER):
        patch.set_facecolor(SEASON_COLORS[season])
    ax.set_title("Temperature Distribution by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Temperature (°C)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_seasonal_temperature_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 01_seasonal_temperature_boxplot.png")


# ── 2. Monthly Average Temperature Line Chart ────────────────────────────────
def plot_monthly_temperature(df: pd.DataFrame) -> None:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_avg = df.groupby("month")["temperature_c"].agg(["mean", "std"]).reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(monthly_avg.index,
                    monthly_avg["mean"] - monthly_avg["std"],
                    monthly_avg["mean"] + monthly_avg["std"],
                    alpha=0.2, color="#3b82d4", label="±1 std dev")
    ax.plot(monthly_avg.index, monthly_avg["mean"], marker="o", color="#3b82d4",
            linewidth=2.2, markersize=6, label="Mean Temp")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title("Monthly Average Temperature")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_monthly_avg_temperature.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 02_monthly_avg_temperature.png")


# ── 3. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh",
                    "precipitation_mm", "cloud_cover_pct"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 10})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 03_correlation_heatmap.png")


# ── 4. Precipitation Distribution by Season ──────────────────────────────────
def plot_seasonal_precipitation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, season in zip(axes, SEASON_ORDER):
        data = df[df["season"] == season]["precipitation_mm"]
        ax.hist(data, bins=25, color=SEASON_COLORS[season], edgecolor="white", alpha=0.85)
        ax.set_title(season)
        ax.set_xlabel("Precipitation (mm)")
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1.2, label=f"Median: {data.median():.1f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Precipitation Distribution by Season", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_seasonal_precipitation_dist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_seasonal_precipitation_dist.png")


# ── 5. Wind Speed vs Temperature Scatter ─────────────────────────────────────
def plot_wind_vs_temp(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for season in SEASON_ORDER:
        subset = df[df["season"] == season]
        ax.scatter(subset["wind_speed_kmh"], subset["temperature_c"],
                   label=season, color=SEASON_COLORS[season], alpha=0.55, s=18)
    ax.set_title("Wind Speed vs Temperature by Season")
    ax.set_xlabel("Wind Speed (km/h)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(title="Season")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_wind_vs_temperature.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 05_wind_vs_temperature.png")


# ── 6. Humidity vs Temperature ────────────────────────────────────────────────
def plot_humidity_vs_temp(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df["temperature_c"], df["humidity_pct"],
                         c=df["precipitation_mm"], cmap="YlGnBu",
                         alpha=0.6, s=16, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Precipitation (mm)")
    clean = df[["temperature_c", "humidity_pct"]].dropna()
    z = np.polyfit(clean["temperature_c"], clean["humidity_pct"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean["temperature_c"].min(), clean["temperature_c"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=1.5, label="Trend")
    ax.set_title("Humidity vs Temperature (colored by Precipitation)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06_humidity_vs_temperature.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 06_humidity_vs_temperature.png")


# ── 7. Monthly Precipitation Bar Chart ───────────────────────────────────────
def plot_monthly_precipitation(df: pd.DataFrame) -> None:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_precip = df.groupby("month")["precipitation_mm"].mean().reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(1, 13), monthly_precip.values, color="#4e91d2", edgecolor="white", width=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title("Average Monthly Precipitation")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg. Precipitation (mm)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "07_monthly_precipitation_bar.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 07_monthly_precipitation_bar.png")


# ── 8. Cloud Cover vs Precipitation ──────────────────────────────────────────
def plot_cloud_vs_precip(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for season in SEASON_ORDER:
        subset = df[df["season"] == season]
        ax.scatter(subset["cloud_cover_pct"], subset["precipitation_mm"],
                   label=season, color=SEASON_COLORS[season], alpha=0.5, s=18)
    ax.set_title("Cloud Cover vs Precipitation by Season")
    ax.set_xlabel("Cloud Cover (%)")
    ax.set_ylabel("Precipitation (mm)")
    ax.legend(title="Season")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "08_cloud_vs_precipitation.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 08_cloud_vs_precipitation.png")


def run():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_data(CLEAN_PATH)
    print("[Visualization] Generating plots...")
    plot_seasonal_temperature(df)
    plot_monthly_temperature(df)
    plot_correlation_heatmap(df)
    plot_seasonal_precipitation(df)
    plot_wind_vs_temp(df)
    plot_humidity_vs_temp(df)
    plot_monthly_precipitation(df)
    plot_cloud_vs_precip(df)
    print(f"\n[Visualization] All 8 plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    run()
