# 🌦️ Weather ML Project

A complete end-to-end machine learning project that ingests a synthetic weather dataset (1 000 rows × 7 columns), cleans it, performs exploratory analysis, visualises seasonal trends, and trains multiple regression models to predict daily temperature.

---

## 📁 Project Structure

```
weather_ml_project/
├── data/
│   ├── weather_data.csv          # Raw generated dataset
│   └── weather_data_clean.csv    # Cleaned dataset
├── models/
│   └── best_model.pkl            # Serialised best-performing model
├── outputs/
│   ├── descriptive_stats.csv
│   ├── seasonal_stats.csv
│   ├── correlation_matrix.csv
│   ├── outlier_report.csv
│   ├── monthly_trends.csv
│   ├── model_metrics.csv
│   └── plots/
│       ├── 01_seasonal_temperature_boxplot.png
│       ├── 02_monthly_avg_temperature.png
│       ├── 03_correlation_heatmap.png
│       ├── 04_seasonal_precipitation_dist.png
│       ├── 05_wind_vs_temperature.png
│       ├── 06_humidity_vs_temperature.png
│       ├── 07_monthly_precipitation_bar.png
│       ├── 08_cloud_vs_precipitation.png
│       ├── 09_prediction_vs_actual.png
│       ├── 10_feature_importance.png
│       └── 11_residuals.png
├── generate_dataset.py           # Synthetic data generator
├── data_cleaning.py              # Data cleaning pipeline
├── exploratory_analysis.py       # EDA & statistics
├── visualization.py              # 8 seasonal trend plots
├── temperature_prediction.py     # ML training & evaluation
├── main.py                       # End-to-end orchestrator
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Column              | Type    | Description                        |
|---------------------|---------|------------------------------------|
| `date`              | date    | Daily timestamp (2020-01-01 …)     |
| `temperature_c`     | float   | Temperature in °C                  |
| `humidity_pct`      | float   | Relative humidity (%)              |
| `wind_speed_kmh`    | float   | Wind speed in km/h                 |
| `precipitation_mm`  | float   | Daily precipitation in mm          |
| `cloud_cover_pct`   | float   | Cloud cover (%)                    |
| `season`            | string  | Winter / Spring / Summer / Autumn  |

~5 % missing values are intentionally injected in `temperature_c`, `humidity_pct`, and `precipitation_mm` to simulate real-world data quality issues.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
cd weather_ml_project
python main.py
```

Or run each module individually:

```bash
python generate_dataset.py       # Step 1 – generate CSV
python data_cleaning.py          # Step 2 – clean data
python exploratory_analysis.py   # Step 3 – EDA
python visualization.py          # Step 4 – plots
python temperature_prediction.py # Step 5 – ML models
```

---

## 🔧 Pipeline Modules

### `generate_dataset.py`
Generates a realistic synthetic dataset using:
- Sinusoidal temperature model with seasonal offsets
- Humidity inversely correlated with temperature
- Exponential precipitation with seasonal scaling
- Cloud cover correlated with precipitation

### `data_cleaning.py`
- Reports & imputes missing values (median strategy)
- Removes duplicate rows
- Validates and clips out-of-range values
- Saves cleaned CSV to `data/weather_data_clean.csv`

### `exploratory_analysis.py`
Produces CSV summaries to `outputs/`:
- Descriptive statistics (mean, std, percentiles)
- Seasonal averages across all numeric features
- Pearson correlation matrix
- IQR-based outlier counts per column
- Monthly temperature, precipitation, and humidity trends

### `visualization.py`
Generates 8 publication-quality plots to `outputs/plots/`:

| # | Plot |
|---|------|
| 1 | Seasonal temperature box plots |
| 2 | Monthly average temperature line chart |
| 3 | Correlation heatmap |
| 4 | Precipitation distribution by season |
| 5 | Wind speed vs temperature scatter |
| 6 | Humidity vs temperature scatter (precipitation-coloured) |
| 7 | Monthly precipitation bar chart |
| 8 | Cloud cover vs precipitation scatter |

### `temperature_prediction.py`
Trains and evaluates four models:

| Model | Notes |
|-------|-------|
| Linear Regression | Scaled baseline |
| Ridge Regression | L2-regularised linear model |
| Random Forest | 200 trees, depth 12 |
| Gradient Boosting | 200 estimators, lr=0.08 |

**Feature engineering:**
- Cyclic encoding of month and day-of-year (sin/cos)
- Label encoding of season

**Metrics reported:** MAE, RMSE, R², 5-fold CV MAE

**Outputs:**
- `outputs/model_metrics.csv` — comparison table
- `models/best_model.pkl` — serialised best model
- Plots 09–11 (predicted vs actual, feature importance, residuals)

---

## 📈 Example Results (typical run)

| Model              | MAE  | RMSE | R²     |
|--------------------|------|------|--------|
| Gradient Boosting  | 0.65 | 0.85 | 0.9940 |
| Random Forest      | 0.71 | 0.93 | 0.9927 |
| Ridge Regression   | 2.18 | 2.74 | 0.9417 |
| Linear Regression  | 2.18 | 2.74 | 0.9416 |

> Results vary slightly each run due to train/test splitting.

---

## 🛠 Requirements

| Package      | Min. Version |
|--------------|-------------|
| numpy        | 1.24.0      |
| pandas       | 2.0.0       |
| matplotlib   | 3.7.0       |
| seaborn      | 0.12.0      |
| scikit-learn | 1.3.0       |
| joblib       | 1.3.0       |

Python ≥ 3.9 recommended.

---

## 📄 License

MIT — free to use and modify.
