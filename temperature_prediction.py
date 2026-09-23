"""
temperature_prediction.py — ML Pipeline for Temperature Prediction
Models:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. Gradient Boosting Regressor (XGBoost-style via sklearn)

Features: humidity_pct, wind_speed_kmh, precipitation_mm, cloud_cover_pct, month, season (encoded)
Target:   temperature_c

Outputs:
  - outputs/model_metrics.csv          — comparison table
  - outputs/plots/09_prediction_vs_actual.png
  - outputs/plots/10_feature_importance.png
  - outputs/plots/11_residuals.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

CLEAN_PATH = "weather_ml_project/data/weather_data_clean.csv"
OUTPUT_DIR = "weather_ml_project/outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
MODEL_DIR = "weather_ml_project/models"

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Data Preparation ──────────────────────────────────────────────────────────
def load_and_prepare(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year

    # Encode season as integer
    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df["season"])

    # Cyclic encoding for month & day_of_year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    feature_cols = [
        "humidity_pct", "wind_speed_kmh", "precipitation_mm",
        "cloud_cover_pct", "season_enc",
        "month_sin", "month_cos", "doy_sin", "doy_cos",
    ]
    target_col = "temperature_c"

    # Drop rows where the target is missing (cannot impute a label)
    df = df.dropna(subset=[target_col])

    # Impute missing feature values with column means
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    X = df[feature_cols]
    y = df[target_col]

    assert X.isna().sum().sum() == 0, "NaNs remain in features after imputation"
    assert y.isna().sum() == 0, "NaNs remain in target after dropna"

    return X, y, feature_cols


# ── Model Definitions ─────────────────────────────────────────────────────────
def build_models():
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=3,
                random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.8, random_state=42
            )),
        ]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="neg_mean_absolute_error", n_jobs=-1)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_mae = -cv_scores.mean()

    print(f"  [{name}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  CV-MAE={cv_mae:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_MAE": cv_mae,
            "y_pred": y_pred}


# ── Plot: Predicted vs Actual ─────────────────────────────────────────────────
def plot_pred_vs_actual(y_test, predictions: dict) -> None:
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ["#3b82d4", "#5ab96b", "#e8a838", "#d2693c"]
    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.4, s=12, color=color, edgecolors="none")
        lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect fit")
        r2 = r2_score(y_test, y_pred)
        ax.set_title(f"{name}\nR² = {r2:.4f}")
        ax.set_xlabel("Actual (°C)")
        ax.set_ylabel("Predicted (°C)")
        ax.legend(fontsize=8)

    fig.suptitle("Predicted vs Actual Temperature", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "09_prediction_vs_actual.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 09_prediction_vs_actual.png")


# ── Plot: Feature Importance (best tree model) ────────────────────────────────
def plot_feature_importance(best_model, feature_cols: list) -> None:
    # Get the actual estimator from pipeline if wrapped
    estimator = best_model
    if hasattr(best_model, "named_steps"):
        estimator = best_model.named_steps.get("model", best_model)

    if not hasattr(estimator, "feature_importances_"):
        print("  [Feature Importance] Not available for this model type.")
        return

    importances = estimator.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#3b82d4", edgecolor="white")
    ax.set_title("Feature Importance (Best Model)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "10_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 10_feature_importance.png")


# ── Plot: Residuals ───────────────────────────────────────────────────────────
def plot_residuals(y_test, best_pred, best_name: str) -> None:
    residuals = y_test.values - best_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Predicted
    axes[0].scatter(best_pred, residuals, alpha=0.4, s=14, color="#7c5cd8", edgecolors="none")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residuals vs Predicted — {best_name}")
    axes[0].set_xlabel("Predicted Temperature (°C)")
    axes[0].set_ylabel("Residual (°C)")

    # Residual distribution
    axes[1].hist(residuals, bins=40, color="#7c5cd8", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (°C)")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "11_residuals.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 11_residuals.png")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[Temperature Prediction] Loading & preparing data...")
    X, y, feature_cols = load_and_prepare(CLEAN_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Features ({len(feature_cols)}): {feature_cols}\n")

    models = build_models()
    results = []
    predictions = {}
    trained_models = {}

    print("[Training & Evaluation]")
    for name, model in models.items():
        result = evaluate(name, model, X_train, X_test, y_train, y_test)
        predictions[name] = result.pop("y_pred")
        results.append(result)
        trained_models[name] = model

    # Save metrics
    metrics_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
    print(f"\n[Metrics saved] → {OUTPUT_DIR}/model_metrics.csv")
    print(metrics_df.to_string(index=False))

    # Best model by R²
    best_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    best_pred = predictions[best_name]
    print(f"\n[Best Model] {best_name}  (R²={metrics_df.iloc[0]['R2']:.4f})")

    # Save best model
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[Model saved] → {model_path}")

    # Plots
    print("\n[Plots]")
    plot_pred_vs_actual(y_test, predictions)
    plot_feature_importance(best_model, feature_cols)
    plot_residuals(y_test, best_pred, best_name)

    print(f"\n[Temperature Prediction] Pipeline complete.")


if __name__ == "__main__":
    run()
