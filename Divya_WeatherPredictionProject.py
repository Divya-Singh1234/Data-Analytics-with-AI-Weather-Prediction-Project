"""
main.py — Orchestrates the full ML pipeline end-to-end:
  1. Generate dataset
  2. Clean data
  3. Exploratory analysis
  4. Visualization
  5. Temperature prediction
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import generate_dataset       # noqa: E402
import data_cleaning          # noqa: E402
import exploratory_analysis   # noqa: E402
import visualization          # noqa: E402
import temperature_prediction # noqa: E402


def main():
    print("=" * 60)
    print("  WEATHER ML PROJECT — Full Pipeline")
    print("=" * 60)

    print("\n[Step 1/5] Generating dataset...")
    os.makedirs("weather_ml_project/data", exist_ok=True)
    generate_dataset  # module-level code runs on import; explicitly call nothing

    print("\n[Step 2/5] Cleaning data...")
    data_cleaning.run()

    print("\n[Step 3/5] Exploratory analysis...")
    exploratory_analysis.run()

    print("\n[Step 4/5] Visualizations...")
    visualization.run()

    print("\n[Step 5/5] Temperature prediction...")
    temperature_prediction.run()

    print("\n" + "=" * 60)
    print("  Pipeline complete! Outputs in weather_ml_project/outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
