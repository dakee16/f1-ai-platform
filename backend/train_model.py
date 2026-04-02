"""
Train a lap time prediction model using FastF1 data.

Key improvement over v1:
- Instead of predicting absolute lap times (which vary wildly by track),
  we predict the DELTA from the race's theoretical fastest lap.
  This lets the model learn degradation patterns that generalize across tracks.
  At prediction time we add the current race's baseline back in.

Run from backend folder:
    python train_model.py
"""

import fastf1
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

fastf1.Cache.enable_cache("../data")

TRAINING_RACES = [
    (2023, "Bahrain"),
    (2023, "Monaco"),
    (2023, "Silverstone"),
    (2023, "Monza"),
    (2023, "Spa"),
    (2023, "Suzuka"),
    (2024, "Bahrain"),
    (2024, "Monaco"),
    (2024, "Silverstone"),
    (2024, "Monza"),
    (2024, "Suzuka"),
]

COMPOUND_MAP = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}


def load_race_laps(year, gp):
    print(f"Loading {year} {gp}...")
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load(telemetry=False, weather=False, messages=False)

        laps = session.laps[[
            "Driver", "LapNumber", "LapTime", "Compound", "TyreLife", "Stint"
        ]].copy()

        laps = laps.dropna(subset=["LapTime", "Compound", "TyreLife"])
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        laps = laps[(laps["LapTimeSeconds"] > 60) & (laps["LapTimeSeconds"] < 200)]

        if len(laps) == 0:
            return None

        # ── Normalize to race baseline ──────────────────────────────────────
        # Baseline = 5th percentile lap time (avoids outlier fastest lap)
        # This represents a "clean air, good conditions" lap
        baseline = laps["LapTimeSeconds"].quantile(0.05)
        laps["LapTimeDelta"] = laps["LapTimeSeconds"] - baseline
        laps["RaceBaseline"] = baseline
        laps["Year"] = year
        laps["GrandPrix"] = gp

        print(f"  {len(laps)} laps, baseline: {baseline:.3f}s")
        return laps

    except Exception as e:
        print(f"  Failed: {e}")
        return None


def build_features(laps):
    df = laps.copy()
    df["CompoundCode"] = df["Compound"].map(COMPOUND_MAP)
    df = df.dropna(subset=["CompoundCode"])
    df["CompoundCode"] = df["CompoundCode"].astype(int)

    max_lap_per_race = df.groupby(["Year", "GrandPrix"])["LapNumber"].transform("max")
    df["LapNumber_normalized"] = df["LapNumber"] / max_lap_per_race
    df["TyreLife_squared"] = df["TyreLife"] ** 2
    df["TyreLife_x_compound"] = df["TyreLife"] * df["CompoundCode"]

    feature_cols = [
        "TyreLife",
        "TyreLife_squared",
        "TyreLife_x_compound",
        "CompoundCode",
        "LapNumber_normalized",
    ]

    X = df[feature_cols]
    y = df["LapTimeDelta"]  # predicting delta, not absolute time

    return X, y, feature_cols


def train():
    all_laps = []
    for year, gp in TRAINING_RACES:
        laps = load_race_laps(year, gp)
        if laps is not None:
            all_laps.append(laps)

    if not all_laps:
        print("No data loaded.")
        return

    combined = pd.concat(all_laps, ignore_index=True)
    print(f"\nTotal laps: {len(combined)}")

    X, y, feature_cols = build_features(combined)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"  MAE on delta: {mae:.3f}s ({mae*1000:.0f}ms)")
    print(f"  This means predicted lap time is within ~{mae:.2f}s of actual")

    importance = dict(zip(feature_cols, model.feature_importances_))
    print(f"\nFeature importances:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    model_data = {
        "model": model,
        "feature_cols": feature_cols,
        "compound_map": COMPOUND_MAP,
        "mae_seconds": mae,
        "trained_on": len(combined),
        "predicts_delta": True,
    }

    with open("lap_time_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nSaved to lap_time_model.pkl")
    print("Done!")


if __name__ == "__main__":
    train()