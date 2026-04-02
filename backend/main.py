from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import pandas as pd
import math
from datetime import datetime
import pickle
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastf1.Cache.enable_cache("../data")

# ─── Per-race cache (key = "YEAR_GP") ────────────────────────────────────────
session_cache = {}
laps_cache = {}
status_cache = {}
stints_cache = {}
positions_cache = {}
overview_cache = {}

def cache_key(year: int, gp: str) -> str:
    return f"{year}_{gp}"

def load_session(year: int, gp: str):
    key = cache_key(year, gp)
    if key not in session_cache:
        session = fastf1.get_session(year, gp, "R")
        session.load()
        session_cache[key] = session
    return session_cache[key]


# ─── Available races ──────────────────────────────────────────────────────────
@app.get("/available-races")
def get_available_races():
    races = {}
    now = datetime.now()

    for year in range(2022, 2027):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)

            # Only include rounds that have already happened
            schedule["EventDate"] = pd.to_datetime(schedule["EventDate"], utc=True)
            cutoff = pd.Timestamp(now, tz="UTC")
            past = schedule[schedule["EventDate"] <= cutoff]

            gp_names = past["EventName"].tolist()
            if gp_names:
                races[str(year)] = gp_names
        except Exception as e:
            print(f"Schedule load failed for {year}: {e}")
            continue

    return races


# ─── Race laps ────────────────────────────────────────────────────────────────
@app.get("/race-laps")
def get_race_laps(year: int = 2024, gp: str = "Monaco"):
    key = cache_key(year, gp)
    if key in laps_cache:
        return laps_cache[key]

    session = load_session(year, gp)
    laps = session.laps[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]].copy()
    laps = laps.dropna(subset=["LapTime"])
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTimeSeconds"] < 200]

    laps_cache[key] = laps.to_dict(orient="records")
    return laps_cache[key]


# ─── Driver status ────────────────────────────────────────────────────────────
@app.get("/driver-status")
def get_driver_status(year: int = 2024, gp: str = "Monaco"):
    key = cache_key(year, gp)
    if key in status_cache:
        return status_cache[key]

    session = load_session(year, gp)
    results = session.results[["Abbreviation", "Status", "ClassifiedPosition"]].copy()

    status_map = {}
    for _, row in results.iterrows():
        abbr = row["Abbreviation"]
        status = str(row["Status"])
        classified = row["ClassifiedPosition"]
        finished = str(classified) != "R"
        status_map[abbr] = {"status": status, "finished": finished}

    status_cache[key] = status_map
    return status_cache[key]


# ─── Stint data ───────────────────────────────────────────────────────────────
@app.get("/stint-data")
def get_stint_data(year: int = 2024, gp: str = "Monaco"):
    key = cache_key(year, gp)
    if key in stints_cache:
        return stints_cache[key]

    session = load_session(year, gp)
    laps = session.laps[["Driver", "LapNumber", "Compound", "Stint"]].copy()
    laps = laps.dropna(subset=["Compound", "Stint"])

    stints = (
        laps.groupby(["Driver", "Stint", "Compound"])
        .agg(StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"))
        .reset_index()
    )

    stints_cache[key] = stints.to_dict(orient="records")
    return stints_cache[key]


# ─── Position data ────────────────────────────────────────────────────────────
@app.get("/position-data")
def get_position_data(year: int = 2024, gp: str = "Monaco"):
    key = cache_key(year, gp)
    if key in positions_cache:
        return positions_cache[key]

    session = load_session(year, gp)
    laps = session.laps[["Driver", "LapNumber", "Position"]].copy()
    laps = laps.dropna(subset=["Position"])
    laps["Position"] = laps["Position"].astype(int)

    positions_cache[key] = laps.to_dict(orient="records")
    return positions_cache[key]


# ─── Race overview ────────────────────────────────────────────────────────────
@app.get("/race-overview")
def get_race_overview(year: int = 2024, gp: str = "Monaco"):
    key = cache_key(year, gp)
    if key in overview_cache:
        return overview_cache[key]

    session = load_session(year, gp)

    laps = session.laps[["Driver", "LapNumber", "LapTime"]].copy()
    laps = laps.dropna(subset=["LapTime"])
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTimeSeconds"] < 200]
    fastest_row = laps.loc[laps["LapTimeSeconds"].idxmin()]

    results = session.results[["Abbreviation", "ClassifiedPosition"]].copy()
    results = results[results["ClassifiedPosition"].apply(lambda x: str(x).isdigit())]
    results["ClassifiedPosition"] = results["ClassifiedPosition"].astype(int)
    results = results.sort_values("ClassifiedPosition")
    podium = results.head(3)[["Abbreviation", "ClassifiedPosition"]].to_dict(orient="records")

    overview_cache[key] = {
        "fastestLap": {
            "driver": fastest_row["Driver"],
            "lapNumber": int(fastest_row["LapNumber"]),
            "timeSeconds": fastest_row["LapTimeSeconds"],
        },
        "podium": podium,
    }
    return overview_cache[key]


# ─── Debug ────────────────────────────────────────────────────────────────────
@app.get("/debug-status")
def debug_status(year: int = 2024, gp: str = "Monaco"):
    session = load_session(year, gp)
    results = session.results[["Abbreviation", "Status", "ClassifiedPosition"]].copy()
    output = []
    for _, row in results.iterrows():
        output.append({
            "driver": row["Abbreviation"],
            "status": str(row["Status"]),
            "classifiedPosition": str(row["ClassifiedPosition"]),
        })
    return output

# ─── Load model once at startup ───────────────────────────────────────────────
model_data = None

def get_model():
    global model_data
    if model_data is None:
        try:
            with open("lap_time_model.pkl", "rb") as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            return None
    return model_data

@app.get("/predict-laptime")
def predict_laptime(
    compound: str = "MEDIUM",
    tyre_life: int = 10,
    lap_number: int = 30,
    total_laps: int = 78,
    baseline_seconds: float = 0,
):
    md = get_model()
    if md is None:
        return {"error": "Model not trained yet. Run train_model.py first."}

    compound_map = md["compound_map"]
    compound_code = compound_map.get(compound.upper(), 1)
    lap_normalized = lap_number / max(total_laps, 1)

    features = np.array([[
        tyre_life,
        tyre_life ** 2,
        tyre_life * compound_code,
        compound_code,
        lap_normalized,
    ]])

    predicted_delta = float(md["model"].predict(features)[0])

    # If a baseline is provided, give absolute time, otherwise just the delta
    predicted_seconds = round(
        baseline_seconds + predicted_delta if baseline_seconds > 0 else predicted_delta,
        3
    )

    return {
        "predictedSeconds": predicted_seconds,
        "predictedDelta": round(predicted_delta, 3),
        "compound": compound,
        "tyreLife": tyre_life,
        "lapNumber": lap_number,
        "modelMAE": round(md["mae_seconds"], 3),
        "hasBaseline": baseline_seconds > 0,
    }

@app.get("/model-info")
def model_info():
    md = get_model()
    if md is None:
        return {"trained": False}
    return {
        "trained": True,
        "trainedOnLaps": md["trained_on"],
        "maeSeconds": round(md["mae_seconds"], 3),
    }

@app.get("/simulate-strategy")
def simulate_strategy(
    year: int = 2024,
    gp: str = "Monaco",
    driver: str = "LEC",
    custom_pit_lap: int = 0,
    custom_compound_1: str = "MEDIUM",
    custom_compound_2: str = "HARD",
):
    md = get_model()
    if md is None:
        return {"error": "Model not trained yet."}

    # Get actual stints for this driver
    key = cache_key(year, gp)
    session = load_session(year, gp)

    laps = session.laps[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]].copy()
    laps = laps.dropna(subset=["LapTime", "Compound"])
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps[(laps["LapTimeSeconds"] > 60) & (laps["LapTimeSeconds"] < 200)]

    driver_laps = laps[laps["Driver"] == driver].copy()
    if driver_laps.empty:
        return {"error": f"No lap data for {driver}"}

    total_laps = int(driver_laps["LapNumber"].max())
    baseline = float(driver_laps["LapTimeSeconds"].quantile(0.05))

    compound_map = md["compound_map"]

    def predict_lap(compound, tyre_life, lap_number):
        compound_code = compound_map.get(compound.upper(), 1)
        lap_normalized = lap_number / max(total_laps, 1)
        features = np.array([[
            tyre_life,
            tyre_life ** 2,
            tyre_life * compound_code,
            compound_code,
            lap_normalized,
        ]])
        delta = float(md["model"].predict(features)[0])
        return round(baseline + delta, 3)

    # ── Simulate actual strategy ──────────────────────────────────────────────
    actual_laps_detail = []
    for _, row in driver_laps.iterrows():
        predicted = predict_lap(row["Compound"], int(row["TyreLife"]), int(row["LapNumber"]))
        actual_laps_detail.append({
            "lap": int(row["LapNumber"]),
            "compound": row["Compound"],
            "tyreLife": int(row["TyreLife"]),
            "actualSeconds": round(row["LapTimeSeconds"], 3),
            "predictedSeconds": predicted,
        })

    actual_predicted_total = sum(l["predictedSeconds"] for l in actual_laps_detail)
    actual_real_total = sum(l["actualSeconds"] for l in actual_laps_detail)

    # Get actual stint structure
    stint_data = session.laps[session.laps["Driver"] == driver][
        ["LapNumber", "Compound", "Stint", "TyreLife"]
    ].dropna()
    actual_stints = (
        stint_data.groupby(["Stint", "Compound"])
        .agg(StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"))
        .reset_index()
        .sort_values("StartLap")
        .to_dict(orient="records")
    )

    # ── Simulate custom strategy ──────────────────────────────────────────────
    custom_laps_detail = []
    if custom_pit_lap > 0 and custom_pit_lap < total_laps:
        # Stint 1: custom_compound_1 from lap 1 to custom_pit_lap
        for lap in range(1, custom_pit_lap + 1):
            tyre_life = lap
            predicted = predict_lap(custom_compound_1, tyre_life, lap)
            custom_laps_detail.append({
                "lap": lap,
                "compound": custom_compound_1,
                "predictedSeconds": predicted,
            })
        # Stint 2: custom_compound_2 from pit lap to end
        for lap in range(custom_pit_lap + 1, total_laps + 1):
            tyre_life = lap - custom_pit_lap
            predicted = predict_lap(custom_compound_2, tyre_life, lap)
            custom_laps_detail.append({
                "lap": lap,
                "compound": custom_compound_2,
                "predictedSeconds": predicted,
            })

    custom_predicted_total = sum(l["predictedSeconds"] for l in custom_laps_detail)

    # Delta: positive = custom is slower, negative = custom is faster
    delta = round(custom_predicted_total - actual_predicted_total, 3) if custom_laps_detail else None

    return {
        "driver": driver,
        "totalLaps": total_laps,
        "baseline": round(baseline, 3),
        "actualStints": actual_stints,
        "actualPredictedTotal": round(actual_predicted_total, 3),
        "actualRealTotal": round(actual_real_total, 3),
        "customPredictedTotal": round(custom_predicted_total, 3) if custom_laps_detail else None,
        "delta": delta,
        "actualLaps": actual_laps_detail,
        "customLaps": custom_laps_detail,
    }