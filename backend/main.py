from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import pandas as pd
import math
from datetime import datetime

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