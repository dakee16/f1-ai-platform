from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastf1.Cache.enable_cache("../data")

# Cache both the laps and the session results
cached_laps = None
cached_status = None


def load_session():
    """Load session once, reuse for both endpoints."""
    session = fastf1.get_session(2024, "Monaco", "R")
    session.load()
    return session

@app.get("/")
def home():
    return {"message": "Backend working ✅"}

@app.get("/race-laps")
def get_race_laps():
    global cached_laps
    if cached_laps is not None:
        return cached_laps

    session = load_session()
    laps = session.laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife']].copy()
    laps = laps.dropna(subset=['LapTime'])
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps = laps[laps['LapTimeSeconds'] < 200]

    cached_laps = laps.to_dict(orient="records")
    return cached_laps

@app.get("/driver-status")
def get_driver_status():
    global cached_status
    if cached_status is not None:
        return cached_status

    session = load_session()

    results = session.results[['Abbreviation', 'Status', 'ClassifiedPosition']].copy()

    status_map = {}
    for _, row in results.iterrows():
        abbr = row['Abbreviation']
        status = str(row['Status'])
        classified = row['ClassifiedPosition']

        # ClassifiedPosition is a real number for finishers/lapped cars, NaN for true DNFs
        finished = str(classified) != "R"

        status_map[abbr] = {
            "status": status,
            "finished": finished
        }

    cached_status = status_map
    return cached_status

@app.get("/stint-data")
def get_stint_data():
    session = load_session()

    laps = session.laps[['Driver', 'LapNumber', 'Compound', 'Stint']].copy()
    laps = laps.dropna(subset=['Compound', 'Stint'])

    # Group by driver + stint number + compound
    # Get the first and last lap of each stint
    stints = (
        laps.groupby(['Driver', 'Stint', 'Compound'])
        .agg(StartLap=('LapNumber', 'min'), EndLap=('LapNumber', 'max'))
        .reset_index()
    )

    return stints.to_dict(orient='records')