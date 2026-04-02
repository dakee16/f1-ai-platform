# F1 AI Analytics Platform 🏎️

A full-stack Formula 1 race analytics dashboard with an XGBoost-powered lap time prediction model and pit stop strategy simulator. Built with React, FastAPI, and FastF1.

**Live Demo:** [f1-ai-platform-7hli.vercel.app](https://f1-ai-platform-7hli.vercel.app)

---

## Features

### Race Analytics
- **Race Overview** — podium display and fastest lap of the race
- **Lap Time Comparison** — multi-driver lap time chart across the full race distance
- **Race Position Chart** — position-by-lap visualization showing overtakes and pit stop drops
- **Tire Strategy** — horizontal stint chart for all 20 drivers, colored by compound (Soft / Medium / Hard)
- **Driver Lap Detail** — full per-lap breakdown with compound, tyre life, and delta from fastest lap
- **DNF Detection** — drivers who did not finish are flagged across all views

### AI / ML Features
- **Lap Time Predictor** — XGBoost regression model that predicts lap time for any compound, tyre age, and lap number
- **Pit Stop Strategy Simulator** — compare a driver's actual strategy against a custom one using ML-predicted lap times, with a race time delta and lap-by-lap chart

### Data Coverage
- Races from **2022 to 2026** via the FastF1 API
- All sessions load from an in-memory cache — first load per race triggers a background download, subsequent loads are instant

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Recharts, deployed on Vercel |
| Backend | FastAPI (Python), deployed on Railway |
| Data | FastF1, pandas |
| ML Model | XGBoost, scikit-learn, NumPy |
| Cache | In-memory Python dicts with background prewarm thread |

---

## ML Model

The lap time prediction model is an XGBoost regressor trained on real F1 telemetry data.

- **Training data:** 11,793 laps across 11 races (2023–2024 seasons)
- **Target:** Lap time delta from race baseline (5th percentile lap time)
- **Features:** Tyre life, tyre life², tyre life × compound code, compound code, normalized lap number
- **Accuracy:** ±1.94s MAE

The model is serialized to `lap_time_model.pkl` and loaded at server startup.

---

## Project Structure

```
f1-ai-platform/
├── backend/
│   ├── main.py              # FastAPI app with all endpoints + prewarm logic
│   ├── train_model.py       # XGBoost training script
│   ├── lap_time_model.pkl   # Trained model (committed to repo)
│   ├── requirements.txt
│   ├── Procfile
│   └── nixpacks.toml
├── frontend/
│   └── src/
│       └── App.js           # Full React app (single file)
└── data/                    # Local FastF1 cache (gitignored)
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /available-races` | Returns all past races 2022–2026 |
| `GET /race-laps` | Lap times, compound, tyre life per driver |
| `GET /driver-status` | DNF detection via classified position |
| `GET /stint-data` | Tyre stint groupby per driver |
| `GET /position-data` | Race position per lap |
| `GET /race-overview` | Fastest lap + podium |
| `GET /predict-laptime` | XGBoost lap time prediction |
| `GET /simulate-strategy` | Pit stop strategy comparison |
| `GET /model-info` | Model metadata |
| `GET /prewarm-status` | Shows how many races are currently cached |

All endpoints accept `?year=YYYY&gp=Grand Prix Name` query params.

---

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

Set the FastF1 cache directory (optional):
```bash
export FASTF1_CACHE=../data
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Make sure `App.js` has the API pointing to localhost:
```javascript
const API = "http://127.0.0.1:8001";
```

### Training the Model

```bash
cd backend
python train_model.py
```

This downloads ~11,000 laps of F1 data, trains the XGBoost model, and saves `lap_time_model.pkl`.

---

## Deployment

- **Frontend** → [Vercel](https://vercel.com) — root directory: `frontend`, build: `npm run build`
- **Backend** → [Railway](https://railway.app) — root directory: `backend`, start command: `python -m uvicorn main:app --host 0.0.0.0 --port 8001`

On server startup, a background thread preloads all available races into memory so that all subsequent requests return instantly.

---

## Screenshots

| Race Overview | Lap Time Comparison |
|---|---|
| Podium + fastest lap card | Multi-driver chart with pit stop spikes |

| Tire Strategy | Strategy Simulator |
|---|---|
| Horizontal stint bars for all 20 drivers | Custom vs actual strategy with delta |

---

## Built By
[Daksh](https://github.com/dakee16)
