"""Carbon-aware scheduler: inference API.

Endpoints
  GET  /health   status and model loading time
  GET  /version  API and model versions
  POST /predict  24 hourly rows in -> 12-hour CO2 forecast + cleanest window out
  GET  /metrics  number of calls and average latency

Configuration comes only from environment variables (no secrets in code):
  MODEL_PATH                              path to the .joblib model
  API_VERSION                             set by CI to the commit SHA
  APPLICATIONINSIGHTS_CONNECTION_STRING   set in Azure only; enables telemetry
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# features.py is shared with training: one file, one truth.
# Locally it lives in <repo>/model; in the Docker image it is copied next to app/.
_model_dir = Path(__file__).resolve().parents[2] / "model"
if _model_dir.exists():
    sys.path.append(str(_model_dir))
from features import HORIZON_HOURS, INPUT_COLUMNS, WINDOW_HOURS, best_window, build_features, to_absolute  # noqa: E402

API_VERSION = os.getenv("API_VERSION", "1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "model/artifacts/model-1.0.0.joblib")

# --- telemetry: only when running in Azure (connection string present) ---
_meter = None
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import metrics

    configure_azure_monitor()
    _meter = metrics.get_meter("carbon-api")
_inference_counter = _meter.create_counter("inferences_total") if _meter else None
_latency_hist = _meter.create_histogram("model_latency_ms", unit="ms") if _meter else None

STATE = {"model": None, "model_version": None, "load_seconds": None,
         "calls": 0, "errors": 0, "total_latency_ms": 0.0, "started": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    bundle = joblib.load(MODEL_PATH)
    STATE["model"] = bundle["model"]
    STATE["model_version"] = bundle["version"]
    STATE["load_seconds"] = round(time.time() - t0, 3)
    yield


app = FastAPI(title="Carbon-aware scheduler API", version=API_VERSION, lifespan=lifespan)


# --- request / response schemas (Pydantic validates every request) ---
class GridHour(BaseModel):
    timestamp: datetime
    consumption_mw: float = Field(gt=0, lt=200_000)
    nuclear_mw: float = Field(ge=0, lt=100_000)
    gas_mw: float = Field(ge=0, lt=50_000)
    wind_mw: float = Field(ge=0, lt=50_000)
    solar_mw: float = Field(ge=0, lt=50_000)
    hydro_mw: float = Field(ge=0, lt=50_000)
    co2_g_per_kwh: float = Field(ge=0, lt=1_000)


class PredictRequest(BaseModel):
    rows: list[GridHour] = Field(min_length=WINDOW_HOURS, max_length=WINDOW_HOURS,
                                 description="Exactly 24 hourly rows, oldest first")
    job_hours: int = Field(default=2, ge=1, le=6, description="Duration of the job to schedule")


class HourForecast(BaseModel):
    timestamp: datetime
    co2_g_per_kwh: float


class PredictResponse(BaseModel):
    model_version: str
    forecast: list[HourForecast]
    best_window_start: datetime
    best_window_end: datetime
    best_window_avg_co2: float
    run_now_avg_co2: float
    co2_saved_pct: float
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if STATE["model"] is not None else "loading",
            "model_loaded": STATE["model"] is not None,
            "model_load_seconds": STATE["load_seconds"]}


@app.get("/version")
def version() -> dict:
    return {"api_version": API_VERSION, "model_version": STATE["model_version"]}


@app.get("/metrics")
def get_metrics() -> dict:
    calls = STATE["calls"]
    return {"calls": calls, "errors": STATE["errors"],
            "avg_latency_ms": round(STATE["total_latency_ms"] / calls, 2) if calls else 0.0,
            "uptime_seconds": round(time.time() - STATE["started"])}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    rows = sorted(req.rows, key=lambda r: r.timestamp)
    df = pd.DataFrame([r.model_dump() for r in rows])[INPUT_COLUMNS]
    gaps = df["timestamp"].diff().dropna()
    if not (gaps == timedelta(hours=1)).all():
        STATE["errors"] += 1
        raise HTTPException(status_code=422, detail="rows must be 24 consecutive hours")

    last_co2 = float(df["co2_g_per_kwh"].iloc[-1])
    change = STATE["model"].predict(build_features(df)[None, :])[0]
    fc = to_absolute(change, last_co2)

    start = rows[-1].timestamp + timedelta(hours=1)
    hours = [start + timedelta(hours=h) for h in range(HORIZON_HOURS)]
    i, best_avg = best_window(fc, req.job_hours)
    now_avg = float(np.mean(fc[:req.job_hours]))
    saved = 0.0 if now_avg <= 0 else max(0.0, 100 * (1 - best_avg / now_avg))

    latency = (time.perf_counter() - t0) * 1000
    STATE["calls"] += 1
    STATE["total_latency_ms"] += latency
    if _inference_counter:
        _inference_counter.add(1, {"model_version": STATE["model_version"]})
        _latency_hist.record(latency, {"model_version": STATE["model_version"]})

    return PredictResponse(
        model_version=STATE["model_version"],
        forecast=[HourForecast(timestamp=t, co2_g_per_kwh=round(float(v), 1)) for t, v in zip(hours, fc)],
        best_window_start=hours[i],
        best_window_end=hours[i] + timedelta(hours=req.job_hours),
        best_window_avg_co2=round(best_avg, 1),
        run_now_avg_co2=round(now_avg, 1),
        co2_saved_pct=round(saved, 1),
        latency_ms=round(latency, 1),
    )
