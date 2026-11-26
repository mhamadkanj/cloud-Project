"""Shared feature engineering for training (model/) and inference (api/).

Input: the last 24 hourly rows of French grid data, oldest first.
The model never sees absolute CO2 levels directly: levels drift a lot between
years (2022: ~52 g/kWh, 2024: ~19 g/kWh), so it works on relative changes
(log-ratios) and on the share of each source in consumption.
"""
import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "timestamp",
    "consumption_mw",
    "nuclear_mw",
    "gas_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "co2_g_per_kwh",
]
NUMERIC_COLUMNS = INPUT_COLUMNS[1:]
WINDOW_HOURS = 24   # we look at the last 24 hours
HORIZON_HOURS = 12  # we predict the next 12 hours


def feature_names() -> list[str]:
    names = [f"log_co2_lag_{i}_vs_last" for i in range(WINDOW_HOURS, 0, -1)]
    names += [f"cons_lag_{i}_vs_last" for i in range(WINDOW_HOURS, 0, -1)]
    names += ["gas_share", "nuclear_share", "wind_share", "solar_share", "hydro_share", "gas_change_share"]
    names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    return names


def build_features(window: pd.DataFrame, target_start: pd.Timestamp | None = None) -> np.ndarray:
    """window: exactly 24 hourly rows, oldest first.
    target_start: first forecast hour (defaults to last timestamp + 1 h)."""
    if len(window) != WINDOW_HOURS:
        raise ValueError(f"expected {WINDOW_HOURS} rows, got {len(window)}")
    log_co2 = np.log1p(window["co2_g_per_kwh"].to_numpy(dtype=float))
    cons = window["consumption_mw"].to_numpy(dtype=float)
    last, prev = window.iloc[-1], window.iloc[-2]
    c = max(float(last["consumption_mw"]), 1.0)
    if target_start is None:
        target_start = pd.Timestamp(last["timestamp"]) + pd.Timedelta(hours=1)
    t = pd.Timestamp(target_start)

    # A) 24 numbers: how CO2 changed compared with now (relative, in log scale)
    feats = list(log_co2 - log_co2[-1])
    # B) 24 numbers: how consumption changed compared with now (relative)
    feats += list((cons - cons[-1]) / c)
    # C) 6 numbers: share of each source right now + is gas rising?
    feats += [float(last["gas_mw"]) / c, float(last["nuclear_mw"]) / c, float(last["wind_mw"]) / c,
              float(last["solar_mw"]) / c, float(last["hydro_mw"]) / c,
              (float(last["gas_mw"]) - float(prev["gas_mw"])) / c]
    # D) 6 numbers: hour, weekday, month as points on a circle (sin/cos)
    feats += [np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24),
              np.sin(2 * np.pi * t.dayofweek / 7), np.cos(2 * np.pi * t.dayofweek / 7),
              np.sin(2 * np.pi * (t.month - 1) / 12), np.cos(2 * np.pi * (t.month - 1) / 12)]
    return np.asarray(feats, dtype=float)


def to_absolute(predicted_log_ratio: np.ndarray, last_co2: float) -> np.ndarray:
    """The model predicts log(1+co2_future) - log(1+co2_last); convert back to g/kWh."""
    return np.clip(np.expm1(np.asarray(predicted_log_ratio, dtype=float) + np.log1p(last_co2)), 0, None)


def best_window(forecast: np.ndarray, job_hours: int = 2) -> tuple[int, float]:
    """Index of the first hour of the cleanest `job_hours` window, and its mean CO2."""
    f = np.asarray(forecast, dtype=float)
    means = np.convolve(f, np.ones(job_hours) / job_hours, mode="valid")
    i = int(means.argmin())
    return i, float(means[i])
