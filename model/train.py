"""Train the CO2-intensity forecaster.

Task: from the last 24 hours of French grid data, predict CO2 intensity
(g/kWh) for each of the next 12 hours.
Split: train on windows fully inside 2022-2023, test on windows fully inside
2024 (never seen in training). Fixed random seed.

Metrics (test set, compared with two naive baselines):
- MAE and RMSE in g/kWh, averaged over the 12 forecast hours
- CO2 saved: run a 2-hour job in the window the forecast says is cleanest,
  instead of running it now; measured on the ACTUAL values.

Usage: python train.py
Output: artifacts/model-1.0.0.joblib, artifacts/metrics.json
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from features import HORIZON_HOURS, WINDOW_HOURS, build_features, feature_names, to_absolute

SEED = 42
MODEL_VERSION = "1.0.0"
JOB_HOURS = 2
HERE = Path(__file__).parent
DATA = HERE / "data" / "eco2mix_hourly_2022_2024.csv"
OUT = HERE / "artifacts"


def make_samples(df: pd.DataFrame, keep) -> dict:
    """Slide a 24 h window over the data: features = the 24 h, answer = the next 12 h."""
    co2 = df["co2_g_per_kwh"].to_numpy(dtype=float)
    years = df["timestamp"].str[:4].to_numpy()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Paris")
    X, Y, last, yday = [], [], [], []
    for end in range(WINDOW_HOURS, len(df) - HORIZON_HOURS + 1):
        if not (keep(years[end - WINDOW_HOURS]) and keep(years[end + HORIZON_HOURS - 1])):
            continue
        window = df.iloc[end - WINDOW_HOURS:end]
        X.append(build_features(window, ts.iloc[end]))
        Y.append(co2[end:end + HORIZON_HOURS])
        last.append(co2[end - 1])
        yday.append(co2[end - WINDOW_HOURS:end - WINDOW_HOURS + HORIZON_HOURS])
    return {"X": np.array(X), "Y": np.array(Y), "last": np.array(last), "yday": np.array(yday)}


def mae(a, b) -> float:
    return round(float(np.abs(a - b).mean()), 2)


def rmse(a, b) -> float:
    return round(float(np.sqrt(((a - b) ** 2).mean())), 2)


def co2_saved_pct(true: np.ndarray, forecast: np.ndarray) -> float:
    """Average % of CO2 saved by running a JOB_HOURS job in the forecast's cleanest window vs now."""
    k = JOB_HOURS
    f_means = np.stack([forecast[:, i:i + k].mean(1) for i in range(HORIZON_HOURS - k + 1)], 1)
    t_means = np.stack([true[:, i:i + k].mean(1) for i in range(HORIZON_HOURS - k + 1)], 1)
    chosen = f_means.argmin(1)
    got = t_means[np.arange(len(chosen)), chosen]
    now = t_means[:, 0]
    return round(float(100 * (1 - got.mean() / now.mean())), 1)


def main() -> None:
    # 1) examples: train = 2022-2023, test = 2024
    df = pd.read_csv(DATA)
    tr = make_samples(df, lambda y: y < "2024")
    te = make_samples(df, lambda y: y == "2024")
    print(f"train samples: {len(tr['X'])}, test samples: {len(te['X'])}")

    # 2) train: target = relative change vs the last observed hour
    y_tr = np.log1p(tr["Y"]) - np.log1p(tr["last"])[:, None]
    model = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED))
    t0 = time.time()
    model.fit(tr["X"], y_tr)
    train_s = time.time() - t0

    # 3) test on 2024, convert back to g/kWh
    pred = np.stack([to_absolute(p, l) for p, l in zip(model.predict(te["X"]), te["last"])])
    t0 = time.time()
    model.predict(te["X"][:1])
    latency_ms = (time.time() - t0) * 1000

    # 4) compare with the two baselines
    true = te["Y"]
    now = np.repeat(te["last"][:, None], HORIZON_HOURS, 1)
    oracle = co2_saved_pct(true, true)
    metrics = {
        "model_version": MODEL_VERSION,
        "task": "Forecast French grid CO2 intensity (g/kWh) for the next 12 hours from the last 24 hours",
        "data": "RTE eCO2mix national consolidated and final data, Licence Ouverte 2.0",
        "train_period": "2022-01-01 to 2023-12-31",
        "test_period": "2024-01-01 to 2024-12-31",
        "train_samples": int(len(tr["X"])),
        "test_samples": int(len(te["X"])),
        "results": {
            "model": {"mae": mae(true, pred), "rmse": rmse(true, pred), "co2_saved_pct": co2_saved_pct(true, pred)},
            "baseline_same_as_now": {"mae": mae(true, now), "rmse": rmse(true, now), "co2_saved_pct": co2_saved_pct(true, now)},
            "baseline_same_hour_yesterday": {"mae": mae(true, te["yday"]), "rmse": rmse(true, te["yday"]),
                                             "co2_saved_pct": co2_saved_pct(true, te["yday"])},
            "perfect_hindsight_co2_saved_pct": oracle,
        },
        "mae_by_horizon_hours": {str(h + 1): mae(true[:, h], pred[:, h]) for h in range(HORIZON_HOURS)},
        "mae_by_horizon_baseline_same_as_now": {str(h + 1): mae(true[:, h], now[:, h]) for h in range(HORIZON_HOURS)},
        "job_hours_for_saving_metric": JOB_HOURS,
        "train_seconds": round(train_s, 1),
        "single_inference_ms": round(latency_ms, 1),
        "seed": SEED,
        "n_features": len(feature_names()),
    }

    # 5) save the model and the results
    OUT.mkdir(exist_ok=True)
    path = OUT / f"model-{MODEL_VERSION}.joblib"
    joblib.dump({"model": model, "version": MODEL_VERSION, "features": feature_names(),
                 "target": "log1p(co2_future) - log1p(co2_last)"}, path, compress=3)
    metrics["model_size_mb"] = round(path.stat().st_size / 1e6, 2)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
