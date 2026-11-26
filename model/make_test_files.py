"""Build the files we upload to Blob input/ during tests and the demo.

valid/   : 24 hourly rows from 2024 (never seen in training), each ending at 18:00.
           Some days are random (seeded); a few are the days where shifting a job
           saves the most, which make a clear demo. The file name says which.
invalid/ : files that the dispatcher must reject (wrong column, negative value,
           wrong number of rows, not a CSV).

Usage: python make_test_files.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import INPUT_COLUMNS, WINDOW_HOURS, HORIZON_HOURS, best_window, build_features, to_absolute

SEED = 42
HERE = Path(__file__).parent
DATA = HERE / "data" / "eco2mix_hourly_2022_2024.csv"
MODEL = HERE / "artifacts" / "model-1.0.0.joblib"
OUT = HERE / "test_files"


def main() -> None:
    df = pd.read_csv(DATA)[INPUT_COLUMNS]
    model = joblib.load(MODEL)["model"]
    df24 = df[df["timestamp"].str.startswith("2024")].reset_index(drop=True)

    # 1) every 24 h window ending at 18:00 in 2024: how much would the advice save?
    ends = [i + 1 for i in range(WINDOW_HOURS - 1, len(df24) - HORIZON_HOURS)
            if df24["timestamp"].iloc[i][11:13] == "18"]
    rows = []
    for end in ends:
        window = df24.iloc[end - WINDOW_HOURS:end]
        last = float(window["co2_g_per_kwh"].iloc[-1])
        fc = to_absolute(model.predict(build_features(window)[None, :])[0], last)
        actual = df24["co2_g_per_kwh"].iloc[end:end + HORIZON_HOURS].to_numpy()
        i, _ = best_window(fc)
        saved = 100 * (1 - actual[i:i + 2].mean() / actual[0:2].mean())
        rows.append((end, saved))

    # 2) pick 3 best days (demo) + 5 random days (honest sample)
    rng = np.random.default_rng(SEED)
    best = sorted(rows, key=lambda r: -r[1])[:3]
    rest = [r for r in rows if r not in best]
    random_pick = [rest[j] for j in rng.choice(len(rest), size=5, replace=False)]

    (OUT / "valid").mkdir(parents=True, exist_ok=True)
    (OUT / "invalid").mkdir(parents=True, exist_ok=True)
    for kind, picked in (("demo", best), ("random", random_pick)):
        for end, saved in picked:
            window = df24.iloc[end - WINDOW_HOURS:end]
            day = window["timestamp"].iloc[-1][:10]
            name = f"grid_{day}_18h_{kind}.csv"
            window.to_csv(OUT / "valid" / name, index=False)
            print(f"{name}: shifting a 2 h job saves {saved:.0f}% (actual values)")

    # 3) invalid files, on purpose
    sample = df24.iloc[ends[0] - WINDOW_HOURS:ends[0]].copy()
    sample.drop(columns=["gas_mw"]).to_csv(OUT / "invalid" / "missing_column_gas.csv", index=False)
    bad = sample.copy()
    bad.loc[bad.index[5], "consumption_mw"] = -500
    bad.to_csv(OUT / "invalid" / "negative_value.csv", index=False)
    sample.head(20).to_csv(OUT / "invalid" / "only_20_rows.csv", index=False)
    (OUT / "invalid" / "not_a_csv.txt").write_text("hello, this is not grid data\n")
    print("invalid files written")

    # 4) actual 2024 values, used by the worker to show predicted vs actual
    df24[["timestamp", "co2_g_per_kwh"]].to_csv(OUT / "actuals_2024.csv", index=False)


if __name__ == "__main__":
    main()
