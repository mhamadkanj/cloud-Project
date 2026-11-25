"""Turn the raw RTE éCO2mix export into a small hourly dataset.

Source: RTE, "Données éCO2mix nationales consolidées et définitives",
https://www.data.gouv.fr/datasets/donnees-eco2mix-nationales-consolidees-et-definitives
Licence Ouverte / Open Licence 2.0 (Etalab).

Usage:  python prepare_data.py --raw data/raw.csv
Output: data/eco2mix_hourly_2022_2024.csv
"""
import argparse
from pathlib import Path

import pandas as pd

# 1) the columns we keep: French name -> English name
RENAME = {
    "Date et Heure": "timestamp",
    "Consommation (MW)": "consumption_mw",
    "Nucléaire (MW)": "nuclear_mw",
    "Gaz (MW)": "gas_mw",
    "Eolien (MW)": "wind_mw",
    "Solaire (MW)": "solar_mw",
    "Hydraulique (MW)": "hydro_mw",
    "Taux de CO2 (g/kWh)": "co2_g_per_kwh",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default=str(Path(__file__).parent / "data" / "eco2mix_hourly_2022_2024.csv"))
    args = parser.parse_args()

    # 2) read only the 8 columns, rename them, drop empty rows
    df = pd.read_csv(args.raw, sep=";", encoding="utf-8-sig", usecols=list(RENAME))
    df = df.rename(columns=RENAME).dropna(subset=["consumption_mw", "co2_g_per_kwh"])

    # 3) one value per hour (average). UTC avoids problems at summer/winter time changes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    hourly = df.set_index("timestamp").sort_index().resample("1h").mean()
    hourly = hourly.interpolate(limit=2).dropna().round(1)

    # 4) back to Paris time, then save
    hourly.index = hourly.index.tz_convert("Europe/Paris")
    hourly = hourly.reset_index()
    hourly["timestamp"] = hourly["timestamp"].map(lambda t: t.isoformat())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.out, index=False)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"{len(hourly)} hourly rows, {hourly['timestamp'].iloc[0]} -> {hourly['timestamp'].iloc[-1]}, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()