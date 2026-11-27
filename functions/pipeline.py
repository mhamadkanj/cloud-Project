"""Pure pipeline logic (no Azure calls), so it can be unit-tested locally.

- validate_csv: the dispatcher's checks on an uploaded file
- item_id / make_item: the summary "card" the worker writes to Cosmos DB
- load_actuals / actual_values: real 2024 CO2 values, to compare with the forecast
"""
import csv
import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

REQUIRED_COLUMNS = [
    "timestamp",
    "consumption_mw",
    "nuclear_mw",
    "gas_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "co2_g_per_kwh",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
EXPECTED_ROWS = 24
MAX_BYTES = 50_000  # a valid file is about 2-3 KB
ACTUALS_FILE = Path(__file__).parent / "actuals_2024.csv"


def validate_csv(name: str, content: bytes) -> tuple[list[dict] | None, str]:
    """Return (rows, "ok") for a valid file, or (None, reason) for a file to reject."""
    if not name.lower().endswith(".csv"):
        return None, "not a .csv file"
    if len(content) > MAX_BYTES:
        return None, f"file too large ({len(content)} bytes, max {MAX_BYTES})"
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, "file is not UTF-8 text"

    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        return None, "missing column(s): " + ", ".join(missing)

    rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        return None, f"expected {EXPECTED_ROWS} rows, got {len(rows)}"

    clean, previous = [], None
    for i, row in enumerate(rows, start=1):
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except ValueError:
            return None, f"row {i}: invalid timestamp"
        if ts.tzinfo is None:
            return None, f"row {i}: timestamp has no time zone"
        if previous is not None and ts - previous != timedelta(hours=1):
            return None, f"row {i}: hours are not consecutive"
        previous = ts
        values = {}
        for col in NUMERIC_COLUMNS:
            try:
                values[col] = float(row[col])
            except (TypeError, ValueError):
                return None, f"row {i}: {col} is not a number"
            if values[col] < 0:
                return None, f"row {i}: {col} is negative"
        if values["consumption_mw"] == 0:
            return None, f"row {i}: consumption_mw is zero"
        if values["co2_g_per_kwh"] >= 1000:
            return None, f"row {i}: co2_g_per_kwh is unrealistic"
        clean.append({"timestamp": row["timestamp"], **values})
    return clean, "ok"


def item_id(name: str, etag: str) -> str:
    """Same file + same version -> same id, so a retry overwrites instead of duplicating."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    safe_etag = re.sub(r"[^A-Za-z0-9]", "", etag)
    return f"{safe_name}-{safe_etag}"


def _utc_key(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(timezone.utc).strftime("%Y-%m-%dT%H")


def load_actuals(path: Path = ACTUALS_FILE) -> dict[str, float]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {_utc_key(r["timestamp"]): float(r["co2_g_per_kwh"]) for r in csv.DictReader(f)}


def actual_values(timestamps: list[str], actuals: dict[str, float]) -> list[float] | None:
    values = [actuals.get(_utc_key(t)) for t in timestamps]
    return None if any(v is None for v in values) else values


def make_item(name: str, etag: str, rows: list[dict], result: dict, attempt: int,
              actuals: dict[str, float]) -> dict:
    """Summary card written to Cosmos DB (partition key: /day)."""
    times = [h["timestamp"] for h in result["forecast"]]
    predicted = [h["co2_g_per_kwh"] for h in result["forecast"]]
    real = actual_values(times, actuals)
    item = {
        "id": item_id(name, etag),
        "day": rows[-1]["timestamp"][:10],
        "file": name,
        "etag": etag,
        "model_version": result["model_version"],
        "forecast_times": times,
        "forecast_co2": predicted,
        "actual_co2": real,
        "best_window_start": result["best_window_start"],
        "best_window_end": result["best_window_end"],
        "best_window_avg_co2": result["best_window_avg_co2"],
        "run_now_avg_co2": result["run_now_avg_co2"],
        "co2_saved_pct": result["co2_saved_pct"],
        "actual_saved_pct": None,
        "api_latency_ms": result["latency_ms"],
        "attempt": attempt,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if real:
        # how much would really have been saved by following the advice
        parsed = [datetime.fromisoformat(t) for t in times]
        begin = datetime.fromisoformat(result["best_window_start"])
        end = datetime.fromisoformat(result["best_window_end"])
        inside = [i for i, t in enumerate(parsed) if begin <= t < end]
        if inside:
            start, hours = inside[0], len(inside)
            now_avg = sum(real[:hours]) / hours
            best_avg = sum(real[start:start + hours]) / hours
            if now_avg > 0:
                item["actual_saved_pct"] = round(100 * (1 - best_avg / now_avg), 1)
    return item
