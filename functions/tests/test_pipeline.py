"""Tests of the pipeline logic. Run from the repository root:  pytest functions/tests -v"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "functions"))

from pipeline import item_id, load_actuals, make_item, validate_csv  # noqa: E402

FILES = ROOT / "model" / "test_files"
DEMO = FILES / "valid" / "grid_2024-10-22_18h_demo.csv"


def check(path: Path):
    return validate_csv(path.name, path.read_bytes())


def test_all_valid_files_pass():
    for path in (FILES / "valid").glob("*.csv"):
        rows, reason = check(path)
        assert rows is not None, f"{path.name}: {reason}"
        assert len(rows) == 24


def test_missing_column_is_rejected():
    rows, reason = check(FILES / "invalid" / "missing_column_gas.csv")
    assert rows is None and "gas_mw" in reason


def test_negative_value_is_rejected():
    rows, reason = check(FILES / "invalid" / "negative_value.csv")
    assert rows is None and "negative" in reason


def test_wrong_row_count_is_rejected():
    rows, reason = check(FILES / "invalid" / "only_20_rows.csv")
    assert rows is None and "24" in reason


def test_not_a_csv_is_rejected():
    rows, reason = check(FILES / "invalid" / "not_a_csv.txt")
    assert rows is None and "csv" in reason


def test_item_id_is_stable_for_retries():
    assert item_id("grid a.csv", '"0x8DD1"') == item_id("grid a.csv", "0x8DD1")
    assert "/" not in item_id("folder/grid.csv", "0x1")


def test_make_item_with_actuals():
    rows, _ = check(DEMO)
    times = [f"2024-10-22T{h:02d}:00:00+02:00" for h in range(19, 24)] + \
            [f"2024-10-23T{h:02d}:00:00+02:00" for h in range(0, 7)]
    result = {
        "model_version": "1.0.0",
        "forecast": [{"timestamp": t, "co2_g_per_kwh": 15.0} for t in times],
        "best_window_start": times[6], "best_window_end": times[8],
        "best_window_avg_co2": 11.3, "run_now_avg_co2": 18.3,
        "co2_saved_pct": 38.4, "latency_ms": 36.9,
    }
    item = make_item(DEMO.name, "0x8DD1", rows, result, 1, load_actuals())
    assert item["day"] == "2024-10-22"
    assert item["actual_co2"] is not None and len(item["actual_co2"]) == 12
    assert item["actual_saved_pct"] is not None
