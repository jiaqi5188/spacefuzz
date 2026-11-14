"""Tests for generic_reader.py and the three mapping templates in data/mappings/."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spacefuzz.readers import generic_reader as gr
from tests.local_data import HAS_ALL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "data" / "mappings"
RAW = ROOT / "data" / "raw"
needs_raw = pytest.mark.skipif(not HAS_ALL, reason="raw data not downloaded")


def write_mapping(tmp_path, mapping: dict) -> Path:
    p = tmp_path / "map.yaml"
    p.write_text(yaml.safe_dump(mapping))
    return p


# ---- unit conversion

@pytest.mark.parametrize("field, unit, value, expected", [
    ("co2", "percent", 0.5, 5000.0),
    ("temp", "degF", 212.0, 100.0),
    ("temp", "K", 300.15, 27.0),
    ("humidity", "fraction", 0.4, 40.0),
    ("dose_rate", "mGy/day", 1.0, 1000 / 24),
    ("hr", "hz", 1.0, 60.0),
    ("rmssd", "s", 0.042, 42.0),
])
def test_unit_conversion(field, unit, value, expected):
    out = gr.convert(pd.Series([value]), field, unit)
    assert out.iloc[0] == pytest.approx(expected)


def test_unknown_unit_is_an_error():
    with pytest.raises(ValueError, match="uSv/h"):
        gr.convert(pd.Series([1.0]), "dose_rate", "uSv/h")  # Sv is not Gy


# ---- timeseries layout on small files

def test_timeseries_csv_with_units_filters_and_extra(tmp_path):
    (tmp_path / "t.csv").write_text(
        "when,co2_pct,tF,flag,note\n"
        "2020-01-01 00:00,0.30,68,0,a\n"
        "2020-01-01 00:05,0.35,77,1,b\n"
        "2020-01-01 00:10,0.40,86,0,c\n")
    m = write_mapping(tmp_path, {"standard": "env", "sources": {"s": {
        "file": "t.csv",
        "time": {"column": "when", "format": "%Y-%m-%d %H:%M"},
        "filters": [{"column": "flag", "equals": 0}],
        "fields": {"co2": {"column": "co2_pct", "unit": "percent"},
                   "temp": {"column": "tF", "unit": "degF"}},
        "extra": {"label": "note"}}}})
    df = gr.read_source(m, "s", base_dir=tmp_path)
    assert list(df.columns) == ["timestamp", "co2", "temp", "label"]
    assert df["co2"].tolist() == pytest.approx([3000.0, 4000.0])
    assert df["temp"].tolist() == pytest.approx([20.0, 30.0])
    assert df["timestamp"].iloc[1] == pd.Timestamp("2020-01-01 00:10")


def test_timeseries_tsv_relative_seconds_and_value_map(tmp_path):
    (tmp_path / "w.tsv").write_text("sub\tlab\tt\tbpm\n"
                                    "A\t1\t0\t60\nA\t2\t60\t90\nA\t7\t120\t70\n")
    m = write_mapping(tmp_path, {"standard": "wearable", "sources": {"w": {
        "file": "w.tsv",
        "time": {"column": "t", "format": "seconds"},
        "ids": {"subject": {"column": "sub"}, "segment_id": {"column": "lab"},
                "condition": {"column": "lab", "value_map": {1: "baseline", 2: "stress"}}},
        "fields": {"hr": {"column": "bpm", "unit": "bpm"}}}}})
    df = gr.read_source(m, "w", base_dir=tmp_path)
    assert df["condition"].tolist() == ["baseline", "stress"]   # label 7 dropped
    assert df["minute"].tolist() == [0.0, 1.0]
    assert np.isnan(df["rmssd"]).all()                          # missing field -> NaN column
    assert list(df.columns) == gr.STANDARD_COLUMNS["wearable"]


def test_xlsx_is_supported(tmp_path):
    pd.DataFrame({"time": ["2020-01-01T00:00:00"], "rh": [0.5]}).to_excel(
        tmp_path / "x.xlsx", index=False)
    m = write_mapping(tmp_path, {"standard": "env", "sources": {"x": {
        "file": "x.xlsx", "time": {"column": "time", "format": "iso"},
        "fields": {"humidity": {"column": "rh", "unit": "fraction"}}}}})
    assert gr.read_source(m, "x", base_dir=tmp_path)["humidity"].iloc[0] == 50.0


def test_missing_column_names_the_column(tmp_path):
    (tmp_path / "t.csv").write_text("a,b\n1,2\n")
    m = write_mapping(tmp_path, {"standard": "env", "sources": {"s": {
        "file": "t.csv", "time": {"column": "a", "format": "seconds"},
        "fields": {"co2": {"column": "CO2", "unit": "ppm"}}}}})
    with pytest.raises(KeyError, match="CO2"):
        gr.read_source(m, "s", base_dir=tmp_path)


def test_mission_day_parsing():
    out = gr.parse_mission_day(pd.Series(["L-92", "R+1", "R+194"]),
                               anchors={"L": "2021-09-16", "R": "2021-09-18"})
    assert out["phase"].tolist() == ["pre_launch", "post_return", "post_return"]
    assert out["day_offset"].tolist() == [-92, 1, 194]
    assert out["date"].iloc[0] == pd.Timestamp("2021-06-16")


# ---- the three templates on real data

@needs_raw
def test_env_template_reads_eda_iss_columns_only():
    df = gr.read_source(MAPS / "env_telemetry.yaml", "eda_rr1")
    assert list(df.columns) == ["timestamp", "co2", "temp", "humidity"]
    assert len(df) == 12_087
    assert 2000 < df["co2"].mean() < 5000          # ISS cabin, not ground (~480 ppm)


@needs_raw
def test_env_template_reads_radlab_and_daily_dose():
    dostel = gr.read_source(MAPS / "env_telemetry.yaml", "radlab_dostel2")
    assert len(dostel) == 13_690
    # Raw row mean; biased high by dense sampling in SAA passes (env_trace time-weights).
    assert dostel["dose_rate"].mean() == pytest.approx(21.7, abs=0.1)
    daily = gr.read_source(MAPS / "env_telemetry.yaml", "eda_rr1_radiation")
    assert {"gcr_mGy_d", "saa_mGy_d", "total_mGy_d"} <= set(daily.columns)
    assert daily["dose_rate"].mean() == pytest.approx(8.5, abs=0.1)   # uGy/h


@needs_raw
def test_clinical_template_reads_all_three_inspiration4_files():
    path = MAPS / "clinical_timepoints.yaml"
    cmp_ = gr.read_source(path, "inspiration4_cmp")
    cbc = gr.read_source(path, "inspiration4_cbc")
    imm = gr.read_source(path, "inspiration4_immune")
    for df in (cmp_, cbc, imm):
        assert list(df.columns) == gr.STANDARD_COLUMNS["clinical"]
        assert set(df["subject"]) <= {"C001", "C002", "C003", "C004"}
        assert df["phase"].isin(["pre_launch", "post_return"]).all()
    assert cmp_["analyte"].nunique() == 19
    hgb = cbc[cbc["analyte"] == "hemoglobin"]
    assert (hgb["unit"] == "g/dL").all() and hgb["value"].between(10, 20).all()
    assert cbc["analyte"].nunique() == 20
    assert cbc.groupby(["subject", "timepoint"]).ngroups == 28
    assert cbc["value"].notna().all()                      # empty cells are not measurements
    assert "activin_a" in set(imm["analyte"]) and (imm["unit"] == "NPQ").all()
    assert imm["timepoint"].nunique() == 6


PROCESSED = ROOT / "data" / "processed" / "wesad_hr_hrv.csv"


@pytest.mark.skipif(not PROCESSED.exists(), reason="run python -m spacefuzz.readers.physio_replay --build first")
def test_wearable_template_reads_wesad_export():
    df = gr.read_source(MAPS / "wearable_series.yaml", "wesad_export")
    assert list(df.columns) == gr.STANDARD_COLUMNS["wearable"]
    assert set(df["condition"]) == {"baseline", "stress", "amusement", "meditation"}
    assert df["hr"].between(40, 180).all()
