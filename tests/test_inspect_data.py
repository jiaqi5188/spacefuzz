"""Tests for scripts/inspect_data.py."""
import hashlib
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from scripts import inspect_data as ins


# ---- time column detection

def detect(df):
    return {c.column: c for c in ins.detect_time_columns(df)}


def test_detects_minute_resolution_strings():
    df = pd.DataFrame({"Controller_Time_GMT": ["2014-09-20 03:59", "2014-09-20 04:04",
                                               "2014-09-20 04:09"], "Temp": [24.1, 24.1, 23.9]})
    found = detect(df)
    assert set(found) == {"Controller_Time_GMT"}
    assert found["Controller_Time_GMT"].fmt == "%Y-%m-%d %H:%M"
    assert "5min" in found["Controller_Time_GMT"].detail or "0 days 00:05:00" in found["Controller_Time_GMT"].detail


def test_detects_iso_timestamps():
    df = pd.DataFrame({"timestamp": ["2017-09-04T00:00:42", "2017-09-04T00:02:22"]})
    assert detect(df)["timestamp"].fmt == "%Y-%m-%dT%H:%M:%S"


def test_detects_mission_relative_day_labels():
    df = pd.DataFrame({"timepoint": ["L-92", "L-44", "R+1", "R+45"]})
    assert "mission-relative" in detect(df)["timepoint"].fmt


def test_detects_mission_day_embedded_in_sample_names():
    df = pd.DataFrame({"Sample Name": ["C001_whole-blood_L-3_cbc", "C001_whole-blood_R+194_cbc",
                                       "C002_whole-blood_L-92_cbc"]})
    found = detect(df)["Sample Name"]
    assert "embedded" in found.fmt
    assert "L-92" in found.detail and "R+194" in found.detail


def test_detects_parsed_datetime_and_epoch_seconds():
    df = pd.DataFrame({"t": pd.date_range("2017-09-04", periods=3, freq="min"),
                       "epoch_time": [1.5e9, 1.5e9 + 60, 1.5e9 + 120]})
    found = detect(df)
    assert found["t"].fmt == "datetime64"
    assert found["epoch_time"].fmt == "epoch seconds"


def test_plain_numbers_and_text_are_not_time():
    df = pd.DataFrame({"VALUE": [83, 25, 1], "ANALYTE": ["GLUCOSE", "BUN", "X"],
                       "code": ["12", "34", "56"]})
    assert detect(df) == {}


# ---- loaders

def test_loads_csv_with_bom(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes("﻿Sample Name,x\nA,1\nB,2\n".encode("utf-8"))
    (name, df), = ins.load_tables(p)[0]
    assert list(df.columns) == ["Sample Name", "x"] and df.shape == (2, 2)


def test_loads_semicolon_file_with_hash_lines(tmp_path):
    p = tmp_path / "quest.csv"
    p.write_text("# Subj;S2;;\n# ORDER;Base;TSST;\n# START;7.08;39.55;\n")
    (name, df), = ins.load_tables(p)[0]
    assert df.shape == (3, 4)
    assert df.iloc[1, 1] == "Base"


def test_loads_tsv(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("a\tb\n1\t2\n")
    (name, df), = ins.load_tables(p)[0]
    assert list(df.columns) == ["a", "b"]


def test_loads_nested_pickle_as_one_table_per_array(tmp_path):
    p = tmp_path / "S9.pkl"
    data = {"signal": {"chest": {"ECG": np.zeros((10, 1)), "ACC": np.ones((10, 3))},
                       "wrist": {"EDA": np.arange(4.0).reshape(4, 1)}},
            "label": np.array([0, 1, 1, 2, 0, 0, 1, 1, 2, 0]), "subject": "S9"}
    with open(p, "wb") as f:
        pickle.dump(data, f)
    tables, notes = ins.load_tables(p)
    shapes = {name: df.shape for name, df in tables}
    assert shapes == {"signal/chest/ECG": (10, 1), "signal/chest/ACC": (10, 3),
                      "signal/wrist/EDA": (4, 1), "label": (10, 1)}
    assert any("subject" in n and "S9" in n for n in notes)


def test_loads_netcdf_variables_along_time(tmp_path):
    p = tmp_path / "x.nc"
    t = pd.date_range("2017-09-04", periods=4, freq="min")
    xr.Dataset({"xrsa_flux": ("time", [1e-8, 2e-8, np.nan, 3e-8]),
                "xrsb_flux": ("time", [1e-7, 1e-7, 1e-7, 1e-7])},
               coords={"time": t}).to_netcdf(p)
    tables, notes = ins.load_tables(p)
    (name, df), = tables
    assert name == "dim: time" and df.shape == (4, 3)
    assert "time" in detect(df)


# ---- full run

def digest(paths):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_run_writes_report_lists_failures_and_leaves_raw_untouched(tmp_path):
    raw = tmp_path / "raw"
    (raw / "ds").mkdir(parents=True)
    (raw / "ds" / "good.csv").write_text("Date,v\n2014-09-21,0.1\n2014-09-22,\n")
    (raw / "ds" / "bad.pkl").write_bytes(b"not a pickle")
    (raw / "ds" / "notes.pdf").write_bytes(b"%PDF")
    before = digest(sorted(raw.rglob("*.*")))

    out = tmp_path / "report.md"
    ins.run(raw, out)

    assert digest(sorted(raw.rglob("*.*"))) == before
    text = out.read_text()
    assert "ds/good.csv" in text and "2 rows × 2 columns" in text
    assert "%Y-%m-%d" in text and "50.0%" in text           # time format, missing ratio
    assert "## Unreadable files" in text and "ds/bad.pkl" in text
    assert "ds/notes.pdf" in text                            # listed as skipped
