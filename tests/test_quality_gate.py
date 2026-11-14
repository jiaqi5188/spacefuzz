"""Tests for quality_gate.py: plausibility, stuck values, missing data -> ok/degraded/untrusted."""
import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.core import quality_gate as qg
from tests.local_data import HAS_ALL  # noqa: E402

CHECKIN = {"systolic_bp": 122, "diastolic_bp": 78, "blood_glucose": 92,
           "sleep_last_3_nights_h": (7.0, 7.2, 7.5), "fatigue_score": 3, "mood": 3, "stress": 2,
           "sleep_last_night_h": 7.2, "reaction_time_ms": 260}


def sensors(hours=4, seed=config.RANDOM_SEED):
    """Realistic 1-minute wearable readings with sensor noise."""
    rng = np.random.default_rng(seed)
    n = hours * 60
    return pd.DataFrame({
        "minute": np.arange(n),
        "heart_rate": np.round(70 + 3 * np.sin(np.arange(n) / 30) + rng.normal(0, 2, n)),
        "hrv_rmssd": np.round(60 + rng.normal(0, 6, n), 1),
        "spo2": rng.integers(96, 100, n).astype(float),
        "core_temp": np.round(36.8 + rng.normal(0, 0.05, n), 2),
    })


# ---- ok

def test_clean_data_is_ok():
    r = qg.check(sensors(), CHECKIN)
    assert r.status == "ok" and r.missing == [] and r.issues == []
    pd.testing.assert_frame_equal(r.clean, sensors())


# ---- acceptance: HR 300, 2 h stuck, missing SpO2

def test_heart_rate_300_spike_is_filtered_and_degrades():
    df = sensors()
    df.loc[100, "heart_rate"] = 300
    r = qg.check(df, CHECKIN)
    assert r.status == "degraded"
    assert "heart_rate" not in r.missing                      # one spike: rest is usable
    assert any(i.field == "heart_rate" and i.kind == "out_of_range" for i in r.issues)
    assert np.isnan(r.clean.loc[100, "heart_rate"])
    assert r.clean["heart_rate"].notna().sum() == len(df) - 1


def test_heart_rate_300_throughout_makes_the_field_unusable():
    df = sensors()
    df["heart_rate"] = 300.0
    r = qg.check(df, CHECKIN)
    assert r.status == "degraded" and r.missing == ["heart_rate"]
    assert r.clean["heart_rate"].isna().all()


def test_two_hours_of_identical_heart_rate_is_stuck():
    df = sensors()
    df.loc[60:179, "heart_rate"] = 72.0                       # 120 minutes unchanged
    r = qg.check(df, CHECKIN)
    assert r.status == "degraded" and r.missing == ["heart_rate"]
    stuck = [i for i in r.issues if i.kind == "stuck"]
    assert stuck and stuck[0].field == "heart_rate" and "120 min" in stuck[0].detail


def test_short_plateau_is_not_stuck():
    df = sensors()
    df.loc[60:79, "heart_rate"] = 72.0                        # 20 minutes
    assert qg.check(df, CHECKIN).status == "ok"


def test_integer_spo2_plateau_under_its_longer_limit_is_not_stuck():
    df = sensors()
    df.loc[0:149, "spo2"] = 98.0                              # 150 min < 180 min SpO2 limit
    assert qg.check(df, CHECKIN).status == "ok"


def test_missing_spo2_column_degrades_and_is_listed():
    r = qg.check(sensors().drop(columns="spo2"), CHECKIN)
    assert r.status == "degraded" and r.missing == ["spo2"]
    assert any(i.field == "spo2" and i.kind == "missing" for i in r.issues)


def test_mostly_empty_spo2_counts_as_missing():
    df = sensors()
    df.loc[: int(len(df) * 0.6), "spo2"] = np.nan
    assert qg.check(df, CHECKIN).missing == ["spo2"]


def test_a_few_gaps_are_tolerated():
    df = sensors()
    df.loc[10:20, "spo2"] = np.nan
    r = qg.check(df, CHECKIN)
    assert r.status == "degraded" and r.missing == []        # noted, still usable


# ---- untrusted

def test_disconnected_device_is_untrusted():
    df = sensors()
    df[list(config.QG_VITAL_SERIES)] = np.nan
    r = qg.check(df, CHECKIN)
    assert r.status == "untrusted"
    assert set(config.QG_VITAL_SERIES) <= set(r.missing)


def test_most_fields_bad_is_untrusted():
    df = sensors()
    df["heart_rate"] = 300.0
    df["hrv_rmssd"] = 55.0                                    # stuck all session
    bad_checkin = {**CHECKIN, "blood_glucose": None, "systolic_bp": 900, "diastolic_bp": None}
    r = qg.check(df.drop(columns="spo2"), bad_checkin)
    assert r.status == "untrusted"


# ---- check-in values

def test_checkin_out_of_range_and_missing():
    # an implausible value the rules read is never simulated: UNKNOWN, re-measure that item
    r = qg.check(sensors(), {**CHECKIN, "blood_glucose": 900, "reaction_time_ms": None})
    assert r.status == "untrusted"
    assert r.missing == ["blood_glucose", "reaction_time_ms"]
    assert r.message.startswith(qg.UNTRUSTED_MESSAGE)
    assert "Re-measure: blood glucose (900 outside 20-600)" in r.message
    assert "reaction" not in r.message                  # not entered: optional, not rejected


def test_reaction_test_not_done_only_degrades():
    r = qg.check(sensors(), {**CHECKIN, "reaction_time_ms": None})
    assert r.status == "degraded" and r.can_forecast and r.missing == ["reaction_time_ms"]


@pytest.mark.parametrize("field,value", [("sleep_last_night_h", 20.0), ("systolic_bp", 300.0),
                                         ("diastolic_bp", 10.0), ("reaction_time_ms", 5000.0),
                                         ("sleep_last_night_h", float("nan"))])
def test_every_rejected_rule_input_makes_the_forecast_unknown(field, value):
    r = qg.check(checkin={**CHECKIN, field: value})
    assert r.status == "untrusted" and field in r.missing
    assert qg.FIELD_LABELS[field] in r.message


def test_checkin_dataclass_is_accepted():
    from spacefuzz.core import state as st
    r = qg.check(sensors(), st.load_checkin("safe_day"))
    assert r.status == "ok"


def test_result_dict_matches_output_format():
    out = qg.check(sensors().drop(columns="spo2"), CHECKIN).to_dict()
    assert out["data_quality"] == "degraded" and out["missing"] == ["spo2"]
    assert out["issues"][0] == {"field": "spo2", "kind": "missing", "detail": "no readings"}


# ---- removed readings count towards the missing fraction

def test_mostly_spikes_makes_heart_rate_unusable():
    df = sensors()
    df.loc[: int(len(df) * 0.6), "heart_rate"] = 300.0      # 60% spikes, 40% real readings
    r = qg.check(df, CHECKIN)
    assert r.missing == ["heart_rate"]
    assert {i.kind for i in r.issues if i.field == "heart_rate"} == {"out_of_range", "missing"}


# ---- real data: the EDA CO2 sensor stuck at 2646 ppm

needs_raw = pytest.mark.skipif(not HAS_ALL,
                               reason="raw data missing")


def eda_window(start, end):
    from spacefuzz.readers import generic_reader as gr
    e = gr.read_source("data/mappings/env_telemetry.yaml", "eda_rr1")   # raw, uncut
    e = e[(e["timestamp"] >= start) & (e["timestamp"] < end)].reset_index(drop=True)
    e["minute"] = (e["timestamp"] - e["timestamp"].iloc[0]).dt.total_seconds() / 60
    return e


@needs_raw
def test_eda_co2_stuck_period_is_caught():
    r = qg.check(eda_window("2014-09-20", "2014-09-29"), series_fields=("co2",))
    assert r.missing == ["co2"]
    stuck = next(i for i in r.issues if i.kind == "stuck")
    assert "9186 min" in stuck.detail   # 09-21 04:24 .. 09-27 13:25 (9181 min) + one 5-min reading


@needs_raw
def test_eda_normal_day_passes():
    assert qg.check(eda_window("2014-10-15", "2014-10-16"), series_fields=("co2",)).status == "ok"


def test_stuck_mask_marks_only_the_frozen_run():
    s = pd.Series([1.0, 2, 3, 3, 3, 3, 3, 4, 5])
    mask = qg.stuck_mask(s, np.arange(9) * 30.0, limit_min=120)
    assert mask.tolist() == [False, False, True, True, True, True, True, False, False]


# ---- untrusted: no forecast, tell the crew to measure again

def test_untrusted_blocks_the_forecast():
    df = sensors()
    df[list(config.QG_VITAL_SERIES)] = np.nan
    r = qg.check(df, CHECKIN)
    assert not r.can_forecast
    assert r.to_dict()["message"] == qg.UNTRUSTED_MESSAGE == r.message   # nothing to name
    with pytest.raises(qg.UntrustedDataError, match="measure again"):
        qg.ensure_trusted(r)


def test_ok_and_degraded_allow_the_forecast():
    assert qg.check(sensors(), CHECKIN).can_forecast
    deg = qg.check(sensors().drop(columns="spo2"), CHECKIN)
    assert deg.can_forecast and qg.ensure_trusted(deg) is deg
