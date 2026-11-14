"""Tests for env_trace.py: clipping to the 15-minute grid and the three mutations."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.readers import env_trace as et
from tests.local_data import HAS_ALL  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
needs_raw = pytest.mark.skipif(not HAS_ALL, reason="raw data not downloaded")


def toy_trace(n=8):
    return pd.DataFrame({"co2": np.full(n, 3000.0), "temp": np.full(n, 22.0),
                         "humidity": np.full(n, 40.0), "dose_rate": np.arange(n, dtype=float)},
                        index=pd.Index(np.arange(n) * 15, name="minute"))


# ---- clip

def test_clip_resamples_to_15_minutes_from_clip_start():
    ts = pd.date_range("2014-09-25 00:00", periods=12, freq="5min")
    src = pd.DataFrame({"timestamp": ts, "co2": np.arange(12, dtype=float)})
    out = et.clip(src, start="2014-09-25 00:00", hours=1)
    assert out.index.tolist() == [0, 15, 30, 45] and out.index.name == "minute"
    assert out["co2"].tolist() == [1.0, 4.0, 7.0, 10.0]          # mean of each 15 min bin


def test_clip_weights_samples_by_time_so_dense_bursts_do_not_dominate():
    # One 15-min bin: 600 s at 10 uGy/h (one sample), then 300 s of a burst sampled
    # every 20 s at 100 uGy/h (15 samples). Time-weighted mean = (600*10 + 300*100) / 900.
    ts = [pd.Timestamp("2017-09-05 00:00")] + list(
        pd.date_range("2017-09-05 00:10", periods=15, freq="20s"))
    src = pd.DataFrame({"timestamp": ts, "dose_rate": [10.0] + [100.0] * 15})
    out = et.clip(src, start="2017-09-05", hours=0.25)
    assert out["dose_rate"].iloc[0] == pytest.approx((600 * 10 + 300 * 100) / 900)


def test_clip_fills_short_gaps_only():
    ts = pd.date_range("2017-09-05", periods=16, freq="15min")
    vals = np.arange(16, dtype=float)
    vals[3:5] = np.nan          # 30 min gap -> filled
    vals[8:12] = np.nan         # 60 min gap -> left empty
    out = et.clip(pd.DataFrame({"timestamp": ts, "dose_rate": vals}),
                  start="2017-09-05", hours=4)
    assert out["dose_rate"].iloc[3:5].tolist() == [3.0, 4.0]
    assert out["dose_rate"].iloc[8:12].isna().all()


def test_clip_outside_data_raises():
    src = pd.DataFrame({"timestamp": pd.date_range("2014-09-25", periods=4, freq="h"),
                        "co2": [1.0, 2, 3, 4]})
    with pytest.raises(ValueError, match="no data"):
        et.clip(src, start="2015-01-01", hours=1)


# ---- real data

@needs_raw
def test_build_trace_one_day_iss():
    tr = et.build_trace(cabin_start="2014-10-15", radiation_start="2017-09-05", radiation="iss")
    assert list(tr.columns) == et.COLUMNS
    assert len(tr) == config.STEPS_PER_DAY and tr.index[-1] == 1425
    assert not tr.isna().any().any()
    assert 2000 < tr["co2"].mean() < 5000 and 15 < tr["temp"].mean() < 30
    # Time-weighted ~8.8 uGy/h. The raw sample mean (21.7) is inflated because
    # DosTel2 samples every ~20 s inside SAA passes vs ~100 s elsewhere.
    assert tr["dose_rate"].mean() == pytest.approx(8.8, rel=0.3)
    assert tr["dose_rate"].max() > 2 * tr["dose_rate"].median()   # SAA passes visible


@needs_raw
def test_default_cabin_day_is_not_a_stuck_sensor_day():
    tr = et.build_trace()
    assert tr["co2"].nunique() > 10        # 2014-09-22..26 CO2 is a flat 2646 ppm


@needs_raw
def test_build_trace_mars_uses_msl_rad():
    tr = et.build_trace(radiation_start="2017-09-11", radiation="mars")
    assert tr["dose_rate"].max() > 15            # solar particle event plateau ~20 uGy/h


@needs_raw
def test_eda_daily_dose_check():
    d = et.load_eda_daily_dose()
    assert {"gcr_mGy_d", "saa_mGy_d", "total_mGy_d", "dose_rate"} <= set(d.columns)
    # Whole days from the first full day on the ISS (pre-ISS launch / transit days dropped)
    assert d["timestamp"].min() == pd.Timestamp(config.EDA_ISS_START).ceil("D")
    assert len(d) == 32
    assert d["total_mGy_d"].mean() == pytest.approx(0.202, abs=0.001)
    assert d["dose_rate"].mean() == pytest.approx(8.42, abs=0.01)


@needs_raw
def test_goes_xrsb_flags_eclipse_and_keeps_flares():
    flux = et.load_goes_xrsb("2017-09-10", "2017-09-11")
    assert flux.loc["2017-09-10 05:30":"2017-09-10 06:20"].isna().all()   # eclipse gap
    assert flux.max() == pytest.approx(1.29e-3, rel=0.01)
    assert et.flare_class(flux.max()) == "X8.2"          # 1.29e-3 * 0.635 = issued class
    assert et.flare_class(1.29e-3, official=False) == "X12.9"


@needs_raw
def test_goes_clip_uses_minutes_from_zero_and_keeps_gaps():
    s = et.clip_goes("2017-09-10", hours=24)
    assert s.index[0] == 0 and s.index[-1] == 1439 and s.index.name == "minute"
    assert s.loc[5 * 60 + 30: 6 * 60 + 20].isna().all()                # eclipse, not filled
    assert s.max() == pytest.approx(1.292e-3, rel=0.001)
    assert et.clip_goes("2017-09-06", hours=24).max() == pytest.approx(1.464e-3, rel=0.001)


# ---- RR-1 launch / transit period is not ISS cabin air

@needs_raw
def test_eda_readings_before_iss_installation_are_dropped():
    eda = et.load_source("eda_rr1")
    assert eda["timestamp"].min() >= pd.Timestamp(config.EDA_ISS_START)
    with pytest.raises(ValueError, match="no data"):
        et.build_trace(cabin_start="2014-09-22")       # Dragon transit, not ISS cabin
