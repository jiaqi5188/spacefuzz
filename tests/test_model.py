"""Tests for model.py: one-day state transition and causal chains 1, 2, 5, 6."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.core import model as md
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from tests.local_data import HAS_ALL  # noqa: E402

BASE = st.make_baseline()
STEPS = config.STEPS_PER_DAY


def flat_trace(co2=3000.0, dose=8.0):
    idx = pd.Index(np.arange(STEPS) * config.STEP_MINUTES, name="minute")
    return pd.DataFrame({"co2": co2, "temp": 22.5, "humidity": 42.0, "dose_rate": dose}, index=idx)


def make_plan(tasks, events=None):
    data = {"date": "Mission Day 1", "environment": "mars_base", "check_window": "morning",
            "comm_delay_min": 20,
            "tasks": [{"start": s, "duration_h": d, "type": t, "kind": "task"} for s, d, t in tasks]}
    if events:
        data["events"] = events
    return pl.load_plan(data)


EVA_DAY = make_plan([("07:00", 0.5, "meal"), ("10:00", 6.5, "EVA"), ("17:00", 0.5, "meal"),
                     ("23:00", 7.5, "sleep")])
EVA_NO_BREAKFAST = make_plan([("10:00", 6.5, "EVA"), ("17:00", 0.5, "meal"), ("23:00", 7.5, "sleep")])
REST_DAY = make_plan([("07:00", 0.5, "meal"), ("17:00", 0.5, "meal"), ("23:00", 7.5, "sleep")])


def checkin(name="safe_day", **changes):
    return dataclasses.replace(st.load_checkin(name), **changes)


def run(plan=EVA_DAY, c=None, trace=None, **kw):
    return md.simulate(plan, trace if trace is not None else flat_trace(), c or checkin(), BASE, **kw)


def window(df, start, end):
    return df[(df["minute"] >= start) & (df["minute"] < end)]


# ---- shape

def test_one_day_is_96_steps_from_the_checkin():
    df = run()
    assert len(df) == STEPS
    assert df["minute"].iloc[0] == 6 * 60 + 30 and df["minute"].diff().dropna().eq(15).all()
    assert not df[md.OUTPUT_COLUMNS].isna().any().any()
    assert df.loc[df["minute"] == 10 * 60 + 30, "activity"].item() == "eva"


def test_deterministic_without_overlay():
    pd.testing.assert_frame_equal(run(), run())


# ---- required: EVA raises HR and lowers HRV (chain 2)

def test_eva_raises_heart_rate_and_lowers_hrv():
    df = run()
    pre = window(df, 8 * 60, 10 * 60)
    eva = window(df, 11 * 60, 16 * 60 + 30)
    assert eva["heart_rate"].mean() > pre["heart_rate"].mean() + 25
    assert eva["hrv_rmssd"].mean() < 0.7 * pre["hrv_rmssd"].mean()
    assert eva["core_temp"].mean() > pre["core_temp"].mean() + 0.3
    assert eva["core_temp"].max() - BASE.core_temp < config.HEAT_STRAIN_ONSET_C  # ordinary EVA
    after = window(df, 19 * 60, 22 * 60)
    assert after["heart_rate"].mean() < eva["heart_rate"].mean() - 25    # recovers


# ---- required: sleep debt slows reaction time (chain 5)

def test_sleep_debt_slows_reaction_time():
    rested = run(c=checkin(reaction_time_ms=None))
    debt = run(c=checkin(sleep_last_night_h=5.0, sleep_last_3_nights_h=(4.5, 5.0, 5.0),
                         reaction_time_ms=None))
    assert (debt["reaction_time_ms"] > rested["reaction_time_ms"]).all()
    gap = debt["reaction_time_ms"] - rested["reaction_time_ms"]
    assert gap.iloc[-30] > gap.iloc[0]          # fatigue builds faster under debt


def test_sleep_debt_with_measured_start_still_diverges():
    rested = run(c=checkin())
    debt = run(c=checkin(sleep_last_night_h=5.0, sleep_last_3_nights_h=(4.5, 5.0, 5.0)))
    assert rested["reaction_time_ms"].iloc[0] == debt["reaction_time_ms"].iloc[0]  # same measurement
    eva_end = window(debt, 16 * 60, 16 * 60 + 30).index[0]
    assert debt["reaction_time_ms"].loc[eva_end] > rested["reaction_time_ms"].loc[eva_end]


def test_sleep_debt_makes_eva_heart_rate_rise_faster_and_recover_slower():
    rested = run()
    debt = run(c=checkin(sleep_last_night_h=5.0, sleep_last_3_nights_h=(4.5, 5.0, 5.0)))
    rise = lambda df: window(df, 10 * 60, 11 * 60)["heart_rate"].mean() - \
        window(df, 9 * 60, 10 * 60)["heart_rate"].mean()
    assert rise(debt) > rise(rested)
    rec = lambda df: window(df, 17 * 60, 18 * 60)["heart_rate"].mean() - \
        window(df, 9 * 60, 10 * 60)["heart_rate"].mean()
    assert rec(debt) > rec(rested)


# ---- required: low glucose before EVA keeps falling (chain 6)

def test_low_glucose_before_eva_keeps_falling():
    df = run(plan=EVA_NO_BREAKFAST, c=checkin(blood_glucose=62))
    start = window(df, 10 * 60, 10 * 60 + 15)["blood_glucose"].item()
    end = window(df, 16 * 60 + 15, 16 * 60 + 30)["blood_glucose"].item()
    assert end < start - 10
    assert window(df, 10 * 60, 16 * 60 + 30)["blood_glucose"].diff().dropna().le(0).all()


def test_normal_glucose_is_held_during_eva():
    df = run(c=checkin(blood_glucose=92))
    eva = window(df, 10 * 60, 16 * 60 + 30)["blood_glucose"]
    assert eva.min() >= config.GLUCOSE_LOW_MG_DL


def test_meal_raises_glucose():
    df = run(plan=REST_DAY, c=checkin(blood_glucose=90))
    assert window(df, 7 * 60, 7 * 60 + 30)["blood_glucose"].max() > 100


# ---- chain 1: acute stress

def test_docking_triggers_stress_response():
    plan = make_plan([("10:00", 2.0, "docking"), ("23:00", 7.5, "sleep")])
    df = run(plan=plan, c=checkin(), trace=flat_trace())
    calm = window(df, 8 * 60, 10 * 60)
    dock = window(df, 11 * 60, 12 * 60)
    assert dock["stress_level"].min() > 0.8
    assert dock["heart_rate"].mean() > calm["heart_rate"].mean() + 0.8 * config.CHAIN1_STRESS_HR_DELTA_BPM
    assert dock["hrv_rmssd"].mean() < calm["hrv_rmssd"].mean() * 0.65
    assert dock["systolic_bp"].mean() > calm["systolic_bp"].mean() + 10


def test_spe_warning_is_an_alarm():
    ev = [{"start": "09:00", "end": "12:00", "type": "spe_warning", "kind": "env"}]
    df = run(plan=make_plan([("23:00", 7.5, "sleep")], events=ev))
    assert window(df, 10 * 60, 12 * 60)["stress_level"].min() > 0.6
    assert window(df, 14 * 60, 16 * 60)["stress_level"].max() < 0.1


# ---- caps and recovery

def test_fatigue_is_capped_and_recovers_with_sleep():
    long_day = make_plan([("07:00", 8.0, "EVA"), ("15:00", 8.0, "docking"), ("23:30", 7.0, "sleep")])
    df = run(plan=long_day, c=checkin("main_sleep_debt_eva"), trace=flat_trace(co2=6000))
    assert df["cognitive_fatigue"].max() <= config.FATIGUE_MAX
    peak = df["cognitive_fatigue"].idxmax()
    assert df["cognitive_fatigue"].iloc[-1] < df["cognitive_fatigue"].loc[peak] - 30


def test_radiation_load_capped_and_recovers_eva_dose_only_grows():
    df = run(trace=flat_trace(dose=400.0))
    assert df["radiation_load"].max() <= config.RAD_LOAD_MAX
    assert df["eva_dose_ugy"].diff().dropna().ge(0).all()
    eva_dose = df["eva_dose_ugy"].iloc[-1]
    assert eva_dose == pytest.approx(400.0 * config.EVA_DOSE_MULTIPLIER * 6.5, rel=0.01)
    quiet = run(trace=flat_trace(dose=8.0))
    assert quiet["radiation_load"].iloc[-1] < quiet["radiation_load"].max()


def test_high_co2_lowers_spo2_and_speeds_fatigue():
    normal = run(trace=flat_trace(co2=3000))
    high = run(trace=flat_trace(co2=6000))
    assert high["spo2"].mean() < normal["spo2"].mean() - 1
    assert high["cognitive_fatigue"].iloc[40] > normal["cognitive_fatigue"].iloc[40]


# ---- real fluctuation overlay

def test_overlay_adds_real_fluctuation_reproducibly():
    pool = md.Fluctuation(hr_dev={"relax": np.array([-2.0, 0.0, 2.0]), "stress": np.array([5.0])},
                          rmssd_rel={"relax": np.array([-0.1, 0.0, 0.1]), "stress": np.array([-0.2])})
    a = run(fluctuation=pool, rng=np.random.default_rng(config.RANDOM_SEED))
    b = run(fluctuation=pool, rng=np.random.default_rng(config.RANDOM_SEED))
    plain = run()
    pd.testing.assert_frame_equal(a, b)
    diff = a["heart_rate"] - plain["heart_rate"]
    assert diff.abs().max() > 0 and diff.abs().max() <= 2.0 + 1e-9
    assert (a["hr_model"] == plain["hr_model"]).all()   # overlay never feeds back into dynamics


@pytest.mark.skipif(not HAS_ALL, reason="WESAD export not built")
def test_fluctuation_pool_from_wesad_export():
    pool = md.Fluctuation.from_wesad()
    assert pool.hr_dev["stress"].std() > pool.hr_dev["relax"].std()
    assert abs(pool.hr_dev["relax"].mean()) < 0.5


# ---- the real main scenario stays physiological

@pytest.mark.skipif(not HAS_ALL, reason="raw data missing")
def test_main_scenario_is_plausible():
    trace, plan = pl.load_scenario("data/plans/main_sleep_debt_eva.json")
    df = md.simulate(plan, trace, st.load_checkin("main_sleep_debt_eva"), BASE,
                     fluctuation=md.Fluctuation.from_wesad(),
                     rng=np.random.default_rng(config.RANDOM_SEED))
    assert df["heart_rate"].between(40, 180).all()
    assert df["spo2"].between(85, 100).all()
    assert df["core_temp"].between(35.5, 39.0).all()
    assert df["blood_glucose"].between(50, 180).all()


def test_unscheduled_awake_time_is_idle_not_rest():
    gap_day = make_plan([("07:00", 0.5, "meal"), ("12:00", 1.0, "rest"), ("23:00", 7.5, "sleep")])
    df = run(plan=gap_day)
    assert df.loc[df["minute"] == 9 * 60, "activity"].item() == "idle"
    idle = window(df, 8 * 60, 12 * 60)["cognitive_fatigue"]
    assert idle.iloc[-1] > idle.iloc[0]                     # awake time costs a little
    rest = window(df, 12 * 60, 13 * 60 + 15)["cognitive_fatigue"]
    assert rest.iloc[-1] < rest.iloc[0]                     # a scheduled break recovers


# ---- design guard: the main plan is near, but not over, the fatigue red line

@pytest.mark.skipif(not HAS_ALL, reason="raw data missing")
def test_main_plan_unmutated_fatigue_is_yellow_not_red():
    """Plan 3.8: the main plan must not violate anything before mutation; risk comes from
    mutations. Peak fatigue sits 10-15 points below the plan 3.5 red line."""
    trace, plan = pl.load_scenario("data/plans/main_sleep_debt_eva.json")
    df = md.simulate(plan, trace, st.load_checkin("main_sleep_debt_eva"), BASE)
    peak = df["cognitive_fatigue"].max()
    assert config.FATIGUE_RED_LINE - 15 <= peak <= config.FATIGUE_RED_LINE - 10
    safe_trace, safe = pl.load_scenario("data/plans/safe_day.json")
    safe_peak = md.simulate(safe, safe_trace, st.load_checkin("safe_day"), BASE)["cognitive_fatigue"].max()
    assert safe_peak < peak - 20


# ---- personal maximum heart rate cap (220 - age)

def test_model_and_overlaid_heart_rate_never_exceed_personal_max():
    b = dataclasses.replace(BASE, age_y=85)                    # max 135 bpm
    hr_max = st.max_heart_rate(b)
    pool = md.Fluctuation(hr_dev={"relax": np.array([40.0]), "stress": np.array([40.0])},
                          rmssd_rel={"relax": np.array([0.0]), "stress": np.array([0.0])})
    fx = md.Effects(eva_heat_rate=3.0)
    df = md.simulate(EVA_DAY, flat_trace(), checkin(), b, fluctuation=pool,
                     rng=np.random.default_rng(1), effects=fx)
    eva = df["activity"] == "eva"
    assert df.loc[eva, "hr_model"].max() == pytest.approx(hr_max)
    assert df["hr_model"].max() <= hr_max + 1e-9
    assert df["heart_rate"].max() <= hr_max + 1e-9
    assert (df.loc[~eva & (df["activity"] == "sleep"), "heart_rate"] < hr_max).all()


def test_cap_leaves_normal_days_unchanged():
    young = dataclasses.replace(BASE, age_y=20)
    a = md.simulate(EVA_DAY, flat_trace(), checkin(), young)
    assert a["hr_model"].max() < st.max_heart_rate(young)      # cap not touched


# ---- WESAD fallback: fitted parameters when the per-minute CSV is not available

@pytest.mark.skipif(not HAS_ALL, reason="WESAD export not built")
def test_fluctuation_load_is_the_real_pool_when_the_csv_is_there():
    real, loaded = md.Fluctuation.from_wesad(), md.Fluctuation.load()
    for k in ("relax", "stress"):                        # same arrays -> same draws, same seeds
        assert np.array_equal(real.hr_dev[k], loaded.hr_dev[k])
        assert np.array_equal(real.rmssd_rel[k], loaded.rmssd_rel[k])
    assert loaded.source == "wesad"


def test_fluctuation_falls_back_to_the_fitted_parameters(monkeypatch):
    def missing():
        raise FileNotFoundError("wesad_hr_hrv.csv")
    monkeypatch.setattr(md.Fluctuation, "from_wesad", classmethod(lambda cls: missing()))
    f = md.Fluctuation.load()
    assert f.source == "wesad_params"
    assert f.hr_dev["stress"].std() > f.hr_dev["relax"].std()   # stress fluctuates more
    assert abs(f.hr_dev["relax"].mean()) < 1.0
    assert (f.rmssd_rel["relax"] > -1).all()                     # RMSSD stays positive
    again = md.Fluctuation.from_params()
    assert np.array_equal(again.hr_dev["relax"], f.hr_dev["relax"])   # reproducible
    rng = np.random.default_rng(1)
    hr, rel = f.draw(True, rng)
    assert np.isfinite(hr) and np.isfinite(rel)


def test_fluctuation_load_none_when_nothing_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(md.Fluctuation, "from_wesad",
                        classmethod(lambda cls: (_ for _ in ()).throw(FileNotFoundError())))
    monkeypatch.setattr(md, "WESAD_PARAMS", tmp_path / "missing.json")
    assert md.Fluctuation.load() is None
