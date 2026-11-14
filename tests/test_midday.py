"""Tests for midday.py: afternoon risk after re-measuring one check-in value at 14:00."""

import numpy as np
import pandas as pd

from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.fuzzing import midday as mdd
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st

BASE = st.make_baseline()
CUT = 14 * 60
TINY = {"probability_samples": 40, "heatmap_samples": 4, "guided_rounds": 20,
        "adjust_samples": 20, "recheck_rounds": 10}


def flat_trace():
    idx = pd.Index(np.arange(config.STEPS_PER_DAY) * config.STEP_MINUTES, name="minute")
    return pd.DataFrame({"co2": 3000.0, "temp": 22.5, "humidity": 42.0, "dose_rate": 7.79},
                        index=idx)


def split_plan():
    t = lambda s, d, k: {"start": s, "duration_h": d, "type": k, "kind": "task"}
    return pl.load_plan({"date": "Mission Day 147", "environment": "mars_base",
                         "check_window": "morning", "comm_delay_min": 20, "tasks": [
                             t("07:00", 1.0, "exercise_ARED"), t("08:15", 0.5, "meal"),
                             t("10:00", 1.5, "sleep"), t("11:30", 2.25, "EVA"),
                             t("13:45", 1.0, "rest"), t("14:45", 2.25, "EVA"),
                             t("17:00", 0.5, "meal"), t("18:00", 2.0, "science_experiment"),
                             t("23:00", 6.0, "sleep")]})


def test_remeasured_sleep_also_replaces_last_night_in_the_three_nights():
    c = st.load_checkin("main_sleep_debt_eva")
    c2 = mdd.remeasured_checkin(c, {"sleep_last_night_h": 5.9})
    assert c2.sleep_last_night_h == 5.9 and c2.sleep_last_3_nights_h == (4.3, 4.5, 5.9)
    assert c2.systolic_bp == c.systolic_bp


def test_morning_record_lists_what_already_happened():
    c = mdd.remeasured_checkin(st.load_checkin("main_sleep_debt_eva"), {"sleep_last_night_h": 5.9})
    rec = mdd.morning_record(split_plan(), flat_trace(), c, BASE, CUT)
    assert [(r["rule"], r["clock"]) for r in rec] == [("rule1_sleep_eva", "11:30")]
    assert "6 h" in rec[0]["text"]


def test_afternoon_candidates_never_touch_the_morning():
    plan = split_plan()
    c = mdd.remeasured_checkin(st.load_checkin("main_sleep_debt_eva"), {"sleep_last_night_h": 5.9})
    cands = mdd.afternoon_candidates(plan, c, CUT)
    assert cands
    morning = {(t.start_min, t.duration_min, t.type) for t in plan.tasks if t.start_min < CUT}
    for cand in cands:
        assert {(t.start_min, t.duration_min, t.type)
                for t in cand.plan.tasks if t.start_min < CUT} == morning, cand.changes
    texts = [" / ".join(x.changes) for x in cands]
    assert any("Cancel the afternoon EVA" in t for t in texts)
    assert any("18:00" in t for t in texts)                      # after the 12-h rule-1 window
    assert any("Move the EVA" in t and "14:45" not in t for t in texts)  # edits hit part 2


def test_forecast_midday_main(monkeypatch, tmp_path):
    monkeypatch.setitem(config.FUZZ_PROFILES, "local", TINY)
    monkeypatch.setattr(en, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mdd, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mdd, "_day_plan", lambda name, profile: split_plan())
    out = mdd.forecast_midday("main_sleep_debt_eva", {"sleep_last_night_h": 5.9}, CUT,
                              profile_name="local")
    assert out["cutoff"] == "14:00" and out["remeasured"] == {"sleep_last_night_h": 5.9}
    assert out["morning_violations"][0]["rule"] == "rule1_sleep_eva"
    assert out["risk_before"] == 1.0                    # part 2 at 14:45 breaks rule 1 for sure
    assert out["rating_before"] == "CRITICAL"           # 45 min after the 14:00 check
    assert out["time_to_crash_before"] == 45
    assert out["by_rule_before"]["rule1_sleep_eva"] == TINY["probability_samples"]
    assert out["risk_after"] < out["risk_before"] and out["changes"]
    pm = [t for t in out["plan_after"]["tasks"] if pl._minutes(t["start"], "") >= CUT]
    assert not any(t["type"] == "EVA" and pl._minutes(t["start"], "") < 18 * 60 for t in pm)
    assert out["from_cache"] is False
    again = mdd.forecast_midday("main_sleep_debt_eva", {"sleep_last_night_h": 5.9}, CUT,
                                profile_name="local")
    assert again["from_cache"] is True and again["risk_after"] == out["risk_after"]


def test_afternoon_rating_counts_time_to_crash_from_the_check():
    c = st.load_checkin("main_sleep_debt_eva")                  # day starts 06:30 (390 min)
    start = en._start_min(c)
    crash_at_1445 = 14 * 60 + 45 - start                        # minutes from the day start
    assert mdd.afternoon_rating(1.0, crash_at_1445, CUT, c, []) == ("CRITICAL", 45)
    late = 14 * 60 + 45 + 120 + 15 - start                      # 2 h 15 min after the check
    assert mdd.afternoon_rating(1.0, late, CUT, c, []) == ("HIGH", 180)
    assert mdd.afternoon_rating(0.002, None, CUT, c, []) == ("LOW", None)
    assert mdd.afternoon_rating(0.002, None, CUT, c, ["outside experience"])[0] == "MEDIUM"


def test_full_cache_midday_demo(monkeypatch):
    """Real engine numbers for the midday demo (main, wristband sleep 5.9 h, 14:00, full
    profile) when the committed cache is present and fresh: rule 1 forbids an EVA within
    12 h of waking after < 6 h sleep, so the afternoon EVA moves to 18:00."""
    import pytest

    from spacefuzz.fuzzing import adjuster as adj
    if not list(en.CACHE_DIR.glob("main_sleep_debt_eva_full_midday_*.json")):
        pytest.skip("no full-profile midday cache")

    class Stale(Exception):
        pass

    def no_compute(*a, **k):
        raise Stale

    monkeypatch.setattr(en, "probability_track", no_compute)
    monkeypatch.setattr(adj, "adjust", no_compute)
    try:
        out = mdd.forecast_midday("main_sleep_debt_eva", {"sleep_last_night_h": 5.9}, CUT,
                                  profile_name="full")
    except (Stale, mdd.MorningMissing):
        pytest.skip("full-profile midday cache is stale")
    assert out["from_cache"] is True
    assert out["risk_before"] == 1.0 and out["rating_before"] == "CRITICAL"
    assert f"{out['risk_after']:.1%}" == "4.9%" and out["rating_after"] == "MEDIUM"
    assert out["changes"] == ["Start the afternoon EVA at 18:00, 12 h after waking (rule 1)"]
