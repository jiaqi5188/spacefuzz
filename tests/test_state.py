"""Tests for state.py: synthetic baseline, check-in loading, initial state."""
import dataclasses

import pytest

from spacefuzz import config
from spacefuzz.core import state as st


def test_synthetic_history_is_reproducible_and_14_days():
    a, b = st.synthetic_history(seed=config.RANDOM_SEED), st.synthetic_history(seed=config.RANDOM_SEED)
    assert len(a) == config.BASELINE_DAYS
    assert a.equals(b)
    assert not a.equals(st.synthetic_history(seed=1))


def test_baseline_lies_inside_the_analysis_ranges():
    base = st.make_baseline()
    lo, hi = config.BASELINE_HR_RANGE_BPM
    assert lo - 3 < base.resting_hr < hi + 3
    lo, hi = config.BASELINE_RMSSD_RANGE_MS
    assert lo < base.rmssd < hi
    lo, hi = config.BASELINE_GLUCOSE_MG_DL_RANGE
    assert lo < base.glucose < hi
    assert base.sleep_h == pytest.approx(config.BASELINE_SLEEP_H, abs=0.5)


def test_load_checkin_and_sleep_debt():
    base = st.make_baseline()
    main = st.load_checkin("main_sleep_debt_eva")
    safe = st.load_checkin("data/checkins/safe_day.json")
    assert main.sleep_last_night_h == 6.2 and main.sleep_last_3_nights_h == (4.3, 4.5, 6.2)
    assert main.check_time_min == 6 * 60 + 30
    assert st.sleep_debt_h(main, base) > 5
    assert st.sleep_debt_h(safe, base) < 0.5


def test_checkin_missing_field_is_an_error(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"plan": "x", "check_window": "morning"}')
    with pytest.raises(st.CheckInError, match="sleep_last_night_h"):
        st.load_checkin(p)


def test_initial_state_uses_checkin_inputs_and_sleep_debt():
    base = st.make_baseline()
    main = st.initial_state(st.load_checkin("main_sleep_debt_eva"), base)
    safe = st.initial_state(st.load_checkin("safe_day"), base)
    assert main.heart_rate > safe.heart_rate            # chain 5: debt raises resting HR
    assert main.hrv_rmssd < safe.hrv_rmssd              # ... and lowers HRV
    assert main.blood_glucose == 88 and main.systolic_bp == 126   # inputs, not modelled
    assert main.cognitive_fatigue > safe.cognitive_fatigue        # from Samn-Perelli score
    assert main.reaction_time_ms == 312                 # measured value anchors the start
    assert main.minute == 6 * 60 + 30


def test_checkin_without_reaction_time_uses_the_model():
    base = st.make_baseline()
    c = dataclasses.replace(st.load_checkin("safe_day"), reaction_time_ms=None)
    s = st.initial_state(c, base)
    assert s.reaction_time_ms == pytest.approx(
        base.reaction_ms * (1 + config.SLEEP_DEBT_RT_PER_H * st.sleep_debt_h(c, base))
        + config.RT_FATIGUE_MS_PER_POINT * s.cognitive_fatigue)


# ---- age-predicted maximum heart rate (220 - age)

def test_baseline_carries_age_and_max_heart_rate():
    b = st.make_baseline()
    assert b.age_y == config.BASELINE_AGE_Y
    assert st.max_heart_rate(b) == pytest.approx(config.HR_MAX_INTERCEPT_BPM - b.age_y)
    assert st.max_heart_rate(dataclasses.replace(b, age_y=45)) == pytest.approx(175.0)
    assert st.max_heart_rate(dataclasses.replace(b, age_y=30)) == pytest.approx(190.0)


# ---- malformed check-in entries are a CheckInError (found by tests/test_fuzz_app.py)

@pytest.mark.parametrize("field, value", [
    ("systolic_bp", ""), ("systolic_bp", None), ("blood_glucose", "abc"), ("mood", float("nan")),
    ("stress", float("inf")), ("fatigue_score", None), ("sleep_last_3_nights_h", None),
    ("sleep_last_3_nights_h", [None, 5, 6]), ("food_water_intake", "x"), ("check_time", None),
])
def test_malformed_checkin_entry_is_a_checkin_error(field, value):
    import json
    data = json.loads((st.ROOT / "data" / "checkins" / "main_sleep_debt_eva.json").read_text())
    data[field] = value
    with pytest.raises(st.CheckInError) as exc:
        st.load_checkin(data)
    assert str(exc.value)
