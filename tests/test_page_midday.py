"""Tests for pages/3_Midday_check.py: re-measure sleep at 14:00, afternoon risk only."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "3_Midday_check.py"
MAIN = "main_sleep_debt_eva"
T = lambda s, d, k: {"start": s, "duration_h": d, "type": k, "kind": "task"}
BEFORE = {"date": "Mission Day 147", "tasks": [T("10:00", 1.5, "sleep"), T("11:30", 2.25, "EVA"),
                                               T("13:45", 1.0, "rest"), T("14:45", 2.25, "EVA"),
                                               T("23:00", 6.0, "sleep")]}
RESULT = {"plan": MAIN, "profile": "full", "cutoff_min": 840, "cutoff": "14:00",
          "remeasured": {"sleep_last_night_h": 5.9},
          "morning_violations": [{"rule": "rule1_sleep_eva", "system": "Cardiovascular",
                                  "minute": 690, "clock": "11:30",
                                  "text": "EVA started within 12 h of waking after < 6 h sleep"}],
          "risk_before": 1.0, "rating_before": "HIGH", "time_to_crash_before": 45,
          "by_rule_before": {"rule1_sleep_eva": 2500, "cardiovascular": 118},
          "changes": ["Cancel the afternoon EVA (14:45); finish it on another day"],
          "risk_after": 0.0476, "rating_after": "MEDIUM",
          "by_rule_after": {"cardiovascular": 118, "cognitive": 1},
          "escalate_to_ground": False, "plan_before": BEFORE,
          "plan_after": {**BEFORE, "tasks": [t for t in BEFORE["tasks"]
                                             if not (t["type"] == "EVA" and t["start"] == "14:45")]},
          "candidates": 38, "selection": [], "from_cache": True}


def text(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                     + [i.value for i in at.info] + [h.proto.body for h in at.get("html")])


def page(ss=None):
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    at.session_state["active_scenario"] = MAIN
    for k, v in (ss or {}).items():
        at.session_state[k] = v
    return at.run()


def test_remeasure_input_is_an_example_and_shows_the_morning_value():
    at = page()
    assert not at.exception, at.exception
    n = next(x for x in at.number_input if x.label.startswith("Last night's sleep"))
    assert n.value == ui_config.MIDDAY_EXAMPLE[MAIN]["sleep_last_night_h"]
    t = text(at)
    assert "example input" in n.help and "6.2 h" in n.help        # the morning self-report
    assert "Midday check 14:00" in t and 'data-testid="illustrative"' in t


def test_run_shows_afternoon_risk_morning_record_and_board():
    at = page({"midday_results": {f"{MAIN}|5.9": RESULT}})
    next(b for b in at.button if b.label == "Run midday forecast").click().run()
    assert not at.exception, at.exception
    t = text(at)
    assert "Afternoon risk · from 14:00" in t and "100.0%" in t and "4.8%" in t
    assert "Already happened" in t and "11:30" in t
    assert "Recorded after the fact and reported to the flight surgeon" in t
    assert "Cancel the afternoon EVA" in t
    assert "Remaining afternoon risk" in t and "heart rate" in t
    assert 'data-testid="midday-done"' in t
    from tests.test_app import frame_docs
    board = frame_docs(at, "sf-plan-board")
    assert board and 'data-testid="now-line"' in board[0]


def test_other_scenarios_point_to_the_main_demo():
    at = page()
    at.session_state["active_scenario"] = "backup_high_co2"
    at.run()
    assert not at.exception
    assert "set up for the main scenario" in text(at)


def test_opening_the_page_never_starts_a_forecast(monkeypatch):
    from spacefuzz.fuzzing import engine as en
    from spacefuzz.fuzzing import midday

    def no_run(*a, **k):
        raise AssertionError("the midday page started a forecast just by opening")

    monkeypatch.setattr(en, "forecast", no_run)
    monkeypatch.setattr(midday.adjuster, "adjust", no_run)
    monkeypatch.setattr(midday, "cached_morning", lambda *a, **k: None)
    at = page()
    assert not at.exception, at.exception
    assert "not in the cache yet" in text(at)


def test_uses_the_app_profile_and_the_local_profile_for_other_values(monkeypatch):
    from spacefuzz.fuzzing import midday
    seen = []
    monkeypatch.setenv("FUZZ_PROFILE", "local")
    monkeypatch.setattr(midday, "cached_morning", lambda n, p: seen.append(("morning", p)))
    monkeypatch.setattr(midday, "forecast_midday",
                        lambda n, r, c, profile_name=None: seen.append(("run", profile_name))
                        or RESULT)
    at = page()
    next(b for b in at.button if b.label == "Run midday forecast").click().run()
    assert not at.exception, at.exception
    assert ("morning", "local") in seen and ("run", "local") in seen


def test_says_when_the_morning_check_in_was_edited():
    from ui import checkin_store as cs
    at = page({"checkin_edits": {MAIN: {**cs.example(MAIN), "systolic_bp": 150}}})
    assert not at.exception
    assert "Check-in edits are not used here" in text(at)
    assert any("example morning check-in" in (c.help or "") for c in at.caption)


def test_run_locks_until_the_board_has_played(monkeypatch):
    from tests.test_app import click_and_hold, frame_docs
    at = page({"midday_results": {f"{MAIN}|5.9": RESULT}})
    click_and_hold(at, "Run midday forecast", monkeypatch)
    assert next(b for b in at.button if b.label == "Running midday forecast...").disabled
    assert 'data-state="play"' in frame_docs(at, "sf-plan-board")[0]
    at.run()
    assert not next(b for b in at.button if b.label == "Run midday forecast").disabled


def test_no_safe_afternoon_change_is_not_called_not_red():
    still_red = {**RESULT, "changes": [], "rating_after": "HIGH"}
    at = page({"midday_results": {f"{MAIN}|5.9": still_red}})
    next(b for b in at.button if b.label == "Run midday forecast").click().run()
    t = text(at)
    assert "not red" not in t and "No safe adjustment" in t


def test_run_uses_the_value_typed_just_before_the_click():
    at = page({"midday_results": {f"{MAIN}|5.5": {**RESULT, "remeasured": {"sleep_last_night_h": 5.5}}}})
    at.number_input(key="midday_sleep").set_value(5.5)
    next(b for b in at.button if b.label == "Run midday forecast").click().run()
    assert not at.exception, at.exception
    assert at.session_state["midday_shown"] == f"{MAIN}|5.5"


def test_missing_morning_result_shows_a_note_and_unlocks(monkeypatch):
    """No morning forecast in the cache: the page says what to do; no traceback, button free."""
    from spacefuzz.fuzzing import midday

    def missing(*a, **k):
        raise midday.MorningMissing("not in the cache")
    monkeypatch.setattr(midday, "forecast_midday", missing)
    at = page()
    next(b for b in at.button if b.label == "Run midday forecast").click().run()
    assert not at.exception, at.exception
    assert any("press Run Forecast" in e.value for e in at.error)   # stays after the rerun
    assert not next(b for b in at.button if b.label == "Run midday forecast").disabled



def test_sleep_input_shows_one_decimal():
    at = page()
    n = next(x for x in at.number_input if x.label.startswith("Last night's sleep"))
    assert n.proto.format == "%.1f"
