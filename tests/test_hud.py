"""Tests for ui/hud.py: in-mission head-up display (visor style, one-line directive)."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from spacefuzz import config  # noqa: E402
from spacefuzz.core import state  # noqa: E402
from ui import hud  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = "main_sleep_debt_eva"
STEPS = config.STEPS_PER_DAY
T = lambda start, h, typ: {"start": start, "duration_h": h, "type": typ, "kind": "task"}


def adjust(tasks, directives, ttc=None, ground=False, events=None):
    minutes = [390 + 15 * k for k in range(STEPS)]
    hr = [80.0] * STEPS
    for k, m in enumerate(minutes):                     # 11:30-13:45: EVA-like heart rate
        if 690 <= m <= 825:
            hr[k] = 130.0
    traj = {"heart_rate": {"p5": hr, "p50": hr, "p95": [h + 15 for h in hr]},
            "core_temp": {"p5": [36.8] * STEPS, "p50": [37.0] * STEPS, "p95": [37.3] * STEPS},
            "spo2": {"p5": [96.0] * STEPS, "p50": [97.5] * STEPS, "p95": [98.0] * STEPS},
            "cognitive_fatigue": {"p5": [10.0] * STEPS, "p50": [40.0] * STEPS,
                                  "p95": [85.0] * STEPS}}
    return {"hud_directives": directives,
            "plan_adjustment": {"time_to_crash_after": ttc, "escalate_to_ground": ground,
                                "rating_after": "HIGH" if ground else "MEDIUM",
                                "risk_after": 0.3 if ground else 0.045},
            "details": {"adjusted_plan": {"tasks": tasks, "events": events or []},
                        "trajectory_after": traj, "trajectory_minutes_after": minutes,
                        "traces_after": {"crashed": [0] * 300}}}


MAIN_TASKS = [T("10:00", 1.5, "sleep"), T("11:30", 2.25, "EVA"), T("13:45", 1.0, "rest"),
              T("14:45", 2.25, "EVA"), T("23:00", 6.0, "sleep")]


def test_mission_task_first_high_load():
    assert hud.mission_task({"tasks": MAIN_TASKS}) == {"type": "EVA", "start": 690, "end": 825}
    backup = [T("13:00", 0.5, "rest"), T("13:30", 1.5, "docking")]
    assert hud.mission_task({"tasks": backup})["type"] == "docking"
    assert hud.mission_task({"tasks": [T("09:00", 1, "meal")]}) is None
    assert hud.mission_task(None) is None


def test_directive_one_line_relevant_to_the_task():
    a = adjust(MAIN_TASKS, ["Return to the airlock if heart rate stays above 140 bpm",
                            "SPE warning: stay in the shielded area"],
               events=[{"start": "09:00", "end": "15:00", "type": "spe_warning"}])
    assert hud.directive(a) == "SPE warning: stay in the shielded area"   # EVA in the warning
    late = [T("15:30", 5.0, "EVA")]
    a = adjust(late, ["Return to the airlock if heart rate stays above 140 bpm",
                      "SPE warning: stay in the shielded area"],
               events=[{"start": "09:00", "end": "15:00", "type": "spe_warning"}])
    assert hud.directive(a) == "Return to the airlock if heart rate stays above 140 bpm"
    a = adjust([T("13:30", 1.5, "docking")], ["Time to risk 11:00:00"], ttc=660, ground=True)
    assert hud.directive(a) == "Follow the adjusted plan"
    assert "Time to risk" not in hud.directive(a)


def test_countdown_from_mission_start():
    a = adjust([T("13:30", 1.5, "docking")], ["Time to risk 11:00:00"], ttc=660)
    assert hud.risk_clock(a, start_min=390) == 390 + 660                    # 17:30
    assert hud.countdown(17 * 60 + 30, 13 * 60 + 30) == "04:00:00"
    assert hud.risk_clock(adjust(MAIN_TASKS, []), start_min=390) is None


def test_mission_series_and_status():
    a = adjust(MAIN_TASKS, [])
    s = hud.mission_series(a, state.make_baseline())
    assert s["minutes"][0] == 690 and s["minutes"][-1] == 825
    hr = s["gauges"]["heart_rate"]
    assert hr["unit"] == "bpm" and hr["p50"][0] == 130.0
    assert hr["threshold"] == pytest.approx(0.8 * state.max_heart_rate(state.make_baseline()))
    assert hr["status"][0] == "yellow"            # median below, upper 5% (145) above 140
    assert s["gauges"]["spo2"]["status"][0] == "green"        # low side: p5 96 > 94
    assert s["gauges"]["cognitive_fatigue"]["status"][0] == "yellow"
    assert "illustrative" in s["note"] and "not live telemetry" in s["note"]
    assert "Median of all forecast runs" in s["note"] and "kept" not in s["note"]


def test_html_play_and_marker():
    a = adjust(MAIN_TASKS, ["Return to the airlock if heart rate stays above 140 bpm"])
    doc = hud.html_doc(a, state.make_baseline(), start_min=390, mission_name="EVA")
    pre = doc.split("<script")[0]
    assert 'data-state="play"' in pre and "sf-hud" in pre
    assert "Return to the airlock if heart rate stays above 140 bpm" in pre
    assert "11:30" in pre and "13:45" in pre
    assert 'data-testid="hud-done"' not in pre                     # added by the replay
    assert "worst 5%: <span class=\"tail\">145</span> bpm (colour) · limit 140 bpm" in pre
    assert "worst 5%: <span class=\"tail\">96.0</span>% (colour)" not in pre      # 1 decimal only for temp
    assert "worst 5%: <span class=\"tail\">96</span>% (colour) · limit 94%" in pre
    assert "http://" not in doc and "https://" not in doc
    none = hud.html_doc(None, state.make_baseline(), start_min=390)
    assert "after the forecast" in none


def test_app_toggle_shows_hud_after_forecast():
    from tests.test_app import fake_results
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    res = fake_results()
    res["adjust"] = {**res["adjust"], **adjust(MAIN_TASKS, ["Return to the airlock if heart "
                                                            "rate stays above 140 bpm"])}
    res["adjust"]["plan_adjustment"] = {**fake_results()["adjust"]["plan_adjustment"],
                                        "time_to_crash_after": None}
    at.session_state["results"] = {MAIN: res}
    at.run()
    tog = next(t for t in at.toggle if t.label == "In-mission HUD")
    tog.set_value(True).run()
    assert not at.exception
    assert any("after the forecast" in c.value for c in at.caption)    # no result yet
    next(b for b in at.button if b.label == "Run Forecast").click().run()
    assert not at.exception                    # no extra rerun: the HUD shows after playback
    from tests.test_app import frame_docs
    assert len(frame_docs(at, "sf-hud")) == 1


def test_task_window_without_trajectory_points_shows_no_gauges():
    a = adjust(MAIN_TASKS, [])
    a["details"]["trajectory_minutes_after"] = [0] * STEPS            # no point inside 11:30-13:45
    assert hud.mission_series(a, state.make_baseline()) is None
    doc = hud.html_doc(a, state.make_baseline(), start_min=390)       # no IndexError
    assert "No EVA, docking or precision task" not in doc and "forecast readings" in doc


def test_unknown_forecast_says_why_there_is_no_hud():
    doc = hud.html_doc(None, state.make_baseline(), start_min=390, forecast_ran=True)
    assert "after the forecast" not in doc and "data quality" in doc


def test_hud_and_plan_table_are_open_by_default():
    """Feature panels start open, so a judge cannot miss them (user, 2026-11-13)."""
    from tests.test_app import fake_results
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.run()
    assert next(t for t in at.toggle if t.label == "In-mission HUD").value is True
    assert at.expander[0].proto.expanded
