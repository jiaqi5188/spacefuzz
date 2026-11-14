"""Tests for app.py (plan 4.1 main screen): pure helpers plus a headless AppTest run.

Engine results are injected into st.session_state (fake, small) so the UI tests do not
depend on cache/; one integration test uses the real full-profile cache when present.
"""
import os
import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"  # skip animation delays

import app  # noqa: E402
from spacefuzz import config  # noqa: E402
from spacefuzz.fuzzing import engine as en  # noqa: E402
import ui_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
MAIN = "main_sleep_debt_eva"


# ---------------------------------------------------------------- fake engine results

def fake_forecast(rating="HIGH", p=0.2096, ttc=420):
    return {
        "plan": MAIN, "profile": "full", "seed": 147, "data_quality": "ok", "missing": [],
        "advice_now": ["Short sleep (6.2 h last night): schedule a 20-minute nap before "
                       "demanding tasks and go to bed early tonight."],
        "action_restrictions": [], "holds": {}, "risk_rating": rating,
        "escalation": "ground", "risk_probability": p, "time_to_crash_minutes": ttc,
        "primary_vulnerable_system": "Cardiovascular",
        "causal_chain": ["Sleep debt 6.9 h", "EVA stores heat faster", "Sustained high HR"],
        "min_crash_perturbation": {"text": "15 min less sleep turns this red"},
        "coverage_warnings": [],
        "probability_track": {"samples": 2500, "crashes": 524, "risk_probability": p,
                              "by_system": {"Cardiovascular": 472, "CNS": 64, "Radiation": 7}},
        "guided_track": {"rounds": 10000, "crash_types": [{"system": "Cardiovascular",
                                                           "rule": "cardiovascular"}] * 5},
        "heatmap": {"sleep_cut_h": [0, 1], "eva_extend_h": [0, 1, 2],
                    "risk": [[0.2, 0.2, 0.2], [1, 1, 1]]},
        "performance": {"seconds": 211.9, "simulations": 100035,
                        "simulations_per_second": 472.1},
    }


def fake_adjust(after=0.0448, rating_after="MEDIUM", ground=False):
    return {
        **{k: v for k, v in fake_forecast().items()
           if k not in ("probability_track", "guided_track", "heatmap", "performance")},
        "min_crash_perturbation": "15 min less sleep turns this red",
        "advice_today": ["Drink water and take carbs before EVA."],
        "measurement_prompts": ["Confirm last night's sleep with the wearable record."],
        "plan_adjustment": {
            "risk_before": 0.2096, "risk_after": after, "rating_after": rating_after,
            "changes": ["Shorten the EVA by 2 h", "Insert a 1 h 30 min nap before the EVA"],
            "escalate_to_ground": ground, "retest": None,
            "safer_option": {"changes": ["Postpone EVA to another day"], "risk": 0.0008,
                             "rating": "LOW", "scope": "Risk for today only.",
                             "note": "Safer option: postpone the EVA."}},
        "hud_directives": [], "concurrency_control": [],
        "details": {"candidates": [{"changes": ["a"], "cost": 1, "screen_risk": 0.2}] * 3,
                    "selection": [{"changes": ["Shorten the EVA by 2 h"], "screen": 0.0448,
                                   "confirm": 0.0448, "result": "chosen"}],
                    "adjusted_plan": {"date": "Mission Day 147", "tasks": [
                        {"start": "10:00", "duration_h": 1.5, "type": "sleep", "kind": "task"},
                        {"start": "11:30", "duration_h": 2.25, "type": "EVA", "kind": "task"}]}},
    }


def fake_results(**kw):
    return {"name": MAIN, "profile": "full", "forecast": fake_forecast(),
            "adjust": fake_adjust(**kw)}


# ---------------------------------------------------------------- helpers

def test_fmt_pct_and_clock():
    assert app.fmt_pct(0.2096) == "21.0%"
    assert app.fmt_pct(0.0448) == "4.5%"
    assert app.fmt_pct(0.0008) == "0.1%"
    assert app.fmt_pct(None) == "--"
    assert app.clock(390 + 420) == "13:30"
    assert app.clock(24 * 60 + 30) == "00:30"


def test_time_to_crash_text():
    assert "13:30" in app.time_to_crash_text(420, 390)
    assert "7 h 00 min" in app.time_to_crash_text(420, 390)
    assert "not shown" in app.time_to_crash_text(None, 390).lower()


def test_timeline_reveals_forecast_before_adjustment_from_real_numbers():
    frames = app.timeline(fake_results())
    stages = [f["stage"] for f in frames]
    assert stages.index("adjust") > max(i for i, s in enumerate(stages) if s == "forecast")
    done = [f.get("done") for f in frames]
    assert done.index("forecast") < done.index("adjust") == len(frames) - 1
    text = " ".join(f["text"] for f in frames)
    assert "2,500" in text and "10,000" in text and "21.0%" in text and "4.5%" in text


def test_timeline_untrusted_has_no_adjustment():
    res = fake_results()
    res["forecast"] = {**res["forecast"], "risk_rating": "UNKNOWN", "risk_probability": None}
    res["adjust"] = None
    frames = app.timeline(res)
    assert [f.get("done") for f in frames][-1] == "adjust"
    assert all(f["stage"] == "forecast" or "No forecast" in f["text"] for f in frames)


def test_indicator_rows_six_with_units_baseline_and_illustrative_thresholds():
    static = app.load_static(MAIN)
    rows = static["indicators"]
    assert [r["key"] for r in rows] == ["heart_rate", "blood_pressure", "blood_glucose",
                                         "body_temperature", "sleep", "mood"]
    units = {"heart_rate": "bpm", "blood_pressure": "mmHg", "blood_glucose": "mg/dL",
             "body_temperature": "°C", "sleep": " h", "mood": "/5"}
    for r in rows:
        assert units[r["key"]] in r["value"]
        if r["key"] != "mood":                          # mood / stress has no personal baseline
            assert units[r["key"]] in r["baseline"]
        assert "illustrative" in r["threshold"]
        assert r["status"] in ("green", "yellow", "red", "grey")
    by = {r["key"]: r for r in rows}
    assert by["sleep"]["status"] == "yellow"            # 6.2 h, 6.9 h debt -> act-now advice
    assert by["blood_pressure"]["status"] == "green"    # 126/82 mmHg in range
    assert by["blood_pressure"]["example_data"] and by["blood_glucose"]["example_data"]


def test_layers_original_reported_adjusted_crew():
    layers = app.layers(fake_adjust())
    assert "flight surgeon" in layers["original"] and "21.0%" in layers["original"]
    assert "crew" in layers["adjusted"].lower() and "4.5%" in layers["adjusted"]
    red = app.layers(fake_adjust(after=0.3232, rating_after="HIGH", ground=True))
    assert "ground" in red["adjusted"].lower()


def test_streamlit_dark_theme_config():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["theme"]["base"] == "dark"
    assert cfg["browser"]["gatherUsageStats"] is False   # offline: no telemetry calls


def test_ui_config_profile_default_is_full():
    assert ui_config.APP_FUZZ_PROFILE in config.FUZZ_PROFILES
    assert ui_config.APP_FUZZ_PROFILE == "full"


def test_ui_config_is_not_part_of_the_engine_cache_key():
    # editing UI settings (colours, timings, name) must not invalidate the cached runs
    assert not any(s.endswith("ui_config.py") for s in en.SOURCES)
    assert "spacefuzz/config.py" in en.SOURCES
    assert all((ROOT / s).exists() for s in en.SOURCES)


def test_ui_config_values():
    assert ui_config.CREW_MEMBER_NAME
    assert ui_config.EDITED_PROFILE in config.FUZZ_PROFILES
    assert ui_config.FRAME_S > 0 and ui_config.REVEAL_S > 0
    assert set(ui_config.COLOR) == {"green", "yellow", "red", "grey"}
    assert set(ui_config.RATING_COLOR) == {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}


# ---------------------------------------------------------------- AppTest

def page_html(at: AppTest) -> str:
    parts = [m.value for m in at.markdown]
    parts += [h.proto.body for h in at.get("html")]
    return "\n".join(parts)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def frame_docs(at, marker: str = "") -> list[str]:
    """Documents of the animated panels (ui/live_frame.py) that contain `marker`."""
    import json
    docs = [json.loads(e.proto.json_args)["doc"] for e in at.get("component_instance")
            if e.proto.component_name.endswith("live_frame")]
    return [d for d in docs if marker in d]


def click_and_hold(at, label, monkeypatch):
    """Click a locking button but stop before its last step (ui/actions.finish), so the page
    shows the state while the job runs: button locked, panels playing."""
    from ui import actions
    monkeypatch.setattr(actions, "finish", lambda: None)
    button(at, label).click().run()
    monkeypatch.undo()


@pytest.fixture
def at():
    a = AppTest.from_file(str(APP_PATH), default_timeout=30)
    a.session_state["results"] = {MAIN: fake_results()}
    a.run()
    assert not a.exception, a.exception
    return a


def test_initial_layout(at):
    h = page_html(at)
    for text in ("Mars Base", "Mission Day 147", "Earth delay 20 min", "Morning",
                 ui_config.CREW_MEMBER_NAME):
        assert text in h
    for tid in ("topbar", "plan-table", "body-map", "indicators", "act-now", "advice-cards",
                "risk-probability", "time-to-crash", "risk-comparison"):
        assert f'data-testid="{tid}"' in h, tid
    assert 'data-testid="forecast-done"' not in h
    assert "illustrative" in h


def test_run_forecast_shows_red_risk_and_adjusted_result(at):
    button(at, "Run Forecast").click().run()
    assert not at.exception, at.exception
    h = page_html(at)
    for tid in ("forecast-done", "adjust-done", "playback-done", "plan-table-adjusted"):
        assert f'data-testid="{tid}"' in h, tid
    assert "21.0%" in h and "HIGH" in h and "4.5%" in h and "MEDIUM" in h
    assert 'data-rating="red"' in h
    assert "13:30" in h                                  # time to crash 420 min after 06:30
    assert "Postpone EVA" in h and "today only" in h
    assert "flight surgeon" in h


def test_full_cache_integration(monkeypatch):
    """Real engine numbers (seed 147, full profile) when the cache is present and fresh."""
    if not list((ROOT / "cache").glob(f"{MAIN}_full_adjust_*.json")):
        pytest.skip("no full-profile cache")

    class Stale(Exception):
        pass

    def no_compute(*a, **k):
        raise Stale

    monkeypatch.setattr(en, "probability_track", no_compute)
    try:
        res = app.load_results(MAIN, "full")
    except Stale:
        pytest.skip("full-profile cache is stale")
    frames = app.timeline(res)
    text = " ".join(f["text"] for f in frames)
    assert "21.0%" in text and "4.9%" in text
    assert "Split the EVA into two parts" in text and "11:00-13:15, 14:15-16:30" in text


def test_one_green_everywhere():
    green = ui_config.COLOR["green"]
    assert green.lower() != "#39ff14"            # user review: the neon green was too bright
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["theme"]["primaryColor"].lower() == green.lower()   # button = same green
    for f in ("app.py", "ui/body_map.py"):
        assert "39ff14" not in (ROOT / f).read_text().lower(), f


def test_pace_profiles_fast_default_demo_by_env(monkeypatch):
    import importlib
    assert set(ui_config.PACES) == {"fast", "demo"}
    for pace in ui_config.PACES.values():
        assert pace["frame_s"] > 0 and pace["reveal_s"] > 0
    monkeypatch.delenv("SPACEFUZZ_PACE", raising=False)
    try:
        assert importlib.reload(ui_config).PACE == "fast"
        assert ui_config.FRAME_S == ui_config.PACES["fast"]["frame_s"]
        monkeypatch.setenv("SPACEFUZZ_PACE", "demo")
        assert importlib.reload(ui_config).REVEAL_S == ui_config.PACES["demo"]["reveal_s"]
        monkeypatch.setenv("SPACEFUZZ_PACE", "slowmo")
        with pytest.raises(ValueError):
            importlib.reload(ui_config)
    finally:
        monkeypatch.delenv("SPACEFUZZ_PACE", raising=False)
        importlib.reload(ui_config)


# ---------------------------------------------------------------- review 2026-11-11

def test_plan_table_calls_a_short_sleep_a_nap():
    plan = {"tasks": [{"start": "10:00", "duration_h": 1.5, "type": "sleep"},
                      {"start": "23:00", "duration_h": 6.0, "type": "sleep"}]}
    h = app.plan_table_html(plan, "t")
    assert "<td>nap</td>" in h and "<td>sleep</td>" in h


def test_act_now_nap_points_to_the_adjusted_plan():
    fc = {"advice_now": ["Short sleep (6.2 h last night, 6.9 h sleep debt over 3 nights): "
                         "schedule a 20-minute nap before demanding tasks and go to bed early "
                         "tonight."], "action_restrictions": []}
    adj = fake_adjust()                       # adjusted plan has a 1.5 h sleep at 10:00
    h = app.act_now_html(fc, adj)
    assert "20-minute" not in h
    assert "take the 1 h 30 min nap the adjusted plan schedules at 10:00" in h
    assert "20-minute nap" in app.act_now_html(fc, None)          # no adjustment: as given


def test_measurement_prompts_below_one_point_are_hidden():
    assert ui_config.MEASURE_MIN_RISK_PP == 1.0
    adj = fake_adjust()
    adj["details"] = {**adj["details"], "measurement_sensitivity": [
        ["sleep_last_night_h", 0.7904], ["diastolic_bp", 0.0048], ["systolic_bp", 0.0028]]}
    lines = app.measurement_lines(adj)
    assert len(lines) == 1 and lines[0].startswith("Confirm last night's sleep")
    adj["details"]["measurement_sensitivity"] = [["diastolic_bp", 0.0048]]
    assert app.measurement_lines(adj) == []


def test_comparison_keeps_the_summary_and_folds_the_long_table():
    h = app.comparison_html(fake_adjust(), {"tasks": []}, True)
    assert "21.0%" in h and "4.5% MEDIUM" in h and "Safer option" in h
    assert "Shorten the EVA by 2 h" in h
    i = h.index("<details")
    assert "open" not in h[i:h.index(">", i)]                    # collapsed by default
    assert h.index('data-testid="plan-table-adjusted"') > i


def test_risk_panel_coverage_is_a_small_grey_summary():
    fc = fake_forecast()
    fc["coverage_warnings"] = [
        "Not covered by the guided track: rules metabolic, rule2_exercise_eva; systems Metabolic",
        "CO2 up to 6000 ppm is above the highest recorded cabin CO2 (4663 ppm); its effects are "
        "extrapolated"]
    h = app.risk_html(fc, 390, True)
    assert "coverage gaps: 2 rules, 1 system · 1 outside experience (see log)" in h
    assert h.count("see log") == 1
    assert "rule2_exercise_eva" not in h and "var(--sf-muted)" in h


def test_act_now_rewrite_tracks_the_engine_text():
    """Guard: the UI rewrite (app.NAP_ADVICE) must match the nap advice advice.py writes today.
    If advice.py's wording changes, this fails instead of the two sources drifting apart."""
    from spacefuzz.fuzzing import advice
    from spacefuzz.fuzzing import mutator as mu
    from spacefuzz.core import quality_gate as qg
    from spacefuzz.core import state
    sc = mu.Scenario.load(MAIN)
    items = advice.act_now(sc.checkin, state.make_baseline(), gate=qg.check(checkin=sc.checkin))
    nap_lines = [a.text for a in items if a.code == "short_sleep"]
    assert nap_lines and all(app.NAP_ADVICE.search(x) for x in nap_lines)


def test_act_now_original_before_the_adjust_reveal_rewritten_after():
    res = fake_results()
    frames = app.timeline(res)
    i_fc = next(i for i, f in enumerate(frames) if f.get("done") == "forecast") + 1
    i_adj = next(i for i, f in enumerate(frames) if f.get("done") == "adjust") + 1
    assert "Shown with the forecast" in app.act_now_view(res, frames, 0)
    before = app.act_now_view(res, frames, i_fc)            # forecast revealed, adjust not yet
    assert "20-minute nap" in before and "adjusted plan" not in before
    after = app.act_now_view(res, frames, i_adj)
    assert "20-minute nap" not in after
    assert "take the 1 h 30 min nap the adjusted plan schedules at 10:00" in after


# ---------------------------------------------------------------- edited check-in (4.3)

def test_edited_checkin_flows_into_the_dashboard():
    from ui import checkin_store as cs
    edits = {MAIN: {**cs.example(MAIN), "systolic_bp": 150, "sleep_last_night_h": 7.0,
                    "sleep_last_3_nights_h": [6.5, 6.8, 7.0]}}
    ss = {"checkin_edits": edits}
    key = cs.key(MAIN, ss)
    assert key != MAIN
    static = app.load_static(MAIN, ss)
    assert static["key"] == key and static["checkin_json"]
    bp = next(r for r in static["indicators"] if r["key"] == "blood_pressure")
    assert bp["value"] == "150/82 mmHg" and bp["status"] == "red"     # act-now: out of range
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.session_state["checkin_edits"] = edits
    at.session_state["results"] = {key: fake_results()}             # results keyed by check-in
    at.run()
    assert not at.exception
    h = page_html(at)
    assert "edited check-in" in h and "150/82 mmHg" in h
    button(at, "Run Forecast").click().run()
    assert not at.exception and 'data-testid="playback-done"' in page_html(at)
    assert at.session_state["active_scenario"] == MAIN


def test_edited_checkin_is_forecast_with_the_local_profile(monkeypatch):
    from spacefuzz.fuzzing import engine
    from ui import checkin_store as cs
    calls = {}
    untrusted = {**fake_forecast(), "risk_rating": "UNKNOWN"}       # skips the adjuster

    def spy(path, scenario=None, profile_name=None, baseline=None, **k):
        calls["forecast"] = (profile_name, scenario.checkin.systolic_bp)
        return untrusted

    monkeypatch.setattr(engine, "forecast", spy)
    j = cs.checkin_json(MAIN, {"checkin_edits": {MAIN: {**cs.example(MAIN), "systolic_bp": 151}}})
    res = app.load_results(MAIN, app.ui_config.app_profile(), checkin_json=j)
    assert calls["forecast"] == ("local", 151.0) and res["profile"] == "local"
    app.load_results(MAIN, app.ui_config.app_profile())                       # example check-in: full
    assert calls["forecast"] == ("full", cs.example(MAIN)["systolic_bp"])


def test_unknown_forecast_names_what_to_re_measure():
    msg = "Data not trustworthy. Please measure again. Re-measure: sleep last night (20 outside 0-16)."
    res = {"forecast": {"risk_rating": "UNKNOWN", "message": msg}, "adjust": None}
    assert app.timeline(res)[0]["text"] == msg
    assert "Re-measure: sleep last night" in app.cards_html(res["forecast"], None, True)


def test_three_columns_of_about_equal_length(at):
    """Left: check-in, act now, trends, plan table; middle: body map, before / after; right:
    forecast, plan board, fuzz log, advice cards (no empty block mid-page, user 2026-11-13)."""
    h = page_html(at)
    order = [h.index(f'data-testid="{t}"') for t in
             ("indicators", "act-now", "plan-table", "body-map", "risk-comparison",
              "risk-probability", "plan-board", "fuzz-log", "advice-cards")]
    assert order == sorted(order)


# ---------------------------------------------------------------- review fixes 2026-11-14

def _no_candidate():
    """The adjuster found no plan change that helps (e.g. a blood-pressure rule): still red."""
    res = fake_results(after=0.2096, rating_after="HIGH", ground=True)
    res["adjust"]["plan_adjustment"]["changes"] = []
    return res


def test_no_safe_adjustment_is_not_called_not_red():
    from ui import common, fuzz_log, plan_board
    res = _no_candidate()
    text = " ".join(f["text"] for f in app.timeline(res))
    assert "not red" not in text and "No safe adjustment" in text
    log = " ".join(x["text"] for x in fuzz_log.log_lines(res, 390))
    assert "not red" not in log and "no safe adjustment" in log
    risk = plan_board.risk_states(res["adjust"])
    doc = plan_board.html_doc("final", plan_board.match_tasks(app.load_static(MAIN)["plan"], None),
                              risk, [])
    assert "not red" not in doc and "No safe adjustment" in doc
    assert common.no_change_text(still_red=False) == "No change needed: the plan is not red."


def test_last_frame_waits_for_the_plan_board_roll(monkeypatch):
    """Playback ends only after the board's slide and risk roll (BOARD_S), so the final
    picture never cuts the roll short."""
    waits = []
    monkeypatch.setattr(app.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(app, "REVEAL_S", 1.2)
    monkeypatch.setattr(app, "BOARD_S", 1.6)
    assert app.frame_wait({"done": "adjust"}, last=True) == 1.6
    assert app.frame_wait({"done": "forecast"}, last=False) == 1.2


def test_a_failed_computation_unlocks_the_button(monkeypatch):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    def boom(*a, **k):
        raise RuntimeError("engine exploded")
    import app as app_module
    monkeypatch.setattr(app_module, "load_results", boom)
    button(at, "Run Forecast").click().run()
    assert any("Forecast failed" in e.value for e in at.error)
    assert not button(at, "Run Forecast").disabled            # not locked after the error


def test_a_long_computation_is_not_dropped_as_stale(monkeypatch):
    import time as _t
    from ui import actions
    import streamlit as st
    st.session_state.clear()
    actions.start("forecast", key="k")
    later = _t.time() + actions.STALE_S + 5
    monkeypatch.setattr(actions.time, "time", lambda: later)
    assert actions.job("forecast", max_age=actions.COMPUTE_STALE_S) is not None
    assert actions.job("forecast") is None
