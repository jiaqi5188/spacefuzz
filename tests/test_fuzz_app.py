"""Fuzz the app's input handling with hypothesis (plan 3.12, manual 4.4).

Malformed task plans and extreme self-report entries must never crash the interface:
- a plan either loads or is refused with a PlanError (a message the UI can show); a plan that
  loads goes through the UI logic (plan table, plan board, simulation, rules, out-of-experience
  check, fuzz log) without an exception;
- an unknown task type is mapped to generic_high_load and flagged in the fuzz log's [cover]
  line ("outside model experience");
- a self-report either parses or is refused with a CheckInError; one that parses goes through
  the quality gate, act-now advice and the dashboard's static part; the check-in page takes
  extreme entries; the reaction-test summary takes any array;
- no online large-language-model call and no network connection happens at any point.
Fixed seed (derandomize) so a failure is reproducible.
"""
import math
import os
import socket
import sys
import types
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from hypothesis import HealthCheck, event, given, settings  # noqa: E402
from hypothesis import strategies as s  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from spacefuzz.fuzzing import advice  # noqa: E402
import app  # noqa: E402
from spacefuzz import config  # noqa: E402
from spacefuzz.fuzzing import engine as en  # noqa: E402
from spacefuzz.core import evaluator as ev  # noqa: E402
from spacefuzz.core import model as md  # noqa: E402
from spacefuzz.fuzzing import mutator as mu  # noqa: E402
from spacefuzz.core import plan as pl  # noqa: E402
from spacefuzz.core import quality_gate as qg  # noqa: E402
from spacefuzz.core import state as st  # noqa: E402
from spacefuzz.core.plan_types import PlanType  # noqa: E402
from ui import checkin_store as cs  # noqa: E402
from ui import fuzz_log, plan_board, reaction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = st.make_baseline()
FUZZ = settings(max_examples=60, deadline=None, derandomize=True,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
SLOW = settings(max_examples=12, deadline=None, derandomize=True,
                suppress_health_check=[HealthCheck.too_slow])
KNOWN_TASKS = [t.value for t in PlanType if t not in (PlanType.SPE_WARNING, PlanType.HIGH_CO2)]


# ---------------------------------------------------------------- no online LLM, no network

class _Poison(types.ModuleType):
    """Stands in for `openai`: any use is recorded and fails loudly."""
    touched: list = []

    def __getattr__(self, name):
        if name.startswith("__"):            # module introspection by tooling, not an API call
            raise AttributeError(name)
        _Poison.touched.append(name)
        raise AssertionError(f"online LLM client used: openai.{name}")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    _Poison.touched = []
    monkeypatch.setitem(sys.modules, "openai", _Poison("openai"))

    def refuse(*a, **k):
        raise AssertionError(f"network connection attempted: {a[1:] if len(a) > 1 else a}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield
    assert not _Poison.touched, f"online LLM client used: {_Poison.touched}"


def test_no_module_imports_an_llm_or_http_client():
    import re
    bad = re.compile(r"^\s*(import|from)\s+(openai|anthropic|requests|httpx|urllib\.request)\b",
                     re.M)
    files = [*ROOT.glob("*.py"), *(ROOT / "ui").glob("*.py"), *(ROOT / "page_files").glob("*.py"),
             *(ROOT / "spacefuzz").rglob("*.py")]
    # ui/assist.py is the one exception: the optional AI hint imports the client inside a
    # function, only with a key set (tests/test_assist.py checks that).
    files = [f for f in files if f != ROOT / "ui" / "assist.py"]
    assert not [f.name for f in files if bad.search(f.read_text())]


# ---------------------------------------------------------------- strategies

TIMES = s.one_of(
    s.builds(lambda h, m: f"{h:02d}:{m:02d}", s.integers(0, 23), s.sampled_from([0, 15, 30, 45])),
    s.sampled_from(["24:00", "7:5", "", "noon", "-01:00", "12:60"]),
    s.integers(-100, 3000), s.none(), s.floats(allow_nan=True))
DURATIONS = s.one_of(s.floats(-5, 30, allow_nan=False), s.sampled_from([0, -1, 24, 24.25, 48]),
                     s.just(float("nan")), s.just(float("inf")), s.text(max_size=3), s.none())
TYPES = s.one_of(s.sampled_from(KNOWN_TASKS),
                 s.sampled_from(["spe_warning", "high_co2", "yoga", "rover_repair", "", "EVA "]),
                 s.text(max_size=12), s.integers(), s.none())
TASK = s.one_of(
    s.fixed_dictionaries({"start": TIMES, "duration_h": DURATIONS, "type": TYPES},
                         optional={"kind": s.sampled_from(["task", "env", "", None])}),
    s.text(max_size=5), s.integers(), s.none(), s.lists(s.integers(), max_size=2))
EVENT = s.one_of(
    s.fixed_dictionaries({"start": TIMES, "end": TIMES,
                          "type": s.sampled_from(["spe_warning", "high_co2", "meteor", None]),
                          "kind": s.sampled_from(["env", "task"])},
                         optional={"co2_ppm": s.one_of(s.integers(-100, 50000), s.none(),
                                                       s.text(max_size=3)),
                                   "co2_pct": s.floats(0, 5)}),
    s.text(max_size=3), s.none())
PLAN = s.fixed_dictionaries(
    {},
    optional={"date": s.one_of(s.text(max_size=20), s.integers(), s.none()),
              "environment": s.sampled_from(["mars_base", "iss", "moon", "", None]),
              "check_window": s.sampled_from(["morning", "midday", "night", None]),
              "comm_delay_min": s.one_of(s.integers(-10, 100), s.text(max_size=3), s.none(),
                                        s.floats(allow_nan=True)),
              "tasks": s.one_of(s.lists(TASK, max_size=8), s.text(max_size=3), s.none(),
                                s.just([])),
              "events": s.one_of(s.lists(EVENT, max_size=3), s.none(), s.text(max_size=2))})

# A plan that loads: well-formed tasks one after another, types known or unknown.
GOOD_TASK = s.tuples(s.integers(15, 240), s.one_of(s.sampled_from(KNOWN_TASKS),
                                                    s.sampled_from(["yoga", "rover_repair",
                                                                    "habitat_tour"])))


@s.composite
def good_plans(draw):
    start = draw(s.integers(6 * 4, 9 * 4)) * 15                      # 06:00-09:00
    tasks, t = [], start
    for minutes, typ in draw(s.lists(GOOD_TASK, min_size=1, max_size=7)):
        minutes = minutes // 15 * 15
        if t + minutes > start + 23 * 60 or t + minutes >= 24 * 60:
            break
        tasks.append({"start": f"{t // 60:02d}:{t % 60:02d}", "duration_h": minutes / 60,
                      "type": typ, "kind": "task"})
        t += minutes + draw(s.sampled_from([0, 15, 30]))
    if not tasks:
        tasks = [{"start": "07:00", "duration_h": 1.0, "type": "meal", "kind": "task"}]
    events = []
    if draw(s.booleans()):
        a = draw(s.integers(7, 18))
        events.append({"start": f"{a:02d}:00", "end": f"{a + draw(s.integers(1, 5)):02d}:00",
                       "type": draw(s.sampled_from(["spe_warning", "high_co2"])), "kind": "env",
                       "co2_ppm": draw(s.integers(3000, 7000))})
    return {"date": "Mission Day 147", "environment": "mars_base", "check_window": "morning",
            "comm_delay_min": 20, "tasks": tasks, "events": events}


def flat_trace():
    idx = pd.Index(np.arange(config.STEPS_PER_DAY) * config.STEP_MINUTES, name="minute")
    return pd.DataFrame({"co2": 3000.0, "temp": 22.5, "humidity": 42.0, "dose_rate": 7.79},
                        index=idx)


# ---------------------------------------------------------------- malformed plans

@FUZZ
@given(PLAN)
def test_a_malformed_plan_is_refused_with_a_plan_error_or_loads(data):
    try:
        plan = pl.load_plan(data)
    except pl.PlanError as exc:
        assert str(exc)                                      # a message the UI can show
        event("plan refused with a message")
        return
    event("plan loaded -> UI logic")
    ui_logic_on(plan)


def ui_logic_on(plan: pl.Plan) -> None:
    """Everything the interface does with a plan, short of a full engine run."""
    d = pl.to_dict(plan)
    assert "data-testid" in app.plan_table_html(d, "plan-table")
    blocks = plan_board.match_tasks(d, d)
    assert "sf-plan-board" in plan_board.html_doc("final", blocks, None, d.get("events", []))
    checkin = st.load_checkin("safe_day")
    trace = flat_trace()
    out = md.simulate(plan, pl.apply_env_events(trace, plan), checkin, BASE)
    assert len(out) == config.STEPS_PER_DAY
    ev.evaluate(out, plan, checkin, BASE)
    ev.evaluate(out, plan, checkin, BASE, count_from=14 * 60)
    warnings = en.out_of_experience(mu.Scenario(plan, trace, checkin), BASE)
    res = {"name": None, "profile": "local", "forecast": {
        "risk_rating": "LOW", "risk_probability": 0.0, "coverage_warnings": warnings,
        "probability_track": {}, "guided_track": {}, "time_to_crash_minutes": None}}
    fuzz_log.log_lines(res, 390)


@SLOW
@given(good_plans())
def test_a_valid_plan_survives_a_short_probability_run(data):
    plan = pl.load_plan(data)
    ui_logic_on(plan)
    p = en.probability_track(mu.Scenario(plan, flat_trace(), st.load_checkin("safe_day")),
                             BASE, 3)
    assert 0.0 <= p["risk_probability"] <= 1.0


# ---------------------------------------------------------------- unknown task types

PLAIN_NAME = s.from_regex(r"[A-Za-z][A-Za-z0-9_ -]{0,18}", fullmatch=True).filter(
    lambda t: t.strip() and t not in {x.value for x in PlanType})


def check_unknown_type_is_flagged(name: str) -> None:
    data = {"date": "Mission Day 147", "environment": "mars_base", "check_window": "morning",
            "comm_delay_min": 20,
            "tasks": [{"start": "09:00", "duration_h": 2.0, "type": name, "kind": "task"}]}
    try:
        plan = pl.load_plan(data)
    except pl.PlanError:
        return                                               # e.g. a name equal to an env type
    task = plan.tasks[0]
    if task.type is not PlanType.GENERIC_HIGH_LOAD:
        return                                               # matched a known type
    assert task.source_type == name
    assert any("generic_high_load" in w for w in plan.warnings)
    warnings = en.out_of_experience(mu.Scenario(plan, flat_trace(), st.load_checkin("safe_day")),
                                    BASE)
    flagged = [w for w in warnings if "outside model experience (treated as high load)" in w]
    assert flagged and name in flagged[0], warnings
    res = {"name": None, "profile": "local", "forecast": {
        "risk_rating": "LOW", "risk_probability": 0.0, "coverage_warnings": warnings,
        "probability_track": {}, "guided_track": {}, "time_to_crash_minutes": None}}
    cover = [ln["text"] for ln in fuzz_log.log_lines(res, 390) if ln["kind"] == "uncovered"]
    assert any(name in c and "high load" in c for c in cover), cover


@FUZZ
@given(s.text(min_size=1, max_size=20).filter(
    lambda t: t.strip() and t not in {x.value for x in PlanType}))
def test_unknown_task_type_becomes_generic_high_load_and_is_flagged_in_the_log(name):
    check_unknown_type_is_flagged(name)


@FUZZ
@given(PLAIN_NAME)
def test_plain_unknown_task_names(name):
    check_unknown_type_is_flagged(name)


@pytest.mark.parametrize("name", ["crew's task", "rover\x1brepair", 'say "hi"'])
def test_unknown_task_type_with_quotes_or_control_characters(name):
    check_unknown_type_is_flagged(name)


# ---------------------------------------------------------------- extreme self-report

NUM = s.one_of(s.floats(-1e6, 1e6, allow_nan=False), s.just(float("nan")),
               s.just(float("inf")), s.just(-float("inf")), s.integers(-10**6, 10**6),
               s.none(), s.text(max_size=4), s.booleans())
CHECKIN = s.fixed_dictionaries({}, optional={
    "systolic_bp": NUM, "diastolic_bp": NUM, "blood_glucose": NUM, "sleep_last_night_h": NUM,
    "sleep_efficiency_pct": NUM, "mood": NUM, "stress": NUM, "fatigue_score": NUM,
    "reaction_time_ms": NUM,
    "sleep_last_3_nights_h": s.one_of(s.lists(NUM, max_size=5), s.none(), s.text(max_size=3)),
    "check_time": TIMES, "wake_time": TIMES,
    "food_water_intake": s.one_of(s.dictionaries(s.sampled_from(["meals", "water_l_24h"]), NUM),
                                  s.none(), s.text(max_size=3))})


@FUZZ
@given(CHECKIN)
def test_an_extreme_self_report_is_refused_with_a_checkin_error_or_handled(changes):
    data = {**cs.example("main_sleep_debt_eva"), **changes}
    try:
        ci = cs.to_checkin(data)
    except st.CheckInError as exc:
        assert str(exc)
        event("self-report refused with a message")
        return
    gate = qg.check(checkin=ci)
    event(f"self-report parsed, gate {gate.status}")
    assert gate.status in ("ok", "degraded", "untrusted")
    items = advice.act_now(ci, BASE, gate=gate)
    advice.advice_now(items)
    if gate.can_forecast:
        out = md.simulate(mu.Scenario.load("main_sleep_debt_eva").plan, flat_trace(), ci, BASE)
        assert len(out) == config.STEPS_PER_DAY
    static = app.load_static("main_sleep_debt_eva", {"checkin_edits": {
        "main_sleep_debt_eva": data}})
    assert len(static["indicators"]) == 6
    assert app.indicators_html(static["indicators"])


PAGE_VALUES = s.fixed_dictionaries({
    "Systolic (mmHg)": s.sampled_from([40.0, 300.0, 181.0, 89.0]),
    "Diastolic (mmHg)": s.sampled_from([20.0, 200.0, 121.0]),
    "Blood glucose (mg/dL)": s.sampled_from([1.0, 800.0, 53.0, 69.0]),
    "Sleep last night (h)": s.sampled_from([0.0, 24.0, 5.9, 16.5]),
    "Sleep efficiency (%)": s.sampled_from([0.0, 100.0]),
    "Water (L, 24 h)": s.sampled_from([0.0, 10.0])})


@SLOW
@given(PAGE_VALUES)
def test_the_check_in_page_takes_extreme_entries(values):
    at = AppTest.from_file(str(ROOT / "page_files" / "1_Morning_check-in.py"), default_timeout=30)
    at.run()
    for label, v in values.items():
        next(n for n in at.number_input if n.label == label).set_value(v)
    next(b for b in at.button if b.label == "Save check-in").click().run()
    assert not at.exception, at.exception
    gate = [h.proto.body for h in at.get("html") if 'data-testid="checkin-gate"' in h.proto.body]
    assert gate and "Data quality:" in gate[0]
    edits = at.session_state["checkin_edits"]
    dash = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    dash.session_state["checkin_edits"] = edits
    dash.run()
    assert not dash.exception, dash.exception


@FUZZ
@given(s.one_of(s.none(), s.fixed_dictionaries(
    {"rts": s.lists(s.one_of(s.floats(allow_nan=True, allow_infinity=True),
                             s.integers(-10**6, 10**6)), max_size=60),
     "false_starts": s.integers(0, 1000), "duration_s": s.sampled_from([60, 180])})))
def test_the_reaction_summary_takes_any_array(result):
    out = reaction.summarise(result)
    if out is None:
        return
    lo, hi = config.QG_PLAUSIBLE_RANGES["reaction_time_ms"]
    assert out["n_valid"] + out["suspicious"] == out["n"]
    if out["mean_ms"] is not None:
        assert lo <= out["mean_ms"] <= hi and not math.isnan(out["mean_ms"])


def test_untrusted_check_in_runs_through_the_dashboard_as_unknown():
    """Garbage sensor entries -> gate untrusted -> grey UNKNOWN, no engine run, no crash."""
    bad = {**cs.example("main_sleep_debt_eva"), "systolic_bp": 900, "diastolic_bp": 5,
           "blood_glucose": 3000, "sleep_last_night_h": -2, "reaction_time_ms": 5}
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["checkin_edits"] = {"main_sleep_debt_eva": bad}
    at.run()
    next(b for b in at.button if b.label == "Run Forecast").click().run()
    assert not at.exception, at.exception
    html = "\n".join(h.proto.body for h in at.get("html"))
    assert 'data-rating="grey"' in html and qg.UNTRUSTED_MESSAGE in html
