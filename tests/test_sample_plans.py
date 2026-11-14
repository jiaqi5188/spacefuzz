"""Checks for the sample plans in data/plans/ and their check-ins in data/checkins/.

The rule helpers below are a minimal reading of the four temporal rules in
plan 3.5, used only to prove each sample violates exactly what it should.
The real rule engine is spacefuzz/core/evaluator.py (thresholds in spacefuzz/config.py).
"""
import json
import re
from pathlib import Path

import pytest

from spacefuzz.core.plan_types import (ENV_TYPES, EXERCISE_TYPES, PRECISION_TYPES, RESTORATIVE_TYPES,
                        TASK_TYPES, PlanType, kind_of, parse_type)

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "data" / "plans"
CHECKINS = ROOT / "data" / "checkins"

# Plan 3.5 rule thresholds, now in config.py (evaluator.py implements the full rules).
from spacefuzz.config import RULE1_MIN_SLEEP_H, RULE1_WINDOW_H, RULE2_GAP_H, RULE3_MAX_WORK_H  # noqa: E402

EXPECTED = {
    "main_sleep_debt_eva": set(),
    "rule2_violation": {2},
    "backup_high_co2": {3},
    "radiation_spe": {4},
    "safe_day": set(),
}
TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def hours(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60


def load(folder: Path, name: str) -> dict:
    with open(folder / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def tasks_of(plan, *types):
    return [t for t in plan["tasks"] if not types or PlanType(t["type"]) in types]


def events_of(plan, etype):
    return [e for e in plan.get("events", []) if e["type"] == etype]


# ---- the four temporal rules (minimal versions)

def rule1(plan, checkin) -> bool:
    """Slept < 6 h: no EVA may start within 12 h of waking."""
    if checkin["sleep_last_night_h"] >= RULE1_MIN_SLEEP_H:
        return False
    wake = hours(checkin["wake_time"])
    return any(0 <= hours(t["start"]) - wake < RULE1_WINDOW_H for t in tasks_of(plan, PlanType.EVA))


def rule2(plan) -> bool:
    """No strength/treadmill training in the 2 h before an EVA starts."""
    for eva in tasks_of(plan, PlanType.EVA):
        s = hours(eva["start"])
        for ex in tasks_of(plan, *EXERCISE_TYPES):
            ex_start = hours(ex["start"])
            ex_end = ex_start + ex["duration_h"]
            if ex_start < s and ex_end > s - RULE2_GAP_H:
                return True
    return False


def in_event(plan, etype, start_h, end_h) -> bool:
    return any(hours(e["start"]) <= start_h and end_h <= hours(e["end"])
               for e in events_of(plan, etype))


def rule3(plan) -> bool:
    """After > 4 h continuous work under high CO2: no docking / precision manual work."""
    run_start = run_end = None
    for t in sorted(plan["tasks"], key=lambda t: hours(t["start"])):
        s = hours(t["start"])
        e = s + t["duration_h"]
        kind = PlanType(t["type"])
        if kind in PRECISION_TYPES and run_end is not None and run_end == s \
                and run_end - run_start > RULE3_MAX_WORK_H \
                and in_event(plan, PlanType.HIGH_CO2.value, run_start, s):
            return True
        if kind in RESTORATIVE_TYPES:
            run_start = run_end = None
        elif run_end is not None and run_end == s:
            run_end = e
        else:
            run_start, run_end = s, e
    return False


def rule4(plan) -> bool:
    """No EVA may start during a solar particle event warning."""
    return any(hours(e["start"]) <= hours(t["start"]) < hours(e["end"])
               for t in tasks_of(plan, PlanType.EVA)
               for e in events_of(plan, PlanType.SPE_WARNING.value))


def violations(name) -> set[int]:
    plan, checkin = load(PLANS, name), load(CHECKINS, name)
    flags = {1: rule1(plan, checkin), 2: rule2(plan), 3: rule3(plan), 4: rule4(plan)}
    return {k for k, v in flags.items() if v}


# ---- plan_types

def test_enum_has_generic_fallback_for_unknown_tasks():
    assert parse_type("EVA") is PlanType.EVA
    assert parse_type("underwater_basket_weaving") is PlanType.GENERIC_HIGH_LOAD
    with pytest.raises(ValueError):
        parse_type("unknown_event", kind="env")
    with pytest.raises(ValueError):
        parse_type("EVA", kind="env")  # a crew task is never an environment condition


def test_every_type_has_exactly_one_kind():
    assert kind_of(PlanType.EVA) == "task"
    assert kind_of(PlanType.HIGH_CO2) == "env" and kind_of(PlanType.SPE_WARNING) == "env"
    assert {kind_of(t) for t in PlanType} == {"task", "env"}


def test_enum_groups_are_consistent():
    assert TASK_TYPES.isdisjoint(ENV_TYPES)
    assert TASK_TYPES | ENV_TYPES == set(PlanType)
    assert EXERCISE_TYPES | PRECISION_TYPES | RESTORATIVE_TYPES <= TASK_TYPES
    assert PlanType.GENERIC_HIGH_LOAD in TASK_TYPES


# ---- files exist, parse, and follow the format

def test_exactly_the_five_plans_each_with_a_checkin():
    plans = {p.stem for p in PLANS.glob("*.json")}
    assert plans == set(EXPECTED)
    assert {p.stem for p in CHECKINS.glob("*.json")} == plans


@pytest.mark.parametrize("name", EXPECTED)
def test_plan_format(name):
    plan = load(PLANS, name)
    assert {"date", "environment", "check_window", "comm_delay_min", "tasks"} <= set(plan)
    assert set(plan) <= {"date", "environment", "check_window", "comm_delay_min", "tasks", "events"}
    for t in plan["tasks"]:
        assert set(t) == {"start", "duration_h", "type", "kind"}
        assert TIME.match(t["start"]) and t["duration_h"] > 0
        assert PlanType(t["type"]) in TASK_TYPES  # no fallback: samples use known types
        assert t["kind"] == "task"
    for e in plan.get("events", []):
        assert TIME.match(e["start"]) and TIME.match(e["end"]) and e["start"] < e["end"]
        assert PlanType(e["type"]) in ENV_TYPES and e["kind"] == "env"
        if e["type"] == PlanType.HIGH_CO2:
            assert set(e) == {"start", "end", "type", "kind", "co2_ppm"}
            assert isinstance(e["co2_ppm"], int) and 400 <= e["co2_ppm"] <= 20000


@pytest.mark.parametrize("name", EXPECTED)
def test_tasks_are_ordered_and_do_not_overlap(name):
    tasks = load(PLANS, name)["tasks"]
    starts = [hours(t["start"]) for t in tasks]
    assert starts == sorted(starts)
    for a, b in zip(tasks, tasks[1:]):
        assert hours(a["start"]) + a["duration_h"] <= hours(b["start"])


@pytest.mark.parametrize("name", EXPECTED)
def test_checkin_fields_and_ranges(name):
    c = load(CHECKINS, name)
    assert c["plan"] == name and c["check_window"] == "morning"
    assert TIME.match(c["wake_time"]) and TIME.match(c["check_time"])
    nights = c["sleep_last_3_nights_h"]
    assert len(nights) == 3 and nights[-1] == c["sleep_last_night_h"]
    assert 0 <= c["sleep_efficiency_pct"] <= 100
    assert 90 <= c["systolic_bp"] <= 160 and 50 <= c["diastolic_bp"] <= 100
    assert 60 <= c["blood_glucose"] <= 160
    assert 1 <= c["mood"] <= 5 and 1 <= c["stress"] <= 5
    assert 1 <= c["fatigue_score"] <= 7
    assert c["food_water_intake"]["meals"] in {"normal", "reduced", "skipped"}
    assert 0 < c["food_water_intake"]["water_l_24h"] < 6
    assert 150 <= c["reaction_time_ms"] <= 600


# ---- each plan violates exactly the intended rules

@pytest.mark.parametrize("name, expected", EXPECTED.items())
def test_rule_violations_match_intent(name, expected):
    assert violations(name) == expected


def test_main_plan_sits_on_the_edge_of_every_risk():
    plan, c = load(PLANS, "main_sleep_debt_eva"), load(CHECKINS, "main_sleep_debt_eva")
    eva, = tasks_of(plan, PlanType.EVA)
    ex, = tasks_of(plan, *EXERCISE_TYPES)
    assert hours(eva["start"]) - (hours(ex["start"]) + ex["duration_h"]) == RULE2_GAP_H
    assert eva["duration_h"] == 6.5
    assert c["sleep_last_night_h"] == 6.2
    assert sum(c["sleep_last_3_nights_h"]) / 3 == pytest.approx(5.0)
    # Small mutations must tip it over: 0.3 h less sleep -> rule 1, training +15 min -> rule 2.
    assert violations("main_sleep_debt_eva") == set()
    assert rule1(plan, {**c, "sleep_last_night_h": c["sleep_last_night_h"] - 0.3})
    shifted = {**plan, "tasks": [{**t, "start": "07:15"} if t is ex else t for t in plan["tasks"]]}
    assert rule2(shifted)


def test_main_plan_risk_is_between_20_and_40_percent():
    """Plan 3.3 target for the demo plan, with the engine's own probability track."""
    from spacefuzz import config
    from spacefuzz.fuzzing import engine as en
    from spacefuzz.fuzzing import mutator as mu
    from spacefuzz.core import state as st
    risk = en.probability_track(mu.Scenario.load("main_sleep_debt_eva"), st.make_baseline(),
                                config.PROBABILITY_SAMPLES,
                                fluctuation=en._wesad())["risk_probability"]
    assert 0.20 <= risk <= 0.40


def test_safe_day_has_margin_on_every_rule():
    plan, c = load(PLANS, "safe_day"), load(CHECKINS, "safe_day")
    assert c["sleep_last_night_h"] >= RULE1_MIN_SLEEP_H + 1
    for eva in tasks_of(plan, PlanType.EVA):
        for ex in tasks_of(plan, *EXERCISE_TYPES):
            gap = hours(eva["start"]) - (hours(ex["start"]) + ex["duration_h"])
            assert gap < 0 or gap >= RULE2_GAP_H + 1
    assert not plan.get("events")
    assert not tasks_of(plan, *PRECISION_TYPES)
