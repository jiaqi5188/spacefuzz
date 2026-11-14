"""Tests for plan.py: loading, validation, CO2 floor and the one-line scenario."""
import copy
import logging
from pathlib import Path

import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.core import plan as pl
from spacefuzz.core.plan_types import PlanType
from tests.local_data import HAS_ALL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "data" / "plans"
needs_raw = pytest.mark.skipif(not HAS_ALL, reason="raw data missing")

BASE = {
    "date": "Mission Day 1", "environment": "mars_base", "check_window": "morning",
    "comm_delay_min": 20,
    "tasks": [
        {"start": "07:00", "duration_h": 1.0, "type": "exercise_ARED", "kind": "task"},
        {"start": "10:00", "duration_h": 6.5, "type": "EVA", "kind": "task"},
        {"start": "23:00", "duration_h": 6.0, "type": "sleep", "kind": "task"},
    ],
}


def variant(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    return d


def with_task(i, **changes):
    d = copy.deepcopy(BASE)
    d["tasks"][i].update(changes)
    return d


# ---- loading

@pytest.mark.parametrize("path", sorted(PLANS.glob("*.json")), ids=lambda p: p.stem)
def test_all_sample_plans_load_without_warnings(path):
    p = pl.load_plan(path)
    assert p.warnings == []
    assert all(isinstance(t.type, PlanType) for t in p.tasks)


def test_times_become_minutes_and_tasks_keep_order():
    p = pl.load_plan(BASE)
    assert [(t.start_min, t.end_min) for t in p.tasks] == [(420, 480), (600, 990), (1380, 1740)]
    assert p.tasks[1].type is PlanType.EVA and p.environment == "mars_base"


def test_unknown_task_type_maps_to_generic_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="plan"):
        p = pl.load_plan(with_task(1, type="lunar_golf"))
    assert p.tasks[1].type is PlanType.GENERIC_HIGH_LOAD
    assert p.tasks[1].source_type == "lunar_golf"
    assert len(p.warnings) == 1 and "lunar_golf" in p.warnings[0]
    assert "lunar_golf" in caplog.text


# ---- hard errors

@pytest.mark.parametrize("data, match", [
    (with_task(1, start="07:30"), "overlap"),
    (with_task(1, duration_h=-1.0), "duration"),
    (with_task(1, duration_h=0), "duration"),
    (with_task(1, duration_h=25), "24 hours"),
    (with_task(2, duration_h=10.0), "24 hours"),           # 07:00 -> 09:00 next day = 26 h
    (with_task(0, start="25:00"), "HH:MM"),
    (with_task(0, kind="env"), "kind"),
    ({k: v for k, v in BASE.items() if k != "tasks"}, "tasks"),
])
def test_invalid_plans_raise(data, match):
    with pytest.raises(pl.PlanError, match=match):
        pl.load_plan(data)


def test_missing_kind_is_an_error():
    d = copy.deepcopy(BASE)
    del d["tasks"][0]["kind"]
    with pytest.raises(pl.PlanError, match="kind"):
        pl.load_plan(d)


# ---- environment events

def test_events_are_env_and_validated():
    ev = {"start": "09:00", "end": "15:00", "type": "spe_warning", "kind": "env"}
    p = pl.load_plan(variant(events=[ev]))
    assert p.events[0].type is PlanType.SPE_WARNING and p.events[0].kind == "env"
    assert (p.events[0].start_min, p.events[0].end_min) == (540, 900)
    for bad, match in (({**ev, "kind": "task"}, "kind"), ({**ev, "type": "EVA"}, "env"),
                       ({**ev, "type": "meteor"}, "meteor"), ({**ev, "end": "08:00"}, "end")):
        with pytest.raises(pl.PlanError, match=match):
            pl.load_plan(variant(events=[bad]))


def test_high_co2_needs_ppm_not_percent():
    ev = {"start": "07:00", "end": "19:00", "type": "high_co2", "kind": "env"}
    assert pl.load_plan(variant(events=[{**ev, "co2_ppm": 6000}])).events[0].co2_ppm == 6000
    with pytest.raises(pl.PlanError, match="co2_ppm"):
        pl.load_plan(variant(events=[{**ev, "co2_pct": 0.6}]))
    with pytest.raises(pl.PlanError, match="co2_ppm"):
        pl.load_plan(variant(events=[ev]))


def test_co2_floor_takes_the_larger_value_inside_the_event_only():
    trace = pd.DataFrame({"co2": [3000.0, 7000.0, 3000.0, 3000.0], "temp": 22.0,
                          "humidity": 40.0, "dose_rate": 20.0},
                         index=pd.Index([0, 15, 30, 45], name="minute"))
    ev = {"start": "00:00", "end": "00:30", "type": "high_co2", "kind": "env", "co2_ppm": 5000}
    out = pl.apply_env_events(trace, pl.load_plan(variant(events=[ev])))
    assert out["co2"].tolist() == [5000.0, 7000.0, 3000.0, 3000.0]
    assert trace["co2"].iloc[0] == 3000.0                        # input untouched


def test_docstring_states_the_co2_rule():
    assert "larger" in pl.__doc__ and "co2_ppm" in pl.__doc__


# ---- acceptance: one line gives a day's environment + a plan

@needs_raw
def test_one_line_scenario():
    trace, p = pl.load_scenario("data/plans/backup_high_co2.json")
    assert len(trace) == config.STEPS_PER_DAY and not trace.isna().any().any()
    assert p.environment == "mars_base"
    in_event = trace.loc[7 * 60:19 * 60 - 15, "co2"]
    assert (in_event >= 6000).all()                               # plan event acts as a floor
    assert trace.attrs["radiation_source"] == "radlab_msl_rad"    # mars_base -> Mars surface


# ---- moving items (used by adjuster.py later)

def test_env_events_cannot_be_moved():
    ev = {"start": "09:00", "end": "15:00", "type": "spe_warning", "kind": "env"}
    p = pl.load_plan(variant(events=[ev]))
    with pytest.raises(pl.PlanError, match="env"):
        pl.move_item(p, p.events[0], 120)


def test_moving_a_task_revalidates_and_leaves_the_original():
    p = pl.load_plan(BASE)
    moved = pl.move_item(p, p.tasks[1], 60)               # EVA 10:00 -> 11:00
    assert moved.tasks[1].start_min == 660 and p.tasks[1].start_min == 600
    with pytest.raises(pl.PlanError, match="overlap"):
        pl.move_item(p, p.tasks[1], -150)                 # onto the 07:00 training


# ---- malformed input is a PlanError, never a crash (found by tests/test_fuzz_app.py)

@pytest.mark.parametrize("data", [
    variant(tasks=[0]),                                       # a task that is not an object
    variant(tasks=[{"start": "07:00", "duration_h": 1.0, "type": "meal", "kind": "task"}, "x"]),
    variant(events=-1.0),                                     # events not a list
    variant(events=[{"start": "08:00", "end": "09:00", "type": "spe_warning", "kind": "env"}, 3]),
    variant(comm_delay_min=None),
    variant(comm_delay_min=""),
    variant(comm_delay_min=float("inf")),
    variant(comm_delay_min=float("nan")),
    with_task(0, duration_h=float("nan")),
    variant(events=[{"start": "08:00", "end": "09:00", "type": "high_co2", "kind": "env",
                     "co2_ppm": float("nan")}]),
    variant(events=[{"start": "08:00", "end": "09:00", "type": "high_co2", "kind": "env",
                     "co2_ppm": float("inf")}]),
])
def test_malformed_input_is_a_plan_error(data):
    with pytest.raises(pl.PlanError) as exc:
        pl.load_plan(data)
    assert str(exc.value)
