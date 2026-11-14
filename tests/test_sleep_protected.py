"""Sleep is protected in plan adjustments (user review 2026-11-14).

An adjustment that pushes tasks later must not create the sleep debt it is meant to fix:
it first shortens or cancels personal time; when the pushed tasks still reach a planned
sleep, the candidate is not valid. This holds for the morning adjustment and for the
midday check. (The fuzzer's own perturbations in mutator.py still squeeze sleep on
purpose: an EVA overrun eating into the night is one of the what-ifs it tests.)
"""
import dataclasses

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as s

from spacefuzz import config
from spacefuzz.core import plan as pl
from spacefuzz.core.plan_types import PlanType
from spacefuzz.fuzzing import adjuster as adj
from spacefuzz.fuzzing import midday as mdd
from spacefuzz.fuzzing import mutator as mu

SCENARIOS = ("main_sleep_debt_eva", "radiation_spe", "backup_high_co2")
NAP_MAX = int(config.RULE1_NAP_MAX_H * 60)


def T(start, minutes, typ):
    return pl.Task(start, minutes, typ, typ.value)


def sleeps(tasks):
    """Planned sleep (night sleep, and any nap already in the plan): (start, duration)."""
    return sorted((t.start_min, t.duration_min) for t in tasks if t.type is PlanType.SLEEP)


def kept(original, tasks):
    """Every sleep of the original plan is still there, same start and same length."""
    return set(sleeps(original)) <= set(sleeps(tasks))


# ---------------------------------------------------------------- resolve

def test_a_push_shortens_personal_time_first_and_keeps_sleep():
    eva = T(18 * 60, 135, PlanType.EVA)
    tasks = [eva, T(20 * 60, 120, PlanType.SCIENCE_EXPERIMENT),
             T(22 * 60, 90, PlanType.PERSONAL_TIME), T(23 * 60 + 30, 360, PlanType.SLEEP)]
    out = adj.resolve(tasks, eva)
    assert sleeps(out) == [(23 * 60 + 30, 360)]                     # sleep untouched
    pt = [t for t in out if t.type is PlanType.PERSONAL_TIME]
    assert pt and pt[0].start_min == 22 * 60 + 15 and pt[0].end_min == 23 * 60 + 30


def test_personal_time_the_moved_task_lands_in_is_cut_at_it():
    eva = T(20 * 60, 60, PlanType.EVA)                        # moved into 19:00-21:00
    tasks = [T(19 * 60, 120, PlanType.PERSONAL_TIME), eva, T(21 * 60, 420, PlanType.SLEEP)]
    out = adj.resolve(tasks, eva)
    assert [(t.start_min, t.duration_min, t.type) for t in sorted(out, key=lambda t: t.start_min)] \
        == [(19 * 60, 60, PlanType.PERSONAL_TIME), (20 * 60, 60, PlanType.EVA),
            (21 * 60, 420, PlanType.SLEEP)]


def test_personal_time_is_cancelled_when_squeezed_below_one_step():
    eva = T(18 * 60, 300, PlanType.EVA)                       # ends 23:00: no room left
    tasks = [eva, T(22 * 60, 60, PlanType.PERSONAL_TIME), T(23 * 60, 360, PlanType.SLEEP)]
    out = adj.resolve(tasks, eva)
    assert not [t for t in out if t.type is PlanType.PERSONAL_TIME]
    assert sleeps(out) == [(23 * 60, 360)]


def test_a_push_that_reaches_sleep_is_not_a_valid_plan():
    eva = T(20 * 60, 240, PlanType.EVA)
    tasks = [eva, T(23 * 60, 360, PlanType.SLEEP)]
    with pytest.raises(adj.SleepConflict):
        adj.resolve(tasks, eva)


def test_a_conflicting_combination_is_dropped_not_built():
    plan = pl.load_plan({"date": "D", "environment": "mars_base", "check_window": "morning",
                         "comm_delay_min": 20, "tasks": [
                             {"start": "17:00", "duration_h": 5, "type": "EVA", "kind": "task"},
                             {"start": "22:30", "duration_h": 7.5, "type": "sleep",
                              "kind": "task"}]})
    cands = adj.candidates(plan)
    assert cands                                                    # postpone etc. remain
    for c in cands:
        assert kept(plan.tasks, c.plan.tasks), c.changes
    for c in cands:                                                 # delays only while they fit
        evas = [t for t in c.plan.tasks if t.type is PlanType.EVA]
        assert all(e.end_min <= 22 * 60 + 30 for e in evas), c.changes
    delays = {c.changes[0] for c in cands if c.kinds == ("delay",)}
    assert delays == {"Move the EVA 30 min later"}                  # 1 h and 2 h hit the night


def test_a_shortened_personal_time_is_named_in_the_changes():
    plan = pl.load_plan({"date": "D", "environment": "mars_base", "check_window": "morning",
                         "comm_delay_min": 20, "tasks": [
                             {"start": "17:00", "duration_h": 2, "type": "EVA", "kind": "task"},
                             {"start": "19:00", "duration_h": 2, "type": "personal_time",
                              "kind": "task"},
                             {"start": "23:00", "duration_h": 7, "type": "sleep", "kind": "task"}]})
    c = next(c for c in adj.candidates(plan) if c.kinds == ("delay",)
             and any(t.type is PlanType.PERSONAL_TIME and t.duration_min < 120
                     for t in c.plan.tasks))
    assert any("personal time" in x.lower() for x in c.changes), c.changes


# ---------------------------------------------------------------- every real candidate

@pytest.mark.parametrize("name", SCENARIOS)
def test_no_morning_candidate_shortens_or_moves_sleep(name):
    plan = mu.Scenario.load(name).plan
    cands = adj.candidates(plan)
    assert cands
    for c in cands:
        assert kept(plan.tasks, c.plan.tasks), (name, c.changes, sleeps(c.plan.tasks))


@pytest.mark.parametrize("name", SCENARIOS)
def test_no_three_change_candidate_shortens_sleep(name):
    plan = mu.Scenario.load(name).plan
    pairs = [c for c in adj.candidates(plan) if len(c.kinds) == 2][:5]
    for c, _ in adj._beam(plan, [(c, 0.0) for c in pairs], screen=lambda c: 0.0, top=5):
        assert kept(plan.tasks, c.plan.tasks), (name, c.changes)


def test_no_midday_candidate_shortens_or_moves_sleep():
    name = "main_sleep_debt_eva"
    try:
        plan = mdd._day_plan(name, "full")
    except mdd.MorningMissing:
        pytest.skip("morning result not cached")
    sc = mu.Scenario.load(name)
    checkin = mdd.remeasured_checkin(sc.checkin, {"sleep_last_night_h": 5.9})
    cands = mdd.afternoon_candidates(plan, checkin, 14 * 60)
    assert cands
    for c in cands:
        assert kept(plan.tasks, c.plan.tasks), (c.changes, sleeps(c.plan.tasks))


# ---------------------------------------------------------------- property

@settings(max_examples=150, deadline=None, derandomize=True,
          suppress_health_check=[HealthCheck.too_slow])
@given(eva_start=s.integers(8, 18).map(lambda h: h * 60),
       eva_h=s.sampled_from([60, 120, 180, 270, 390]),
       pt=s.sampled_from([0, 30, 60, 120]),
       gap=s.sampled_from([0, 15, 60, 120]),
       sleep_start=s.sampled_from([21 * 60, 22 * 60, 23 * 60]),
       push=s.sampled_from([15, 60, 120, 240]))
def test_resolve_never_shortens_or_moves_a_sleep(eva_start, eva_h, pt, gap, sleep_start, push):
    eva_end = eva_start + eva_h
    pt_start = sleep_start - pt
    if eva_end + gap > pt_start:
        return                                              # not a valid starting plan
    tasks = [T(eva_start, eva_h, PlanType.EVA), T(sleep_start, 420, PlanType.SLEEP)]
    if pt:
        tasks.append(T(pt_start, pt, PlanType.PERSONAL_TIME))
    moved = dataclasses.replace(tasks[0], start_min=eva_start + push)
    try:
        out = adj.resolve([moved] + tasks[1:], moved)
    except adj.SleepConflict:
        assert eva_end + push > sleep_start                  # only when it truly cannot fit
        return
    assert sleeps(out) == [(sleep_start, 420)]
    assert all(a.end_min <= b.start_min for a, b in
               zip(sorted(out, key=lambda t: t.start_min), sorted(out, key=lambda t: t.start_min)[1:]))
