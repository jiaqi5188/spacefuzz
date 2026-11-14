"""Boundary and property tests added by the test-focused review of 2026-11-14.

Each boundary test pins one comparison (< vs <=, n vs n + 1 steps) that a mutation check
showed no other fast test would notice; the property tests (hypothesis) state invariants the
demo relies on: bad data never forecasts, the midday check never rewrites the morning, a plan
that failed its re-test is never reported as fixed, the cache key is the same everywhere.
"""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as s

from spacefuzz import config
from spacefuzz.core import evaluator as ev
from spacefuzz.core import plan as pl
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType
from spacefuzz.fuzzing import adjuster, engine as en, midday, mutator as mu

MAIN = "main_sleep_debt_eva"
BASE = st.make_baseline()
STEP = config.STEP_MINUTES
FAST = settings(max_examples=60, deadline=None,
                suppress_health_check=[HealthCheck.function_scoped_fixture])


def task(start, minutes, typ):
    return pl.Task(start, minutes, typ, typ.value)


def feed(machine, tasks, start, end, co2=800.0):
    """Feed a rule machine one observation per step, as evaluator.run_temporal does."""
    for m in range(start, end, STEP):
        t = next((x for x in tasks if x.start_min <= m < x.end_min), None)
        machine.observe(ev.Obs(minute=m, task=t, starts=t is not None and t.start_min == m,
                               co2=co2))
    return machine


# ---------------------------------------------------------------- rule 1: 12 h, nap

WAKE = 6 * 60
WINDOW = int(config.RULE1_WINDOW_H * 60)


@pytest.mark.parametrize("eva_start, violated", [
    (WAKE + WINDOW - STEP, True),      # 11 h 45 min after waking: inside the window
    (WAKE + WINDOW, False),            # exactly 12 h: allowed ("within 12 h" excludes it)
    (WAKE + WINDOW + STEP, False),
])
def test_rule1_window_boundary_is_exactly_12_hours(eva_start, violated):
    eva = task(eva_start, 60, PlanType.EVA)
    m = feed(ev.Rule1SleepEva(5.0, WAKE), [eva], WAKE, eva_start + 60)
    assert (m.violated_at is not None) is violated


@pytest.mark.parametrize("nap_min, resets", [
    (int(config.RULE1_NAP_MAX_H * 60), False),          # exactly the nap limit: still a nap
    (int(config.RULE1_NAP_MAX_H * 60) + STEP, True),    # one step longer: a real sleep
])
def test_rule1_nap_boundary(nap_min, resets):
    """A nap does not replace the night's sleep: the short night still blocks the EVA. A
    longer sleep resets waking time and sleep length (too short itself -> still a violation,
    but counted from the new wake time)."""
    nap = task(WAKE + 60, nap_min, PlanType.SLEEP)
    eva = task(nap.end_min + 60, 60, PlanType.EVA)
    m = feed(ev.Rule1SleepEva(5.0, WAKE), [nap, eva], WAKE, eva.end_min)
    assert m.violated_at == eva.start_min              # 5 h night, or the short new sleep
    assert (m.wake == nap.end_min) is resets
    assert math.isclose(m.sleep_h, nap_min / 60 if resets else 5.0)


# ---------------------------------------------------------------- rule 3: CO2 threshold

@pytest.mark.parametrize("co2, counts", [
    (config.RULE3_CO2_PPM, True),          # at the line counts ("CO2 >= line")
    (config.RULE3_CO2_PPM - 1, False),
])
def test_rule3_co2_at_the_line_counts(co2, counts):
    limit = int(config.RULE3_MAX_WORK_H * 60)
    work = task(8 * 60, limit + STEP, PlanType.MAINTENANCE)       # one step over 4 h
    dock = task(work.end_min, 60, PlanType.DOCKING)
    m = feed(ev.Rule3Co2Precision(), [work, dock], 8 * 60, dock.end_min, co2=co2)
    assert (m.violated_at is not None) is counts


def test_rule3_exactly_four_hours_is_allowed():
    limit = int(config.RULE3_MAX_WORK_H * 60)
    work = task(8 * 60, limit, PlanType.MAINTENANCE)              # exactly 4 h: not "more than"
    dock = task(work.end_min, 60, PlanType.DOCKING)
    m = feed(ev.Rule3Co2Precision(), [work, dock], 8 * 60, dock.end_min, co2=6000.0)
    assert m.violated_at is None


# ---------------------------------------------------------------- rule 4: clock times

def test_rule4_sees_an_eva_pushed_past_midnight_into_a_clock_warning():
    """Events are clock times of the plan day (model.simulate applies them modulo 24 h); an
    EVA that a mutation pushed past midnight (minute 1500 = 01:00) inside a 00:30-02:00
    warning breaks rule 4, as in the model."""
    eva = task(1500, 60, PlanType.EVA)
    m = feed(ev.Rule4SpeEva([(30, 120)]), [eva], 1440, 1560)
    assert m.violated_at == 1500
    day = feed(ev.Rule4SpeEva([(30, 120)]), [task(600, 60, PlanType.EVA)], 540, 660)
    assert day.violated_at is None                      # 10:00 is not inside 00:30-02:00


# ---------------------------------------------------------------- quality gate

def frame(n=240):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"minute": np.arange(n),
                         "heart_rate": 70 + rng.normal(0, 2, n).round(3),
                         "hrv_rmssd": 60 + rng.normal(0, 6, n).round(3),
                         "spo2": 97 + rng.normal(0, 0.5, n).round(3),
                         "core_temp": 36.8 + rng.normal(0, 0.05, n).round(4)})


@pytest.mark.parametrize("minutes, stuck", [
    (config.QG_STUCK_MINUTES["heart_rate"], True),         # exactly the limit: stuck
    (config.QG_STUCK_MINUTES["heart_rate"] - 1, False),
])
def test_stuck_boundary_is_at_the_limit(minutes, stuck):
    df = frame()
    df.loc[30:30 + minutes - 1, "heart_rate"] = 71.5      # a value the noise cannot produce
    kinds = {i.kind for i in qg.check(df).issues if i.field == "heart_rate"}
    assert ("stuck" in kinds) is stuck


def good_checkin() -> dict:
    return dataclasses.asdict(mu.Scenario.load(MAIN).checkin)


def test_text_in_an_optional_field_is_rejected_not_just_missing():
    """Reaction time may be left out (degraded), but a value that is not a number is a bad
    entry: untrusted, like any other invalid entry."""
    c = good_checkin()
    assert qg.check(checkin={**c, "reaction_time_ms": None}).status == "degraded"
    g = qg.check(checkin={**c, "reaction_time_ms": "fast"})
    assert g.status == "untrusted" and not g.can_forecast


JUNK = s.one_of(s.none(), s.text(max_size=6), s.booleans(),
                s.floats(allow_nan=True, allow_infinity=True),
                s.integers(-10**6, 10**6), s.lists(s.floats(allow_nan=True), max_size=4))


@FAST
@given(field=s.sampled_from(config.QG_CHECKIN_FIELDS), value=JUNK)
def test_gate_never_crashes_and_never_passes_a_bad_value(field, value):
    """Whatever is typed into one check-in field: no exception, and the result is "ok" only if
    that value really is a plausible number (or plausible numbers, for the 3 nights)."""
    g = qg.check(checkin={**good_checkin(), field: value})
    assert g.status in {"ok", "degraded", "untrusted"}
    lo, hi = config.QG_PLAUSIBLE_RANGES[field]
    items = value if isinstance(value, list) else [value]
    plausible = bool(items) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)
        and lo <= v <= hi for v in items)
    if isinstance(value, str):
        try:
            plausible = lo <= float(value) <= hi
        except ValueError:
            plausible = False
    if not plausible:
        assert g.status != "ok" and field in g.missing
        if field not in config.QG_OPTIONAL_CHECKIN_FIELDS or value is not None:
            assert not g.can_forecast or (value is None or (isinstance(value, float)
                                                             and math.isnan(value)))


# ---------------------------------------------------------------- plan grid (property)

@FAST
@given(start=s.integers(0, 23 * 60), minutes=s.integers(1, 6 * 60))
def test_a_plan_loads_exactly_when_its_times_sit_on_the_grid(start, minutes):
    d = {"date": "Mission Day 1", "environment": "mars_base", "check_window": "morning",
         "comm_delay_min": 20,
         "tasks": [{"kind": "task", "start": f"{start // 60:02d}:{start % 60:02d}",
                    "duration_h": minutes / 60, "type": "EVA"}]}
    on_grid = start % STEP == 0 and minutes % STEP == 0
    if on_grid:
        p = pl.load_plan(d)
        assert p.tasks[0].start_min == start and p.tasks[0].duration_min == minutes
    else:
        with pytest.raises(pl.PlanError):
            pl.load_plan(d)


# ---------------------------------------------------------------- mutator (properties)

@FAST
@given(a=s.integers(0, 2 * 1440 - STEP).map(lambda x: x - x % STEP),
       length=s.integers(1, 96).map(lambda k: k * STEP))
def test_clock_events_cover_the_same_minutes_once(a, length):
    b = a + length
    parts = mu.clock_events(a, b)
    assert all(0 <= x < y <= 1440 for x, y in parts)
    minutes = [m for x, y in parts for m in range(x, y, STEP)]
    assert len(minutes) == len(set(minutes)) == length // STEP
    assert sorted(minutes) == sorted({m % 1440 for m in range(a, b, STEP)})


def split_day() -> pl.Plan:
    """The morning-adjusted main plan (nap, EVA split 11:30 / 14:45)."""
    d = pl.to_dict(mu.Scenario.load(MAIN).plan)
    tasks = [t for t in d["tasks"] if t["type"] != "EVA" and t["start"] != "16:45"] + [
        {"kind": "task", "start": "17:00", "duration_h": 0.5, "type": "meal"},
        {"kind": "task", "start": "10:00", "duration_h": 1.5, "type": "sleep"},
        {"kind": "task", "start": "11:30", "duration_h": 2.25, "type": "EVA"},
        {"kind": "task", "start": "13:45", "duration_h": 1.0, "type": "rest"},
        {"kind": "task", "start": "14:45", "duration_h": 2.25, "type": "EVA"}]
    return pl.load_plan({**d, "tasks": tasks})


CUT = 14 * 60


@pytest.mark.parametrize("seed", range(6))
def test_no_operator_rewrites_the_morning_at_the_midday_check(seed):
    """With every operator switched on, 40 samples per seed: nothing before 14:00 changes
    (a task under way at 14:00 keeps its start; only its remaining part may be cut short by
    an inserted task) and no new event starts before 14:00."""
    sc = dataclasses.replace(mu.Scenario.load(MAIN), plan=split_day())
    m = mu.Mutator(sc, BASE, seed=seed, not_before=CUT,
                   params={"MUT_DAILY_PROB": {op: 1.0 for op in "ABCD"}})
    past = lambda p: [(t.start_min, min(t.end_min, CUT), t.type) for t in p.tasks
                      if t.start_min < CUT]
    morning = past(sc.plan)
    seen = set()
    for _ in range(40):
        smp = m.draw()
        seen |= {o.op for o in smp.ops}
        assert past(smp.plan) == morning, [o.params for o in smp.ops]
        new_events = [e for e in smp.plan.events if e not in sc.plan.events]
        for o in smp.ops:
            if o.op == "D" and o.params["warning"]:
                assert o.params["warning"][0] >= CUT
        assert all(e.type is PlanType.SPE_WARNING for e in new_events)
    assert seen == set("ABCD")                           # every operator really ran


def test_operator_b_leaves_work_before_the_check_alone():
    sc = mu.Scenario.load(MAIN)                          # science_experiment at 18:00
    late = mu.Mutator(sc, BASE, not_before=19 * 60)
    ctx = lambda: {"trace": sc.trace, "effects": {}}
    assert all(late._op_b(sc.plan, ctx())[1]["work_task"] is None for _ in range(10))
    early = mu.Mutator(sc, BASE)
    assert early._op_b(sc.plan, ctx())[1]["work_task"] == "science_experiment"


def test_spo2_check_under_more_co2_is_skipped_only_for_a_shortened_eva():
    ref = pd.DataFrame({"activity": ["eva", "eva"], "hr_model": [150.0, 150.0],
                        "core_temp": [37.5, 37.5], "co2": [800.0, 800.0], "spo2": [96.0, 96.0]})
    out = ref.assign(spo2=[97.5, 97.5], co2=[900.0, 900.0])
    b = mu.OpRecord("B", {"co2_excursion": 0.1})
    longer = mu.OpRecord("A", {"duration_delta_min": 60})
    shorter = mu.OpRecord("A", {"duration_delta_min": -60})
    assert any("SpO2" in v for v in mu.check_chains(out, ref, [b]))
    assert any("SpO2" in v for v in mu.check_chains(out, ref, [longer, b]))    # still checked
    assert not any("SpO2" in v for v in mu.check_chains(out, ref, [shorter, b]))


# ---------------------------------------------------------------- adjuster.select (property)

@FAST
@given(data=s.lists(s.tuples(s.floats(0, 1), s.floats(0, 1), s.booleans(), s.floats(0.1, 9)),
                    min_size=1, max_size=7))
def test_select_never_reports_a_red_or_failed_plan_as_fixed(data):
    """Random candidates (screen risk, confirm risk, passes the re-test, cost): whenever the
    decision does not go to the ground, the chosen plan is confirmed not red and passed its
    re-test; if a plan is fine on all three counts, the cheapest such plan is chosen."""
    cands = [adjuster.Candidate(None, (f"k{i}",), (f"c{i}",), cost)
             for i, (_, _, _, cost) in enumerate(data)]
    info = dict(zip(cands, data))
    d = adjuster.select(cands, lambda c: info[c][0], lambda c: info[c][1],
                        lambda c: (info[c][2], "why"))
    red = config.RISK_RED_ABOVE
    good = [c for c in cands if info[c][0] <= red and info[c][1] <= red and info[c][2]]
    assert d.chosen in cands
    if good:
        assert not d.escalate_to_ground and d.chosen in good
        assert d.chosen.cost == min(c.cost for c in good)
    if not d.escalate_to_ground:
        assert info[d.chosen][1] <= red and info[d.chosen][2]
    assert d.risk_after == info[d.chosen][1]


# ---------------------------------------------------------------- cache key

def test_cache_key_is_the_same_whatever_the_terminal_looks_like():
    """The key must not depend on how pandas prints a table (terminal width, display options):
    a fresh clone on another machine has to find the committed cache."""
    sc, prof = mu.Scenario.load(MAIN), en.profile("local")
    key = en._cache_key(sc, prof)
    with pd.option_context("display.max_columns", 3, "display.width", 40,
                           "display.max_rows", 4, "display.precision", 2):
        assert en._cache_key(sc, prof) == key
    assert en._cache_key(sc, prof, st.make_baseline()) == key          # default baseline


def test_cache_key_follows_the_baseline_history():
    sc, prof = mu.Scenario.load(MAIN), en.profile("local")
    hist = BASE.history.copy()
    hist.iloc[0, 0] += 1.0
    other = dataclasses.replace(BASE, history=hist)
    assert en._cache_key(sc, prof, other) != en._cache_key(sc, prof, BASE)


# ---------------------------------------------------------------- midday without a morning

def test_midday_without_a_morning_result_is_a_named_error(monkeypatch):
    monkeypatch.setattr(midday, "cached_morning", lambda name, profile=None: None)
    with pytest.raises(midday.MorningMissing):
        midday.scenario_for(MAIN, {"sleep_last_night_h": 5.9}, "full")
    assert issubclass(midday.MorningMissing, RuntimeError)


def test_prune_lists_names_even_without_a_morning_result(monkeypatch):
    """The clean-up script must not crash (or compute) when the morning result is missing:
    the midday file then simply has no current name."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "prune_cache.py"
    spec = importlib.util.spec_from_file_location("prune_cache_review", path)
    prune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prune)
    monkeypatch.setattr(midday, "cached_morning", lambda name, profile=None: None)
    names = prune.current_names("full")
    assert names and not any("midday" in n for n in names)
    assert sum("_adjust_" in n for n in names) >= len(prune.DEMO_SCENARIOS)
