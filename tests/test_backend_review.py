"""Backend fixes from the code review of 2026-11-14 (one test per finding; the cache was
re-run once after all of them)."""
import dataclasses
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.core import evaluator as ev
from spacefuzz.core import plan as pl
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state as st
from spacefuzz.core import model as md
from spacefuzz.core.plan_types import PlanType
from spacefuzz.fuzzing import adjuster, engine as en, midday, mutator as mu
from spacefuzz.readers import env_trace, generic_reader as gr, physio_replay as pr

MAIN = "main_sleep_debt_eva"
BASE = st.make_baseline()


def plan_dict(tasks, events=()):
    return {"date": "Mission Day 1", "environment": "mars_base", "check_window": "morning",
            "comm_delay_min": 20,
            "tasks": [{"kind": "task", **t} for t in tasks],
            "events": [{"kind": "env", **e} for e in events]}


# ---------------------------------------------------------------- core/plan.py

@pytest.mark.parametrize("task", [
    {"start": "10:05", "duration_h": 2, "type": "EVA"},       # off the 15-minute grid
    {"start": "10:00", "duration_h": 0.1, "type": "EVA"},     # 6 min: not whole steps
    {"start": "10:00", "duration_h": 0.004, "type": "EVA"},   # rounds to 0 minutes
    {"start": "10:00", "duration_h": True, "type": "EVA"},    # a bool is not a duration
])
def test_plan_times_must_sit_on_the_15_minute_grid(task):
    pl.load_plan(plan_dict([{"start": "10:00", "duration_h": 2, "type": "EVA"}]))   # control
    with pytest.raises(pl.PlanError, match="15-minute|duration must be"):
        pl.load_plan(plan_dict([task]))


def test_event_times_must_sit_on_the_grid():
    with pytest.raises(pl.PlanError, match="15-minute"):
        pl.load_plan(plan_dict([{"start": "10:00", "duration_h": 1, "type": "EVA"}],
                               [{"start": "09:10", "end": "11:00", "type": "spe_warning"}]))


# ---------------------------------------------------------------- core/evaluator.py

def test_glucose_fall_counts_the_last_eva_step():
    """Each row holds the state before its step, so the last EVA step's drop shows on the
    first row after the EVA; it belongs to the EVA's fall."""
    plan = pl.load_plan(plan_dict([{"start": "10:15", "duration_h": 0.75, "type": "EVA"}]))
    o = pd.DataFrame({"minute": [600, 615, 630, 645, 660],
                      "activity": ["idle", "eva", "eva", "eva", "idle"],
                      "blood_glucose": [100.0, 100.0, 95.0, 90.0, 79.0]})
    hit, _, _ = ev._metabolic(o, BASE, plan)
    assert hit[4]                        # 5 + 5 + 11 = 21 >= 20 mg/dL
    assert not hit[:4].any()


# ---------------------------------------------------------------- core/model.py

def test_spe_warning_uses_the_clock_like_high_co2():
    """Events are clock times of the plan day; one before the check-in applies to the coming
    night (the simulation wraps past midnight), for SPE warnings as for high CO2."""
    sc = mu.Scenario.load(MAIN)
    night = pl.load_plan({**pl.to_dict(sc.plan), "events": [
        {"kind": "env", "type": "spe_warning", "start": "01:00", "end": "02:00"}]})
    plain = md.simulate(sc.plan, sc.trace, sc.checkin, BASE)
    warned = md.simulate(night, sc.trace, sc.checkin, BASE)
    at = plain.index[plain["minute"] == 24 * 60 + 90][0]           # 01:30 the coming night
    assert warned.loc[at, "stress_level"] > plain.loc[at, "stress_level"]


# ---------------------------------------------------------------- core/state.py + quality gate

def test_fatigue_score_zero_does_not_start_negative():
    c = dataclasses.replace(mu.Scenario.load(MAIN).checkin, fatigue_score=0)
    assert st.initial_state(c, BASE).cognitive_fatigue >= 0.0


@pytest.mark.parametrize("field, value", [
    ("systolic_bp", "abc"), ("blood_glucose", "999"), ("sleep_last_3_nights_h", (20, 20, 20)),
    ("fatigue_score", 0), ("mood", 9), ("stress", 0),
])
def test_gate_rejects_bad_check_in_values_without_crashing(field, value):
    c = dataclasses.asdict(mu.Scenario.load(MAIN).checkin)
    c[field] = value
    g = qg.check(checkin=c)
    assert g.status == "untrusted" and field in g.missing
    assert field in g.message or qg.FIELD_LABELS.get(field, field) in g.message


def test_gate_accepts_numbers_written_as_text():
    c = dataclasses.asdict(mu.Scenario.load(MAIN).checkin)
    c["systolic_bp"] = "126"
    assert qg.check(checkin=c).status in ("ok", "degraded")


# ---------------------------------------------------------------- readers

def test_build_trace_refuses_steps_it_could_not_fill(monkeypatch):
    real = env_trace.clip

    def gappy(*a, **k):
        out = real(*a, **k)
        if "dose_rate" in out:
            out.iloc[10:20, out.columns.get_loc("dose_rate")] = np.nan
        return out
    monkeypatch.setattr(env_trace, "clip", gappy)
    with pytest.raises(ValueError, match="dose_rate"):
        env_trace.build_trace(radiation="mars")


def test_iso_times_with_a_zone_become_naive_utc():
    kind, t = gr.parse_time(pd.Series(["2017-09-05T00:00:00Z", "2017-09-05T08:00:00+08:00"]),
                            "iso")
    assert kind == "timestamp" and t.dt.tz is None
    assert list(t) == [pd.Timestamp("2017-09-05 00:00"), pd.Timestamp("2017-09-05 00:00")]


def test_fallback_fluctuation_spreads_around_each_segment_mean():
    """The real pool measures deviations from each segment's own mean; the fitted fallback
    must use the same spread (not the spread across segments)."""
    df = pd.DataFrame({"subject": ["S1"] * 4, "wesad_label": [4, 4, 4, 4],
                       "segment": [1, 1, 2, 2], "hr_bpm": [60.0, 62.0, 80.0, 82.0],
                       "rmssd_ms": [50.0, 50.0, 40.0, 40.0]})
    p = pr.fit_params(df)["subjects"]["S1"]["meditation"]
    assert p["hr_dev_sd"] == pytest.approx(np.std([-1, 1, -1, 1], ddof=1), abs=0.01)
    assert p["hr_std"] > 5                                     # across segments: much wider


def test_fallback_pool_is_drawn_with_the_within_segment_spread(tmp_path):
    cond = {"hr_mean": 70.0, "hr_std": 50.0, "rmssd_mean": 60.0, "rmssd_std": 40.0,
            "hr_dev_sd": 2.0, "rmssd_dev_cv": 0.1, "minutes": 3000}
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"subjects": {"S1": {"baseline": cond, "stress": cond}}}))
    f = md.Fluctuation.from_params(path)
    assert f.source == "wesad_params"
    assert np.std(f.hr_dev["relax"]) == pytest.approx(2.0, abs=0.2)      # not 50
    assert np.std(f.rmssd_rel["stress"]) == pytest.approx(0.1, abs=0.02)  # not 40 / 60


# ---------------------------------------------------------------- fuzzing/mutator.py

def _mutator(**kw):
    return mu.Mutator(mu.Scenario.load(MAIN), BASE, **kw)


def test_operators_never_touch_what_has_already_happened():
    """Midday: nothing before the check may change (EVA already flown, no task inserted in the
    past, no particle event starting in the past)."""
    m = _mutator(not_before=11 * 60)                 # the 10:00 EVA is under way
    assert "A" not in m.applicable(m.base.plan)
    assert all(s >= 11 * 60 for s in m._free_slots(m.base.plan, 15))
    for _ in range(40):
        plan, p = m._op_d(m.base.plan, {"trace": m.base.trace, "effects": {"fatigue_rate": ()}})
        assert p["warning"] is None or p["warning"][0] >= 11 * 60


def test_a_nap_is_not_bedtime():
    sc = mu.Scenario.load(MAIN)
    nap = pl.load_plan({**pl.to_dict(sc.plan), "tasks": [
        *pl.to_dict(sc.plan)["tasks"],
        {"kind": "task", "start": "17:15", "duration_h": 0.5, "type": "sleep"}]})
    m = mu.Mutator(dataclasses.replace(sc, plan=nap), BASE)
    assert m._awake_end(nap) == 23 * 60


def test_a_shorter_eva_may_lower_its_heart_rate_peak():
    """Chain 2 is checked only for an EVA made longer or harder; a shortened EVA lowering the
    peak is the expected effect, not a broken chain."""
    ref = pd.DataFrame({"activity": ["eva", "eva"], "hr_model": [150.0, 150.0],
                        "core_temp": [37.5, 37.5], "co2": [800.0, 800.0], "spo2": [97.0, 97.0]})
    out = ref.assign(hr_model=[140.0, 140.0], core_temp=[37.3, 37.3])
    shorter = [mu.OpRecord("A", {"duration_delta_min": -60, "eva_heat_rate": 0.3})]
    longer = [mu.OpRecord("A", {"duration_delta_min": 60, "eva_heat_rate": 0.3})]
    assert not mu.check_chains(out, ref, shorter)
    assert mu.check_chains(out, ref, longer)


def test_operator_c_skips_a_day_without_a_free_slot():
    m = _mutator()
    full = dataclasses.replace(m.base.plan, tasks=[pl.Task(m.start_min, 24 * 60, PlanType.SLEEP,
                                                           "sleep")])
    plan, params = m._op_c(full, {"effects": {"fatigue_jumps": ()}})
    assert plan is full and params["task"] is None


def test_a_particle_warning_past_midnight_is_kept_as_clock_events():
    assert mu.clock_events(20 * 60, 22 * 60) == [(20 * 60, 22 * 60)]
    assert mu.clock_events(23 * 60, 25 * 60) == [(23 * 60, 24 * 60), (0, 60)]
    assert mu.clock_events(25 * 60, 26 * 60) == [(60, 120)]


# ---------------------------------------------------------------- fuzzing/adjuster.py

def test_a_plan_that_failed_its_retest_is_never_reported_as_fixed():
    a = adjuster.Candidate(None, ("x",), ("A",), 1.0)
    b = adjuster.Candidate(None, ("y",), ("B",), 2.0)
    risk = {a: 0.01, b: 0.30}
    d = adjuster.select([a, b], risk.get, risk.get, lambda c: (c is not a, "slip"))
    assert d.chosen is b and d.escalate_to_ground
    d = adjuster.select([a], risk.get, risk.get, lambda c: (False, "slip"))
    assert d.escalate_to_ground                  # only a failed plan left: still the ground


def test_minimum_eva_length_is_a_config_value(monkeypatch):
    plan = mu.Scenario.load(MAIN).plan                        # one 6.5 h EVA
    kinds = lambda: {k for c in adjuster.candidates(plan) for k in c.kinds}
    assert "shorten" in kinds()
    monkeypatch.setattr(config, "ADJ_MIN_EVA_MIN", 6 * 60)   # 6.5 h - 1 h < 6 h: cannot shorten
    assert "shorten" not in kinds()


# ---------------------------------------------------------------- fuzzing/midday.py

def test_midday_reads_the_morning_plan_from_the_cache_only(monkeypatch):
    monkeypatch.setattr(midday, "cached_morning", lambda name, profile=None: None)
    monkeypatch.setattr(adjuster, "adjust", lambda *a, **k: pytest.fail("must not compute"))
    with pytest.raises(RuntimeError, match="morning"):
        midday._day_plan(MAIN, "full")


def _split_day():
    """The morning-adjusted main plan (nap, EVA split 11:30 / 14:45), built by hand so the
    test does not need the cache."""
    d = pl.to_dict(mu.Scenario.load(MAIN).plan)
    tasks = [t for t in d["tasks"] if t["type"] != "EVA" and t["start"] != "16:45"] + [
        {"kind": "task", "start": "17:00", "duration_h": 0.5, "type": "meal"},
        {"kind": "task", "start": "10:00", "duration_h": 1.5, "type": "sleep"},
        {"kind": "task", "start": "11:30", "duration_h": 2.25, "type": "EVA"},
        {"kind": "task", "start": "13:45", "duration_h": 1.0, "type": "rest"},
        {"kind": "task", "start": "14:45", "duration_h": 2.25, "type": "EVA"}]
    return pl.load_plan({**d, "tasks": tasks})


def test_midday_rule1_window_start_sits_on_the_grid():
    sc = dataclasses.replace(mu.Scenario.load(MAIN), plan=_split_day())
    c = dataclasses.replace(midday.remeasured_checkin(sc.checkin, {"sleep_last_night_h": 5.9}),
                            wake_time_min=6 * 60 + 10)                # window ends 18:10
    after = [x for x in midday.afternoon_candidates(sc.plan, c, 14 * 60)
             if "after_window" in x.kinds]
    assert after and "18:15" in after[0].changes[0]                   # rounded up to the grid
    assert all(t.start_min % 15 == 0 for t in after[0].plan.tasks)


def test_midday_track_mutates_only_the_afternoon(monkeypatch):
    seen = {}
    real = mu.Mutator

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(mu, "Mutator", spy)
    en.probability_track(mu.Scenario.load(MAIN), BASE, 2, count_from=14 * 60)
    assert seen.get("not_before") == 14 * 60


# ---------------------------------------------------------------- fuzzing/engine.py

def test_cache_key_covers_baseline_and_committed_data():
    sc, prof = mu.Scenario.load(MAIN), en.profile("local")
    other = dataclasses.replace(BASE, resting_hr=BASE.resting_hr + 5)
    assert en._cache_key(sc, prof, BASE) != en._cache_key(sc, prof, other)
    for f in ("data/processed/op_b_co2_excursions.csv", "data/processed/spe_profile_msl_rad.csv",
              "data/processed/wesad_params.json", "spacefuzz/readers/physio_replay.py",
              "spacefuzz/core/plan_types.py"):
        assert f in en.SOURCES, f
    assert not any("wesad_hr_hrv" in f for f in en.SOURCES)    # local only: clone key must match


def test_guided_root_fitness_uses_the_same_overlay_as_its_children(monkeypatch):
    """The root seed is simulated like its children (with the fluctuation overlay), through
    Mutator.simulate_base, before the first round."""
    calls = []
    real = mu.Mutator.simulate_base

    def spy(self):
        calls.append(self.fluctuation)
        return real(self)
    monkeypatch.setattr(mu.Mutator, "simulate_base", spy)
    fl = md.Fluctuation(hr_dev={"relax": np.array([3.0]), "stress": np.array([3.0])},
                        rmssd_rel={"relax": np.array([0.0]), "stress": np.array([0.0])})
    en.guided_track(mu.Scenario.load(MAIN), BASE, rounds=2, fluctuation=fl)
    assert calls and calls[0] is fl


# ---------------------------------------------------------------- docs in code

def test_config_comments_match_the_code():
    src = inspect.getsource(config)
    assert "will replace" not in src
    assert "per day" in src.split("RAD_EVA_DOSE_LIMIT_UGY")[1].split("\n")[0]
    assert "eva_start_min" not in inspect.signature(
        __import__("spacefuzz.fuzzing.advice", fromlist=["x"]).prevent_today).parameters
