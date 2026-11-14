"""Tests for adjuster.py: candidates, selection, guided re-test, plan 3.10 output."""
import dataclasses
import json
import socket

import numpy as np
import pandas as pd
import pytest

from spacefuzz.fuzzing import adjuster as aj
from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import EXERCISE_TYPES, PlanType
from tests.local_data import HAS_ALL  # noqa: E402

BASE = st.make_baseline()
STEPS = config.STEPS_PER_DAY
RAW = HAS_ALL
FIELDS_3_10 = {"risk_rating", "escalation", "risk_probability", "time_to_crash_minutes",
               "primary_vulnerable_system", "causal_chain", "min_crash_perturbation",
               "advice_now", "advice_today", "measurement_prompts", "action_restrictions",
               "plan_adjustment", "concurrency_control", "hud_directives", "coverage_warnings",
               "data_quality"}


def flat_trace():
    idx = pd.Index(np.arange(STEPS) * config.STEP_MINUTES, name="minute")
    return pd.DataFrame({"co2": 3000.0, "temp": 22.5, "humidity": 42.0, "dose_rate": 7.79}, index=idx)


def scenario(name="main_sleep_debt_eva"):
    return mu.Scenario(pl.load_plan(f"data/plans/{name}.json"), flat_trace(), st.load_checkin(name))


def eva_minutes(plan):
    return sum(t.duration_min for t in plan.tasks if t.type is PlanType.EVA)


def valid(plan):
    ts = sorted(plan.tasks, key=lambda t: t.start_min)
    return all(t.duration_min > 0 for t in ts) and all(a.end_min <= b.start_min for a, b in zip(ts, ts[1:]))


# ---- candidates

def test_candidates_cover_every_kind_of_plan_3_9():
    cands = aj.candidates(scenario().plan)
    kinds = {k for c in cands for k in c.kinds}
    assert kinds >= {"delay", "shorten", "split", "rest", "nap", "move_training", "postpone"}
    assert all(valid(c.plan) for c in cands)
    assert any(set(c.kinds) == {"split", "nap"} for c in cands)        # combinations
    assert any(set(c.kinds) == {"split", "shorten"} for c in cands)
    assert all(len(c.kinds) <= config.ADJ_MAX_COMBINED for c in cands)


def test_candidate_edits_keep_or_change_eva_as_named():
    base = scenario().plan
    for c in aj.candidates(base):
        if "postpone" in c.kinds:
            assert eva_minutes(c.plan) == 0 and c.kinds == ("postpone",)
        elif "shorten" in c.kinds:
            assert eva_minutes(c.plan) < eva_minutes(base)
        else:
            assert eva_minutes(c.plan) == eva_minutes(base)      # moved, split, rested: same work
        assert c.changes and all(isinstance(t, str) and t.isascii() for t in c.changes)


def test_split_has_rest_between_two_eva_parts():
    c = next(c for c in aj.candidates(scenario().plan)
             if c.kinds == ("split",))
    evas = [t for t in c.plan.tasks if t.type is PlanType.EVA]
    assert len(evas) == 2
    rest = [t for t in c.plan.tasks if evas[0].end_min <= t.start_min < evas[1].start_min]
    assert rest and rest[0].type is PlanType.REST
    assert rest[0].duration_min == config.ADJ_SPLIT_REST_MIN


def test_training_moves_after_the_eva():
    c = next(c for c in aj.candidates(scenario().plan)
             if c.kinds == ("move_training",))
    eva = next(t for t in c.plan.tasks if t.type is PlanType.EVA)
    for t in c.plan.tasks:
        if t.type in EXERCISE_TYPES:
            assert t.start_min >= eva.end_min


def test_nap_counts_as_a_nap_and_postpone_costs_most():
    cands = aj.candidates(scenario().plan)
    for c in cands:
        if "nap" in c.kinds:
            nap = [t for t in c.plan.tasks if t.type is PlanType.SLEEP and t.duration_min <= 120]
            assert nap and all(t.duration_min / 60 <= config.RULE1_NAP_MAX_H for t in nap)
    post = next(c for c in cands if c.kinds == ("postpone",))
    assert post.cost == max(c.cost for c in cands)
    assert post.cost > 10 * max(c.cost for c in cands if c.kinds != ("postpone",))


def test_no_eva_plan_gets_only_rest_before_precision_work():
    cands = aj.candidates(scenario("backup_high_co2").plan)
    assert cands and {k for c in cands for k in c.kinds} == {"rest"}
    for c in cands:
        dock = next(t for t in c.plan.tasks if t.type is PlanType.DOCKING)
        before = [t for t in c.plan.tasks if t.end_min == dock.start_min]
        assert before and before[0].type is PlanType.REST


def test_held_eva_leaves_only_plans_without_eva():
    cands = aj.candidates(scenario().plan,
                          held={PlanType.EVA, PlanType.DOCKING})
    assert cands and all(eva_minutes(c.plan) == 0 for c in cands)


# ---- selection (stub risks: the logic of plan 3.9)

def cands_with(costs):
    p = scenario().plan
    return [aj.Candidate(p, (f"k{i}",), (f"change {i}",), cost) for i, cost in enumerate(costs)]


def test_select_smallest_change_that_reaches_yellow():
    cs = cands_with([1.0, 2.0, 3.0, 100.0])
    risk = {"change 0": 0.20, "change 1": 0.04, "change 2": 0.004, "change 3": 0.0}
    d = aj.select(cs, screen=lambda c: risk[c.changes[0]], confirm=lambda c: risk[c.changes[0]],
                  recheck=lambda c: (True, "ok"))
    assert d.chosen.changes == ("change 1",) and not d.escalate_to_ground


def test_select_skips_candidates_that_fail_confirmation_or_recheck():
    cs = cands_with([1.0, 2.0, 3.0])
    screen = {"change 0": 0.03, "change 1": 0.03, "change 2": 0.03}
    confirm = {"change 0": 0.07, "change 1": 0.03, "change 2": 0.03}
    recheck = {"change 1": (False, "training 15 min later crashes"), "change 2": (True, "ok")}
    d = aj.select(cs, screen=lambda c: screen[c.changes[0]], confirm=lambda c: confirm[c.changes[0]],
                  recheck=lambda c: recheck[c.changes[0]])
    assert d.chosen.changes == ("change 2",)
    assert [r["result"] for r in d.log] == ["confirmed red", "re-test failed", "chosen"]


def test_select_falls_back_to_lowest_risk_and_ground():
    cs = cands_with([1.0, 2.0])
    risk = {"change 0": 0.30, "change 1": 0.12}
    d = aj.select(cs, screen=lambda c: risk[c.changes[0]], confirm=lambda c: risk[c.changes[0]],
                  recheck=lambda c: (True, "ok"))
    assert d.chosen.changes == ("change 1",) and d.escalate_to_ground
    assert d.risk_after == pytest.approx(0.12)


def test_select_with_no_candidates_escalates():
    d = aj.select([], screen=None, confirm=None, recheck=None)
    assert d.chosen is None and d.escalate_to_ground


# ---- guided re-test: a normal schedule slip must not crash

def test_slip_margin_main_is_fragile_until_training_moves():
    s = scenario()
    assert aj.slip_margin(s, BASE)["minutes"] == 15               # training 15 min later
    moved = next(c for c in aj.candidates(s.plan) if c.kinds == ("move_training",))
    m = aj.slip_margin(dataclasses.replace(s, plan=moved.plan), BASE)
    assert m is None or m["minutes"] > config.ADJ_SLIP_MARGIN_MIN


def test_sleep_is_not_a_schedule_slip():
    s = scenario()
    m = aj.slip_margin(s, BASE)
    assert m["change"] != "less sleep"


# ---- measurement prompts

def test_measurement_sensitivity_main_sleep_first():
    sens = aj.measurement_sensitivity(scenario(), BASE, n=60)
    fields = [f for f, _ in sens]
    assert set(fields) == set(config.MEASUREMENT_ERROR)
    assert fields[0] == "sleep_last_night_h"                        # 6.2 h sits 0.2 h above 6 h
    assert all(d >= 0 for _, d in sens)


# ---- full run: plan 3.10 output

TINY = {"probability_samples": 60, "heatmap_samples": 4, "guided_rounds": 20,
        "adjust_samples": 30, "recheck_rounds": 10}


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    monkeypatch.setitem(config.FUZZ_PROFILES, "local", TINY)
    monkeypatch.setattr(config, "HEATMAP_SLEEP_CUT_H", (0.0,))
    monkeypatch.setattr(config, "HEATMAP_EVA_EXTEND_H", (0.0,))
    monkeypatch.setattr(config, "ADJ_DELAY_MIN", (60,))
    monkeypatch.setattr(config, "ADJ_REST_BEFORE_MIN", (60,))
    monkeypatch.setattr(config, "ADJ_NAP_BEFORE_MIN", (60,))
    monkeypatch.setattr(config, "ADJ_SHORTEN_H", (2.0,))
    monkeypatch.setattr(en, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(aj, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("FUZZ_PROFILE", raising=False)
    return tmp_path


def test_output_has_exactly_the_plan_3_10_fields(tiny):
    out = aj.adjust("data/plans/main_sleep_debt_eva.json", scenario=scenario())
    assert FIELDS_3_10 <= set(out)
    assert set(out) - FIELDS_3_10 <= {"plan", "details", "from_cache"}
    assert out["risk_rating"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN")
    assert out["escalation"] in ("self", "crew", "ground")
    assert out["primary_vulnerable_system"] in ("Cardiovascular", "CNS", "Respiratory", "Metabolic",
                                                "Radiation", "None")
    pa = out["plan_adjustment"]
    assert set(pa) >= {"risk_before", "risk_after", "changes", "rating_after", "escalate_to_ground",
                       "time_to_crash_after"}
    det = out["details"]                                           # stage 4: fan after adjustment
    assert set(det["traces_after"]) == set(config.ENGINE_TRACE_FIELDS) | {"crashed"}
    assert set(det["trajectory_after"]) == set(config.ENGINE_TRAJECTORY_FIELDS)
    assert len(det["trajectory_minutes_after"]) == config.STEPS_PER_DAY
    assert isinstance(out["min_crash_perturbation"], (str, type(None)))
    text = json.dumps(out)
    assert text.isascii()                                            # English only


def test_offline_run_is_identical(tiny, monkeypatch):
    a = aj.adjust("data/plans/main_sleep_debt_eva.json", scenario=scenario(), use_cache=False)

    def no_network(*args, **kwargs):
        raise OSError("network disabled for this test")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    b = aj.adjust("data/plans/main_sleep_debt_eva.json", scenario=scenario(), use_cache=False)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("details", "from_cache")}
    assert strip(a) == strip(b)


def test_untrusted_gives_unknown_and_no_adjustment(tiny):
    s = scenario()
    bad = dataclasses.replace(s.checkin, systolic_bp=900.0, diastolic_bp=500.0, blood_glucose=5.0)
    out = aj.adjust("data/plans/main_sleep_debt_eva.json", scenario=dataclasses.replace(s, checkin=bad))
    assert out["risk_rating"] == "UNKNOWN" and out["data_quality"] == "untrusted"
    assert out["plan_adjustment"] is None
    assert any("measure again" in t.lower() for t in out["advice_now"])


def test_cli_prints_before_after_and_changes(tiny, capsys):
    aj.main(["--plan", "data/plans/main_sleep_debt_eva.json", "--no-cache", "--synthetic"])
    text = capsys.readouterr().out
    assert "before" in text and "after" in text and "changes" in text.lower()


# ---- acceptance on the real scenarios

@pytest.fixture
def scratch_cache(tmp_path, monkeypatch):
    """Local-profile acceptance runs write their cache to a temporary folder, not cache/."""
    monkeypatch.setattr(en, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(aj, "CACHE_DIR", tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not RAW, reason="raw data missing")
def test_main_goes_from_red_to_yellow_or_green(scratch_cache):
    out = aj.adjust("data/plans/main_sleep_debt_eva.json")
    assert out["risk_rating"] in ("HIGH", "CRITICAL")
    pa = out["plan_adjustment"]
    assert pa["rating_after"] in ("LOW", "MEDIUM") and not pa["escalate_to_ground"]
    assert pa["risk_after"] < pa["risk_before"] and pa["changes"]
    # smallest change first: a same-day plan reaches yellow, so the EVA is not postponed;
    # postponing is offered as the safer option
    assert not any("Postpone" in t for t in pa["changes"])
    assert pa["safer_option"]["changes"] == ["Postpone EVA to another day"]
    assert pa["safer_option"]["risk"] < pa["risk_after"]


@pytest.mark.slow
@pytest.mark.skipif(not RAW, reason="raw data missing")
def test_backup_high_co2_stays_red_and_goes_to_ground(scratch_cache):
    out = aj.adjust("data/plans/backup_high_co2.json")
    pa = out["plan_adjustment"]
    assert pa["rating_after"] in ("HIGH", "CRITICAL") and pa["escalate_to_ground"]
    assert out["escalation"] == "ground"


# ---- same-day alternative and the scope of the risk after postponing

def test_postponed_output_carries_alternative_and_scope(tiny, monkeypatch):
    def force_postpone(cands, screen, confirm, recheck):
        post = next(c for c in cands if c.kinds == ("postpone",))
        return aj.Decision(post, confirm(post), False, [{"result": "chosen"}], "ok")
    monkeypatch.setattr(aj, "select", force_postpone)
    out = aj.adjust("data/plans/main_sleep_debt_eva.json", scenario=scenario(), use_cache=False)
    pa = out["plan_adjustment"]
    assert "today" in pa["scope"].lower() and "tomorrow" in pa["scope"].lower()
    alt = pa["same_day_alternative"]
    assert alt["changes"] and not any("Postpone" in t for t in alt["changes"])
    assert alt["rating"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert alt["escalate_to_ground"] == (alt["risk"] > config.RISK_RED_ABOVE)
    assert "must happen today" in alt["note"]


def test_no_alternative_when_not_postponed_or_eva_on_hold():
    assert aj.needs_same_day_alternative(("split", "shorten"), held=set()) is False
    assert aj.needs_same_day_alternative(("postpone",), held=set()) is True
    assert aj.needs_same_day_alternative(("postpone",), held={PlanType.EVA}) is False


# ---- three-change candidates in the main selection (beam of ADJ_SAME_DAY_TOP)

def test_beam_adds_three_change_candidates_from_the_best_pairs():
    s = scenario()
    risk = lambda c: 0.2 - 0.05 * len(c.kinds) + 0.001 * c.cost
    pool = aj.candidates_with_beam(s.plan, risk)
    three = [c for c in pool if len(c.kinds) == 3]
    assert three and all("postpone" not in c.kinds for c in three)
    pairs = sorted((c for c in pool if len(c.kinds) == 2), key=lambda c: (risk(c), c.cost))
    parents = {tuple(sorted(c.kinds)) for c in pairs[:config.ADJ_SAME_DAY_TOP]}
    for c in three:      # every triple extends one of the best pairs
        assert any(set(p) < set(c.kinds) for p in parents)
    assert any(c.kinds == ("postpone",) for c in pool)


def test_same_day_choice_offers_postponing_as_the_safer_option(tiny, monkeypatch):
    def pick_split(cands, screen, confirm, recheck):
        c = next(c for c in cands if c.kinds == ("split",))
        return aj.Decision(c, confirm(c), False, [{"result": "chosen"}], "ok")
    monkeypatch.setattr(aj, "select", pick_split)
    out = aj.adjust("data/plans/main_sleep_debt_eva.json", scenario=scenario(), use_cache=False)
    pa = out["plan_adjustment"]
    assert "same_day_alternative" not in pa and "scope" not in pa
    safer = pa["safer_option"]
    assert safer["changes"] == ["Postpone EVA to another day"]
    assert "tomorrow" in safer["scope"].lower()
    assert safer["rating"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ---- EVA after an overlapping environment event (general rule)

def test_eva_moves_after_an_overlapping_env_event():
    s = scenario("radiation_spe")                 # SPE warning 09:00-15:00, EVA 10:00-15:00
    cands = aj.candidates(s.plan)
    after = [c for c in cands if c.kinds == ("after_event",)]
    assert len(after) == 1
    eva = next(t for t in after[0].plan.tasks if t.type is PlanType.EVA)
    assert eva.start_min == 15 * 60 and eva.duration_min == 5 * 60
    assert valid(after[0].plan) and "after" in after[0].changes[0] and "spe_warning" in after[0].changes[0]
    assert after[0].cost == pytest.approx(config.ADJ_COST["delay_per_h"] * 5)   # moved 5 h
    assert not any({"after_event", "delay"} <= set(c.kinds) for c in cands)    # both move the EVA
    assert any(set(c.kinds) == {"after_event", "shorten"} for c in cands)


def test_no_after_event_candidate_without_overlap():
    for name in ("main_sleep_debt_eva", "safe_day"):
        s = scenario(name)
        assert not any("after_event" in c.kinds for c in aj.candidates(s.plan))


def test_after_event_is_generic_for_any_env_event():
    raw = pl.to_dict(scenario("safe_day").plan)
    raw["events"] = [{"start": "11:00", "end": "13:30", "type": "spe_warning", "kind": "env"}]
    p = pl.load_plan(raw)                         # EVA 12:00-16:00 overlaps
    c = next(c for c in aj.candidates(p) if c.kinds == ("after_event",))
    assert next(t for t in c.plan.tasks if t.type is PlanType.EVA).start_min == 13 * 60 + 30
