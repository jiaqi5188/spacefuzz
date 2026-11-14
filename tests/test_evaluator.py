"""Tests for evaluator.py: composite rules, temporal state machines, fitness."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.core import evaluator as ev
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from tests.local_data import HAS_ALL  # noqa: E402

BASE = st.make_baseline()
STEPS = config.STEPS_PER_DAY
RAW = HAS_ALL
PLANS = ["main_sleep_debt_eva", "rule2_violation", "backup_high_co2", "radiation_spe", "safe_day"]
TEMPORAL = {"rule1_sleep_eva", "rule2_exercise_eva", "rule3_co2_precision", "rule4_spe_eva"}
COMPOSITE = {"cognitive", "cardiovascular", "metabolic", "radiation_dose"}
EXPECTED = {
    "main_sleep_debt_eva": set(),
    "rule2_violation": {"rule2_exercise_eva"},
    "backup_high_co2": {"rule3_co2_precision"},
    "radiation_spe": {"rule4_spe_eva"},
    "safe_day": set(),
}


def flat_trace(co2=3000.0, dose=7.79):
    idx = pd.Index(np.arange(STEPS) * config.STEP_MINUTES, name="minute")
    return pd.DataFrame({"co2": co2, "temp": 22.5, "humidity": 42.0, "dose_rate": dose}, index=idx)


def scenario(name):
    if RAW:
        return mu.Scenario.load(name)
    return mu.Scenario(pl.load_plan(f"data/plans/{name}.json"), flat_trace(), st.load_checkin(name))


def run(name, fluctuation=None, checkin=None, plan=None):
    s = scenario(name)
    p = plan or s.plan
    c = checkin or s.checkin
    out = md.simulate(p, pl.apply_env_events(s.trace, p), c, BASE, fluctuation=fluctuation,
                      rng=np.random.default_rng(config.RANDOM_SEED) if fluctuation else None)
    return ev.evaluate(out, p, c, BASE), out, p, c


def make_plan(tasks, events=None):
    data = {"date": "Mission Day 1", "environment": "mars_base", "check_window": "morning",
            "comm_delay_min": 20,
            "tasks": [{"start": s, "duration_h": d, "type": t, "kind": "task"} for s, d, t in tasks]}
    if events:
        data["events"] = events
    return pl.load_plan(data)


def sim(plan, c=None, trace=None):
    c = c or st.load_checkin("safe_day")
    return md.simulate(plan, pl.apply_env_events(trace if trace is not None else flat_trace(), plan),
                       c, BASE), c


def names(result):
    return {v.rule for v in result.violations}


# ---- config

def test_rule_thresholds_in_config():
    assert config.RULE1_MIN_SLEEP_H == 6.0 and config.RULE1_WINDOW_H == 12.0
    assert config.RULE2_GAP_H == 2.0 and config.RULE3_MAX_WORK_H == 4.0
    assert config.RULE3_CO2_PPM == config.ENV_CO2_HIGH_PPM
    assert config.COG_FATIGUE_ABOVE == 80 and config.COG_SPO2_BELOW_PCT == 94


# ---- the five sample plans

@pytest.mark.parametrize("name", PLANS)
def test_sample_plans_trigger_exactly_their_rule(name):
    result, *_ = run(name)
    assert names(result) == EXPECTED[name]


@pytest.mark.skipif(not HAS_ALL, reason="WESAD export not built")
@pytest.mark.parametrize("name", PLANS)
def test_sample_plans_with_real_fluctuation(name):
    result, *_ = run(name, fluctuation=md.Fluctuation.from_wesad())
    assert names(result) == EXPECTED[name]


def test_main_is_much_closer_to_crash_than_safe_day():
    main, *_ = run("main_sleep_debt_eva")
    safe, *_ = run("safe_day")
    assert not main.crashed and not safe.crashed
    assert main.fitness < safe.fitness - 0.5
    assert main.steps_to_violation["rule1_sleep_eva"] == 1      # 15 min less sleep
    assert main.steps_to_violation["rule2_exercise_eva"] == 1   # training 15 min later
    assert safe.steps_to_violation["rule1_sleep_eva"] == 8      # 7.8 h: 2 h of margin
    assert safe.steps_to_violation["rule2_exercise_eva"] == 7   # gap 3.5 h
    assert main.steps_to_violation["rule3_co2_precision"] is None   # no precision task
    assert main.steps_to_violation["rule4_spe_eva"] is None         # no SPE warning


def test_violated_rule_has_zero_distance_and_a_time():
    result, *_ = run("rule2_violation")
    (v,) = result.violations
    assert v.rule == "rule2_exercise_eva" and v.system == "Cardiovascular"
    assert v.minute == 10 * 60                                  # at EVA start
    assert result.distance["rule2_exercise_eva"] == 0.0
    assert result.steps_to_violation["rule2_exercise_eva"] == 0
    assert result.first_crash_minute == 10 * 60


def test_fitness_is_the_sum_of_rule_distances():
    result, *_ = run("safe_day")
    assert set(result.distance) == TEMPORAL | COMPOSITE
    assert all(0.0 <= d <= 1.0 for d in result.distance.values())
    assert result.fitness == pytest.approx(sum(result.distance.values()))


def test_backup_rule3_and_radiation_rule4_details():
    backup, *_ = run("backup_high_co2")
    (v,) = backup.violations
    assert v.minute == 13 * 60 and v.system == "CNS"           # docking after 5 h high-CO2 work
    rad, *_ = run("radiation_spe")
    (v,) = rad.violations
    assert v.minute == 10 * 60 and v.system == "Radiation"


# ---- temporal state machines

def test_rule1_short_sleep_then_eva_within_12_h():
    plan = make_plan([("10:00", 4, "EVA"), ("23:00", 7, "sleep")])
    out, c = sim(plan, dataclasses.replace(st.load_checkin("safe_day"), sleep_last_night_h=5.9))
    r = ev.evaluate(out, plan, c, BASE)
    assert "rule1_sleep_eva" in names(r)
    late = make_plan([("18:15", 3, "EVA"), ("23:00", 7, "sleep")])    # 12 h 15 min after waking
    out, c = sim(late, c)
    r = ev.evaluate(out, late, c, BASE)
    assert "rule1_sleep_eva" not in names(r)
    assert r.steps_to_violation["rule1_sleep_eva"] == 2         # move EVA 30 min earlier


def test_rule1_machine_steps_directly():
    m = ev.Rule1SleepEva(last_sleep_h=6.0, wake_min=360)
    assert m.state == "rested"
    m.observe(ev.Obs(minute=600, task=pl.Task(600, 240, pl.PlanType.EVA, "EVA"), starts=True))
    assert m.violated_at is None and m.distance == 1           # 6.0 h is not "< 6 h"
    m = ev.Rule1SleepEva(last_sleep_h=5.0, wake_min=360)
    assert m.state == "short_sleep_window"
    m.observe(ev.Obs(minute=600, task=pl.Task(600, 240, pl.PlanType.EVA, "EVA"), starts=True))
    assert m.state == "violated" and m.violated_at == 600 and m.distance == 0


def test_rule2_counts_steps_since_training():
    for gap_h, expected in ((2.0, 1), (2.5, 3), (1.75, 0)):
        start = 10 - gap_h - 1
        hh, mm = int(start), int(round((start % 1) * 60))
        plan = make_plan([(f"{hh:02d}:{mm:02d}", 1, "exercise_T2"), ("10:00", 4, "EVA")])
        out, c = sim(plan)
        r = ev.evaluate(out, plan, c, BASE)
        assert r.steps_to_violation["rule2_exercise_eva"] == expected, gap_h
        assert ("rule2_exercise_eva" in names(r)) == (expected == 0)


def test_rule3_needs_more_than_4_h_of_high_co2_work():
    ev_high = [{"start": "07:00", "end": "19:00", "type": "high_co2", "kind": "env", "co2_ppm": 6000}]
    for work_h, violated, steps in ((4.0, False, 1), (4.25, True, 0), (2.0, False, 9)):
        end = 8 + work_h
        plan = make_plan([("08:00", work_h, "maintenance"),
                          (f"{int(end):02d}:{int(round(end % 1 * 60)):02d}", 1, "docking")], ev_high)
        out, c = sim(plan)
        r = ev.evaluate(out, plan, c, BASE)
        assert ("rule3_co2_precision" in names(r)) == violated, work_h
        assert r.steps_to_violation["rule3_co2_precision"] == steps, work_h


def test_rule3_threshold_reads_co2_and_rest_breaks_the_run():
    plan = make_plan([("08:00", 5, "maintenance"), ("13:00", 1, "docking")])
    out, c = sim(plan, trace=flat_trace(co2=3000))              # normal cabin air
    assert "rule3_co2_precision" not in names(ev.evaluate(out, plan, c, BASE))
    out, c = sim(plan, trace=flat_trace(co2=config.RULE3_CO2_PPM + 1))
    assert "rule3_co2_precision" in names(ev.evaluate(out, plan, c, BASE))
    rested = make_plan([("08:00", 3, "maintenance"), ("11:00", 0.5, "rest"),
                        ("11:30", 1.5, "science_experiment"), ("13:00", 1, "docking")])
    out, c = sim(rested, trace=flat_trace(co2=6000))
    assert "rule3_co2_precision" not in names(ev.evaluate(out, rested, c, BASE))


def test_rule4_reads_plan_events():
    warn = [{"start": "12:00", "end": "15:00", "type": "spe_warning", "kind": "env"}]
    plan = make_plan([("10:00", 4, "EVA")], warn)
    out, c = sim(plan)
    r = ev.evaluate(out, plan, c, BASE)
    assert "rule4_spe_eva" not in names(r)
    assert r.steps_to_violation["rule4_spe_eva"] == 8          # EVA start 2 h later
    plan = make_plan([("12:30", 2, "EVA")], warn)
    out, c = sim(plan)
    assert "rule4_spe_eva" in names(ev.evaluate(out, plan, c, BASE))
    after = make_plan([("15:00", 2, "EVA")], warn)              # warning ends 15:00 (exclusive)
    out, c = sim(after)
    r = ev.evaluate(out, after, c, BASE)
    assert "rule4_spe_eva" not in names(r) and r.steps_to_violation["rule4_spe_eva"] == 1


# ---- composite rules (outputs overwritten to isolate each rule)

def base_out():
    plan = make_plan([("07:00", 0.5, "meal"), ("10:00", 4, "EVA"), ("23:00", 7, "sleep")])
    out, c = sim(plan)
    return out, plan, c


def test_cognitive_needs_both_conditions_at_the_same_step():
    out, plan, c = base_out()
    o = out.copy()
    o.loc[20:22, "cognitive_fatigue"] = 85.0
    assert "cognitive" not in names(ev.evaluate(o, plan, c, BASE))
    o.loc[22:24, "spo2"] = 93.0
    r = ev.evaluate(o, plan, c, BASE)
    (v,) = [x for x in r.violations if x.rule == "cognitive"]
    assert v.minute == o.loc[22, "minute"] and v.system == "CNS"


def test_cardiovascular_sustained_hr_and_low_hrv_on_overlaid_values():
    out, plan, c = base_out()
    thr = config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(BASE)
    o = out.copy()
    o.loc[20, "heart_rate"] = thr + 5
    o.loc[20, "hrv_rmssd"] = 10
    assert "cardiovascular" not in names(ev.evaluate(o, plan, c, BASE))   # one step only
    o.loc[20:21, "heart_rate"] = thr + 5
    o.loc[20:21, "hrv_rmssd"] = 10
    assert "cardiovascular" in names(ev.evaluate(o, plan, c, BASE))
    m = out.copy()                                    # model column high, readings normal
    m.loc[20:23, "hr_model"] = thr + 20
    assert "cardiovascular" not in names(ev.evaluate(m, plan, c, BASE))


def test_blood_pressure_only_judged_at_settled_rest():
    out, plan, c = base_out()
    o = out.copy()
    eva = o.index[o["activity"] == "eva"]
    o.loc[eva, "systolic_bp"] = 160.0                 # exertion: normal rise
    assert "cardiovascular" not in names(ev.evaluate(o, plan, c, BASE))
    settled = o.index[(o["activity"] == "idle") & (o["minute"] > 15 * 60)]
    o.loc[settled, "systolic_bp"] = 150.0
    r = ev.evaluate(o, plan, c, BASE)
    assert "cardiovascular" in names(r)


def test_metabolic_low_glucose_before_eva_or_unbroken_fall():
    out, plan, c = base_out()
    o = out.copy()
    start = o.index[o["activity"] == "eva"][0]
    o.loc[start, "blood_glucose"] = 65.0
    (v,) = [x for x in ev.evaluate(o, plan, c, BASE).violations if x.rule == "metabolic"]
    assert v.minute == 600 and v.system == "Metabolic"
    f = out.copy()
    eva = f.index[f["activity"] == "eva"]
    f.loc[eva, "blood_glucose"] = 100 - 2.0 * np.arange(len(eva))    # 30 mg/dL unbroken fall
    assert "metabolic" in names(ev.evaluate(f, plan, c, BASE))
    g = f.copy()
    g.loc[eva[5], "blood_glucose"] += 15                              # a meal-like bump breaks it
    g.loc[eva[6:], "blood_glucose"] = g.loc[eva[5], "blood_glucose"] - 1.0 * np.arange(len(eva) - 6)
    assert "metabolic" not in names(ev.evaluate(g, plan, c, BASE))


def test_radiation_dose_rule():
    out, plan, c = base_out()
    o = out.copy()
    o["eva_dose_ugy"] = np.linspace(0, config.RAD_EVA_DOSE_LIMIT_UGY + 10, len(o))
    r = ev.evaluate(o, plan, c, BASE)
    (v,) = [x for x in r.violations if x.rule == "radiation_dose"]
    assert v.system == "Radiation"
    assert r.distance["radiation_dose"] == 0.0


def test_rules_reported_separately():
    """backup-like case: rule 3 and the cognitive rule together give two violations."""
    result, out, plan, c = run("backup_high_co2")
    o = out.copy()
    o.loc[40:42, "cognitive_fatigue"] = 90.0
    o.loc[40:42, "spo2"] = 93.0
    r = ev.evaluate(o, plan, c, BASE)
    assert names(r) == {"rule3_co2_precision", "cognitive"}
    assert {v.system for v in r.violations} == {"CNS"}
    assert r.systems() == ["CNS"]


def test_cardio_threshold_follows_age_not_resting_hr():
    out, plan, c = base_out()
    o = out.copy()
    o.loc[20:21, "heart_rate"] = 120.0                 # ordinary hard exercise
    o.loc[20:21, "hrv_rmssd"] = 10.0
    fit = dataclasses.replace(BASE, resting_hr=54.0)   # low resting HR, same age
    assert "cardiovascular" not in names(ev.evaluate(o, plan, c, fit))
    assert config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(dataclasses.replace(BASE, age_y=45)) \
        == pytest.approx(140.0)
    old = dataclasses.replace(BASE, age_y=70)          # max 150 -> threshold 120
    assert "cardiovascular" in names(ev.evaluate(o, plan, c, old))


# ---- with the mutator

def test_operator_e_on_main_trips_rule1():
    s = scenario("main_sleep_debt_eva")
    m = mu.Mutator(s, BASE, params={"MUT_OPS_PER_SAMPLE": (1,), "MUT_OPS_PER_SAMPLE_WEIGHTS": (1.0,),
                                    "MUT_OPERATOR_WEIGHTS": {"E": 1.0}})
    for r in m.generate(5):
        res = ev.evaluate(r.output, r.sample.plan, r.sample.checkin, BASE)
        assert "rule1_sleep_eva" in names(res)


def test_rule1_a_nap_does_not_count_as_the_night():
    plan = make_plan([("10:00", 1.5, "sleep"), ("11:30", 4, "EVA"), ("23:00", 7, "sleep")])
    out, c = sim(plan, dataclasses.replace(st.load_checkin("safe_day"), sleep_last_night_h=7.5))
    r = ev.evaluate(out, plan, c, BASE)
    assert "rule1_sleep_eva" not in names(r)
    long_sleep = make_plan([("08:00", 3, "sleep"), ("13:00", 4, "EVA"), ("23:00", 7, "sleep")])
    out, c = sim(long_sleep, c)
    assert "rule1_sleep_eva" in names(ev.evaluate(out, long_sleep, c, BASE))   # a 3 h main sleep


def test_mutator_with_no_applicable_operator_says_so():
    p = make_plan([("07:00", 0.5, "meal"), ("23:00", 7, "sleep")])
    s = mu.Scenario(p, flat_trace(), st.load_checkin("safe_day"))
    m = mu.Mutator(s, BASE, params={"MUT_OPERATOR_WEIGHTS": {"A": 1.0}})
    with pytest.raises(ValueError, match="no applicable operator"):
        m.draw()



# ---- operator A heat storage: target behaviours (they set the illustrative parameters)

RATES = np.linspace(*config.OP_A_EVA_HEAT_RATE_C_PER_H, 21)   # the A distribution, evenly


def heat_runs(name, plan=None):
    """(output, evaluation) for every heat rate, EVA timing as planned (no fluctuation)."""
    s = scenario(name)
    p = plan or s.plan
    trace = pl.apply_env_events(s.trace, p)
    runs = []
    for r in RATES:
        out = md.simulate(p, trace, s.checkin, BASE, effects=md.Effects(eva_heat_rate=float(r)))
        runs.append((out, ev.evaluate(out, p, s.checkin, BASE)))
    return runs, p


def eva_peak(out):
    return out.loc[out["activity"] == "eva", "hr_model"].max()


def cardio_share(runs):
    return np.mean(["cardiovascular" in names(e) for _, e in runs])


def split_main():
    raw = pl.to_dict(scenario("main_sleep_debt_eva").plan)
    tasks = [t for t in raw["tasks"] if t["type"] != "EVA"]
    tasks += [{"start": "10:00", "duration_h": 3.0, "type": "EVA", "kind": "task"},
              {"start": "13:00", "duration_h": 1.0, "type": "rest", "kind": "task"},
              {"start": "14:00", "duration_h": 2.75, "type": "EVA", "kind": "task"}]
    return pl.load_plan({**raw, "tasks": tasks})


def test_rested_plan_with_a_stays_below_140():
    runs, _ = heat_runs("safe_day")
    peaks = np.array([eva_peak(o) for o, _ in runs])
    assert 120 <= np.median(peaks) <= 135           # rested: a harder EVA stays below
    assert peaks.max() < 140                         # the 140 bpm impairment line
    assert eva_peak(run("safe_day")[1]) < 120


def test_main_with_a_crosses_140_late_in_a_continuous_eva():
    runs, p = heat_runs("main_sleep_debt_eva")
    start = next(t.start_min for t in p.tasks if t.type is pl.PlanType.EVA)
    thr = config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(BASE)
    assert cardio_share(runs) >= 0.8
    firsts = [out.loc[out["hr_model"] >= thr, "minute"].iloc[0] - start
              for out, e in runs if "cardiovascular" in names(e)]
    assert np.median(firsts) >= 3 * 60                 # typically in the later part of the EVA
    assert min(firsts) >= 2 * 60                       # never in the first 2 h


def test_core_rise_is_capped():
    runs, _ = heat_runs("main_sleep_debt_eva")
    assert max(o["core_temp"].max() for o, _ in runs) - BASE.core_temp <= config.HEAT_MAX_CORE_RISE_C + 1e-6


def test_core_signal_and_heart_rate_signal_come_from_the_same_heat():
    runs, _ = heat_runs("main_sleep_debt_eva")
    thr = config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(BASE)
    for out, _ in runs:
        over = out["hr_model"] >= thr
        if over.any():
            assert (out.loc[over, "core_temp"] - BASE.core_temp > config.HEAT_STRAIN_ONSET_C).all()


def test_splitting_the_eva_with_rest_clearly_lowers_cardio_violations():
    continuous, _ = heat_runs("main_sleep_debt_eva")
    split, _ = heat_runs("main_sleep_debt_eva", plan=split_main())
    c, s = cardio_share(continuous), cardio_share(split)
    assert s <= 0.6 * c
    assert s > 0                                        # some cannot be planned away


# ---- count_from (midday check, plan 1.2): state machines run from the morning, violations
# before count_from are not recorded

MIDDAY = 14 * 60
SPLIT_EVA = [("07:00", 1.0, "exercise_ARED"), ("10:00", 1.5, "sleep"), ("11:30", 2.25, "EVA"),
             ("13:45", 1.0, "rest"), ("14:45", 2.25, "EVA"), ("23:00", 6.0, "sleep")]


def test_count_from_records_the_afternoon_violation_when_both_halves_break_rule_1():
    p = make_plan(SPLIT_EVA)
    c = dataclasses.replace(st.load_checkin("main_sleep_debt_eva"), sleep_last_night_h=5.9,
                            sleep_last_3_nights_h=(4.3, 4.5, 5.9))
    out = md.simulate(p, flat_trace(), c, BASE)
    first = {v.rule: v.minute for v in ev.evaluate(out, p, c, BASE).violations}
    assert first["rule1_sleep_eva"] == 11 * 60 + 30                  # morning EVA, recorded
    pm = {v.rule: v.minute for v in ev.evaluate(out, p, c, BASE, count_from=MIDDAY).violations}
    assert pm["rule1_sleep_eva"] == 14 * 60 + 45                     # the 14:45 one, not dropped
    assert all(v >= MIDDAY for v in pm.values())


def test_count_from_keeps_morning_history_for_the_state_machines():
    # rule 3 needs > 4 h of high-CO2 cabin work before docking: work from 10:00, docking 14:30
    p = make_plan([("10:00", 4.5, "maintenance"), ("14:30", 1.0, "docking")])
    c = st.load_checkin("safe_day")
    out = md.simulate(p, flat_trace(co2=6000.0), c, BASE)
    pm = {v.rule: v.minute for v in ev.evaluate(out, p, c, BASE, count_from=MIDDAY).violations}
    assert pm.get("rule3_co2_precision") == 14 * 60 + 30             # history from 10:00 kept


def test_count_from_counts_a_sustained_condition_completed_after_the_cut():
    """Heart rate high and HRV low from 13:50: the 30-min sustained rule completes after
    14:00, so it belongs to the afternoon."""
    e0, out, p, c = run("safe_day")
    assert "cardiovascular" not in {v.rule for v in e0.violations}
    hot = out.copy()
    window = hot["minute"].between(13 * 60 + 45, 15 * 60)            # from the 13:45 step on
    hot.loc[window, "heart_rate"] = 175.0
    hot.loc[window, "hrv_rmssd"] = 5.0
    full = {v.rule: v.minute for v in ev.evaluate(hot, p, c, BASE).violations}
    done = full["cardiovascular"]
    assert MIDDAY <= done <= 14 * 60 + 30                            # completed after 14:00
    pm = {v.rule: v.minute for v in ev.evaluate(hot, p, c, BASE, count_from=MIDDAY).violations}
    assert pm["cardiovascular"] == done


def test_count_from_none_is_the_default_behaviour():
    for name in PLANS:
        e, out, p, c = run(name)
        e2 = ev.evaluate(out, p, c, BASE, count_from=None)
        assert e2.violations == e.violations and e2.distance == e.distance
