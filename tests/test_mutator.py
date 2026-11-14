"""Tests for mutator.py: operators A-F, 1-3 stacked per sample, causal-chain discard."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType
from tests.local_data import HAS_ALL  # noqa: E402

BASE = st.make_baseline()
STEPS = config.STEPS_PER_DAY
EXCURSIONS = pd.read_csv(pl.ROOT / config.OP_B_CO2_EXCURSIONS_PATH)["excursion"].to_numpy()
PROFILE = pd.read_csv(pl.ROOT / config.OP_D_SPE_PROFILE_PATH)["excess_over_baseline"].to_numpy()
RAW = HAS_ALL


def wavy_trace(dose=7.79):
    """Synthetic raw trace: CO2 3000-4600 ppm over the day, flat dose."""
    idx = pd.Index(np.arange(STEPS) * config.STEP_MINUTES, name="minute")
    co2 = 3800 + 800 * np.sin(np.arange(STEPS) / STEPS * 2 * np.pi)
    return pd.DataFrame({"co2": co2, "temp": 22.5, "humidity": 42.0, "dose_rate": dose}, index=idx)


def scenario(name="main_sleep_debt_eva", trace=None):
    return mu.Scenario(plan=pl.load_plan(f"data/plans/{name}.json"),
                       trace=trace if trace is not None else wavy_trace(),
                       checkin=st.load_checkin(name))


def mutator(name="main_sleep_debt_eva", seed=config.RANDOM_SEED, **kw):
    return mu.Mutator(scenario(name), BASE, seed=seed, **kw)


def only(op, name="main_sleep_debt_eva", seed=config.RANDOM_SEED, n=1):
    """Draw n samples that use exactly the one operator `op`."""
    m = mutator(name, seed=seed, params={"MUT_OPS_PER_SAMPLE": (1,),
                                         "MUT_OPS_PER_SAMPLE_WEIGHTS": (1.0,),
                                         "MUT_OPERATOR_WEIGHTS": {op: 1.0}})
    return m, [m.draw() for _ in range(n)]


def eva(plan):
    return next(t for t in plan.tasks if t.type is PlanType.EVA)


def no_overlap(plan):
    ts = sorted(plan.tasks, key=lambda t: t.start_min)
    return all(a.end_min <= b.start_min for a, b in zip(ts, ts[1:]))


# ---- config

def test_operator_distributions_in_config():
    assert set(config.MUT_OPERATOR_WEIGHTS) == set("ABCDEF")
    assert tuple(config.MUT_OPS_PER_SAMPLE) == (1, 2, 3)
    assert sum(config.MUT_OPS_PER_SAMPLE_WEIGHTS) == pytest.approx(1.0)
    for name in ("OP_A_EVA_HEAT_RATE_C_PER_H", "OP_A_EVA_DURATION_DELTA_H",
                 "OP_B_WORK_EXTEND_H", "OP_B_SPO2_DROP_PCT", "OP_C_TASK_DURATION_H",
                 "OP_C_FATIGUE_JUMP", "OP_D_FATIGUE_PER_H", "OP_E_SLEEP_CUT_H",
                 "OP_E_CHRONIC_NIGHT_H", "OP_F_MISSING_FRACTION", "OP_F_DROPOUT_MIN",
                 "OP_F_STUCK_MIN", "OP_F_SPIKES", "OP_F_SPIKE_FACTOR", "OP_F_NETWORK_DOWN_MIN"):
        lo, hi = getattr(config, name)
        assert 0 < lo < hi, name


# ---- model hook (model.Effects)

def test_default_effects_change_nothing():
    s = scenario()
    trace = pl.apply_env_events(s.trace, s.plan)
    a = md.simulate(s.plan, trace, s.checkin, BASE)
    b = md.simulate(s.plan, trace, s.checkin, BASE, effects=md.Effects())
    pd.testing.assert_frame_equal(a, b)


def test_effects_eva_heat_storage_raises_core_then_heart_rate():
    s = scenario()
    trace = pl.apply_env_events(s.trace, s.plan)
    a = md.simulate(s.plan, trace, s.checkin, BASE)
    b = md.simulate(s.plan, trace, s.checkin, BASE, effects=md.Effects(eva_heat_rate=0.2))
    on = a["activity"] == "eva"
    rise = b.loc[on, "core_temp"] - a.loc[on, "core_temp"]
    assert rise.is_monotonic_increasing and rise.iloc[-1] > 0.8      # heat keeps building
    assert b.loc[on, "hr_model"].max() > a.loc[on, "hr_model"].max() + 10
    # heart rate follows core temperature: no extra HR until the core passes the 0.6 degC line
    below = on & (b["core_temp"] - BASE.core_temp <= config.HEAT_STRAIN_ONSET_C)
    np.testing.assert_allclose(b.loc[below, "hr_model"], a.loc[below, "hr_model"])
    before = a["minute"] < eva(s.plan).start_min
    pd.testing.assert_series_equal(a.loc[before, "hr_model"], b.loc[before, "hr_model"])
    # after the EVA the stored heat dissipates and heart rate comes back down
    late = b.index[-1]
    assert b.loc[late, "core_temp"] == pytest.approx(a.loc[late, "core_temp"], abs=0.05)


def test_effects_spo2_drop_only_above_high_co2():
    s = scenario()
    trace = pl.apply_env_events(s.trace, s.plan)
    a = md.simulate(s.plan, trace, s.checkin, BASE)
    b = md.simulate(s.plan, trace, s.checkin, BASE, effects=md.Effects(spo2_drop_high_co2=4.0))
    high = a["co2"] > config.ENV_CO2_HIGH_PPM
    assert high.any() and (~high).any()
    assert b["spo2"].min() < a["spo2"].min() - 3
    first_high = a.index[high][0]
    pd.testing.assert_series_equal(a.loc[:first_high, "spo2"], b.loc[:first_high, "spo2"])


def test_effects_fatigue_jump_and_rate():
    s = scenario()
    trace = pl.apply_env_events(s.trace, s.plan)
    a = md.simulate(s.plan, trace, s.checkin, BASE)
    b = md.simulate(s.plan, trace, s.checkin, BASE,
                    effects=md.Effects(fatigue_jumps=((600, 25.0),)))
    at = a.index[a["minute"] == 600][0]
    assert b.loc[at, "cognitive_fatigue"] == pytest.approx(
        min(config.FATIGUE_MAX, a.loc[at, "cognitive_fatigue"] + 25.0))
    assert b.loc[at, "reaction_time_ms"] > a.loc[at, "reaction_time_ms"]
    pd.testing.assert_series_equal(a.loc[:at - 1, "cognitive_fatigue"],
                                   b.loc[:at - 1, "cognitive_fatigue"])
    c = md.simulate(s.plan, trace, s.checkin, BASE,
                    effects=md.Effects(fatigue_rate=((600, 720, 4.0),)))
    end = a.index[a["minute"] == 720][0]
    gain = c.loc[end, "cognitive_fatigue"] - a.loc[end, "cognitive_fatigue"]
    assert gain == pytest.approx(4.0 * 2, abs=0.5)


def test_effects_dose_median_override():
    s = scenario()
    trace = pl.apply_env_events(s.trace, s.plan)
    spiky = trace.assign(dose_rate=trace["dose_rate"] * 3)       # whole day at 3x
    a = md.simulate(s.plan, spiky, s.checkin, BASE)               # median moves with it
    b = md.simulate(s.plan, spiky, s.checkin, BASE,
                    effects=md.Effects(dose_median=float(trace["dose_rate"].median())))
    assert b["radiation_load"].max() > a["radiation_load"].max()
    assert b["stress_level"].max() > a["stress_level"].max()


# ---- stacking and reproducibility

def test_same_seed_same_samples_and_outputs():
    a = mutator().generate(25)
    b = mutator().generate(25)
    assert [r.sample.ops for r in a] == [r.sample.ops for r in b]
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.output, y.output)
        pd.testing.assert_frame_equal(x.readings, y.readings)


def test_same_seed_with_real_fluctuation_is_reproducible():
    fl = md.Fluctuation(hr_dev={"relax": np.array([-2.0, 2.0]), "stress": np.array([4.0, -4.0])},
                        rmssd_rel={"relax": np.array([0.1, -0.1]), "stress": np.array([-0.2])})
    a = mutator(fluctuation=fl).generate(10)
    b = mutator(fluctuation=fl).generate(10)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.output, y.output)


def test_different_seed_differs():
    a = [r.sample.ops for r in mutator(seed=1).generate(10)]
    b = [r.sample.ops for r in mutator(seed=2).generate(10)]
    assert a != b


def test_each_sample_stacks_one_to_three_distinct_operators():
    results = mutator().generate(200)
    sizes = {len(r.sample.ops) for r in results}
    assert sizes == {1, 2, 3}
    for r in results:
        names = [o.op for o in r.sample.ops]
        assert len(names) == len(set(names))
    assert {o.op for r in results for o in r.sample.ops} == set("ABCDEF")


def test_operator_a_needs_an_eva():
    results = mutator("backup_high_co2").generate(100)
    assert "A" not in {o.op for r in results for o in r.sample.ops}


def test_base_scenario_is_not_mutated():
    s = scenario()
    before = (pl.to_dict(s.plan), s.trace.copy(), s.checkin)
    m = mu.Mutator(s, BASE, seed=3)
    m.generate(30)
    assert pl.to_dict(s.plan) == before[0]
    pd.testing.assert_frame_equal(s.trace, before[1])
    assert s.checkin == before[2]


# ---- operator A: EVA load

def test_operator_a_eva_duration_and_intensity():
    base = eva(scenario().plan)
    _, samples = only("A", n=30)
    deltas = set()
    for s in samples:
        p = s.ops[0].params
        lo, hi = config.OP_A_EVA_HEAT_RATE_C_PER_H
        assert lo <= s.effects.eva_heat_rate <= hi
        d = eva(s.plan).duration_min - base.duration_min
        assert d == p["duration_delta_min"] and d % config.STEP_MINUTES == 0
        assert 60 <= abs(d) <= 120
        assert eva(s.plan).start_min == base.start_min
        assert no_overlap(s.plan)
        deltas.add(np.sign(d))
    assert deltas == {-1, 1}


def test_operator_a_longer_eva_pushes_later_tasks_and_sleep_keeps_its_end():
    plan = scenario().plan
    tasks = mu.resize_task(plan.tasks, plan.tasks.index(eva(plan)), +120)
    by_type = {t.type: t for t in tasks}
    assert by_type[PlanType.EVA].end_min == 10 * 60 + 6.5 * 60 + 120
    meal = [t for t in tasks if t.type is PlanType.MEAL][1]
    assert meal.start_min == by_type[PlanType.EVA].end_min           # pushed right behind
    assert sorted(tasks, key=lambda t: t.start_min) == tasks
    assert all(a.end_min <= b.start_min for a, b in zip(tasks, tasks[1:]))
    sleep_old = next(t for t in plan.tasks if t.type is PlanType.SLEEP)
    assert by_type[PlanType.SLEEP].end_min == sleep_old.end_min


def test_overrun_into_sleep_shortens_sleep():
    plan = scenario().plan
    sci = next(i for i, t in enumerate(plan.tasks) if t.type is PlanType.SCIENCE_EXPERIMENT)
    tasks = mu.resize_task(plan.tasks, sci, +240)                     # 18:00-24:00
    sleep = next(t for t in tasks if t.type is PlanType.SLEEP)
    old = next(t for t in plan.tasks if t.type is PlanType.SLEEP)
    assert sleep.start_min > old.start_min and sleep.end_min == old.end_min
    assert all(a.end_min <= b.start_min for a, b in zip(tasks, tasks[1:]))


def test_operator_a_raises_eva_heart_rate_within_the_personal_max():
    m, samples = only("A", n=10)
    ref = m.reference.loc[m.reference["activity"] == "eva", "hr_model"].max()
    for s in samples:
        out = m.simulate(s)
        hr = out.loc[out["activity"] == "eva", "hr_model"].max()
        assert ref - 1e-6 <= hr <= st.max_heart_rate(BASE) + 1e-9


# ---- operator B: CO2 build-up

def test_operator_b_samples_observed_excursions_by_frequency():
    _, samples = only("B", n=400)
    es = np.array([s.ops[0].params["co2_excursion"] for s in samples])
    assert set(np.round(es, 6)) <= set(np.round(EXCURSIONS, 6))
    assert es.max() > config.OP_B_CO2_AMPLIFY_RANGE[1]               # real tail is kept
    usual = np.mean((es >= config.OP_B_CO2_AMPLIFY_RANGE[0]) & (es <= config.OP_B_CO2_AMPLIFY_RANGE[1]))
    observed = np.mean((EXCURSIONS >= config.OP_B_CO2_AMPLIFY_RANGE[0])
                       & (EXCURSIONS <= config.OP_B_CO2_AMPLIFY_RANGE[1]))
    assert usual == pytest.approx(observed, abs=0.08)


def test_operator_b_scales_raw_co2_then_applies_event_floor():
    s0 = scenario("backup_high_co2")
    _, samples = only("B", name="backup_high_co2", n=5)
    floor = s0.plan.events[0].co2_ppm
    for s in samples:
        e = s.ops[0].params["co2_excursion"]
        expected = pl.apply_env_events(s0.trace.assign(co2=s0.trace["co2"] * (1 + e)), s.plan)
        pd.testing.assert_series_equal(s.trace["co2"], expected["co2"])
        ev = s0.plan.events[0]
        inside = (s.trace.index >= ev.start_min) & (s.trace.index < ev.end_min)
        assert (s.trace.loc[inside, "co2"] >= floor).all()


def test_operator_b_extends_continuous_work_and_drops_spo2():
    base = scenario().plan
    m, samples = only("B", n=20)
    for s in samples:
        p = s.ops[0].params
        assert 60 <= p["work_extend_min"] <= 240 and p["work_extend_min"] % 15 == 0
        old = next(t for t in base.tasks if t.source_type == p["work_task"])
        new = next(t for t in s.plan.tasks if t.source_type == p["work_task"]
                   and t.start_min == old.start_min)
        assert new.duration_min == old.duration_min + p["work_extend_min"]
        lo, hi = config.OP_B_SPO2_DROP_PCT
        assert lo <= s.effects.spo2_drop_high_co2 <= hi
        assert no_overlap(s.plan)


# ---- operator C: multi-task overload

def test_operator_c_inserts_high_load_task_and_fatigue_jump():
    m, samples = only("C", n=30)
    for s in samples:
        p = s.ops[0].params
        new = [t for t in s.plan.tasks if t.start_min == p["start_min"]
               and t.source_type in config.OP_C_INSERT_TYPES]
        assert len(new) == 1
        lo, hi = config.OP_C_TASK_DURATION_H
        assert lo * 60 <= new[0].duration_min <= hi * 60
        for t in s.plan.tasks:                     # never on top of EVA or sleep
            if t.type in (PlanType.EVA, PlanType.SLEEP):
                assert t.end_min <= new[0].start_min or t.start_min >= new[0].end_min
        assert new[0].start_min >= s.checkin.check_time_min
        assert no_overlap(s.plan)
        (minute, jump), = s.effects.fatigue_jumps
        assert minute == new[0].start_min
        assert config.OP_C_FATIGUE_JUMP[0] <= jump <= config.OP_C_FATIGUE_JUMP[1]


def test_insert_task_splits_the_task_it_lands_in():
    plan = scenario().plan
    eva_task = eva(plan)
    tasks = mu.insert_task(plan.tasks, PlanType.DOCKING, 18 * 60 + 30, 60)  # inside science
    sci = [t for t in tasks if t.type is PlanType.SCIENCE_EXPERIMENT]
    assert [(t.start_min, t.end_min) for t in sci] == [(18 * 60, 18 * 60 + 30), (19 * 60 + 30, 20 * 60)]
    assert eva_task in tasks


# ---- operator D: solar particle event

def test_operator_d_superimposes_real_event_dose():
    s0 = scenario()
    _, samples = only("D", n=20)
    median = float(s0.trace["dose_rate"].median())
    for s in samples:
        p = s.ops[0].params
        added = (s.trace["dose_rate"] - s0.trace["dose_rate"]).to_numpy()
        # walk the simulated day from the check-in step and compare with the profile
        k0 = (s.checkin.check_time_min // 15) % STEPS
        sim = np.array([added[(k0 + k) % STEPS] for k in range(STEPS)])
        n = min(STEPS - p["onset_step"], len(PROFILE) - p["profile_offset"])
        expected = PROFILE[p["profile_offset"]:p["profile_offset"] + n] * median
        np.testing.assert_allclose(sim[p["onset_step"]:p["onset_step"] + n], expected)
        assert np.allclose(sim[:p["onset_step"]], 0)
        assert s.effects.dose_median == pytest.approx(median)
        lo, hi = config.OP_D_FATIGUE_PER_H
        assert all(lo <= r <= hi for _, _, r in s.effects.fatigue_rate)


def test_operator_d_adds_spe_warning_over_the_event():
    _, samples = only("D", n=40)
    seen = 0
    for s in samples:
        p = s.ops[0].params
        warn = [e for e in s.plan.events if e.type is PlanType.SPE_WARNING]
        if p["warning"] is None:
            assert not warn
            continue
        seen += 1
        (start, end) = p["warning"]                  # minutes of the simulated window
        clock = [(e.start_min, e.end_min) for e in warn]
        assert all(x in clock for x in mu.clock_events(start, end))   # past midnight: split
        assert start < end and all(0 <= a < b <= 24 * 60 for a, b in clock)
    assert seen > 0


def test_operator_d_raises_dose_and_stress():
    m, samples = only("D", n=5)
    ref = m.reference
    for s in samples:
        out = m.simulate(s)
        assert out["dose_cum_ugy"].iloc[-1] > ref["dose_cum_ugy"].iloc[-1]
        assert out["stress_level"].max() >= ref["stress_level"].max()


# ---- operator E: sleep debt

def test_operator_e_cuts_sleep_and_keeps_check_in_consistent():
    c0 = st.load_checkin("main_sleep_debt_eva")
    _, samples = only("E", n=40)
    chronic = 0
    for s in samples:
        c, p = s.checkin, s.ops[0].params
        lo, hi = config.OP_E_SLEEP_CUT_H
        assert lo <= p["sleep_cut_h"] <= hi
        cut = max(0.0, c0.sleep_last_night_h - p["sleep_cut_h"])
        if p["chronic"]:
            assert c.sleep_last_night_h <= cut + 1e-9
        else:
            assert c.sleep_last_night_h == pytest.approx(cut)
        assert c.sleep_last_3_nights_h[-1] == c.sleep_last_night_h
        assert all(n <= o + 1e-9 for n, o in zip(c.sleep_last_3_nights_h, c0.sleep_last_3_nights_h))
        if p["chronic"]:
            chronic += 1
            assert all(n < 6.0 for n in c.sleep_last_3_nights_h)
        assert c.stress == min(5, c0.stress + config.OP_E_STRESS_SELF_DELTA)
        # measured reaction time slows exactly as the model says for the extra debt
        d0, d1 = st.sleep_debt_h(c0, BASE), st.sleep_debt_h(c, BASE)
        f = st.initial_state(c0, BASE).cognitive_fatigue
        slower = st.model_reaction_ms(BASE, d1, f) - st.model_reaction_ms(BASE, d0, f)
        assert c.reaction_time_ms == pytest.approx(c0.reaction_time_ms + slower)
    assert 0 < chronic < 40


# ---- operator F: sensor faults

def readings():
    s = scenario()
    return md.simulate(s.plan, pl.apply_env_events(s.trace, s.plan), s.checkin, BASE)


def test_fault_stuck_is_caught_by_the_gate():
    r = readings()
    bad = mu.apply_faults(r, [mu.Fault("stuck", ("heart_rate",), 600, 120)])
    on = (bad["minute"] >= 600) & (bad["minute"] < 720)
    assert bad.loc[on, "heart_rate"].nunique() == 1
    gate = qg.check(bad)
    assert any(i.field == "heart_rate" and i.kind == "stuck" for i in gate.issues)


def test_fault_network_down_blanks_every_field():
    r = readings()
    bad = mu.apply_faults(r, [mu.Fault("network_down", config.OP_F_FIELDS, 600, 240)])
    on = (bad["minute"] >= 600) & (bad["minute"] < 840)
    assert bad.loc[on, list(config.OP_F_FIELDS)].isna().all().all()
    assert bad.loc[~on, list(config.OP_F_FIELDS)].notna().all().all()
    assert qg.check(bad).status == "degraded"


def test_fault_spike_and_missing_and_dropout():
    r = readings()
    spiked = mu.apply_faults(r, [mu.Fault("spike", ("heart_rate",), indices=(10, 20), factor=4.0)])
    assert spiked.loc[10, "heart_rate"] == pytest.approx(4.0 * r.loc[10, "heart_rate"])
    gate = qg.check(spiked)
    assert any(i.field == "heart_rate" and i.kind == "out_of_range" for i in gate.issues)
    missing = mu.apply_faults(r, [mu.Fault("missing", ("spo2",), indices=tuple(range(0, 96, 2)))])
    assert missing["spo2"].isna().sum() == 48
    drop = mu.apply_faults(r, [mu.Fault("dropout", ("core_temp",), 900, 60)])
    assert drop["core_temp"].isna().sum() == 4
    pd.testing.assert_frame_equal(r, readings())                     # input untouched


def test_operator_f_corrupts_readings_not_the_body():
    m, samples = only("F", n=20)
    kinds = set()
    for s in samples:
        assert 1 <= len(s.faults) <= 2
        kinds |= {f.kind for f in s.faults}
    assert kinds <= set(config.OP_F_KINDS) and len(kinds) >= 3
    results = m.generate(5)
    for r in results:
        pd.testing.assert_frame_equal(r.output, m.reference)          # body unchanged
        assert not r.readings.equals(r.output)


# ---- causal-chain check and discard

def test_reference_run_passes_the_chain_check():
    m = mutator()
    assert mu.check_chains(m.reference, m.reference, ["A", "B", "C", "D", "E"]) == []


def test_sleep_more_violates_chain_5():
    m = mutator()
    c = dataclasses.replace(m.base.checkin, sleep_last_night_h=8.0, sleep_last_3_nights_h=(8.0, 8.0, 8.0))
    out = m.simulate(mu.Sample(ops=(mu.OpRecord("E", {}),), plan=m.base.plan,
                               trace=pl.apply_env_events(m.base.trace, m.base.plan),
                               checkin=c, effects=m.base_effects))
    v = mu.check_chains(out, m.reference, ["E"])
    assert v and all(x.startswith("chain5") for x in v)


def test_weaker_eva_violates_chain_2():
    m = mutator()
    s = mu.Sample(ops=(mu.OpRecord("A", {}),), plan=m.base.plan,
                  trace=pl.apply_env_events(m.base.trace, m.base.plan), checkin=m.base.checkin,
                  effects=dataclasses.replace(m.base_effects, eva_heat_rate=-0.3))
    v = mu.check_chains(m.simulate(s), m.reference, ["A"])
    assert any(x.startswith("chain2") for x in v)


def test_violating_samples_are_discarded_and_counted():
    # an operator E that adds sleep instead of cutting it breaks chain 5 every time
    m = mutator(params={"OP_E_SLEEP_CUT_H": (-3.0, -1.0), "OP_E_CHRONIC_PROB": 0.0,
                        "MUT_OPS_PER_SAMPLE": (1,), "MUT_OPS_PER_SAMPLE_WEIGHTS": (1.0,),
                        "MUT_OPERATOR_WEIGHTS": {"E": 1.0, "D": 1.0}})
    results = m.generate(30)
    assert len(results) == 30
    assert all([o.op for o in r.sample.ops] == ["D"] for r in results)
    st_ = m.stats
    assert st_.discarded > 0 and st_.accepted == 30
    assert st_.attempted == st_.accepted + st_.discarded
    assert st_.by_reason["chain5"] == st_.discarded
    assert st_.discard_rate == pytest.approx(st_.discarded / st_.attempted)
    text = st_.summary()
    assert "discard rate" in text and f"{st_.discard_rate:.1%}" in text


def test_generate_gives_up_when_everything_is_discarded():
    m = mutator(params={"OP_E_SLEEP_CUT_H": (-3.0, -1.0), "OP_E_CHRONIC_PROB": 0.0,
                        "MUT_OPS_PER_SAMPLE": (1,), "MUT_OPS_PER_SAMPLE_WEIGHTS": (1.0,),
                        "MUT_OPERATOR_WEIGHTS": {"E": 1.0}})
    results = m.generate(4)
    assert results == []
    assert m.stats.attempted == 4 * config.MUT_MAX_ATTEMPTS_FACTOR


def test_normal_operators_rarely_discard():
    m = mutator()
    m.generate(150)
    assert m.stats.discard_rate < 0.05


def test_summary_is_logged(caplog):
    m = mutator()
    with caplog.at_level("INFO", logger="mutator"):
        m.generate(5)
    assert "discard rate" in caplog.text


# ---- real data

@pytest.mark.skipif(not RAW, reason="raw data missing")
def test_scenario_from_files_and_real_fluctuation():
    s = mu.Scenario.load("main_sleep_debt_eva")
    trace, plan = pl.load_scenario("data/plans/main_sleep_debt_eva.json")
    pd.testing.assert_frame_equal(pl.apply_env_events(s.trace, s.plan), trace, check_like=True)
    m = mu.Mutator(s, BASE, fluctuation=md.Fluctuation.from_wesad())
    a = m.generate(20)
    b = mu.Mutator(s, BASE, fluctuation=md.Fluctuation.from_wesad()).generate(20)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.output, y.output)


# ---- seeds for the guided track (mutating an already mutated sample)

def test_sample_keeps_raw_trace_and_can_become_a_seed():
    m = mutator()
    s = m.draw()
    pd.testing.assert_frame_equal(pl.apply_env_events(s.raw_trace, s.plan), s.trace)
    seed = mu.Scenario.from_sample(s)
    assert seed.plan is s.plan and seed.checkin == s.checkin and seed.effects == s.effects


def test_mutating_a_seed_builds_on_its_effects():
    _, (s,) = only("C")
    seed = mu.Scenario.from_sample(s)
    m = mu.Mutator(seed, BASE, params={"MUT_OPS_PER_SAMPLE": (1,), "MUT_OPS_PER_SAMPLE_WEIGHTS": (1.0,),
                                        "MUT_OPERATOR_WEIGHTS": {"C": 1.0}})
    child = m.draw()
    assert len(child.effects.fatigue_jumps) == 2                   # parent jump kept
    assert child.effects.fatigue_jumps[0] == s.effects.fatigue_jumps[0]
    pd.testing.assert_frame_equal(m.reference, m.simulate(dataclasses.replace(
        child, plan=seed.plan, trace=pl.apply_env_events(seed.trace, seed.plan),
        checkin=seed.checkin, effects=seed.effects)))               # reference = the parent


def test_seed_keeps_the_root_dose_median():
    _, (s,) = only("D")
    seed = mu.Scenario.from_sample(s)
    m = mu.Mutator(seed, BASE)
    assert m.base_effects.dose_median == pytest.approx(float(np.nanmedian(scenario().trace["dose_rate"])))


def test_shared_generator_is_used_when_given():
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    a = mu.Mutator(scenario(), BASE, rng=rng_a).draw()
    b = mu.Mutator(scenario(), BASE, rng=rng_b).draw()
    assert a.ops == b.ops


def test_apply_sleep_cut_matches_operator_e_without_chronic():
    c0 = st.load_checkin("main_sleep_debt_eva")
    c = mu.apply_sleep_cut(c0, BASE, 1.0)
    assert c.sleep_last_night_h == pytest.approx(5.2)
    assert c.sleep_last_3_nights_h == (4.3, 4.5, pytest.approx(5.2))
    assert c.reaction_time_ms > c0.reaction_time_ms and c.stress == c0.stress


# ---- independent daily draws (probability track)

DAILY = {"A": 0.5, "B": 0.3, "C": 0.0, "D": 0.1, "E": 0.0, "F": 0.2}


def test_daily_mode_draws_each_operator_independently():
    m = mutator(params={"MUT_DAILY_PROB": DAILY})
    samples = [m.draw() for _ in range(2000)]
    share = {op: np.mean([op in [o.op for o in s.ops] for s in samples]) for op in DAILY}
    for op, p in DAILY.items():
        assert share[op] == pytest.approx(p, abs=0.035), op
    none = np.mean([len(s.ops) == 0 for s in samples])
    assert none == pytest.approx(0.5 * 0.7 * 0.9 * 0.8, abs=0.035)   # ordinary days exist


def test_daily_mode_skips_operators_that_do_not_apply():
    m = mutator("backup_high_co2", params={"MUT_DAILY_PROB": {**DAILY, "A": 1.0}})
    assert all("A" not in [o.op for o in m.draw().ops] for _ in range(50))


def test_daily_mode_is_reproducible_and_an_empty_sample_is_the_real_day():
    a = mutator(params={"MUT_DAILY_PROB": DAILY}).generate(40)
    b = mutator(params={"MUT_DAILY_PROB": DAILY}).generate(40)
    assert [r.sample.ops for r in a] == [r.sample.ops for r in b]
    plain = mutator(params={"MUT_DAILY_PROB": {op: 0.0 for op in DAILY}})
    (r,) = plain.generate(1)
    assert r.sample.ops == ()
    pd.testing.assert_frame_equal(r.output, plain.reference)


def test_overrun_through_a_short_nap_keeps_pushing():
    """A nap swallowed by an overrun is dropped and the push carries on to the next task."""
    plan = pl.load_plan({"date": "D", "environment": "mars_base", "check_window": "morning",
                         "comm_delay_min": 20, "tasks": [
                             {"start": "07:30", "duration_h": 1.5, "type": "science_experiment", "kind": "task"},
                             {"start": "10:00", "duration_h": 1.0, "type": "sleep", "kind": "task"},
                             {"start": "11:00", "duration_h": 5.0, "type": "EVA", "kind": "task"}]})
    tasks = mu.resize_task(plan.tasks, 0, 300)                       # science until 14:00
    assert all(a.end_min <= b.start_min for a, b in zip(tasks, tasks[1:]))
    assert [t.type for t in tasks] == [PlanType.SCIENCE_EXPERIMENT, PlanType.EVA]
    assert tasks[1].start_min == 14 * 60
