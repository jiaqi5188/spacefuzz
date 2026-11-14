"""Tests for engine.py: probability track, guided track, rating, cache, CLI."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacefuzz.fuzzing import advice
from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from tests.local_data import HAS_ALL  # noqa: E402

BASE = st.make_baseline()
STEPS = config.STEPS_PER_DAY
RAW = HAS_ALL
raw_only = pytest.mark.skipif(not RAW, reason="raw data missing")


def flat_trace(co2=3000.0, dose=7.79):
    idx = pd.Index(np.arange(STEPS) * config.STEP_MINUTES, name="minute")
    return pd.DataFrame({"co2": co2, "temp": 22.5, "humidity": 42.0, "dose_rate": dose}, index=idx)


def scenario(name="main_sleep_debt_eva"):
    return mu.Scenario(pl.load_plan(f"data/plans/{name}.json"), flat_trace(), st.load_checkin(name))


# ---- rating (plan 1.3), boundaries written out

@pytest.mark.parametrize("p, ttc, expected", [
    (0.0, None, "LOW"),
    (0.0099, None, "LOW"),
    (0.01, None, "MEDIUM"),         # exactly 1% is yellow
    (0.03, None, "MEDIUM"),
    (0.05, 100, "MEDIUM"),          # exactly 5% is yellow (red is > 5%)
    (0.0501, 300, "HIGH"),
    (0.37, 121, "HIGH"),
    (0.37, 120, "CRITICAL"),        # red and time to crash <= 2 h
    (0.37, 0, "CRITICAL"),
])
def test_risk_rating_boundaries(p, ttc, expected):
    assert en.risk_rating(p, ttc) == expected


def test_untrusted_is_unknown_and_escalations():
    assert en.risk_rating(0.5, 10, untrusted=True) == "UNKNOWN"
    assert [advice.RATING_ESCALATION[r] for r in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN")] == \
        ["self", "crew", "ground", "ground", "self"]


def test_time_to_crash_is_first_time_cumulative_share_reaches_5_percent():
    firsts = [30] * 2 + [90] * 3 + [200] * 5 + [None] * 90          # 10% crash overall
    assert en.time_to_crash(firsts, 100) == 90                       # 2% at 30, 5% at 90
    assert en.time_to_crash([60] * 5 + [None] * 95, 100) is None      # risk 5%: not shown
    assert en.time_to_crash([None] * 100, 100) is None


# ---- profiles

def test_profiles_switch_with_environment(monkeypatch):
    monkeypatch.delenv("FUZZ_PROFILE", raising=False)
    local = en.profile()
    assert local["name"] == "local" and local["heatmap_samples"] == 300
    assert local["guided_rounds"] == 2000 and local["probability_samples"] == 2500
    monkeypatch.setenv("FUZZ_PROFILE", "full")
    full = en.profile()
    assert full["name"] == "full" and full["heatmap_samples"] == 2500
    assert full["guided_rounds"] > 2000 and full["probability_samples"] == 2500
    monkeypatch.setenv("FUZZ_PROFILE", "huge")
    with pytest.raises(ValueError):
        en.profile()


# ---- probability track

def test_probability_track_is_reproducible_and_complete():
    a = en.probability_track(scenario(), BASE, n=60)
    b = en.probability_track(scenario(), BASE, n=60)
    assert a == b
    assert a["track"] == "probability" and a["samples"] == 60
    assert a["risk_probability"] == pytest.approx(a["crashes"] / 60, abs=1e-6)
    assert len(a["cumulative_crash_share"]) == STEPS
    assert a["cumulative_crash_share"][-1] == pytest.approx(a["risk_probability"])
    for f in config.ENGINE_TRAJECTORY_FIELDS:
        tr = a["trajectory"][f]
        assert len(tr["p50"]) == STEPS
        assert all(lo <= mid <= hi for lo, mid, hi in zip(tr["p5"], tr["p50"], tr["p95"]))
    assert set(a["by_system"]) <= {"Cardiovascular", "CNS", "Respiratory", "Metabolic", "Radiation"}
    assert "discard rate" in a["mutator"]


def test_probability_track_draws_daily_probabilities(monkeypatch):
    monkeypatch.setattr(config, "PROB_OPERATOR_DAILY", {"A": 0, "B": 1.0, "C": 0, "D": 0, "E": 0, "F": 0})
    a = en.probability_track(scenario(), BASE, n=30)
    assert a["coverage"]["operators"] == {"B": 30}
    assert config.PROB_OPERATOR_DAILY is not None
    monkeypatch.setattr(config, "PROB_OPERATOR_DAILY", {op: 0.0 for op in "ABCDEF"})
    quiet = en.probability_track(scenario(), BASE, n=30)
    assert quiet["crashes"] == 0 and quiet["coverage"]["operators"] == {}


def test_e_is_not_drawn_in_the_probability_track():
    assert config.PROB_OPERATOR_DAILY["E"] == 0.0
    assert config.PROB_OPERATOR_DAILY["B"] == config.OP_B_DAILY_PROB
    assert config.PROB_OPERATOR_DAILY["D"] == config.OP_D_DAILY_PROB


def test_causal_chain_and_primary_system():
    a = en.probability_track(scenario(), BASE, n=200)
    assert a["crashes"] > 0
    assert a["primary_vulnerable_system"] == max(a["by_system"], key=a["by_system"].get)
    chain = a["causal_chain"]
    assert 2 <= len(chain) <= 5 and all(isinstance(x, str) and x for x in chain)


def test_coverage_records_rules_systems_tasks_and_ranges():
    a = en.probability_track(scenario(), BASE, n=200)
    cov = a["coverage"]
    assert set(cov) >= {"rules", "systems", "task_types", "indicator_ranges", "operators",
                        "quality_gate"}
    assert "EVA" in cov["task_types"] and "sleep" in cov["task_types"]
    assert cov["rules"] and cov["systems"]
    hr = cov["indicator_ranges"]["heart_rate"]
    assert hr and all(isinstance(b, str) for b in hr)
    assert set(cov["quality_gate"]) <= {"ok", "degraded", "untrusted"}


def test_tracks_never_share_samples():
    alone = en.probability_track(scenario(), BASE, n=40)
    en.guided_track(scenario(), BASE, rounds=50)        # must not disturb the probability track
    after = en.probability_track(scenario(), BASE, n=40)
    assert alone == after


# ---- guided track

def test_guided_track_finds_deduplicated_crash_types():
    g = en.guided_track(scenario(), BASE, rounds=300)
    assert g["track"] == "guided" and g["rounds"] == 300
    keys = [(c["system"], c["rule"]) for c in g["crash_types"]]
    assert len(keys) == len(set(keys)) and keys
    for c in g["crash_types"]:
        assert c["count"] >= 1 and c["first_round"] >= 1 and c["operators"]
    assert g["seeds"] >= 1 and g["seeds"] <= config.GUIDED_POOL_MAX
    no_clock = lambda d: {k: v for k, v in d.items() if k != "timing"}  # timing is wall-clock
    assert no_clock(g) == no_clock(en.guided_track(scenario(), BASE, rounds=300))  # reproducible


def test_guided_track_prefers_the_closest_seed(monkeypatch):
    monkeypatch.setattr(config, "GUIDED_EXPLORE_PROB", 0.0)
    g = en.guided_track(scenario(), BASE, rounds=100)
    assert g["explore_rounds"] == 0
    monkeypatch.setattr(config, "GUIDED_EXPLORE_PROB", 1.0)
    g = en.guided_track(scenario(), BASE, rounds=100)
    assert g["explore_rounds"] == 100


def test_heatmap_grid():
    h = en.heatmap(scenario(), BASE, samples=20, sleep_cut_h=(0.0, 1.0, 2.0), eva_extend_h=(0.0, 1.0))
    assert h["sleep_cut_h"] == [0.0, 1.0, 2.0] and h["eva_extend_h"] == [0.0, 1.0]
    r = np.array(h["risk"])
    assert r.shape == (3, 2) and ((r >= 0) & (r <= 1)).all()
    assert r[2, 1] >= r[0, 0]
    assert h["samples_per_cell"] == 20


def test_heatmap_without_eva_has_one_column():
    h = en.heatmap(scenario("backup_high_co2"), BASE, samples=10, sleep_cut_h=(0.0, 1.0))
    assert h["eva_extend_h"] == [0.0] and np.array(h["risk"]).shape == (2, 1)


def test_min_crash_perturbation():
    m = en.min_crash_perturbation(scenario(), BASE)
    assert m["minutes"] == 15 and m["rule"] in ("rule1_sleep_eva", "rule2_exercise_eva")
    assert "15 min" in m["text"]
    safe = en.min_crash_perturbation(scenario("safe_day"), BASE)
    assert safe["minutes"] > 60


# ---- forecast, gate, cache, CLI

TINY = {"probability_samples": 40, "heatmap_samples": 5, "guided_rounds": 30}


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    monkeypatch.setitem(config.FUZZ_PROFILES, "local", TINY)
    monkeypatch.setattr(config, "HEATMAP_SLEEP_CUT_H", (0.0, 1.0))
    monkeypatch.setattr(config, "HEATMAP_EVA_EXTEND_H", (0.0, 1.0))
    monkeypatch.setattr(en, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("FUZZ_PROFILE", raising=False)
    return tmp_path


def test_forecast_output_and_cache(tiny):
    a = en.forecast("data/plans/main_sleep_debt_eva.json", scenario=scenario())
    assert a["from_cache"] is False
    for key in ("risk_rating", "escalation", "risk_probability", "time_to_crash_minutes",
                "primary_vulnerable_system", "causal_chain", "min_crash_perturbation",
                "advice_now", "action_restrictions", "data_quality", "probability_track",
                "guided_track", "heatmap", "performance"):
        assert key in a, key
    assert a["performance"]["simulations_per_second"] > 0
    assert list(tiny.glob("*.json"))
    b = en.forecast("data/plans/main_sleep_debt_eva.json", scenario=scenario())
    assert b["from_cache"] is True
    assert {k: v for k, v in a.items() if k not in ("from_cache", "performance")} == \
        {k: v for k, v in b.items() if k not in ("from_cache", "performance")}


def test_untrusted_check_in_never_forecasts(tiny, monkeypatch):
    s = scenario()
    bad = dataclasses.replace(s.checkin, systolic_bp=900.0, diastolic_bp=500.0, blood_glucose=5.0)
    called = []
    monkeypatch.setattr(en, "probability_track", lambda *a, **k: called.append(1))
    out = en.forecast("data/plans/main_sleep_debt_eva.json", scenario=dataclasses.replace(s, checkin=bad))
    assert out["risk_rating"] == "UNKNOWN" and out["data_quality"] == "untrusted"
    assert out["risk_probability"] is None and "measure again" in out["message"].lower()
    assert not called


def test_cli_prints_summary(tiny, capsys):
    en.main(["--plan", "data/plans/safe_day.json", "--no-cache"] + ([] if RAW else ["--synthetic"]))
    text = capsys.readouterr().out
    assert "risk" in text.lower() and "simulations/s" in text


# ---- acceptance on the real scenarios (2,500 probability samples)

@pytest.fixture(scope="module")
def real_risk():
    """Probability-track risk (2,500 samples) for the real main and safe scenarios."""
    if not RAW:
        pytest.skip("raw data missing")
    fl = md.Fluctuation.from_wesad()
    out = {}
    for name in ("main_sleep_debt_eva", "safe_day"):
        s = mu.Scenario.load(name)
        out[name] = en.probability_track(s, BASE, n=2500, fluctuation=fl)
    no_eva = mu.Scenario.load("safe_day")              # test-only variant: EVA removed
    no_eva = dataclasses.replace(no_eva, plan=dataclasses.replace(
        no_eva.plan, tasks=[t for t in no_eva.plan.tasks if t.type is not pl.PlanType.EVA]))
    out["safe_day_without_eva"] = en.probability_track(no_eva, BASE, n=2500, fluctuation=fl)
    return out


@raw_only
def test_main_plan_is_red_from_mutation_not_a_deterministic_violation(real_risk):
    s = mu.Scenario.load("main_sleep_debt_eva")
    from spacefuzz.core import evaluator as ev
    base = md.simulate(s.plan, pl.apply_env_events(s.trace, s.plan), s.checkin, BASE)
    assert not ev.evaluate(base, s.plan, s.checkin, BASE).crashed
    p = real_risk["main_sleep_debt_eva"]
    assert p["risk_probability"] > config.RISK_RED_ABOVE
    assert en.risk_rating(p["risk_probability"], p["time_to_crash_minutes"]) in ("HIGH", "CRITICAL")


@raw_only
def test_safe_day_is_not_red_and_an_order_of_magnitude_below_main(real_risk):
    """Acceptance (revised 2026-11-09): an EVA day carries the real SPE floor (~0.4%), so
    safe_day sits near the green / yellow line; it must not be red and must be >= 10x
    below the main plan."""
    safe = real_risk["safe_day"]["risk_probability"]
    main = real_risk["main_sleep_debt_eva"]["risk_probability"]
    assert safe <= config.RISK_RED_ABOVE
    assert en.risk_rating(safe, real_risk["safe_day"]["time_to_crash_minutes"]) in ("LOW", "MEDIUM")
    assert main >= 10 * safe


@raw_only
def test_green_is_reachable_a_safe_day_without_eva_is_green(real_risk):
    p = real_risk["safe_day_without_eva"]
    assert p["risk_probability"] < config.RISK_GREEN_BELOW
    assert en.risk_rating(p["risk_probability"], p["time_to_crash_minutes"]) == "LOW"


# ---- out of model experience (plan 3.12)

def test_normal_days_are_inside_experience():
    for name in ("main_sleep_debt_eva", "safe_day"):
        assert en.out_of_experience(scenario(name), BASE) == []


def test_out_of_experience_checks():
    s = scenario("safe_day")
    c = dataclasses.replace(s.checkin, blood_glucose=130.0)
    w = en.out_of_experience(dataclasses.replace(s, checkin=c), BASE)
    assert len(w) == 1 and "blood_glucose" in w[0] and "130" in w[0]
    hot = dataclasses.replace(s, trace=s.trace.assign(co2=config.ENV_CO2_MAX_OBSERVED_PPM + 500))
    assert any("CO2" in x for x in en.out_of_experience(hot, BASE))
    i = next(k for k, t in enumerate(s.plan.tasks) if t.type is pl.PlanType.EVA)
    long_eva = dataclasses.replace(s, plan=dataclasses.replace(
        s.plan, tasks=mu.resize_task(s.plan.tasks, i, 5 * 60)))       # 4 h -> 9 h
    assert any("EVA" in x for x in en.out_of_experience(long_eva, BASE))
    tired = dataclasses.replace(s, checkin=dataclasses.replace(
        s.checkin, sleep_last_3_nights_h=(2.0, 2.0, 3.5), sleep_last_night_h=3.5))
    assert any("sleep debt" in x for x in en.out_of_experience(tired, BASE))


def test_backup_event_co2_is_out_of_experience():
    w = en.out_of_experience(scenario("backup_high_co2"), BASE)
    assert any("6000" in x and "4663" in x for x in w)


def test_unknown_task_type_warns():
    raw = pl.to_dict(scenario("safe_day").plan)
    raw["tasks"].append({"start": "21:30", "duration_h": 1, "type": "robot_repair", "kind": "task"})
    s = dataclasses.replace(scenario("safe_day"), plan=pl.load_plan(raw))
    assert any("robot_repair" in x for x in en.out_of_experience(s, BASE))


def test_out_of_experience_is_never_green():
    assert en.experience_floor("LOW", ["x"]) == "MEDIUM"
    assert en.experience_floor("LOW", []) == "LOW"
    for r in ("MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"):
        assert en.experience_floor(r, ["x"]) == r


def test_uncovered_lists_rules_and_systems_never_reached():
    cov = {"rules": {"cardiovascular": 3}, "systems": {"Cardiovascular": 3}}
    text = en.uncovered(cov)
    assert "metabolic" in text and "rule2_exercise_eva" in text and "cardiovascular," not in text
    assert "Metabolic" in text and "Respiratory" not in text     # no rule targets Respiratory


def test_forecast_reports_coverage_warnings_and_floors_rating(tiny):
    s = scenario("safe_day")
    s = dataclasses.replace(s, checkin=dataclasses.replace(s.checkin, blood_glucose=130.0))
    out = en.forecast("data/plans/safe_day.json", scenario=s, use_cache=False)
    assert any("blood_glucose" in w for w in out["coverage_warnings"])
    assert any(w.startswith("Not covered") for w in out["coverage_warnings"])
    assert out["risk_rating"] != "LOW" and out["escalation"] in ("crew", "ground")


# ---- records for the UI (stage 4: fuzz log, charts, plan board, HUD)

def test_guided_track_records_history_best_events_and_timing(monkeypatch):
    monkeypatch.setattr(config, "GUIDED_TIMING_EVERY", 50)
    g = en.guided_track(scenario(), BASE, rounds=120)
    h = g["history"]
    assert set(h) == {"fitness", "crashed", "discarded", "pool", "sleep_cut_h", "eva_extend_h"}
    assert all(len(v) == 120 for v in h.values())
    assert set(h["crashed"]) <= {0, 1} and set(h["discarded"]) <= {0, 1}
    for i in range(120):
        if h["discarded"][i]:
            assert h["fitness"][i] is None and h["crashed"][i] == 0
        else:
            assert h["fitness"][i] is not None
    assert h["pool"] == sorted(h["pool"]) and h["pool"][-1] == g["seeds"]   # never shrinks
    assert all(c >= 0 for c in h["sleep_cut_h"])                 # cut relative to the plan
    ev_ = g["best_events"]
    assert ev_ and [e["round"] for e in ev_] == sorted(e["round"] for e in ev_)
    fits = [e["fitness"] for e in ev_]
    assert fits == sorted(fits, reverse=True) and len(set(fits)) == len(fits)  # new lows only
    for e in ev_:
        assert {"round", "fitness", "steps_to_violation", "nearest_rule", "ops", "pool"} <= set(e)
        assert e["fitness"] == h["fitness"][e["round"] - 1] and not h["crashed"][e["round"] - 1]
    assert fits[-1] == pytest.approx(min(f for f, c in zip(h["fitness"], h["crashed"])
                                         if f is not None and not c))
    rounds = [r for r, _ in g["timing"]]
    assert rounds == [50, 100, 120]
    secs = [t for _, t in g["timing"]]
    assert secs == sorted(secs) and secs[0] > 0


def test_probability_track_keeps_individual_traces(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_TRACE_SAMPLES", 30)
    p = en.probability_track(scenario(), BASE, 50)
    tr = p["traces"]
    assert set(tr) == set(config.ENGINE_TRACE_FIELDS) | {"crashed"}
    assert len(tr["crashed"]) == 30
    for f in config.ENGINE_TRACE_FIELDS:
        assert len(tr[f]) == 30 and all(len(x) == STEPS for x in tr[f])
    assert sum(tr["crashed"]) <= p["crashes"]
    small = en.probability_track(scenario(), BASE, 10)
    assert len(small["traces"]["crashed"]) == 10                # fewer samples than the cap


def test_best_events_carry_the_full_mutation_lineage():
    g = en.guided_track(scenario(), BASE, rounds=150)
    assert g["best_events"]
    for e in g["best_events"]:
        lin = e["lineage"]
        assert lin[-len(e["ops"]):] == e["ops"]                 # ends with this round's ops
        assert all(set(x) == {"op", "params"} for x in lin)
    for c in g["crash_types"]:                                   # letters kept as before
        assert all(isinstance(o, str) for o in c["operators"])


def test_probability_track_count_from_only_counts_later_violations():
    sc = scenario("main_sleep_debt_eva")
    full = en.probability_track(sc, BASE, 60)
    pm = en.probability_track(sc, BASE, 60, count_from=20 * 60)
    assert pm["crashes"] <= full["crashes"]                        # same samples, later window
    assert pm["count_from"] == 20 * 60 and full["count_from"] is None
    assert en.probability_track(sc, BASE, 60, count_from=None)["crashes"] == full["crashes"]


@pytest.mark.parametrize("name", ["crew's task", "rover\x1brepair", 'say "hi"'])
def test_unknown_task_name_is_reported_exactly(name):
    p = pl.load_plan({"date": "Mission Day 1", "environment": "mars_base",
                      "check_window": "morning", "comm_delay_min": 20,
                      "tasks": [{"start": "09:00", "duration_h": 2.0, "type": name,
                                 "kind": "task"}]})
    w = en.out_of_experience(mu.Scenario(p, flat_trace(), st.load_checkin("safe_day")), BASE)
    assert f"Task type {name} is outside model experience (treated as high load)" in w


def test_engine_wesad_uses_the_shared_loader(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(md.Fluctuation, "load", classmethod(lambda cls: sentinel))
    assert en._wesad() is sentinel


def test_a_rejected_check_in_value_is_never_simulated(monkeypatch):
    # sleep 20 h would make the day look safe: the forecast is UNKNOWN and names the item
    def no_run(*a, **k):
        raise AssertionError("simulated a rejected value")
    monkeypatch.setattr(en, "probability_track", no_run)
    sc = en._synthetic("main_sleep_debt_eva")
    sc = dataclasses.replace(sc, checkin=dataclasses.replace(sc.checkin, sleep_last_night_h=20.0))
    r = en.forecast("data/plans/main_sleep_debt_eva.json", scenario=sc, use_cache=False)
    assert r["risk_rating"] == "UNKNOWN" and r["risk_probability"] is None
    assert "Re-measure: sleep last night (20 outside 0-16)" in r["message"]


def test_fitted_fluctuation_reads_the_committed_cache_but_never_overwrites_it(monkeypatch,
                                                                          tmp_path):
    # a fresh clone (no WESAD series) must use the committed real-WESAD results at once, and a
    # run there writes its own file instead of overwriting them under the same name
    monkeypatch.setattr(en, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(en, "_fluctuation_source", lambda: "wesad")
    real = tmp_path / "main_full_abcdef123456.json"
    assert en.cache_files("main_full_", "abcdef1234567890") == (real, real)
    monkeypatch.setattr(en, "_fluctuation_source", lambda: "wesad_params")
    own = tmp_path / "main_full_wesad_params_abcdef123456.json"
    assert en.cache_files("main_full_", "abcdef1234567890") == (own, own)    # nothing committed
    real.write_text("{}")
    assert en.cache_files("main_full_", "abcdef1234567890") == (real, own)   # read committed


def test_the_cache_key_does_not_depend_on_the_fluctuation_source(monkeypatch):
    sc, prof = en._synthetic("safe_day"), en.profile("local")
    monkeypatch.setattr(en, "_fluctuation_source", lambda: "wesad")
    real = en._cache_key(sc, prof)
    monkeypatch.setattr(en, "_fluctuation_source", lambda: "wesad_params")
    assert en._cache_key(sc, prof) == real
