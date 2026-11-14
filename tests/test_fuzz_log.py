"""Tests for ui/fuzz_log.py: typewriter fuzz log replayed from the real engine results."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import fuzz_log as fl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = "main_sleep_debt_eva"

GUIDED = {
    "rounds": 10000, "explore_rounds": 1960, "seeds": 198, "best_fitness": 1.242672,
    "plateau_round": 528,
    "mutator": "mutator: 10000 drawn, 8873 accepted, 1127 discarded (discard rate 11.3%; "
               "chain1 264, chain2 865)",
    "best_events": [
        {"round": 3, "fitness": 2.1, "steps_to_violation": None, "nearest_rule": "cardiovascular",
         "ops": [{"op": "A", "params": {"duration_delta_min": 60, "eva_heat_rate": 0.2}}],
         "pool": 2},
        # test fixture only, deliberately unlike the plan's example line
        {"round": 40, "fitness": 1.24, "steps_to_violation": 3,
         "nearest_rule": "rule1_sleep_eva",
         "ops": [{"op": "E", "params": {"sleep_cut_h": 0.7}},
                 {"op": "A", "params": {"duration_delta_min": 45, "eva_heat_rate": 0.1}}],
         "pool": 17}],
    "timing": [[5000, 10.0], [10000, 25.0]],
    "history": {"pool": [1] * 5000 + [150] * 4999 + [198],
                "sleep_cut_h": [0.0] * 39 + [0.7] + [0.0] * 9960,
                "eva_extend_h": [0.0] * 39 + [0.75] + [0.0] * 9960,
                "crashed": [0] * 10000, "discarded": [0] * 10000},
    "crash_types": [
        {"system": "Cardiovascular", "rule": "rule1_sleep_eva", "first_round": 1,
         "description": "EVA started within 12 h of waking after < 6 h sleep",
         "last_mutation": [{"op": "E", "params": {"sleep_cut_h": 2.86, "chronic": False}},
                           {"op": "B", "params": {"co2_excursion": 0.162,
                                                  "work_task": "science_experiment",
                                                  "work_extend_min": 165, "spo2_drop": 3.9}}]},
        {"system": "Cardiovascular", "rule": "cardiovascular", "first_round": 2,
         "description": "Sustained high heart rate with low HRV",
         "last_mutation": [{"op": "A", "params": {"duration_delta_min": 90,
                                                  "eva_heat_rate": 0.292}}]},
        {"system": "CNS", "rule": "rule3_co2_precision", "first_round": 228,
         "description": "docking / precision work after > 4 h of high-CO2 cabin work",
         "last_mutation": [{"op": "C", "params": {"task": "docking", "start_min": 1440,
                                                  "duration_min": 60, "fatigue_jump": 29.5}}]},
    ],
}


def results():
    from tests.test_app import fake_results
    res = fake_results()
    res["forecast"] = {**res["forecast"], "guided_track": GUIDED,
                       "coverage_warnings": ["Not covered by the guided track: rules metabolic; "
                                             "systems Metabolic"],
                       "min_crash_perturbation": {"text": "15 min less sleep turns this red"}}
    res["forecast"]["probability_track"] = {**res["forecast"]["probability_track"],
                                            "mutator": "mutator: 2500 drawn, 2500 accepted, 0 "
                                                       "discarded (discard rate 0.0%; none)"}
    return res


def test_describe_ops_plain_language():
    ops = [{"op": "E", "params": {"sleep_cut_h": 2.86}},
           {"op": "A", "params": {"duration_delta_min": -75, "eva_heat_rate": 0.195}},
           {"op": "B", "params": {"co2_excursion": 0.162, "work_task": "science_experiment",
                                  "work_extend_min": 165, "spo2_drop": 3.9}},
           {"op": "C", "params": {"task": "docking", "duration_min": 60, "fatigue_jump": 29.5}},
           {"op": "D", "params": {"warning": [810, 1439]}},
           {"op": "F", "params": {"faults": ["missing", "network_down"]}}]
    text = fl.describe_ops(ops)
    for part in ("sleep -2.9 h", "EVA -1 h 15 min", "heat 0.20 °C/h",
                 "CO2 excursion +16% -> science_experiment +2 h 45 min, SpO2 -3.9%",
                 "docking 1 h 00 min inserted",
                 "fatigue +30", "SPE warning from 13:30", "sensor fault: missing, network_down"):
        assert part in text, part


def test_log_lines_from_real_results_in_order():
    lines = fl.log_lines(results(), start_min=390)
    text = [ln["text"] for ln in lines]
    joined = "\n".join(text)
    new = [ln for ln in lines if ln["kind"] == "new"]
    assert len(new) == 3                                       # one per crash type
    assert "round 1 " in new[0]["text"] and "round 228" in new[-1]["text"]
    assert "sleep -2.9 h" in new[0]["text"]
    assert any(ln["kind"] == "uncovered" and "Not covered" in ln["text"] for ln in lines)
    assert "plateau" in joined and "528" in joined
    assert "seed pool 198" in joined and "lowest fitness 1.24" in joined
    assert "1,127" in joined and "11.3%" in joined              # causal-chain discards
    assert "472 simulations/s" in joined
    assert "21.0%" in joined and "4.5%" in joined
    # forecast lines first, then adjustment lines; times rise and stay in [0, 1]
    stages = [ln["stage"] for ln in lines]
    assert stages == sorted(stages, key=["forecast", "adjust"].index)
    for stage in ("forecast", "adjust"):
        ts = [ln["t"] for ln in lines if ln["stage"] == stage]
        assert ts == sorted(ts) and 0 <= ts[0] and ts[-1] <= 1


def test_round_counter_mapping_is_monotonic_and_spreads_early_events():
    ts = [fl.round_to_t(r, 10000) for r in (1, 2, 6, 59, 228, 528, 10000)]
    assert ts == sorted(ts) and ts[-1] == pytest.approx(1.0)
    assert ts[3] > 0.3                                         # round 59 not crammed at 0


def test_log_untrusted():
    res = results()
    res["forecast"] = {**res["forecast"], "risk_rating": "UNKNOWN",
                       "message": "Data not trustworthy. Please measure again."}
    res["adjust"] = None
    text = " ".join(ln["text"] for ln in fl.log_lines(res, start_min=390))
    assert "not trustworthy" in text and "round" not in text


def test_stage_times_match_app_playback():
    frames = [{"stage": "forecast"}] * 3 + [{"stage": "forecast", "done": "forecast"},
                                             {"stage": "adjust"},
                                             {"stage": "adjust", "done": "adjust"}]
    assert fl.stage_times(frames, frame_s=0.7, reveal_s=1.2) == pytest.approx((2.1, 4.0))


def test_html_modes():
    lines = fl.log_lines(results(), start_min=390)
    play = fl.html_doc("play", lines, rounds=10000, forecast_s=2.1, adjust_s=4.0)
    final = fl.html_doc("final", lines, rounds=10000)
    idle = fl.html_doc("idle", [], rounds=0)
    assert 'data-state="play"' in play and "2100" in play and "4000" in play
    markup = lambda h: h.split("<script")[0]                   # the JS names the marker too
    assert 'data-testid="fuzz-log-done"' in markup(final)
    assert 'data-testid="fuzz-log-done"' not in markup(idle) + markup(play)
    assert 'class="ln new"' in final and 'class="ln uncovered"' in final
    assert "http://" not in final and "https://" not in final
    assert "&lt;" in fl.html_doc("final", [{"text": "<b>", "kind": "info", "stage": "forecast",
                                            "t": 0}], rounds=1)


def test_app_plays_then_final(monkeypatch):
    from tests.test_app import click_and_hold, fake_results, frame_docs
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.run()
    assert not at.exception

    def doc():
        f = frame_docs(at, "sf-fuzz-log")
        assert len(f) == 1
        return f[0]

    assert 'data-state="idle"' in doc()
    click_and_hold(at, "Run Forecast", monkeypatch)       # one run: play, drawn once
    assert not at.exception and 'data-state="play"' in doc()
    at.run()                                               # playback ends: final picture
    assert 'data-state="final"' in doc()
    assert 'data-testid="fuzz-log-done"' in doc().split("<script")[0]


def test_log_shows_lowest_fitness_samples_and_both_discard_rates():
    lines = fl.log_lines(results(), start_min=390)
    best = [ln for ln in lines if ln["kind"] == "best"]
    assert [("round 3 " in b["text"], "round 40" in b["text"]) for b in best] == \
        [(True, False), (False, True)]
    t = best[-1]["text"]
    assert "change vs plan: sleep -0.7 h, EVA +0.8 h" in t            # cumulative, from history
    assert "last step: sleep -0.7 h + EVA +0 h 45 min" in t           # only this round's ops
    assert "a time-based rule is 3 steps (45 min of change) away" in t
    assert "seed pool 17" in t
    text = "\n".join(ln["text"] for ln in lines)
    assert "guided track: causal-chain check discarded 1,127" in text
    assert "probability track: causal-chain check discarded 0" in text


def test_best_events_are_thinned_but_keep_first_and_last():
    many = [{"round": r, "fitness": 3 - r / 100, "steps_to_violation": None,
             "nearest_rule": "cognitive", "ops": [], "pool": r} for r in range(1, 60)]
    kept = fl.thin_events(many, keep=6)
    assert len(kept) == 6 and kept[0] is many[0] and kept[-1] is many[-1]


def test_header_series_pool_and_rate():
    s = fl.header_series(GUIDED | {"history": {"pool": [1] * 5000 + [150] * 4999 + [198]}})
    assert s["pool"][0] == 1 and s["pool"][-1] == 198 and len(s["pool"]) <= 101
    assert s["rate"] == [[5000, 500.0], [10000, 333.3]]       # rounds per second per segment


def test_no_near_miss_when_every_guided_sample_crashed():
    res = results()
    g = dict(res["forecast"]["guided_track"], best_events=[],
             history={"pool": [1] * 10, "crashed": [1] * 9 + [0], "discarded": [0] * 9 + [1],
                      "sleep_cut_h": [0.0] * 10, "eva_extend_h": [0.0] * 10})
    res["forecast"] = {**res["forecast"], "guided_track": g}
    text = "\n".join(ln["text"] for ln in fl.log_lines(res, start_min=390))
    assert "all 9 accepted guided samples crashed, so there is no near miss to show" in text
    assert "plan as written already breaks" not in text          # not true for every plan


def test_best_line_says_as_planned_when_nothing_cumulative_changed():
    b = {"round": 1, "fitness": 2.0, "steps_to_violation": 1, "nearest_rule": "cognitive",
         "ops": [{"op": "F", "params": {"faults": ["stuck"]}}], "pool": 3}
    line = fl._best_line(b, {"sleep_cut_h": [0.0], "eva_extend_h": [0.0]})
    assert "change vs plan: sleep and EVA as planned" in line and "-0.0" not in line
    assert "a time-based rule is 1 step (15 min of change) away" in line


def test_plan_as_written_steps_from_data():
    steps = fl._plan_eval(MAIN)[0]
    assert steps["rule1_sleep_eva"] == 1 and steps["rule2_exercise_eva"] == 1
    line = fl.plan_line(steps, crashed=False, best_steps=[1, 1, 1])
    assert "plan as written is 1 step (15 min of change) from" in line
    assert "rule 1" in line and "rule 2" in line
    assert "every near miss below is 1 step away" in line
    assert "15 min less sleep" not in line or "closest crash" in line
    far = fl.plan_line({"rule3_co2_precision": 17, "rule1_sleep_eva": None}, crashed=False,
                       best_steps=[])
    assert "17 steps (4 h 15 min of change)" in far and "every near miss" not in far
    hit = fl.plan_line({"rule4_spe_eva": 0}, crashed=True, best_steps=[])
    assert "plan as written already breaks rule 4" in hit


def test_log_has_plan_line_before_guided_events():
    lines = fl.log_lines(results(), start_min=390)
    i_plan = next(i for i, ln in enumerate(lines) if ln["text"].startswith("[plan ]"))
    i_first = next(i for i, ln in enumerate(lines) if ln["kind"] in ("new", "best"))
    assert i_plan < i_first


def _b(lineage, ops=None):
    return {"round": 1, "fitness": 2.0, "steps_to_violation": 1, "nearest_rule": "cardiovascular",
            "ops": ops if ops is not None else lineage[-1:], "lineage": lineage, "pool": 5}


ZERO = {"sleep_cut_h": [0.0], "eva_extend_h": [0.0]}


def test_sensor_fault_only_lineage_is_plan_unchanged():
    line = fl._best_line(_b([{"op": "F", "params": {"faults": ["stuck"]}},
                             {"op": "F", "params": {"faults": ["spike"]}}]), ZERO)
    assert "plan unchanged (sensor fault only)" in line
    assert "change vs plan" not in line


def test_lineage_single_changes_in_full_repeats_counted_faults_flagged():
    lin = [{"op": "D", "params": {"warning": [690, 1439]}},
           {"op": "F", "params": {"faults": ["network_down"]}},
           {"op": "C", "params": {"task": "docking", "duration_min": 45, "fatigue_jump": 31.1}},
           {"op": "F", "params": {"faults": ["stuck"]}}]
    line = fl._best_line(_b(lin), ZERO)
    assert ("change vs plan: sleep and EVA as planned + SPE warning from 11:30 "     # lineage order
            "+ docking 0 h 45 min inserted, fatigue +31 (+ sensor faults)") in line
    assert "last step: sensor fault: stuck" in line


def test_deep_lineage_is_summarised_by_type_and_count():
    lin = ([{"op": "C", "params": {"task": "precision_manual", "duration_min": 45,
                                   "fatigue_jump": 26}}] * 6
           + [{"op": "C", "params": {"task": "docking", "duration_min": 60,
                                     "fatigue_jump": 21}}] * 3
           + [{"op": "B", "params": {"co2_excursion": 0.18, "work_task": "science_experiment",
                                     "work_extend_min": 75, "spo2_drop": 2.2}},
              {"op": "B", "params": {"co2_excursion": 0.08, "work_task": "science_experiment",
                                     "work_extend_min": 60, "spo2_drop": 2.1}}]
           + [{"op": "D", "params": {"warning": [900, 1439]}}] * 9
           + [{"op": "D", "params": {"warning": None}}]
           + [{"op": "F", "params": {"faults": ["stuck"]}}])
    line = fl._best_line(_b(lin), ZERO)
    assert ("change vs plan: sleep and EVA as planned + 9 tasks inserted (precision_manual x6, "
            "docking x3) + 2 CO2 excursions (up to +18%) + 10 solar particle events "
            "(9 with a warning) (+ sensor faults)") in line
    assert "22 mutations stacked" in line                 # 6 + 3 + 2 + 9 + 1 + 1


def test_lineage_sleep_and_eva_from_history_heat_from_last_a():
    lin = [{"op": "E", "params": {"sleep_cut_h": 0.7}},
           {"op": "A", "params": {"duration_delta_min": 30, "eva_heat_rate": 0.12}},
           {"op": "A", "params": {"duration_delta_min": 15, "eva_heat_rate": 0.21}}]
    line = fl._best_line(_b(lin), {"sleep_cut_h": [0.7], "eva_extend_h": [0.75]})
    assert "change vs plan: sleep -0.7 h, EVA +0.8 h, EVA heat 0.21 °C/h" in line


def test_simplest_near_miss_fewest_physical_changes():
    F = {"op": "F", "params": {"faults": ["stuck"]}}
    B = {"op": "B", "params": {"co2_excursion": 0.23, "work_task": "science_experiment",
                               "work_extend_min": 105, "spo2_drop": 4.8}}
    D = {"op": "D", "params": {"warning": None}}
    C = {"op": "C", "params": {"task": "docking", "duration_min": 60, "fatigue_jump": 21}}
    ev_ = [dict(_b([B, D, F, F, F]), round=3, fitness=3.35),       # 2 physical, 5 total
           dict(_b([C]), round=9, fitness=3.50, steps_to_violation=2),  # 1 physical, 2 steps
           dict(_b([B, D, C]), round=50, fitness=2.0),             # 3 physical
           dict(_b([D, B]), round=70, fitness=3.0)]                # 2 physical, lower fitness
    hist = {"sleep_cut_h": [0.0] * 100, "eva_extend_h": [0.0] * 100}
    line = fl.simplest_near_miss(ev_, hist, plan_fitness=4.08, plan_min_steps=1)
    assert "round 70" in line                          # nearest (1 step), fewest physical, then fitness
    assert "closer to a crash than the plan as written" in line
    assert "among the 4 recorded new lows" in line and "fitness 4.08 -> 3.00" in line
    assert "2 changes vs plan:" in line
    assert "\n   ① solar particle event (no warning)" in line            # numbered, grouped
    assert ("\n   ② CO2 excursion +23% -> science_experiment +1 h 45 min, SpO2 -4.8%"
            in line)
    assert "sleep and EVA as planned" in line
    assert "still 1 step (15 min of change) from a time-based rule, like the plan as written" in line
    assert fl.simplest_near_miss([], hist) is None


def test_simplest_near_miss_line_is_in_the_log():
    res = results()
    lin = [{"op": "C", "params": {"task": "docking", "duration_min": 60, "fatigue_jump": 21}}]
    g = dict(res["forecast"]["guided_track"])
    g["best_events"] = [dict(e, lineage=lin) for e in g["best_events"]]
    res["forecast"] = {**res["forecast"], "guided_track": g}
    lines = fl.log_lines(res, start_min=390)
    near = [ln for ln in lines if ln["kind"] == "near"]
    assert len(near) == 1 and near[0]["text"].startswith("[near ] fewest changes")
    best_t = max(ln["t"] for ln in lines if ln["kind"] == "best")
    assert near[0]["t"] > best_t                     # a summary after the search process


def test_speed_labels_now_in_header_average_in_perf_line():
    lines = fl.log_lines(results(), start_min=390)
    perf = next(ln["text"] for ln in lines if ln["text"].startswith("[perf ]"))
    assert perf.endswith("472 simulations/s average")
    final = fl.html_doc("final", lines, rounds=10000,
                        series=fl.header_series(GUIDED))
    head = final.split('<div class="body">')[0]
    assert "sim/s now" in head and "average" not in head
    assert "333 sim/s now" in head                          # last measured segment
    assert "sim/s now" in fl._JS                            # the live header keeps the label
