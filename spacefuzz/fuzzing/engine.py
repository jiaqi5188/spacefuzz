"""Two-track fuzzing engine (plan 3.6, 3.7) and the command-line forecast.

Probability track: N unbiased samples, each operator drawn independently with its daily
probability (config.PROB_OPERATOR_DAILY; a sample may be an ordinary day), no guidance.
Risk probability = crashed samples / N; time to crash = first time the cumulative crash
share reaches 5% (minutes from the check-in; shown only when risk > 5%); primary
vulnerable system; most common causal chain; trajectory percentiles for the UI.

Guided track: a seed pool, starting from the real day. Each round mutates the seed
closest to a crash (lowest evaluator fitness) or, with GUIDED_EXPLORE_PROB, a random
seed; a closer child joins the pool, a crash is recorded by (system, rule). Outputs the
crash types found, the minimum crashing perturbation and heatmap data (risk per cell of
"less sleep" x "longer EVA", each cell its own unbiased sampling).

The tracks never share samples: each has its own mutator and random generator, and
guided samples are oversampled on purpose, so only the probability track gives risk.
Coverage (rules, systems, task types, indicator ranges reached) is kept per track.

Rating (plan 1.3): green < 1% <= yellow <= 5% < red; red with time to crash <= 2 h is
CRITICAL; untrusted data is UNKNOWN and nothing is simulated. Out of model experience
(plan 3.12: check-in or plan outside the ranges the model was built on) is never green;
those warnings and the rules / systems the guided track never reached go to
coverage_warnings. Results are cached in
cache/ under a key of the inputs, the profile, the seed and the source code.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from spacefuzz.fuzzing import advice
from spacefuzz import config
from spacefuzz.core import evaluator as ev
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType

log = logging.getLogger("engine")

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / config.ENGINE_CACHE_DIR
STEP = config.STEP_MINUTES
# Everything committed that changes a result: engine sources and the data files the operators
# and the fluctuation fallback read. The local-only WESAD series is left out on purpose, so a
# fresh clone computes the same keys and finds the committed cache (cache_names covers it).
SOURCES = ("spacefuzz/fuzzing/engine.py", "spacefuzz/fuzzing/mutator.py",
           "spacefuzz/core/model.py", "spacefuzz/core/evaluator.py", "spacefuzz/core/state.py",
           "spacefuzz/core/plan.py", "spacefuzz/core/plan_types.py",
           "spacefuzz/readers/env_trace.py", "spacefuzz/readers/physio_replay.py",
           "spacefuzz/config.py", "spacefuzz/fuzzing/advice.py", "spacefuzz/core/quality_gate.py",
           "data/processed/op_b_co2_excursions.csv", "data/processed/spe_profile_msl_rad.csv",
           "data/processed/wesad_params.json")

OP_TEXT = {
    "A": "EVA stores heat faster (harder work or poorer cooling)",
    "B": "Cabin CO2 builds up",
    "C": "Extra docking / precision task (overload)",
    "D": "Solar particle event",
    "E": "Less sleep last night",
}
RULE_TEXT = {
    "cognitive": "Fatigue above 80 with SpO2 below 94%",
    "cardiovascular": "Sustained high heart rate with low HRV (or resting BP out of range)",
    "metabolic": "Low or steadily falling glucose around EVA",
    "radiation_dose": "EVA radiation dose above the single-EVA load signal",
    **ev.RULE_TEXT,
}


# ---------------------------------------------------------------- profile and rating

def profile(name: str | None = None) -> dict:
    name = name or os.environ.get("FUZZ_PROFILE") or config.FUZZ_PROFILE_DEFAULT
    if name not in config.FUZZ_PROFILES:
        raise ValueError(f"unknown FUZZ_PROFILE {name!r}; known: {sorted(config.FUZZ_PROFILES)}")
    return {"name": name, **config.FUZZ_PROFILES[name]}


def risk_rating(p: float, time_to_crash_min: int | None, untrusted: bool = False) -> str:
    """Plan 1.3. Exactly 1% and exactly 5% are MEDIUM (yellow is 1%-5%, ends included)."""
    if untrusted:
        return "UNKNOWN"
    if p > config.RISK_RED_ABOVE:
        if time_to_crash_min is not None and time_to_crash_min <= config.CRITICAL_TIME_TO_CRASH_MIN:
            return "CRITICAL"
        return "HIGH"
    return "MEDIUM" if p >= config.RISK_GREEN_BELOW else "LOW"


def time_to_crash(first_minutes, n: int) -> int | None:
    """First time the cumulative crash share reaches 5%; None unless risk > 5%."""
    crashed = sorted(m for m in first_minutes if m is not None)
    if n == 0 or len(crashed) / n <= config.RISK_RED_ABOVE:
        return None
    need = config.TIME_TO_CRASH_FRACTION * n
    for i, m in enumerate(crashed, start=1):
        if i >= need - 1e-9:
            return int(m)
    return None


# ---------------------------------------------------------------- helpers

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    return x


def _bin_label(edges, value: float) -> str:
    i = int(np.searchsorted(edges, value, side="right"))
    if i == 0:
        return f"<{edges[0]:g}"
    if i == len(edges):
        return f">={edges[-1]:g}"
    return f"{edges[i - 1]:g}-{edges[i]:g}"


class Coverage:
    """What a track reached: rules, systems, task types, indicator ranges, operators, gate."""

    def __init__(self):
        self.rules, self.systems, self.tasks = Counter(), Counter(), Counter()
        self.ranges = {f: set() for f in config.COVERAGE_BINS}
        self.ops, self.gate = Counter(), Counter()

    def add(self, out: pd.DataFrame, e: ev.Evaluation, sample: mu.Sample,
            gate_status: str | None) -> None:
        for v in {(v.rule, v.system) for v in e.violations}:
            self.rules[v[0]] += 1
            self.systems[v[1]] += 1
        self.tasks.update(set(out["task"].dropna()))
        for f, edges in config.COVERAGE_BINS.items():
            vals = out[f].to_numpy(dtype=float)
            idx = np.searchsorted(edges, vals[~np.isnan(vals)], side="right")
            for i in set(idx.tolist()):
                self.ranges[f].add(i)
        self.ops.update(o.op for o in sample.ops)
        if gate_status is not None:
            self.gate[gate_status] += 1

    def to_dict(self) -> dict:
        def label(f, i):
            edges = config.COVERAGE_BINS[f]
            return _bin_label(edges, edges[0] - 1 if i == 0 else edges[i - 1])
        return {"rules": dict(self.rules), "systems": dict(self.systems),
                "task_types": dict(self.tasks), "operators": dict(self.ops),
                "indicator_ranges": {f: [label(f, i) for i in sorted(s)]
                                     for f, s in self.ranges.items()},
                "quality_gate": dict(self.gate)}


def _gate_status(r: mu.Result) -> str | None:
    """Operator F samples: does the gate catch the corrupted readings?"""
    return qg.check(r.readings).status if r.sample.faults else None


_start_min = st.start_min               # used by app.py, ui/, midday.py and adjuster.py


def _chain(key, debt_h: float) -> list[str]:
    ops, rule = key
    chain = []
    if debt_h >= config.ACT_NOW_SLEEP_DEBT_H:
        chain.append(f"Sleep debt {debt_h:.1f} h over the last 3 nights")
    chain += [OP_TEXT[o] for o in ops]
    chain.append(RULE_TEXT.get(rule, rule))
    return chain


def _wesad() -> md.Fluctuation | None:
    """Real WESAD fluctuation, or the fitted-parameter pool without the series (model.py)."""
    return md.Fluctuation.load()


def _fluctuation_source() -> str:
    """Which pool Fluctuation.load() would use, without loading it (for the cache key)."""
    from spacefuzz.readers import physio_replay as pr
    if pr.EXPORT.exists():
        return "wesad"
    return "wesad_params" if md.WESAD_PARAMS.exists() else "none"


# ---------------------------------------------------------------- probability track

def probability_track(scenario: mu.Scenario, baseline: st.Baseline, n: int,
                      seed: int = config.RANDOM_SEED,
                      fluctuation: md.Fluctuation | None = None,
                      count_from: int | None = None) -> dict:
    """Risk = share of sampled days that crash. count_from (midday check): only violations
    at or after that minute count; the day is still simulated from the check-in."""
    m = mu.Mutator(scenario, baseline, seed=seed, fluctuation=fluctuation,
                   params={"MUT_DAILY_PROB": dict(config.PROB_OPERATOR_DAILY)},
                   not_before=count_from)          # midday: the morning is not mutated
    results = m.generate(n)
    start = _start_min(scenario.checkin)
    cov = Coverage()
    firsts, by_rule, by_system, chains = [], Counter(), Counter(), Counter()
    traj = {f: [] for f in config.ENGINE_TRAJECTORY_FIELDS}
    for r in results:
        e = ev.evaluate(r.output, r.sample.plan, r.sample.checkin, baseline, count_from)
        cov.add(r.output, e, r.sample, _gate_status(r))
        for f in traj:
            traj[f].append(r.output[f].to_numpy(dtype=float))
        if e.crashed:
            firsts.append(e.first_crash_minute - start)
            by_rule.update({v.rule for v in e.violations})
            by_system.update(e.systems())
            first = min(e.violations, key=lambda v: v.minute)
            chains[(tuple(o.op for o in r.sample.ops if o.op != "F"), first.rule)] += 1
        else:
            firsts.append(None)
    total = len(results)
    crashes = sum(f is not None for f in firsts)
    risk = crashes / total if total else 0.0
    grid = np.arange(config.STEPS_PER_DAY) * STEP
    crashed = np.array([f for f in firsts if f is not None], dtype=float)
    cum = [float((crashed <= t).sum() / total) if total else 0.0 for t in grid]
    debt = st.sleep_debt_h(scenario.checkin, baseline)
    out = {
        "track": "probability", "samples": total, "crashes": crashes, "count_from": count_from,
        "risk_probability": risk, "time_to_crash_minutes": time_to_crash(firsts, total),
        "by_rule": dict(by_rule), "by_system": dict(by_system),
        "primary_vulnerable_system": max(by_system, key=by_system.get) if by_system else "None",
        "causal_chain": _chain(chains.most_common(1)[0][0], debt) if chains else [],
        "cumulative_crash_share": cum,
        "trajectory": {f: {q: np.percentile(np.vstack(v), p, axis=0).round(2).tolist()
                           for q, p in (("p5", 5), ("p50", 50), ("p95", 95))}
                       for f, v in traj.items()} if total else {},
        "trajectory_minutes": (grid + start).tolist(),
        "traces": _traces(traj, firsts),
        "coverage": cov.to_dict(), "mutator": m.stats.summary(),
    }
    log.info("probability track: risk %.1f%% over %d samples", 100 * risk, total)
    return _jsonable(out)


def _traces(traj: dict, firsts: list) -> dict:
    """First ENGINE_TRACE_SAMPLES individual traces + crash flags (UI trajectory fan)."""
    k = min(config.ENGINE_TRACE_SAMPLES, len(firsts))
    out = {f: [np.round(x, 1).tolist() for x in traj[f][:k]] for f in config.ENGINE_TRACE_FIELDS}
    out["crashed"] = [int(f is not None) for f in firsts[:k]]
    return out


def _eva_minutes(plan: pl.Plan) -> int:
    return sum(t.duration_min for t in plan.tasks if t.type is PlanType.EVA)


# ---------------------------------------------------------------- guided track

@dataclasses.dataclass
class _Seed:
    scenario: mu.Scenario
    fitness: float
    history: tuple
    mutator: mu.Mutator | None = None
    lineage: tuple = ()              # every mutation from the plan to this seed, with params


def guided_track(scenario: mu.Scenario, baseline: st.Baseline, rounds: int,
                 seed: int = config.RANDOM_SEED + 1,
                 fluctuation: md.Fluctuation | None = None) -> dict:
    rng = np.random.default_rng(seed)
    params = {"MUT_OPERATOR_WEIGHTS": dict(config.MUT_OPERATOR_WEIGHTS)}
    start = _start_min(scenario.checkin)

    def mutator_for(sd: _Seed) -> mu.Mutator:
        if sd.mutator is None:
            sd.mutator = mu.Mutator(sd.scenario, baseline, rng=rng, fluctuation=fluctuation,
                                    params=params)
        return sd.mutator

    root = _Seed(scenario, 0.0, ())
    # scored like its children: with the same fluctuation overlay
    root.fitness = ev.evaluate(mutator_for(root).simulate_base(), scenario.plan,
                               scenario.checkin, baseline).fitness
    pool = [root]
    crash_types: dict = {}
    cov = Coverage()
    stats = mu.Stats()
    explore_rounds, last_new, plateau_round = 0, 0, None
    # Records for the UI (fuzz log, boundary line): one entry per round, new lowest-fitness
    # samples, and wall-clock checkpoints. Sleep cut and EVA extension are relative to the plan.
    hist = {k: [] for k in ("fitness", "crashed", "discarded", "pool", "sleep_cut_h",
                            "eva_extend_h")}
    best_events, timing, best = [], [], float("inf")
    sleep0, eva0 = scenario.checkin.sleep_last_night_h, _eva_minutes(scenario.plan)
    t0 = time.perf_counter()

    def record(rnd, child, fitness, crashed, discarded):
        hist["fitness"].append(fitness)
        hist["crashed"].append(int(crashed))
        hist["discarded"].append(int(discarded))
        hist["pool"].append(len(pool))
        hist["sleep_cut_h"].append(round(max(0.0, sleep0 - child.checkin.sleep_last_night_h), 3))
        hist["eva_extend_h"].append(round((_eva_minutes(child.plan) - eva0) / 60, 3))
        if rnd % config.GUIDED_TIMING_EVERY == 0 or rnd == rounds:
            timing.append([rnd, round(time.perf_counter() - t0, 3)])

    for rnd in range(1, rounds + 1):
        explore = rng.random() < config.GUIDED_EXPLORE_PROB
        explore_rounds += explore
        parent = pool[int(rng.integers(len(pool)))] if explore else min(pool, key=lambda s: s.fitness)
        m = mutator_for(parent)
        child = m.draw()
        out = m.checked(child, stats)
        if out is None:
            record(rnd, child, None, False, True)
            continue
        e = ev.evaluate(out, child.plan, child.checkin, baseline)
        r = mu.Result(child, out, mu.apply_faults(out, child.faults))
        cov.add(out, e, child, _gate_status(r))
        history = parent.history + tuple(o.op for o in child.ops)
        lineage = parent.lineage + tuple({"op": o.op, "params": o.params} for o in child.ops)
        if e.crashed:
            for v in e.violations:
                key = (v.system, v.rule)
                if key not in crash_types:
                    last_new = rnd
                    crash_types[key] = {
                        "system": v.system, "rule": v.rule, "first_round": rnd, "count": 0,
                        "operators": list(history), "minutes_after_check_in": v.minute - start,
                        "description": RULE_TEXT.get(v.rule, v.rule),
                        "last_mutation": [{"op": o.op, "params": o.params} for o in child.ops]}
                crash_types[key]["count"] += 1
        elif e.fitness < parent.fitness:
            pool.append(_Seed(mu.Scenario.from_sample(child), e.fitness, history,
                              lineage=lineage))
            if len(pool) > config.GUIDED_POOL_MAX:
                pool.sort(key=lambda s: s.fitness)
                del pool[config.GUIDED_POOL_MAX:]
        fit = round(e.fitness, 4)
        record(rnd, child, fit, e.crashed, False)
        if not e.crashed and fit < best:
            best = fit
            steps = [v for v in e.steps_to_violation.values() if v is not None]
            best_events.append({
                "round": rnd, "fitness": fit, "steps_to_violation": min(steps) if steps else None,
                "nearest_rule": min(e.distance, key=e.distance.get),
                "ops": [{"op": o.op, "params": o.params} for o in child.ops],
                "lineage": list(lineage), "pool": len(pool)})
        if plateau_round is None and rnd - last_new >= config.GUIDED_PLATEAU_ROUNDS:
            plateau_round = rnd
            log.info("guided track: plateau, no new crash type for %d rounds (round %d)",
                     config.GUIDED_PLATEAU_ROUNDS, rnd)
    out = {
        "track": "guided", "rounds": rounds, "explore_rounds": explore_rounds,
        "seeds": len(pool), "best_fitness": min(s.fitness for s in pool),
        "crash_types": sorted(crash_types.values(), key=lambda c: c["first_round"]),
        "plateau_round": plateau_round, "coverage": cov.to_dict(), "mutator": stats.summary(),
        "history": hist, "best_events": best_events, "timing": timing,
    }
    log.info("guided track: %d crash types in %d rounds", len(crash_types), rounds)
    return _jsonable(out)


# ---------------------------------------------------------------- heatmap, minimum perturbation

def _eva_index(plan: pl.Plan) -> int | None:
    return next((i for i, t in enumerate(plan.tasks) if t.type is PlanType.EVA), None)


def _variant(scenario: mu.Scenario, baseline: st.Baseline, sleep_cut_h: float,
             eva_extend_h: float) -> mu.Scenario:
    c = mu.apply_sleep_cut(scenario.checkin, baseline, sleep_cut_h) if sleep_cut_h else scenario.checkin
    p = scenario.plan
    i = _eva_index(p)
    if eva_extend_h and i is not None:
        p = dataclasses.replace(p, tasks=mu.resize_task(p.tasks, i, int(round(eva_extend_h * 60))))
    return dataclasses.replace(scenario, plan=p, checkin=c)


def heatmap(scenario: mu.Scenario, baseline: st.Baseline, samples: int, sleep_cut_h=None,
            eva_extend_h=None, seed: int = config.RANDOM_SEED + 2,
            fluctuation: md.Fluctuation | None = None) -> dict:
    """Risk per cell (less sleep x longer EVA); each cell is its own unbiased sampling."""
    rows = list(sleep_cut_h if sleep_cut_h is not None else config.HEATMAP_SLEEP_CUT_H)
    cols = list(eva_extend_h if eva_extend_h is not None else config.HEATMAP_EVA_EXTEND_H)
    if _eva_index(scenario.plan) is None:
        cols = [0.0]
    risk, det = [], []
    for i, s in enumerate(rows):
        risk.append([])
        det.append([])
        for j, e in enumerate(cols):
            cell = _variant(scenario, baseline, s, e)
            m = mu.Mutator(cell, baseline, seed=seed + 100 * i + j, fluctuation=fluctuation,
                           params={"MUT_DAILY_PROB": dict(config.PROB_OPERATOR_DAILY)})
            res = m.generate(samples)
            crashed = sum(ev.evaluate(r.output, r.sample.plan, r.sample.checkin, baseline).crashed
                          for r in res)
            risk[-1].append(crashed / len(res) if res else 0.0)
            det[-1].append(ev.evaluate(m.reference, cell.plan, cell.checkin, baseline).crashed)
    return _jsonable({"sleep_cut_h": rows, "eva_extend_h": cols, "risk": risk,
                      "unmutated_crash": det, "samples_per_cell": samples})


def _fmt_minutes(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h} h {mm} min" if h and mm else (f"{h} h" if h else f"{mm} min")


def min_crash_perturbation(scenario: mu.Scenario, baseline: st.Baseline,
                           schedule_only: bool = False) -> dict | None:
    """Smallest single deterministic change (15-min steps) that makes the day crash.
    schedule_only: only schedule slips (no "less sleep": sleep is measured at check-in)."""
    def crash_rule(sc):
        out = md.simulate(sc.plan, pl.apply_env_events(sc.trace, sc.plan), sc.checkin, baseline,
                          effects=sc.effects)
        e = ev.evaluate(out, sc.plan, sc.checkin, baseline)
        return (min(e.violations, key=lambda v: v.minute).rule if e.crashed else None), e

    base_rule, base_eval = crash_rule(scenario)
    if base_rule is not None:
        return {"minutes": 0, "change": "none", "rule": base_rule,
                "text": f"Already crashes as planned ({RULE_TEXT.get(base_rule, base_rule)})"}
    found = []
    steps = int(config.MIN_PERTURBATION_MAX_H * 60 // STEP)
    for k in range(1, 0 if schedule_only else steps + 1):    # less sleep
        rule, _ = crash_rule(_variant(scenario, baseline, k * STEP / 60, 0.0))
        if rule:
            found.append((k * STEP, "less sleep", rule, "{} less sleep turns this red"))
            break
    if _eva_index(scenario.plan) is not None:
        for k in range(1, steps + 1):                   # longer EVA
            rule, _ = crash_rule(_variant(scenario, baseline, 0.0, k * STEP / 60))
            if rule:
                found.append((k * STEP, "longer EVA", rule, "An EVA {} longer turns this red"))
                break
    phrases = {"rule2_exercise_eva": ("training later", "Training {} later turns this red"),
               "rule3_co2_precision": ("more high-CO2 work",
                                       "{} more high-CO2 work before docking turns this red"),
               "rule4_spe_eva": ("EVA moved into the SPE warning",
                                 "Moving the EVA {} into the SPE warning turns this red")}
    for rule, (change, text) in phrases.items():
        k = base_eval.steps_to_violation.get(rule)
        if k:
            found.append((k * STEP, change, rule, text))
    if not found:
        return None
    minutes, change, rule, text = min(found, key=lambda f: f[0])
    return {"minutes": minutes, "change": change, "rule": rule,
            "text": f"{text.format(_fmt_minutes(minutes))} ({RULE_TEXT.get(rule, rule)})"}


# ---------------------------------------------------------------- model experience (plan 3.12)

def out_of_experience(scenario: mu.Scenario, baseline: st.Baseline) -> list[str]:
    """Where today's check-in or plan leaves the ranges the model was built on."""
    out = []
    c = scenario.checkin
    for name, (lo, hi) in config.OOE_CHECKIN_RANGES.items():
        v = getattr(c, name)
        if v is not None and not lo <= v <= hi:
            out.append(f"Check-in {name} {v:g} is outside the model's experience ({lo:g}-{hi:g})")
    debt = st.sleep_debt_h(c, baseline)
    if debt > config.OOE_SLEEP_DEBT_MAX_H:
        out.append(f"3-night sleep debt {debt:.1f} h is beyond the model's experience "
                   f"(up to {config.OOE_SLEEP_DEBT_MAX_H:g} h)")
    co2 = float(pl.apply_env_events(scenario.trace, scenario.plan)["co2"].max())
    if co2 > config.ENV_CO2_MAX_OBSERVED_PPM:
        out.append(f"CO2 up to {co2:.0f} ppm is above the highest recorded cabin CO2 "
                   f"({config.ENV_CO2_MAX_OBSERVED_PPM} ppm); its effects are extrapolated")
    for t in scenario.plan.tasks:
        if t.type is PlanType.EVA and t.duration_min > config.OOE_EVA_MAX_H * 60:
            out.append(f"EVA of {t.duration_min / 60:.1f} h is longer than modelled EVAs "
                       f"(up to {config.OOE_EVA_MAX_H:g} h)")
    for t in scenario.plan.tasks:              # the name as written (quotes, odd characters kept)
        if t.type is PlanType.GENERIC_HIGH_LOAD and t.source_type != PlanType.GENERIC_HIGH_LOAD.value:
            out.append(f"Task type {t.source_type} is outside model experience "
                       f"(treated as high load)")
    return out


def experience_floor(rating: str, warnings: list[str]) -> str:
    """Out of experience: never green (plan 3.12)."""
    return "MEDIUM" if warnings and rating == "LOW" else rating


def uncovered(coverage: dict) -> str | None:
    """Rules and systems the guided track never reached ("not covered")."""
    rules = sorted(set(ev.SYSTEM) - set(coverage.get("rules", {})))
    systems = sorted(set(ev.SYSTEM.values()) - set(coverage.get("systems", {})))
    if not rules and not systems:
        return None
    parts = ([f"rules {', '.join(rules)}"] if rules else []) + \
        ([f"systems {', '.join(systems)}"] if systems else [])
    return "Not covered by the guided track: " + "; ".join(parts)


# ---------------------------------------------------------------- forecast

def _cache_key(scenario: mu.Scenario, prof: dict, baseline: st.Baseline | None = None) -> str:
    h = hashlib.sha256()
    # The baseline by value: its numbers and the bytes of its history table. Never str(table):
    # how pandas prints a table depends on the terminal, and the key must be the same on every
    # machine (a fresh clone has to find the committed cache).
    base = baseline or st.make_baseline()
    h.update(json.dumps({f.name: getattr(base, f.name) for f in dataclasses.fields(base)
                         if f.name != "history"}, sort_keys=True).encode())
    if base.history is not None:
        h.update(",".join(map(str, base.history.columns)).encode())
        h.update(pd.util.hash_pandas_object(base.history).to_numpy().tobytes())
    h.update(json.dumps(pl.to_dict(scenario.plan), sort_keys=True).encode())
    h.update(json.dumps(dataclasses.asdict(scenario.checkin), sort_keys=True, default=str).encode())
    h.update(pd.util.hash_pandas_object(scenario.trace).to_numpy().tobytes())
    h.update(json.dumps(prof, sort_keys=True).encode())
    h.update(json.dumps([config.RANDOM_SEED, config.PROB_OPERATOR_DAILY,
                         config.MUT_OPERATOR_WEIGHTS], sort_keys=True).encode())
    for src in SOURCES:
        h.update((ROOT / src).read_bytes())
    return h.hexdigest()


def cache_names(stem: str, key: str) -> list[str]:
    """[real-WESAD name, name for the source in use]: the same file when the WESAD series is
    there. Without it (a fresh clone) results carry the source in their name, so they never
    overwrite the committed real-WESAD results that share the inputs."""
    src = _fluctuation_source()
    real = f"{stem}{key[:12]}.json"
    return [real, real if src == "wesad" else f"{stem}{src}_{key[:12]}.json"]


def cache_files(stem: str, key: str) -> tuple[Path, Path]:
    """(file to read, file to write): the committed real-WESAD result is read when it exists."""
    real, own = (Path(CACHE_DIR) / n for n in cache_names(stem, key))
    return (real if real.exists() else own), own


def forecast(plan_path, scenario: mu.Scenario | None = None, profile_name: str | None = None,
             use_cache: bool = True, baseline: st.Baseline | None = None) -> dict:
    """Full forecast for one plan and its check-in (plan 3.10 fields + both tracks)."""
    name = Path(plan_path).stem
    scenario = scenario or mu.Scenario.load(name)
    baseline = baseline or st.make_baseline()
    prof = profile(profile_name)
    gate = qg.check(checkin=scenario.checkin)
    items = advice.act_now(scenario.checkin, baseline, gate=gate)
    common = {"plan": name, "profile": prof["name"], "seed": config.RANDOM_SEED,
              "data_quality": gate.status, "missing": gate.missing,
              "advice_now": advice.advice_now(items),
              "action_restrictions": advice.action_restrictions(items),
              "holds": {t.value: code for t, code in advice.holds(items).items()}}
    try:
        qg.ensure_trusted(gate)
    except qg.UntrustedDataError:
        return {**common, "risk_rating": "UNKNOWN", "escalation": "self",
                "risk_probability": None, "time_to_crash_minutes": None,
                "primary_vulnerable_system": None, "causal_chain": [],
                "min_crash_perturbation": None, "message": gate.message,
                "from_cache": False}

    read, path = cache_files(f"{name}_{prof['name']}_", _cache_key(scenario, prof, baseline))
    if use_cache and read.exists():
        return {**json.loads(read.read_text()), "from_cache": True}

    fl = _wesad()
    t0 = time.perf_counter()
    prob = probability_track(scenario, baseline, prof["probability_samples"], fluctuation=fl)
    guided = guided_track(scenario, baseline, prof["guided_rounds"], fluctuation=fl)
    heat = heatmap(scenario, baseline, prof["heatmap_samples"], fluctuation=fl)
    minp = min_crash_perturbation(scenario, baseline)
    seconds = time.perf_counter() - t0
    sims = (prob["samples"] + guided["rounds"]
            + len(heat["risk"]) * len(heat["eva_extend_h"]) * (heat["samples_per_cell"] + 1))
    ooe = out_of_experience(scenario, baseline)
    rating = experience_floor(risk_rating(prob["risk_probability"], prob["time_to_crash_minutes"]),
                              ooe)
    not_covered = uncovered(guided["coverage"])
    escalation = min((advice.RATING_ESCALATION[rating], advice.overall_escalation(items)),
                     key=advice.ESCALATION_ORDER.get)
    result = {
        **common, "risk_rating": rating, "escalation": escalation,
        "risk_probability": prob["risk_probability"],
        "time_to_crash_minutes": prob["time_to_crash_minutes"],
        "primary_vulnerable_system": prob["primary_vulnerable_system"],
        "causal_chain": prob["causal_chain"], "min_crash_perturbation": minp,
        "coverage_warnings": ooe + ([not_covered] if not_covered else []),
        "probability_track": prob, "guided_track": guided, "heatmap": heat,
        "performance": {"seconds": round(seconds, 2), "simulations": sims,
                        "simulations_per_second": round(sims / seconds, 1)},
    }
    result = _jsonable(result)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    log.info("forecast cached in %s", path)
    return {**result, "from_cache": False}


def _synthetic(name: str) -> mu.Scenario:
    """Plan and check-in with a flat environment trace (no raw data needed)."""
    idx = pd.Index(np.arange(config.STEPS_PER_DAY) * STEP, name="minute")
    trace = pd.DataFrame({"co2": 3000.0, "temp": 22.5, "humidity": 45.0,
                          "dose_rate": config.RAD_BASELINE_MARS_UGY_H}, index=idx)
    return mu.Scenario(pl.load_plan(f"data/plans/{name}.json"), trace, st.load_checkin(name))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SpaceFuzz forecast for one plan.")
    ap.add_argument("--plan", required=True, help="e.g. data/plans/main_sleep_debt_eva.json")
    ap.add_argument("--profile", choices=sorted(config.FUZZ_PROFILES), default=None,
                    help="overrides FUZZ_PROFILE (default local)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="flat environment, no raw data")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    name = Path(args.plan).stem
    r = forecast(args.plan, scenario=_synthetic(name) if args.synthetic else None,
                 profile_name=args.profile, use_cache=not args.no_cache)
    print(f"plan {name} | profile {r['profile']} | data quality {r['data_quality']}")
    if r["risk_rating"] == "UNKNOWN":
        print(f"risk UNKNOWN: {r['message']}")
        return
    ttc = r["time_to_crash_minutes"]
    print(f"risk {r['risk_probability']:.1%} -> {r['risk_rating']} (escalation {r['escalation']})"
          + (f", time to crash {ttc} min after check-in" if ttc is not None else ""))
    print(f"primary system: {r['primary_vulnerable_system']}")
    print("causal chain: " + " -> ".join(r["causal_chain"]))
    if r["min_crash_perturbation"]:
        print(f"min crash perturbation: {r['min_crash_perturbation']['text']}")
    p = r["probability_track"]
    print("by rule: " + ", ".join(f"{k} {v / p['samples']:.1%}" for k, v in
                                  sorted(p["by_rule"].items(), key=lambda kv: -kv[1])))
    g = r["guided_track"]
    print(f"guided: {len(g['crash_types'])} crash types in {g['rounds']} rounds: "
          + ", ".join(f"{c['system']}/{c['rule']}" for c in g["crash_types"]))
    perf = r["performance"]
    print(f"{perf['simulations']} simulations in {perf['seconds']} s = "
          f"{perf['simulations_per_second']} simulations/s"
          + (" (cached result)" if r["from_cache"] else ""))


if __name__ == "__main__":
    main()
