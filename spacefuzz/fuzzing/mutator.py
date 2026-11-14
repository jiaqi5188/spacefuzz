"""Mutation operators A-F (plan 1.3, 3.3): perturb one day's inputs before a simulation.

Two ways to choose a sample's operators, among the ones that apply to the plan (A needs
an EVA, C a free awake slot):
- guided track (default): 1-3 distinct operators stacked (config.MUT_OPS_PER_SAMPLE);
- probability track (params MUT_DAILY_PROB): each operator independently with the chance
  it happens on a given day (config.PROB_OPERATOR_DAILY), so a sample may have none.
Magnitudes are drawn from the distributions in config (OP_*):

- A  EVA physical load: the EVA stores heat faster (harder work or poorer cooling), so
     core temperature keeps rising through a continuous EVA and heart rate follows it
     above the 0.6 degC heat-strain line (model.Effects.eva_heat_rate); EVA 1-2 h longer
     or shorter. A longer task pushes the following tasks later; a
     pushed sleep keeps its end time (it gets shorter), as wake-up time is fixed.
- B  cabin CO2 build-up: the raw cabin CO2 scaled by (1 + e), e drawn from the 34
     observed daily excursions (A1, data/processed/op_b_co2_excursions.csv) by their
     observed frequency, so the real 31-40% tail stays; plan high_co2 events are applied
     after scaling (CO2 = max(event, trace)). A science / maintenance task runs up to
     4 h longer, and SpO2 drops by up to 5 points while CO2 is above ENV_CO2_HIGH_PPM.
- C  multi-task overload: a docking or precision_manual task is inserted in the awake
     part of the day (never over EVA or sleep; the task it lands in is cut around it),
     with a cognitive fatigue jump of 20-40 points when it starts.
- D  solar particle event: the real MSL-RAD event (A2, data/processed/spe_profile_msl_rad.csv,
     excess over baseline per 15 min) is added to the dose rate from a random onset and a
     random point in the event (onset, peak or decay), scaled by the habitat's usual dose
     rate. An spe_warning env event covers the part above 1.25x baseline, and fatigue
     rises slowly while it lasts. GOES flares are Earth-orbit data used as a proxy.
- E  isolation and sleep debt: last night 0.5-3 h shorter; in some samples all three
     nights under 6 h (never longer than recorded). The measured reaction time slows by
     what the model gives for the extra debt, and self-rated stress goes up by one.
- F  sensor faults: missing readings, signal dropout, stuck values, spikes, device
     offline. F changes the wearable readings only (Result.readings, the input of the
     quality gate), never the body (Result.output): it tests the system, not the crew.

Causal-chain check (plan 2.3): every sample is simulated and compared with the unmutated
reference run; each applied operator must move its chain in the right direction
(check_chains). Samples that do not are discarded and counted; the discard rate is
logged. Checks use the model columns (hr_model, rmssd_model, ...), not the real-fluctuation
overlay, and aggregate values (peaks, totals), because stacked operators move tasks in time.

Reproducible: one numpy Generator seeded with config.RANDOM_SEED drives every draw,
including the fluctuation overlay. The base scenario is never modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from spacefuzz import config
from spacefuzz.readers import env_trace
from spacefuzz.core import model as md
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType

log = logging.getLogger("mutator")

STEP = config.STEP_MINUTES
OPERATORS = ("A", "B", "C", "D", "E", "F")
APPLY_ORDER = ("E", "A", "B", "C", "D", "F")   # C after A / B so its slot is final


# ---------------------------------------------------------------- data records

@dataclass(frozen=True)
class Scenario:
    """The unmutated day. `trace` is the raw environment trace, before plan env events.
    `effects` carries model effects of earlier mutations when a mutated sample is used
    as a seed (guided track); None for a real day."""
    plan: pl.Plan
    trace: pd.DataFrame
    checkin: st.CheckIn
    effects: md.Effects | None = None

    @classmethod
    def from_sample(cls, sample: "Sample") -> "Scenario":
        """A mutated sample as the starting point of further mutation."""
        return cls(plan=sample.plan, trace=sample.raw_trace, checkin=sample.checkin,
                   effects=sample.effects)

    @classmethod
    def load(cls, name: str, cabin_start=config.ENV_DEFAULT_CABIN_START,
             radiation_start=config.ENV_DEFAULT_RADIATION_START) -> "Scenario":
        p = pl.load_plan(f"data/plans/{name}.json")
        radiation = pl.RADIATION_FOR_ENVIRONMENT[p.environment]
        trace = env_trace.build_trace(cabin_start, radiation_start, hours=24, radiation=radiation)
        return cls(plan=p, trace=trace, checkin=st.load_checkin(name))


@dataclass(frozen=True)
class OpRecord:
    op: str
    params: dict


@dataclass(frozen=True)
class Fault:
    kind: str                       # missing | dropout | stuck | spike | network_down
    fields: tuple
    start_min: int | None = None    # dropout / stuck / network_down window (absolute minutes)
    duration_min: int | None = None
    indices: tuple = ()             # missing / spike: row labels of the readings frame
    factor: float = 1.0             # spike multiplier


@dataclass(frozen=True)
class Sample:
    ops: tuple                      # OpRecord per applied operator, in APPLY_ORDER
    plan: pl.Plan
    trace: pd.DataFrame             # mutated trace with plan env events applied
    checkin: st.CheckIn
    effects: md.Effects
    faults: tuple = ()
    raw_trace: pd.DataFrame | None = None   # mutated trace before plan env events (for seeds)


@dataclass(frozen=True)
class Result:
    sample: Sample
    output: pd.DataFrame            # simulated day (the body)
    readings: pd.DataFrame          # what the wearable reports (output + operator F faults)


@dataclass
class Stats:
    attempted: int = 0
    accepted: int = 0
    discarded: int = 0
    by_reason: Counter = field(default_factory=Counter)

    @property
    def discard_rate(self) -> float:
        return self.discarded / self.attempted if self.attempted else 0.0

    def summary(self) -> str:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(self.by_reason.items())) or "none"
        return (f"mutator: {self.attempted} drawn, {self.accepted} accepted, "
                f"{self.discarded} discarded (discard rate {self.discard_rate:.1%}; {reasons})")


# ---------------------------------------------------------------- plan edits

def _sorted(tasks) -> list:
    return sorted(tasks, key=lambda t: t.start_min)


def resize_task(tasks, index: int, delta_min: int) -> list:
    """Change one task's duration. Later tasks it now overlaps are pushed back in turn; a
    pushed sleep keeps its end time and gets shorter (dropped if under one step)."""
    ts = _sorted(tasks)
    t = ts[index]
    ts[index] = dataclasses.replace(t, duration_min=max(STEP, t.duration_min + delta_min))
    end = ts[index].end_min
    for j in range(index + 1, len(ts)):
        u = ts[j]
        if u.start_min >= end:
            break
        if u.type is PlanType.SLEEP:
            if u.end_min - end < STEP:       # swallowed (e.g. a nap): drop it, keep pushing
                ts[j] = None
                continue
            ts[j] = dataclasses.replace(u, start_min=end, duration_min=u.end_min - end)
            break
        ts[j] = dataclasses.replace(u, start_min=end)
        end = ts[j].end_min
    return _sorted(t for t in ts if t is not None)


def insert_task(tasks, typ: PlanType, start: int, duration: int) -> list:
    """Insert a task; any task it lands in is cut around it (kept before and after)."""
    end = start + duration
    out = []
    for t in tasks:
        if t.end_min <= start or t.start_min >= end:
            out.append(t)
            continue
        if t.start_min < start:
            out.append(dataclasses.replace(t, duration_min=start - t.start_min))
        if t.end_min > end:
            out.append(dataclasses.replace(t, start_min=end, duration_min=t.end_min - end))
    out.append(pl.Task(start, duration, typ, typ.value))
    return _sorted(out)


def _replace_plan(plan: pl.Plan, tasks=None, events=None) -> pl.Plan:
    new = dataclasses.replace(plan, tasks=list(tasks if tasks is not None else plan.tasks),
                              events=list(events if events is not None else plan.events))
    for a, b in zip(new.tasks, new.tasks[1:]):
        if a.end_min > b.start_min:
            raise pl.PlanError(f"mutated plan overlaps: {a} / {b}")
    return new


def _with_sleep(c: st.CheckIn, baseline: st.Baseline, nights: tuple) -> st.CheckIn:
    """Check-in with new sleep nights (last = last night); the measured reaction time
    slows by what the model gives for the extra debt (chain 5)."""
    new = dataclasses.replace(c, sleep_last_night_h=nights[-1], sleep_last_3_nights_h=nights)
    if c.reaction_time_ms is not None:
        f = st.initial_state(c, baseline).cognitive_fatigue
        d0, d1 = st.sleep_debt_h(c, baseline), st.sleep_debt_h(new, baseline)
        new = dataclasses.replace(new, reaction_time_ms=c.reaction_time_ms + (
            st.model_reaction_ms(baseline, d1, f) - st.model_reaction_ms(baseline, d0, f)))
    return new


def apply_sleep_cut(c: st.CheckIn, baseline: st.Baseline, hours: float) -> st.CheckIn:
    """Last night `hours` shorter (deterministic part of operator E, for the heatmap)."""
    nights = tuple(c.sleep_last_3_nights_h[:-1]) + (max(0.0, c.sleep_last_night_h - hours),)
    return _with_sleep(c, baseline, nights)


# ---------------------------------------------------------------- sensor faults

def apply_faults(readings: pd.DataFrame, faults) -> pd.DataFrame:
    """Corrupt a copy of the wearable readings (operator F)."""
    out = readings.copy()
    minutes = out["minute"].to_numpy()
    for f in faults:
        cols = list(f.fields)
        if f.kind in ("dropout", "stuck", "network_down"):
            on = (minutes >= f.start_min) & (minutes < f.start_min + f.duration_min)
            if f.kind == "stuck":
                for c in cols:
                    first = out.loc[on, c].iloc[0] if on.any() else np.nan
                    out.loc[on, c] = first
            else:
                out.loc[on, cols] = np.nan
        elif f.kind == "missing":
            out.loc[list(f.indices), cols] = np.nan
        elif f.kind == "spike":
            out.loc[list(f.indices), cols] = out.loc[list(f.indices), cols] * f.factor
        else:
            raise ValueError(f"unknown fault kind {f.kind!r}")
    return out


# ---------------------------------------------------------------- causal-chain check

def clock_events(a: int, b: int) -> list[tuple[int, int]]:
    """An event from minute a to b of the simulated window (b may pass midnight) as plan-day
    clock intervals: plan events are clock times, like the trace rows (model.simulate)."""
    day = 24 * 60
    start, length = a % day, min(b - a, day)
    if start + length <= day:
        return [(start, start + length)]
    return [(start, day), (0, start + length - day)]


def check_chains(out: pd.DataFrame, ref: pd.DataFrame, ops) -> list[str]:
    """Directions each applied operator must produce versus the unmutated run (plan 2.3).

    ops: the sample's operator records (OpRecord) or bare operator names. A shortened EVA
    (operator A with a negative duration change) may lower the EVA peaks and raise the day's
    SpO2 low point, so the EVA-peak checks and B's SpO2 check are skipped then (only then).
    Returns a list of violations, each starting with the chain ("chain5: ...")."""
    tol = config.MUT_CHAIN_TOLERANCE
    v = []
    params = {getattr(o, "op", o): getattr(o, "params", {}) or {} for o in ops}
    ops = set(params)
    eva_shortened = params.get("A", {}).get("duration_delta_min", 0) < 0
    if "E" in ops:   # chain 5: more sleep debt -> resting HR up, HRV down, reaction slower
        a, b = out.iloc[0], ref.iloc[0]
        if a["sleep_debt_h"] < b["sleep_debt_h"] - tol:
            v.append(f"chain5: sleep debt fell ({b['sleep_debt_h']:.1f} -> {a['sleep_debt_h']:.1f} h)")
        if a["hr_model"] < b["hr_model"] - tol:
            v.append("chain5: resting heart rate fell")
        if a["rmssd_model"] > b["rmssd_model"] + tol:
            v.append("chain5: resting HRV rose")
        if a["reaction_time_ms"] < b["reaction_time_ms"] - tol:
            v.append("chain5: reaction time got faster")
    if "A" in ops and not eva_shortened:   # chain 2: longer, hotter EVA -> higher HR, warmer
        on, ron = out["activity"] == "eva", ref["activity"] == "eva"
        if on.any() and ron.any():
            if out.loc[on, "hr_model"].max() < ref.loc[ron, "hr_model"].max() - tol:
                v.append("chain2: EVA heart rate peak fell")
            if out.loc[on, "core_temp"].max() < ref.loc[ron, "core_temp"].max() - tol:
                v.append("chain2: EVA core temperature peak fell")
    if "B" in ops:   # chain 2: more CO2 -> lower SpO2
        if out["co2"].max() < ref["co2"].max() - tol:
            v.append("chain2: CO2 peak fell")
        if not eva_shortened and out["spo2"].min() > ref["spo2"].min() + tol:
            v.append("chain2: SpO2 low point rose under more CO2")
    if "C" in ops:   # overload -> more cognitive fatigue
        if out["cognitive_fatigue"].max() < ref["cognitive_fatigue"].max() - tol:
            v.append("chain1: fatigue peak fell under overload")
    if "D" in ops:   # chain 1: radiation alarm -> stress up; more dose
        if out["dose_rate"].sum() < ref["dose_rate"].sum() - tol:
            v.append("chain1: dose rate fell under a particle event")
        if out["stress_level"].max() < ref["stress_level"].max() - tol:
            v.append("chain1: stress peak fell under a particle event")
    return v


# ---------------------------------------------------------------- data files

@lru_cache(maxsize=None)
def _excursions() -> np.ndarray:
    return pd.read_csv(pl.ROOT / config.OP_B_CO2_EXCURSIONS_PATH)["excursion"].to_numpy(float)


@lru_cache(maxsize=None)
def _spe_profile() -> np.ndarray:
    return pd.read_csv(pl.ROOT / config.OP_D_SPE_PROFILE_PATH)["excess_over_baseline"].to_numpy(float)


# ---------------------------------------------------------------- the mutator

class Mutator:
    """Draws mutated samples of one scenario. `params` overrides config names (tests)."""

    def __init__(self, base: Scenario, baseline: st.Baseline, seed: int = config.RANDOM_SEED,
                 fluctuation: md.Fluctuation | None = None, params: dict | None = None,
                 rng: np.random.Generator | None = None, not_before: int | None = None):
        """not_before (minute of the plan day): nothing before it may change. The midday
        check passes its cut-off: the morning has been flown, so no operator may edit a task,
        insert one or start an event before it."""
        self.base = base
        self.baseline = baseline
        self.fluctuation = fluctuation
        self.params = dict(params or {})
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.stats = Stats()
        fx = base.effects or md.Effects()
        if fx.dose_median is None:      # a seed keeps the root day's usual dose rate
            fx = dataclasses.replace(fx, dose_median=float(np.nanmedian(base.trace["dose_rate"])))
        self.base_effects = fx
        self.base_trace = pl.apply_env_events(base.trace, base.plan)
        self.reference = md.simulate(base.plan, self.base_trace, base.checkin, baseline,
                                     effects=self.base_effects)
        self.start_min = st.start_min(base.checkin)
        self.not_before = self.start_min if not_before is None else max(self.start_min,
                                                                         not_before)

    # -- helpers
    def cfg(self, name: str):
        return self.params.get(name, getattr(config, name))

    def _uniform(self, name: str) -> float:
        lo, hi = self.cfg(name)
        return float(self.rng.uniform(lo, hi))

    def _steps_min(self, hours: float) -> int:
        return int(round(hours * 60 / STEP)) * STEP

    def _awake_end(self, plan: pl.Plan) -> int:
        """Bedtime: the first sleep longer than a nap (a nap is blocked, not the night)."""
        nap_max = config.RULE1_NAP_MAX_H * 60
        sleeps = [t.start_min for t in plan.tasks if t.type is PlanType.SLEEP
                  and t.start_min >= self.start_min and t.duration_min > nap_max]
        return min(sleeps) if sleeps else self.start_min + 24 * 60

    def _first_eva(self, plan: pl.Plan) -> int | None:
        """Index of the first EVA that has not started yet (not_before), or None."""
        return next((k for k, t in enumerate(plan.tasks)
                     if t.type is PlanType.EVA and t.start_min >= self.not_before), None)

    def _free_slots(self, plan: pl.Plan, duration: int) -> list[int]:
        blocked = [(t.start_min, t.end_min) for t in plan.tasks
                   if t.type in (PlanType.EVA, PlanType.SLEEP)]
        return [s for s in range(self.not_before, self._awake_end(plan) - duration + 1, STEP)
                if all(s + duration <= a or s >= b for a, b in blocked)]

    def applicable(self, plan: pl.Plan, weights: dict | None = None) -> list[str]:
        weights = self.cfg("MUT_OPERATOR_WEIGHTS") if weights is None else weights
        ops = []
        for op in OPERATORS:
            if weights.get(op, 0.0) <= 0:
                continue
            if op == "A" and self._first_eva(plan) is None:
                continue
            if op == "C" and not self._free_slots(plan, STEP):
                continue
            ops.append(op)
        return ops

    # -- operators
    def _op_a(self, plan, ctx):
        i = self._first_eva(plan)
        sign = 1 if self.rng.random() < self.cfg("OP_A_LONGER_PROB") else -1
        delta = sign * max(STEP, self._steps_min(self._uniform("OP_A_EVA_DURATION_DELTA_H")))
        delta = max(delta, STEP - plan.tasks[i].duration_min)
        ctx["effects"]["eva_heat_rate"] = self._uniform("OP_A_EVA_HEAT_RATE_C_PER_H")
        return (_replace_plan(plan, resize_task(plan.tasks, i, delta)),
                {"duration_delta_min": delta, "eva_heat_rate": ctx["effects"]["eva_heat_rate"]})

    def _op_b(self, plan, ctx):
        e = float(self.rng.choice(_excursions()))
        ctx["trace"] = ctx["trace"].assign(co2=ctx["trace"]["co2"] * (1.0 + e))
        work = [k for k, t in enumerate(plan.tasks)
                if t.type.value in self.cfg("OP_B_WORK_TYPES") and t.start_min >= self.not_before]
        params = {"co2_excursion": e, "work_task": None, "work_extend_min": 0}
        if work:
            k = work[int(self.rng.integers(len(work)))]
            ext = self._steps_min(self._uniform("OP_B_WORK_EXTEND_H"))
            params.update(work_task=plan.tasks[k].source_type, work_extend_min=ext)
            plan = _replace_plan(plan, resize_task(plan.tasks, k, ext))
        ctx["effects"]["spo2_drop_high_co2"] = self._uniform("OP_B_SPO2_DROP_PCT")
        params["spo2_drop"] = ctx["effects"]["spo2_drop_high_co2"]
        return plan, params

    def _op_c(self, plan, ctx):
        typ = PlanType(self.rng.choice(list(self.cfg("OP_C_INSERT_TYPES"))))
        dur = max(STEP, self._steps_min(self._uniform("OP_C_TASK_DURATION_H")))
        slots = self._free_slots(plan, dur)
        while not slots and dur > STEP:
            dur -= STEP
            slots = self._free_slots(plan, dur)
        if not slots:                  # no free time left today: nothing to insert
            return plan, {"task": None, "start_min": None, "duration_min": 0, "fatigue_jump": 0.0}
        start = int(self.rng.choice(slots))
        jump = self._uniform("OP_C_FATIGUE_JUMP")
        ctx["effects"]["fatigue_jumps"] = tuple(ctx["effects"]["fatigue_jumps"]) + ((start, jump),)
        return (_replace_plan(plan, insert_task(plan.tasks, typ, start, dur)),
                {"task": typ.value, "start_min": start, "duration_min": dur, "fatigue_jump": jump})

    def _op_d(self, plan, ctx):
        prof = _spe_profile()
        offset = int(self.rng.integers(len(prof)))
        first = (self.not_before - self.start_min) // STEP     # 0 except at the midday check
        onset = int(self.rng.integers(first, config.STEPS_PER_DAY))
        n = min(config.STEPS_PER_DAY - onset, len(prof) - offset)
        seg = prof[offset:offset + n]
        trace = ctx["trace"].copy()
        col = trace.columns.get_loc("dose_rate")
        k0 = self.start_min // STEP
        rows = [(k0 + onset + k) % len(trace) for k in range(n)]
        trace.iloc[rows, col] = trace.iloc[rows, col].to_numpy() + seg * self.base_effects.dose_median
        ctx["trace"] = trace
        rate = self._uniform("OP_D_FATIGUE_PER_H")
        above = np.flatnonzero(seg >= self.cfg("OP_D_SPE_THRESHOLD_EXCESS"))
        warning, events = None, list(plan.events)
        if above.size:
            a = self.start_min + (onset + int(above[0])) * STEP
            b = self.start_min + (onset + int(above[-1]) + 1) * STEP
            ctx["effects"]["fatigue_rate"] = tuple(ctx["effects"]["fatigue_rate"]) + ((a, b, rate),)
            warning = (a, b)
            events += [pl.EnvEvent(x, y, PlanType.SPE_WARNING) for x, y in clock_events(a, b)]
        return (_replace_plan(plan, events=events),
                {"onset_step": onset, "profile_offset": offset, "fatigue_per_h": rate,
                 "warning": warning})

    def _op_e(self, plan, ctx):
        c = ctx["checkin"]
        cut = self._uniform("OP_E_SLEEP_CUT_H")
        last = max(0.0, c.sleep_last_night_h - cut)
        nights = list(c.sleep_last_3_nights_h[:-1])
        chronic = bool(self.rng.random() < self.cfg("OP_E_CHRONIC_PROB"))
        if chronic:
            lo, hi = self.cfg("OP_E_CHRONIC_NIGHT_H")
            nights = [min(n, float(self.rng.uniform(lo, hi))) for n in nights]
            last = min(last, float(self.rng.uniform(lo, hi)))
        new = _with_sleep(c, self.baseline, tuple(nights) + (last,))
        new = dataclasses.replace(new, stress=min(5, c.stress + self.cfg("OP_E_STRESS_SELF_DELTA")))
        ctx["checkin"] = new
        return plan, {"sleep_cut_h": cut, "chronic": chronic,
                      "nights": new.sleep_last_3_nights_h}

    def _op_f(self, plan, ctx):
        fields = tuple(self.cfg("OP_F_FIELDS"))
        lo, hi = self.cfg("OP_F_FAULTS_PER_SAMPLE")
        count = int(self.rng.integers(lo, hi + 1))
        kinds = list(self.rng.choice(list(self.cfg("OP_F_KINDS")), size=count, replace=False))
        n = config.STEPS_PER_DAY
        faults = []
        for kind in kinds:
            field_ = (str(self.rng.choice(list(fields))),)
            if kind == "missing":
                frac = self._uniform("OP_F_MISSING_FRACTION")
                idx = np.sort(self.rng.choice(n, size=max(1, int(round(frac * n))), replace=False))
                faults.append(Fault(kind, field_, indices=tuple(int(i) for i in idx)))
            elif kind == "spike":
                lo_, hi_ = self.cfg("OP_F_SPIKES")
                k = int(self.rng.integers(lo_, hi_ + 1))
                idx = np.sort(self.rng.choice(n, size=k, replace=False))
                faults.append(Fault(kind, field_, indices=tuple(int(i) for i in idx),
                                    factor=self._uniform("OP_F_SPIKE_FACTOR")))
            else:
                name = {"dropout": "OP_F_DROPOUT_MIN", "stuck": "OP_F_STUCK_MIN",
                        "network_down": "OP_F_NETWORK_DOWN_MIN"}[kind]
                dur = max(STEP, self._steps_min(self._uniform(name) / 60))
                start = self.start_min + STEP * int(self.rng.integers(n - dur // STEP + 1))
                faults.append(Fault(kind, fields if kind == "network_down" else field_,
                                    start, dur))
        ctx["faults"] = tuple(faults)
        return plan, {"faults": [f.kind for f in faults]}

    # -- sampling
    def _choose(self) -> set:
        daily = self.params.get("MUT_DAILY_PROB")
        if daily is not None:        # probability track: each operator independently, may be none
            usable = set(self.applicable(self.base.plan, weights=daily))
            draws = {op: self.rng.random() for op in OPERATORS}   # fixed stream per sample
            return {op for op in OPERATORS if op in usable and draws[op] < daily.get(op, 0.0)}
        ops = self.applicable(self.base.plan)
        if not ops:
            raise ValueError("no applicable operator for this plan and these weights")
        sizes = list(self.cfg("MUT_OPS_PER_SAMPLE"))
        k = int(self.rng.choice(sizes, p=np.asarray(self.cfg("MUT_OPS_PER_SAMPLE_WEIGHTS"))))
        k = min(k, len(ops))
        w = np.array([self.cfg("MUT_OPERATOR_WEIGHTS")[o] for o in ops], dtype=float)
        return set(self.rng.choice(ops, size=k, replace=False, p=w / w.sum()))

    def draw(self) -> Sample:
        """One mutated sample (no simulation). Guided mode: 1-3 distinct applicable operators;
        daily mode (params MUT_DAILY_PROB): each operator with its daily probability."""
        chosen = self._choose()
        ctx = {"trace": self.base.trace, "checkin": self.base.checkin, "faults": (),
               "effects": dataclasses.asdict(self.base_effects)}
        plan = self.base.plan
        records = []
        for op in APPLY_ORDER:
            if op in chosen:
                plan, params = getattr(self, f"_op_{op.lower()}")(plan, ctx)
                records.append(OpRecord(op, params))
        return Sample(ops=tuple(records), plan=plan,
                      trace=pl.apply_env_events(ctx["trace"], plan), checkin=ctx["checkin"],
                      effects=md.Effects(**ctx["effects"]), faults=ctx["faults"],
                      raw_trace=ctx["trace"])

    def simulate_base(self) -> pd.DataFrame:
        """The unmutated day with the same fluctuation overlay as the samples (a seed's fitness
        must be comparable with its children's)."""
        return md.simulate(self.base.plan, self.base_trace, self.base.checkin, self.baseline,
                           fluctuation=self.fluctuation,
                           rng=self.rng if self.fluctuation is not None else None,
                           effects=self.base_effects)

    def simulate(self, sample: Sample) -> pd.DataFrame:
        return md.simulate(sample.plan, sample.trace, sample.checkin, self.baseline,
                           fluctuation=self.fluctuation,
                           rng=self.rng if self.fluctuation is not None else None,
                           effects=sample.effects)

    def checked(self, sample: Sample, stats: "Stats | None" = None) -> pd.DataFrame | None:
        """Simulate a sample and run the causal-chain check; None (counted as discarded in
        `stats`, default this mutator's) when an operator moved its chain the wrong way."""
        stats = self.stats if stats is None else stats
        out = self.simulate(sample)
        stats.attempted += 1
        violations = check_chains(out, self.reference, sample.ops)
        if violations:
            stats.discarded += 1
            stats.by_reason.update({x.split(":")[0] for x in violations})
            log.debug("discarded %s: %s", [o.op for o in sample.ops], violations)
            return None
        stats.accepted += 1
        return out

    def generate(self, n: int) -> list[Result]:
        """Up to n accepted samples; gives up after n x MUT_MAX_ATTEMPTS_FACTOR draws."""
        results: list[Result] = []
        for _ in range(n * self.cfg("MUT_MAX_ATTEMPTS_FACTOR")):
            if len(results) == n:
                break
            sample = self.draw()
            out = self.checked(sample)
            if out is not None:
                results.append(Result(sample, out, apply_faults(out, sample.faults)))
        log.info(self.stats.summary())
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw mutated samples and print the discard rate.")
    ap.add_argument("--plan", default="main_sleep_debt_eva")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()
    m = Mutator(Scenario.load(args.plan), st.make_baseline(), seed=args.seed,
                fluctuation=md.Fluctuation.load())
    results = m.generate(args.n)
    counts = Counter(o.op for r in results for o in r.sample.ops)
    print(m.stats.summary())
    print("operators used:", ", ".join(f"{k} {counts[k]}" for k in OPERATORS))


if __name__ == "__main__":
    main()
