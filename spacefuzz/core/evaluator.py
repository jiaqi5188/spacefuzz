"""Crash rules and fitness (plan 3.5, 3.6): judge one simulated day.

Composite rules, read from the model output (overlaid heart_rate / hrv_rmssd, never
hr_model / rmssd_model: real readings fluctuate and that is part of the risk):
- cognitive:      fatigue > 80 and SpO2 < 94% at the same step;
- cardiovascular: heart rate >= CARDIO_HR_MAX_FRACTION x personal max HR (220 - age) and RMSSD
                  <= CARDIO_HRV_RATIO x baseline, both for CARDIO_SUSTAIN_MIN; or blood
                  pressure outside CARDIO_BP_RANGE after CARDIO_BP_SETTLE_MIN at rest
                  (exertion raises BP normally, the range is a resting range);
- metabolic:      glucose < 70 mg/dL when an EVA starts, or an unbroken fall of
                  METAB_EVA_FALL_MG_DL during an EVA;
- radiation_dose: EVA dose of the day (all EVAs) above RAD_EVA_DOSE_LIMIT_UGY (not a NASA
                  limit: NASA's formal radiation limits are career limits).

Temporal rules, each a small hand-written state machine fed one 15-min step at a time:
1. after < 6 h sleep, no EVA may start within 12 h of waking;
2. no strength (ARED) / treadmill (T2) training in the 2 h before an EVA starts;
3. after > 4 h of continuous cabin work at CO2 >= RULE3_CO2_PPM, no docking / precision
   manual work (EVA is suit air, not cabin air, so it ends a run; so does rest or a gap);
4. no EVA may start during an spe_warning event (read from the plan's events).

"Steps to violation" counts the smallest change, in 15-min steps, that would make a rule
fire: 15 min less sleep or an EVA 15 min earlier (rule 1), training 15 min later (2),
15 more minutes of high-CO2 work (3), an EVA start shifted 15 min (4). 0 = violated;
None = not reachable by shifting (e.g. no precision task, no SPE warning).

Fitness (plan 3.6): each rule gives a distance in [0, 1], 0 = violated, and fitness is
their sum (lower = closer to a crash). Temporal: min(steps, RULE_DISTANCE_CAP_STEPS) /
cap (None -> 1). Composite: the gap between the value and the threshold divided by the
gap between the personal baseline and the threshold (1 = at baseline or better),
taking the worst step; an "and" rule needs every part, so its gap is the largest part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spacefuzz import config
from spacefuzz.core import model as md
from spacefuzz.core import state as st
from spacefuzz.core.plan import Plan, Task
from spacefuzz.core.plan_types import EXERCISE_TYPES, PRECISION_TYPES, RESTORATIVE_TYPES, PlanType

STEP = config.STEP_MINUTES

SYSTEM = {
    "cognitive": "CNS", "cardiovascular": "Cardiovascular", "metabolic": "Metabolic",
    "radiation_dose": "Radiation", "rule1_sleep_eva": "Cardiovascular",
    "rule2_exercise_eva": "Cardiovascular", "rule3_co2_precision": "CNS",
    "rule4_spe_eva": "Radiation",
}

RULE_TEXT = {
    "rule1_sleep_eva": "EVA started within 12 h of waking after < 6 h sleep",
    "rule2_exercise_eva": "strength / treadmill training in the 2 h before EVA",
    "rule3_co2_precision": "docking / precision work after > 4 h of high-CO2 cabin work",
    "rule4_spe_eva": "EVA started during a solar particle event warning",
}


@dataclass(frozen=True)
class Violation:
    rule: str
    system: str
    minute: int       # absolute plan-day minute when it fires
    detail: str


@dataclass
class Evaluation:
    violations: list[Violation]
    distance: dict[str, float]                   # per rule, [0, 1], 0 = violated
    steps_to_violation: dict[str, int | None]    # temporal rules only
    fitness: float = field(init=False)

    def __post_init__(self):
        self.fitness = float(sum(self.distance.values()))

    @property
    def crashed(self) -> bool:
        return bool(self.violations)

    @property
    def first_crash_minute(self) -> int | None:
        return min((v.minute for v in self.violations), default=None)

    def systems(self) -> list[str]:
        return sorted({v.system for v in self.violations})


# ---------------------------------------------------------------- temporal machines

@dataclass(frozen=True)
class Obs:
    """One step seen by the temporal machines."""
    minute: int
    task: Task | None = None
    starts: bool = False          # the task starts at this step
    co2: float = 0.0


def _is(task: Task | None, *types) -> bool:
    return task is not None and task.type in types


class _Counted:
    """Shared bookkeeping of the temporal rules. count_from: violations before this minute are
    not recorded (midday check: the machines still run from the morning, so wake time and
    work history are kept)."""
    count_from: int | None = None
    violated_at: int | None = None
    _best: int | None = None

    def _counts(self, minute: int) -> bool:
        return self.count_from is None or minute >= self.count_from

    def _record(self, minute: int, d: int) -> bool:
        """Keep the smallest distance seen; True when this is the first violation."""
        first = d == 0 and self.violated_at is None
        if first:
            self.violated_at = minute
        self._best = d if self._best is None else min(self._best, d)
        return first

    @property
    def distance(self) -> int | None:
        return self._best


class Rule1SleepEva(_Counted):
    """States: rested -> short_sleep_window (slept < 6 h, within 12 h of waking) -> violated.
    A planned sleep longer than RULE1_NAP_MAX_H restarts the machine when it ends; a nap does not."""
    name = "rule1_sleep_eva"

    def __init__(self, last_sleep_h: float, wake_min: int, count_from: int | None = None):
        self.sleep_h, self.wake = last_sleep_h, wake_min
        self.count_from = count_from
        self._sleep_start: int | None = None

    @property
    def state(self) -> str:
        if self.violated_at is not None:
            return "violated"
        return "short_sleep_window" if self.sleep_h < config.RULE1_MIN_SLEEP_H else "rested"

    def _sleep_steps(self) -> int:
        """15-min sleep cuts needed to fall below the minimum."""
        if self.sleep_h < config.RULE1_MIN_SLEEP_H:
            return 0
        return math.floor((self.sleep_h - config.RULE1_MIN_SLEEP_H) * 60 / STEP + 1e-9) + 1

    def observe(self, o: Obs) -> None:
        if _is(o.task, PlanType.SLEEP):
            if self._sleep_start is None:
                self._sleep_start = o.minute
            return
        if self._sleep_start is not None:          # just woke up from a planned sleep
            slept = (o.minute - self._sleep_start) / 60
            if slept > config.RULE1_NAP_MAX_H:     # a nap does not replace the night's sleep
                self.sleep_h, self.wake = slept, o.minute
            self._sleep_start = None
        if not (o.starts and _is(o.task, PlanType.EVA)):
            return
        since = o.minute - self.wake
        window = config.RULE1_WINDOW_H * 60
        time_steps = 0 if since < window else int((since - window) // STEP) + 1
        d = self._sleep_steps() + time_steps
        if self._counts(o.minute):
            self._record(o.minute, d)


class Rule2ExerciseEva(_Counted):
    """States: no_training -> training -> after_training(minutes since it ended)."""
    name = "rule2_exercise_eva"

    def __init__(self, count_from: int | None = None):
        self.count_from = count_from
        self.state = "no_training"
        self.ended: int | None = None

    def observe(self, o: Obs) -> None:
        training = _is(o.task, *EXERCISE_TYPES)
        if o.starts and _is(o.task, PlanType.EVA):
            gap = config.RULE2_GAP_H * 60
            if self.state == "training":
                d = 0
            elif self.state == "after_training":
                since = o.minute - self.ended
                d = 0 if since < gap else int((since - gap) // STEP) + 1
            else:
                d = None
            if d is not None and self._counts(o.minute) and self._record(o.minute, d):
                self.state = "violated"
        if self.state == "violated":
            return
        if training:
            self.state = "training"
        elif self.state == "training":
            self.state, self.ended = "after_training", o.minute


class Rule3Co2Precision(_Counted):
    """States: idle -> working(n steps of high-CO2 cabin work) -> violated."""
    name = "rule3_co2_precision"

    def __init__(self, co2_ppm: float | None = None, count_from: int | None = None):
        self.count_from = count_from
        self.co2_ppm = config.RULE3_CO2_PPM if co2_ppm is None else co2_ppm   # read at run time
        self.n = 0

    @property
    def state(self) -> str:
        return "violated" if self.violated_at is not None else ("working" if self.n else "idle")

    def observe(self, o: Obs) -> None:
        if o.starts and _is(o.task, *PRECISION_TYPES):
            limit = int(config.RULE3_MAX_WORK_H * 60 // STEP)    # more than this many steps
            d = max(0, limit + 1 - self.n)
            if self._counts(o.minute):
                self._record(o.minute, d)
        cabin_work = (o.task is not None and o.task.type not in RESTORATIVE_TYPES
                      and o.task.type is not PlanType.EVA)
        self.n = self.n + 1 if cabin_work and o.co2 >= self.co2_ppm else 0


class Rule4SpeEva(_Counted):
    """States: clear / warning (from the plan's spe_warning events) -> violated."""
    name = "rule4_spe_eva"

    def __init__(self, warnings: list[tuple[int, int]], count_from: int | None = None):
        self.count_from = count_from
        self.warnings = warnings
        self.state = "clear"

    def observe(self, o: Obs) -> None:
        # warnings are clock times of the plan day, as model.simulate applies them: past
        # midnight the same clock times apply again
        clock = o.minute % (24 * 60)
        inside = any(a <= clock < b for a, b in self.warnings)
        if self.state != "violated":
            self.state = "warning" if inside else "clear"
        if not (o.starts and _is(o.task, PlanType.EVA)) or not self.warnings:
            return
        steps = []
        for a, b in self.warnings:            # shift of the EVA start into [a, b)
            if clock < a:
                steps.append(-(-(a - clock) // STEP))
            elif clock >= b:
                steps.append(int((clock - b) // STEP) + 1)
            else:
                steps.append(0)
        if self._counts(o.minute) and self._record(o.minute, min(steps)):
            self.state = "violated"


def run_temporal(output: pd.DataFrame, plan: Plan, checkin: st.CheckIn,
                 count_from: int | None = None) -> list:
    """Feed the simulated day, step by step, to the four machines (all from the first step;
    count_from only stops them recording violations before that minute)."""
    warnings = [(e.start_min, e.end_min) for e in plan.events if e.type is PlanType.SPE_WARNING]
    machines = [Rule1SleepEva(checkin.sleep_last_night_h, checkin.wake_time_min, count_from),
                Rule2ExerciseEva(count_from), Rule3Co2Precision(count_from=count_from),
                Rule4SpeEva(warnings, count_from)]
    for m_, co2 in zip(output["minute"].to_numpy(), output["co2"].to_numpy()):
        task = md._task_at(plan, int(m_))
        o = Obs(minute=int(m_), task=task, starts=task is not None and task.start_min == m_,
                co2=float(co2))
        for machine in machines:
            machine.observe(o)
    return machines


# ---------------------------------------------------------------- composite rules

def _gap(value, threshold, reference) -> np.ndarray:
    """(distance to threshold) / (distance from personal reference to threshold), >= 0."""
    scale = abs(reference - threshold) or 1.0
    return np.clip(np.asarray(value, dtype=float) / scale, 0.0, None)


def _first(mask: np.ndarray, minutes: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(minutes[idx[0]]) if idx.size else None


def _cognitive(o, base):
    f, s = o["cognitive_fatigue"].to_numpy(), o["spo2"].to_numpy()
    hit = (f > config.COG_FATIGUE_ABOVE) & (s < config.COG_SPO2_BELOW_PCT)
    gap = np.maximum(_gap(config.COG_FATIGUE_ABOVE - f, config.COG_FATIGUE_ABOVE, 0.0),
                     _gap(s - config.COG_SPO2_BELOW_PCT, config.COG_SPO2_BELOW_PCT, base.spo2))
    return hit, gap, "fatigue > {:.0f} and SpO2 < {:.0f}%".format(config.COG_FATIGUE_ABOVE,
                                                                  config.COG_SPO2_BELOW_PCT)


def _cardiovascular(o, base):
    hr, rmssd = o["heart_rate"].to_numpy(), o["hrv_rmssd"].to_numpy()
    hr_thr = config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(base)
    hrv_thr = config.CARDIO_HRV_RATIO * base.rmssd
    step_hit = (hr >= hr_thr) & (rmssd <= hrv_thr)
    step_gap = np.maximum(_gap(hr_thr - hr, hr_thr, base.resting_hr),
                          _gap(rmssd - hrv_thr, hrv_thr, base.rmssd))
    # "sustained N minutes" = N / STEP consecutive rows: each row stands for its whole
    # 15-minute step (30 min = 2 rows).
    k = max(1, config.CARDIO_SUSTAIN_MIN // STEP)
    hit = pd.Series(step_hit).rolling(k).min().fillna(0).to_numpy().astype(bool)
    gap = pd.Series(step_gap).rolling(k, min_periods=1).max().to_numpy()

    # blood pressure: only at steps settled at rest (the day starts settled, at the check-in)
    rest = o["activity"].isin(md.RECOVERY).to_numpy()
    minutes = o["minute"].to_numpy()
    settled = np.zeros(len(o), dtype=bool)
    since = -np.inf
    for i in range(len(o)):
        if not rest[i]:
            since = None
        elif since is None:
            since = minutes[i]
        settled[i] = since is not None and minutes[i] - since >= config.CARDIO_BP_SETTLE_MIN
    (s_lo, s_hi), (d_lo, d_hi) = config.CARDIO_BP_RANGE["sbp"], config.CARDIO_BP_RANGE["dbp"]
    sbp, dbp = o["systolic_bp"].to_numpy(), o["diastolic_bp"].to_numpy()
    bp_out = (sbp > s_hi) | (sbp < s_lo) | (dbp > d_hi) | (dbp < d_lo)
    bp_gap = np.minimum.reduce([_gap(s_hi - sbp, s_hi, base.sbp), _gap(sbp - s_lo, s_lo, base.sbp),
                                _gap(d_hi - dbp, d_hi, base.dbp), _gap(dbp - d_lo, d_lo, base.dbp)])
    bp_gap = np.where(settled, bp_gap, np.inf)
    return (hit | (bp_out & settled), np.minimum(gap, bp_gap),
            "sustained high heart rate with low HRV, or resting blood pressure out of range")


def _metabolic(o, base, plan):
    g = o["blood_glucose"].to_numpy()
    minutes = o["minute"].to_numpy()
    eva = (o["activity"] == "eva").to_numpy()
    starts = {t.start_min for t in plan.tasks if t.type is PlanType.EVA}
    at_start = np.isin(minutes, list(starts)) & eva
    low = config.METAB_PRE_EVA_GLUCOSE_MG_DL
    hit_low = at_start & (g < low)
    gap_low = np.where(at_start, _gap(g - low, low, base.glucose), np.inf)
    # Unbroken fall during an EVA. Each row holds the state before its step, so the change
    # from row i-1 to row i happened during step i-1: it counts when step i-1 was EVA (the
    # last EVA step's drop shows on the first row after the EVA).
    fall = np.zeros(len(o))
    for i in range(1, len(o)):
        if eva[i - 1] and g[i] <= g[i - 1]:
            fall[i] = fall[i - 1] + g[i - 1] - g[i]
    lim = config.METAB_EVA_FALL_MG_DL
    hit_fall = fall >= lim
    during = eva | np.r_[False, eva[:-1]]
    gap_fall = np.where(during, _gap(lim - fall, lim, 0.0), np.inf)
    return (hit_low | hit_fall, np.minimum(gap_low, gap_fall),
            f"glucose < {low:.0f} at EVA start or unbroken fall of {lim:.0f} mg/dL during EVA")


def _radiation(o, base):
    dose = o["eva_dose_ugy"].to_numpy()
    lim = config.RAD_EVA_DOSE_LIMIT_UGY
    return dose > lim, _gap(lim - dose, lim, 0.0), f"EVA dose above {lim:.0f} uGy"


def evaluate(output: pd.DataFrame, plan: Plan, checkin: st.CheckIn,
             baseline: st.Baseline, count_from: int | None = None) -> Evaluation:
    """Judge one simulated day (model.simulate output) against every rule.

    count_from (minutes after midnight, midday check): the whole day is still judged from the
    morning (state machines keep wake time and work history; a sustained condition that
    completes after count_from counts), but only violations at or after count_from are
    recorded and distances use those steps only. None = the whole day (default)."""
    minutes = output["minute"].to_numpy()
    after = np.ones(len(minutes), bool) if count_from is None else minutes >= count_from
    violations, distance, steps = [], {}, {}

    composite = {"cognitive": _cognitive(output, baseline),
                 "cardiovascular": _cardiovascular(output, baseline),
                 "metabolic": _metabolic(output, baseline, plan),
                 "radiation_dose": _radiation(output, baseline)}
    for name, (hit, gap, text) in composite.items():
        hit, gap = hit & after, np.where(after, gap, np.inf)
        at = _first(hit, minutes)
        if at is not None:
            violations.append(Violation(name, SYSTEM[name], at, text))
            distance[name] = 0.0
        else:
            finite = gap[np.isfinite(gap)]
            distance[name] = float(min(1.0, finite.min())) if finite.size else 1.0

    cap = config.RULE_DISTANCE_CAP_STEPS
    for m in run_temporal(output, plan, checkin, count_from):
        steps[m.name] = m.distance
        distance[m.name] = 1.0 if m.distance is None else min(m.distance, cap) / cap
        if m.violated_at is not None:
            violations.append(Violation(m.name, SYSTEM[m.name], m.violated_at, RULE_TEXT[m.name]))
    violations.sort(key=lambda v: v.minute)
    return Evaluation(violations=violations, distance=distance, steps_to_violation=steps)
