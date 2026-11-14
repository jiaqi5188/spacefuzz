"""One-day state transition: 96 steps of 15 minutes from the check-in (plan 3.2).

Each step reads what the astronaut is doing (plan task -> activity; unscheduled
awake time is "idle", which costs a little fatigue rather than recovering it), the cabin
environment (env_trace row) and active env events, then moves every modelled
quantity part of the way toward a target (first-order response, time constants
in config.MODEL_TAU_MIN). Causal chains (plan 2.3):

- Chain 1 acute stress: a high-load task (docking, precision manual work), an SPE
  warning or a dose-rate spike raises a 0-1 stress level; at full stress HR rises
  by CHAIN1_STRESS_HR_DELTA_BPM and RMSSD is multiplied by CHAIN1_STRESS_RMSSD_RATIO
  (both fitted in A3), BP rises and glucose creeps up (cortisol).
- Chain 2 energy and heat: EVA / exercise raise HR, lower RMSSD, warm the core
  towards a plateau and cost a little SpO2; stored heat (operator A) keeps warming the
  core during a continuous EVA, and above HEAT_STRAIN_ONSET_C heart rate follows core
  temperature (HEAT_HR_PER_C); rest lets the heat, and with it the heart rate, come down; CO2 above the A1 high level adds HR,
  lowers SpO2 and speeds fatigue.
- Chain 5 sleep debt: debt (last 3 nights) raises resting HR, lowers resting
  RMSSD, slows reaction time, makes fatigue build faster, makes exertion HR rise
  faster and recover slower. Sleep repays part of it.
- Chain 6 glucose and load: during exertion glucose holds (small drift) while it
  is >= 70 mg/dL; below that it keeps falling. Glucose below 70 mg/dL adds fatigue in
  any activity. Meals raise it; otherwise it returns to the check-in level.

Caps and recovery: cognitive fatigue is 0-100 and falls during rest and sleep;
radiation_load (0-100) grows with exposure above the trace's usual dose rate and
recovers at rest, while eva_dose_ugy / dose_cum_ugy are physical doses that never
decrease.

Heart rate cap: the model heart rate and the overlaid reading never exceed the personal
maximum, 220 - age (state.max_heart_rate); additive effects (EVA, sleep debt, stress,
CO2, operator A) would otherwise stack past what a body can reach.

Real fluctuation: heart rate and RMSSD outputs add a deviation drawn from WESAD
(per-minute deviation from its segment mean, stress pool when stress > 0.5). The
overlay is output only; it never feeds back into the dynamics. The evaluator judges
rules on the overlaid outputs (heart_rate, hrv_rmssd), not on hr_model / rmssd_model:
real sensor readings fluctuate, and that fluctuation is itself a source of risk.

Mutation hook: simulate(effects=Effects(...)) lets mutator.py apply the operator effects
that are not plan, trace or check-in changes (EVA heat storage, SpO2 drop at high CO2, fatigue
jumps and slow rises, the habitat's usual dose rate). The default Effects() changes nothing.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spacefuzz import config
from spacefuzz.core import state as st
from spacefuzz.core.plan import Plan
from spacefuzz.core.plan_types import PlanType

DT_MIN = config.STEP_MINUTES
DT_H = DT_MIN / 60

ACTIVITY_OF = {
    PlanType.EVA: "eva",
    PlanType.EXERCISE_T2: "treadmill",
    PlanType.EXERCISE_ARED: "strength",
    PlanType.DOCKING: "high_load",
    PlanType.PRECISION_MANUAL: "high_load",
    PlanType.GENERIC_HIGH_LOAD: "high_load",
    PlanType.SCIENCE_EXPERIMENT: "work",
    PlanType.MAINTENANCE: "work",
    PlanType.COMM_SESSION: "work",
    PlanType.HEALTH_CHECK: "work",
    PlanType.MEAL: "meal",
    PlanType.SLEEP: "sleep",
    PlanType.REST: "rest",
    PlanType.PERSONAL_TIME: "rest",
}
EXERTION = {"eva", "treadmill", "strength"}
RECOVERY = {"idle", "rest", "meal", "sleep"}   # radiation_load recovers

OUTPUT_COLUMNS = ["heart_rate", "hrv_rmssd", "hr_model", "rmssd_model", "spo2", "core_temp",
                  "cognitive_fatigue", "blood_glucose", "systolic_bp", "diastolic_bp",
                  "reaction_time_ms", "stress_level", "sleep_debt_h", "radiation_load",
                  "eva_dose_ugy", "dose_cum_ugy", "co2", "dose_rate"]


@dataclass(frozen=True)
class StepContext:
    activity: str
    task: str | None            # plan type string, None when nothing is scheduled
    co2: float                  # ppm, after plan high_co2 floors (plan.apply_env_events)
    dose_rate: float            # uGy/h inside the habitat
    dose_median: float          # the trace's usual dose rate
    spe_warning: bool
    meal_steps: int = 1         # length of the current meal, to spread its glucose rise
    rest_glucose: float = 90.0  # check-in glucose: where glucose returns outside exertion
    rest_sbp: float = 118.0
    rest_dbp: float = 76.0
    temp_extra: float = 0.0        # mutation effects (Effects), already limited to their window
    spo2_shift: float = 0.0
    fatigue_rate_extra: float = 0.0


@dataclass(frozen=True)
class Effects:
    """Operator effects applied during a simulation (mutator.py); minutes are absolute
    plan-day minutes, as in AstronautState.minute."""
    eva_heat_rate: float = 0.0         # operator A: heat stored per hour of continuous EVA [degC/h]
    spo2_drop_high_co2: float = 0.0    # operator B: SpO2 drop [pct pt] while CO2 > ENV_CO2_HIGH_PPM
    fatigue_jumps: tuple = ()          # operator C: ((minute, points), ...)
    fatigue_rate: tuple = ()           # operator D: ((start, end, points per h), ...)
    dose_median: float | None = None   # usual habitat dose rate; None = the trace median


def _approach(x: float, target: float, tau_min: float) -> float:
    return x + (target - x) * (1.0 - math.exp(-DT_MIN / tau_min))


def step(s: st.AstronautState, base: st.Baseline, ctx: StepContext) -> st.AstronautState:
    """Advance one 15-minute step; returns a new state."""
    tau = config.MODEL_TAU_MIN
    eff = config.ACTIVITY_EFFECTS[ctx.activity]
    debt = s.sleep_debt_h
    exertion = ctx.activity in EXERTION

    # Chain 1: stress level from triggers
    trig = config.STRESS_TRIGGERS
    stress_target = max(
        trig["high_load_task"] if ctx.activity == "high_load" else 0.0,
        trig["spe_warning"] if ctx.spe_warning else 0.0,
        trig["dose_spike"] if ctx.dose_rate > config.DOSE_SPIKE_FACTOR * ctx.dose_median else 0.0)
    stress = _approach(s.stress_level, stress_target, tau["stress"])

    # Chain 2 (+5): heart rate and HRV
    co2_excess = max(0.0, ctx.co2 - config.ENV_CO2_HIGH_PPM) / 1000.0
    gain = 1.0 + config.SLEEP_DEBT_EXERTION_GAIN_PER_H * debt if exertion else 1.0
    heat = max(0.0, s.core_temp - base.core_temp - config.HEAT_STRAIN_ONSET_C)
    hr_target = (st.resting_hr(base, debt) + eff["hr"] * gain + config.HEAT_HR_PER_C * heat
                 + stress * config.CHAIN1_STRESS_HR_DELTA_BPM
                 + co2_excess * config.CO2_EFFECT_PER_1000_PPM["hr"])
    if hr_target > s.heart_rate:
        hr_tau = tau["hr_rise"] / (1.0 + config.SLEEP_DEBT_EXERTION_GAIN_PER_H * debt)
    else:
        hr_tau = tau["hr_recover"] * (1.0 + config.SLEEP_DEBT_RECOVERY_SLOWDOWN_PER_H * debt)
    hr = min(st.max_heart_rate(base), _approach(s.heart_rate, hr_target, hr_tau))
    rmssd_target = (st.resting_rmssd(base, debt) * eff["rmssd"]
                    * config.CHAIN1_STRESS_RMSSD_RATIO ** stress)
    rmssd = _approach(s.hrv_rmssd, rmssd_target, hr_tau)

    # Chain 2: core temperature, SpO2
    temp_rise = eff["temp"] + min(ctx.temp_extra, max(0.0, config.HEAT_MAX_CORE_RISE_C - eff["temp"]))
    temp = _approach(s.core_temp, base.core_temp + temp_rise, tau["core_temp"])
    spo2_target = (base.spo2 + eff["spo2"] + co2_excess * config.CO2_EFFECT_PER_1000_PPM["spo2"]
                   - ctx.spo2_shift)
    spo2 = min(config.SPO2_MAX_PCT, max(config.SPO2_MIN_PCT,
                                        _approach(s.spo2, spo2_target, tau["spo2"])))

    # Chain 1 + 2: blood pressure around the check-in values
    sbp = _approach(s.systolic_bp, ctx.rest_sbp + eff["sbp"] + stress * config.STRESS_SBP_DBP[0],
                    tau["bp"])
    dbp = _approach(s.diastolic_bp, ctx.rest_dbp + eff["dbp"] + stress * config.STRESS_SBP_DBP[1],
                    tau["bp"])

    # Chain 6: glucose
    g = s.blood_glucose
    low = g < config.GLUCOSE_LOW_MG_DL
    if exertion:
        g -= eff["glucose_drift"] * DT_H
        if low:
            g -= config.GLUCOSE_LOW_FALL_PER_H * DT_H
    else:
        g = _approach(g, ctx.rest_glucose, tau["glucose_return"])
    if ctx.activity == "meal":
        g += config.MEAL_GLUCOSE_RISE_MG_DL / max(1, ctx.meal_steps)
    g += config.STRESS_GLUCOSE_PER_H * stress * DT_H
    g = max(config.GLUCOSE_FLOOR_MG_DL, g)

    # Fatigue (chains 5, 6, CO2), capped, recovers at rest / sleep
    rate = eff["fatigue"]
    if rate > 0:
        rate *= 1.0 + config.SLEEP_DEBT_FATIGUE_GAIN_PER_H * debt
    rate += co2_excess * config.CO2_EFFECT_PER_1000_PPM["fatigue"]
    if low:
        rate += config.GLUCOSE_LOW_FATIGUE_PER_H
    rate += ctx.fatigue_rate_extra
    fatigue = min(config.FATIGUE_MAX, max(0.0, s.cognitive_fatigue + rate * DT_H))

    # Sleep repays debt (chain 5)
    if ctx.activity == "sleep":
        debt = max(0.0, debt - config.SLEEP_DEBT_REPAY_PER_SLEEP_H * DT_H)

    # Radiation: physical dose, plus a capped strain index that recovers at rest
    dose_rate = ctx.dose_rate * (config.EVA_DOSE_MULTIPLIER if ctx.activity == "eva" else 1.0)
    dose = dose_rate * DT_H
    load = s.radiation_load + max(0.0, dose_rate - ctx.dose_median) * DT_H * config.RAD_LOAD_PER_UGY
    if ctx.activity in RECOVERY:
        load -= config.RAD_LOAD_RECOVERY_PER_H * DT_H
    load = min(config.RAD_LOAD_MAX, max(0.0, load))

    rt = st.model_reaction_ms(base, debt, fatigue) + s.rt_offset_ms

    return dataclasses.replace(
        s, minute=s.minute + DT_MIN, heart_rate=hr, hrv_rmssd=rmssd, spo2=spo2, core_temp=temp,
        cognitive_fatigue=fatigue, blood_glucose=g, systolic_bp=sbp, diastolic_bp=dbp,
        reaction_time_ms=rt, stress_level=stress, sleep_debt_h=debt, radiation_load=load,
        eva_dose_ugy=s.eva_dose_ugy + (dose if ctx.activity == "eva" else 0.0),
        dose_cum_ugy=s.dose_cum_ugy + dose)


# ---------------------------------------------------------------- real fluctuation

WESAD_PARAMS = Path(__file__).resolve().parents[2] / "data" / "processed" / "wesad_params.json"


@dataclass(frozen=True)
class Fluctuation:
    """Pools of per-minute deviations: hr_dev [bpm], rmssd_rel (relative).

    source "wesad": the real per-minute WESAD series (data/processed/wesad_hr_hrv.csv, kept local:
    WESAD is for academic, non-commercial use). source "wesad_params": generated from the fitted
    per-subject, per-condition statistics (wesad_params.json, committed) when the series is not
    available, e.g. in a fresh clone of the public repository."""
    hr_dev: dict
    rmssd_rel: dict
    source: str = "wesad"

    @classmethod
    def load(cls) -> "Fluctuation | None":
        """The real pool when the series is there (unchanged, so forecasts and their random
        draws are identical), else the fitted-parameter pool, else None (no overlay)."""
        try:
            return cls.from_wesad()
        except (FileNotFoundError, OSError):
            pass
        try:
            return cls.from_params()
        except (FileNotFoundError, OSError, KeyError, ValueError):
            return None

    @classmethod
    def from_params(cls, path: Path | None = None) -> "Fluctuation":
        """Normal per-minute deviations with each subject's per-condition SD (as many minutes
        as WESAD has), from a generator of its own: the forecast's random stream is untouched."""
        data = json.loads(Path(path or WESAD_PARAMS).read_text(encoding="utf-8"))
        rng = np.random.default_rng(config.RANDOM_SEED)
        hr = {"relax": [], "stress": []}
        rel = {"relax": [], "stress": []}
        for subject in sorted(data["subjects"]):
            for cond, p in sorted(data["subjects"][subject].items()):
                key = ("relax" if cond in config.RELAX_CONDITIONS
                       else "stress" if cond == "stress" else None)
                if key is None:
                    continue
                n = int(p["minutes"])
                # spread around each segment's own mean, as in the real pool (from_wesad)
                hr[key].append(rng.normal(0.0, p["hr_dev_sd"], n))
                rel[key].append(np.clip(rng.normal(0.0, p["rmssd_dev_cv"], n),
                                        config.FLUCT_RMSSD_REL_MIN, None))
        return cls(hr_dev={k: np.concatenate(v) for k, v in hr.items()},
                   rmssd_rel={k: np.concatenate(v) for k, v in rel.items()},
                   source="wesad_params")

    @classmethod
    def from_wesad(cls) -> "Fluctuation":
        from spacefuzz.readers import physio_replay as pr

        f = pr.PhysioReplay.load().frame
        seg = f.groupby(["subject", "segment_id"])[["hr", "rmssd"]].transform("mean")
        f = f.assign(hr_dev=f["hr"] - seg["hr"], rmssd_rel=f["rmssd"] / seg["rmssd"] - 1)
        pools = {"relax": f[f["condition"].isin(config.RELAX_CONDITIONS)],
                 "stress": f[f["condition"] == "stress"]}
        return cls(hr_dev={k: v["hr_dev"].to_numpy() for k, v in pools.items()},
                   rmssd_rel={k: v["rmssd_rel"].to_numpy() for k, v in pools.items()})

    def draw(self, stressed: bool, rng: np.random.Generator) -> tuple[float, float]:
        key = "stress" if stressed else "relax"
        i = int(rng.integers(len(self.hr_dev[key])))
        j = int(rng.integers(len(self.rmssd_rel[key])))
        return float(self.hr_dev[key][i]), float(self.rmssd_rel[key][j])


# ---------------------------------------------------------------- simulation

def _task_at(plan: Plan, minute: int):
    for t in plan.tasks:
        if t.start_min <= minute < t.end_min:
            return t
    return None


def simulate(plan: Plan, trace: pd.DataFrame, checkin: st.CheckIn, baseline: st.Baseline,
             fluctuation: Fluctuation | None = None, rng: np.random.Generator | None = None,
             steps: int = config.STEPS_PER_DAY, effects: Effects | None = None) -> pd.DataFrame:
    """Run `steps` steps from the check-in; row k is the state at check-in + 15 k minutes."""
    if fluctuation is not None and rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)
    fx = effects or Effects()
    jumps = dict(fx.fatigue_jumps)
    co2 = trace["co2"].to_numpy()
    dose = trace["dose_rate"].to_numpy()
    n_trace = len(trace)
    dose_median = float(np.nanmedian(dose)) if fx.dose_median is None else float(fx.dose_median)
    start = st.start_min(checkin)
    hr_max = st.max_heart_rate(baseline)   # the overlaid reading cannot exceed it either
    s = dataclasses.replace(st.initial_state(checkin, baseline), minute=start)

    rows = []
    eva_run_min = 0                         # minutes into the current continuous EVA
    for _ in range(steps):
        m = s.minute
        if m in jumps:   # operator C: overload adds fatigue at once
            f = min(config.FATIGUE_MAX, max(0.0, s.cognitive_fatigue + jumps[m]))
            s = dataclasses.replace(s, cognitive_fatigue=f, reaction_time_ms=st.model_reaction_ms(
                baseline, s.sleep_debt_h, f) + s.rt_offset_ms)
        task = _task_at(plan, m)
        activity = ACTIVITY_OF.get(task.type, "idle") if task else "idle"
        i = (m % (24 * 60)) // DT_MIN % n_trace
        is_eva = activity == "eva"
        eva_run_min = eva_run_min + DT_MIN if is_eva else 0
        ctx = StepContext(
            activity=activity, task=task.source_type if task else None,
            co2=float(co2[i]), dose_rate=float(dose[i]), dose_median=dose_median,
            # env events are clock times of the plan day, like the trace rows they overlay
            # (high CO2): past midnight the same clock times apply again
            spe_warning=any(e.type is PlanType.SPE_WARNING and e.start_min <= m % 1440 < e.end_min
                            for e in plan.events),
            meal_steps=max(1, task.duration_min // DT_MIN) if task else 1,
            rest_glucose=checkin.blood_glucose, rest_sbp=checkin.systolic_bp,
            rest_dbp=checkin.diastolic_bp,
            temp_extra=fx.eva_heat_rate * eva_run_min / 60 if is_eva else 0.0,
            spo2_shift=fx.spo2_drop_high_co2 if co2[i] > config.ENV_CO2_HIGH_PPM else 0.0,
            fatigue_rate_extra=sum(r for a, b, r in fx.fatigue_rate if a <= m < b))

        stressed = s.stress_level > config.FLUCT_STRESS_POOL_LEVEL
        hr_noise, rmssd_noise = (fluctuation.draw(stressed, rng)
                                 if fluctuation is not None else (0.0, 0.0))
        rows.append({
            "minute": m, "clock": f"{m // 60 % 24:02d}:{m % 60:02d}", "task": ctx.task,
            "activity": activity,
            "heart_rate": min(hr_max, s.heart_rate + hr_noise),
            "hrv_rmssd": max(config.HRV_READING_FLOOR_MS, s.hrv_rmssd * (1.0 + rmssd_noise)),
            "hr_model": s.heart_rate, "rmssd_model": s.hrv_rmssd,
            "spo2": s.spo2, "core_temp": s.core_temp, "cognitive_fatigue": s.cognitive_fatigue,
            "blood_glucose": s.blood_glucose, "systolic_bp": s.systolic_bp,
            "diastolic_bp": s.diastolic_bp, "reaction_time_ms": s.reaction_time_ms,
            "stress_level": s.stress_level, "sleep_debt_h": s.sleep_debt_h,
            "radiation_load": s.radiation_load, "eva_dose_ugy": s.eva_dose_ugy,
            "dose_cum_ugy": s.dose_cum_ugy, "co2": ctx.co2, "dose_rate": ctx.dose_rate,
        })
        s = step(s, baseline, ctx)
    return pd.DataFrame(rows)
