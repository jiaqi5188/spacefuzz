"""Astronaut state, synthetic personal baseline, and morning check-in loading.

- Baseline: means of a 14-day synthetic history generated with a fixed seed
  (plan 3.1: no public per-astronaut history exists). Person-level centres come
  from the middle half of the ranges found in analyses A3 (HR, RMSSD) and A4
  (glucose); other fields use illustrative values from config.
- CheckIn: a data/checkins/*.json file (sleep, BP, glucose, self-reports,
  reaction time) - the starting point of a forecast.
- AstronautState: the per-step state the model advances. Sleep debt (chain 5)
  shifts the starting HR, HRV and reaction time; BP, glucose and self-reports are
  taken from the check-in as entered.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spacefuzz import config

ROOT = Path(__file__).resolve().parents[2]
CHECKIN_DIR = ROOT / "data" / "checkins"
TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
REQUIRED = ("plan", "check_window", "check_time", "wake_time", "sleep_last_night_h",
            "sleep_last_3_nights_h", "sleep_efficiency_pct", "systolic_bp", "diastolic_bp",
            "blood_glucose", "mood", "stress", "fatigue_score", "food_water_intake")


class CheckInError(ValueError):
    """The check-in file cannot be used as given."""


# ---------------------------------------------------------------- baseline

@dataclass(frozen=True)
class Baseline:
    resting_hr: float
    rmssd: float
    spo2: float
    core_temp: float
    sleep_h: float
    reaction_ms: float
    glucose: float
    sbp: float
    dbp: float
    history: pd.DataFrame = field(repr=False, compare=False, default=None)
    age_y: float = config.BASELINE_AGE_Y   # personal profile, sets the maximum heart rate


def _centre(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    q_lo, q_hi = config.BASELINE_CENTER_QUANTILES
    return float(lo + (hi - lo) * rng.uniform(q_lo, q_hi))


def synthetic_history(seed: int = config.RANDOM_SEED, days: int = config.BASELINE_DAYS
                      ) -> pd.DataFrame:
    """Daily values for the last `days` days around a seeded person-level centre."""
    rng = np.random.default_rng(seed)
    centre = {
        "resting_hr": _centre(rng, config.BASELINE_HR_RANGE_BPM),
        "rmssd": _centre(rng, config.BASELINE_RMSSD_RANGE_MS),
        "glucose": _centre(rng, config.BASELINE_GLUCOSE_MG_DL_RANGE),
        "spo2": config.BASELINE_SPO2_PCT,
        "core_temp": config.BASELINE_CORE_TEMP_C,
        "sleep_h": config.BASELINE_SLEEP_H,
        "reaction_ms": config.BASELINE_REACTION_MS,
        "sbp": config.BASELINE_SBP_DBP[0],
        "dbp": config.BASELINE_SBP_DBP[1],
    }
    sd = config.BASELINE_DAILY_SD
    rows = {k: v + rng.normal(0.0, sd[k], days) for k, v in centre.items()}
    hist = pd.DataFrame(rows, index=pd.RangeIndex(-days, 0, name="day"))
    hist["spo2"] = hist["spo2"].clip(upper=config.SPO2_MAX_PCT)
    return hist.round(2)


def make_baseline(seed: int = config.RANDOM_SEED) -> Baseline:
    hist = synthetic_history(seed)
    m = hist.mean()
    return Baseline(**{k: float(m[k]) for k in hist.columns}, history=hist)


# ---------------------------------------------------------------- check-in

@dataclass(frozen=True)
class CheckIn:
    plan: str
    check_window: str
    check_time_min: int
    wake_time_min: int
    sleep_last_night_h: float
    sleep_last_3_nights_h: tuple[float, float, float]
    sleep_efficiency_pct: float
    systolic_bp: float
    diastolic_bp: float
    blood_glucose: float
    mood: int
    stress: int
    fatigue_score: int
    food_water_intake: dict
    reaction_time_ms: float | None = None


def _minutes(value, key: str) -> int:
    m = TIME.match(str(value))
    if not m:
        raise CheckInError(f"{key}: {value!r} is not HH:MM")
    return int(m.group(1)) * 60 + int(m.group(2))


def load_checkin(source) -> CheckIn:
    """Load a check-in by plan name ("main_sleep_debt_eva"), path, or dict."""
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if path.suffix != ".json":
            path = CHECKIN_DIR / f"{source}.json"
        elif not path.is_absolute():
            path = ROOT / path
        data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise CheckInError(f"check-in is missing {missing}")
    try:
        return _checkin_from(data)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise CheckInError(f"check-in entry cannot be used: {exc}") from None


def _checkin_from(data: dict) -> CheckIn:
    nights = tuple(float(x) for x in data["sleep_last_3_nights_h"])
    if len(nights) != 3:
        raise CheckInError("sleep_last_3_nights_h needs exactly 3 values")
    return CheckIn(
        plan=str(data["plan"]), check_window=str(data["check_window"]),
        check_time_min=_minutes(data["check_time"], "check_time"),
        wake_time_min=_minutes(data["wake_time"], "wake_time"),
        sleep_last_night_h=float(data["sleep_last_night_h"]), sleep_last_3_nights_h=nights,
        sleep_efficiency_pct=float(data["sleep_efficiency_pct"]),
        systolic_bp=float(data["systolic_bp"]), diastolic_bp=float(data["diastolic_bp"]),
        blood_glucose=float(data["blood_glucose"]), mood=int(data["mood"]),
        stress=int(data["stress"]), fatigue_score=int(data["fatigue_score"]),
        food_water_intake=dict(data["food_water_intake"]),
        reaction_time_ms=float(data["reaction_time_ms"])
        if data.get("reaction_time_ms") is not None else None,
    )


def start_min(checkin: CheckIn) -> int:
    """First simulated minute: the check-in time floored to the 15-min grid."""
    return checkin.check_time_min - checkin.check_time_min % config.STEP_MINUTES


def sleep_debt_h(checkin: CheckIn, baseline: Baseline) -> float:
    """Chain 5: hours short of personal sleep need over the last 3 nights."""
    return float(sum(max(0.0, baseline.sleep_h - n) for n in checkin.sleep_last_3_nights_h))


# ---------------------------------------------------------------- state

@dataclass
class AstronautState:
    minute: int                  # absolute minutes since 00:00 of the plan day
    heart_rate: float            # model value [bpm] (real fluctuation is added on output)
    hrv_rmssd: float             # model value [ms]
    spo2: float                  # [%]
    core_temp: float             # [degC]
    cognitive_fatigue: float     # 0-100
    blood_glucose: float         # [mg/dL]
    systolic_bp: float           # [mmHg]
    diastolic_bp: float
    reaction_time_ms: float
    stress_level: float          # 0-1, chain 1 arousal
    sleep_debt_h: float
    radiation_load: float        # 0-100 strain index, recovers at rest
    eva_dose_ugy: float          # physical dose absorbed during EVA, never decreases
    dose_cum_ugy: float          # physical dose absorbed today
    rt_offset_ms: float          # measured minus modelled reaction time at check-in
    mood: int                    # self-reports, carried from the check-in
    stress_self: int
    fatigue_score: int


def max_heart_rate(baseline: Baseline) -> float:
    """Age-predicted maximum heart rate, 220 - age (illustrative, about +-10 bpm per person)."""
    return config.HR_MAX_INTERCEPT_BPM - baseline.age_y


def resting_hr(baseline: Baseline, debt_h: float) -> float:
    return baseline.resting_hr + config.SLEEP_DEBT_HR_PER_H * debt_h


def resting_rmssd(baseline: Baseline, debt_h: float) -> float:
    return baseline.rmssd * max(0.3, 1.0 - config.SLEEP_DEBT_RMSSD_PER_H * debt_h)


def model_reaction_ms(baseline: Baseline, debt_h: float, fatigue: float) -> float:
    return (baseline.reaction_ms * (1 + config.SLEEP_DEBT_RT_PER_H * debt_h)
            + config.RT_FATIGUE_MS_PER_POINT * fatigue)


def initial_state(checkin: CheckIn, baseline: Baseline) -> AstronautState:
    debt = sleep_debt_h(checkin, baseline)
    fatigue = min(config.FATIGUE_MAX, max(0.0,       # a score below 1 (not on the scale) -> 0
                  (checkin.fatigue_score - 1) * config.FATIGUE_PER_SAMN_PERELLI_POINT))
    rt_model = model_reaction_ms(baseline, debt, fatigue)
    rt = checkin.reaction_time_ms if checkin.reaction_time_ms is not None else rt_model
    return AstronautState(
        minute=checkin.check_time_min,
        heart_rate=resting_hr(baseline, debt), hrv_rmssd=resting_rmssd(baseline, debt),
        spo2=baseline.spo2, core_temp=baseline.core_temp, cognitive_fatigue=fatigue,
        blood_glucose=checkin.blood_glucose, systolic_bp=checkin.systolic_bp,
        diastolic_bp=checkin.diastolic_bp, reaction_time_ms=rt, stress_level=0.0,
        sleep_debt_h=debt, radiation_load=0.0, eva_dose_ugy=0.0, dose_cum_ugy=0.0,
        rt_offset_ms=rt - rt_model, mood=checkin.mood, stress_self=checkin.stress,
        fatigue_score=checkin.fatigue_score)
