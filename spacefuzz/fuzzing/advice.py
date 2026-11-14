"""Advice library (plan 3.11). This module has the "act now" part: advice triggered by an
abnormal morning / midday check-in value, each with its escalation level (plan 1.3):

- self:   the astronaut handles it          (e.g. low glucose -> fast carbs, recheck)
- crew:   alert the crew / a crewmate       (e.g. heart rate above baseline, short sleep)
- ground: report to the ground flight surgeon (e.g. blood pressure still out of range)

Urgent tiers:
- glucose below 54 mg/dL (ADA level-2 hypoglycaemia) is crew level with the ground
  notified, never self-care;
- severe hypertension (>= 180 / >= 120 mmHg, AHA/ACC 2025 term): as AHA advises, recheck
  after 5 minutes; meanwhile today's EVA and high-load tasks are put on hold (a
  spaceflight-specific precaution: the EVA is the same day and cannot wait). If the
  recheck (bp_recheck) is still >= 180 / 120, report to ground and keep the hold.
  Symptoms (chest pain, severe headache, vision changes) are not on the check-in form, so
  the advice text tells the crew to contact the flight surgeon at once if they occur.
Holds carry a fixed status code (HOLD_* below) for the engine and UI; holds() maps each
held task type to its code. Logic must test the code, never the text: the human-readable
`restriction` sentence (action_restrictions output field, plan 3.10) can be reworded freely.

The "prevent today" part (prevent_today -> the advice_today field) is driven by the
forecast: one item per rule that crashed in the probability track, plus a sleep item when
there is sleep debt; its escalation follows the three-level risk (plan 1.3). Measurement
prompts (measurement_prompts) name the check-in entry whose measurement error moves the
risk most (computed by adjuster.py). Thresholds are in config (ACT_NOW_*), all illustrative.

If the data quality gate says "untrusted", act_now returns only the measure-again
advice: no advice is ever given on data the gate rejected. Check-in values the gate
marked unusable are skipped and named in a "remeasure" item instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from spacefuzz import config
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType
from spacefuzz.core.quality_gate import GateResult

# Hold status codes: stable identifiers for the engine and UI.
HOLD_PENDING_REMEASUREMENT = "pending_remeasurement"    # waiting for the 5-minute BP recheck
HOLD_PENDING_FLIGHT_SURGEON = "pending_flight_surgeon"  # recheck confirmed, ground decides

SEVERE_BP_HOLD = (PlanType.EVA, PlanType.DOCKING, PlanType.PRECISION_MANUAL,
                  PlanType.GENERIC_HIGH_LOAD)
SEVERE_BP_SYMPTOMS = ("If you have chest pain, severe headache, or vision changes, "
                      "contact the flight surgeon now.")

ESCALATION_ORDER = {"ground": 0, "crew": 1, "self": 2}


@dataclass(frozen=True)
class Advice:
    code: str          # stable id, e.g. "low_glucose"
    text: str          # English, shown to the crew
    escalation: str    # self | crew | ground
    indicator: str     # the indicator that triggered it
    notify_ground: bool = False  # also tell the flight surgeon, whatever the level
    hold: tuple = ()             # plan task types on hold until resolved
    hold_status: str = ""        # HOLD_* code for `hold`; logic uses this, not the text
    restriction: str = ""        # human-readable text for the action_restrictions field

    @property
    def effective_escalation(self) -> str:
        return "ground" if self.notify_ground else self.escalation


def act_now(checkin: st.CheckIn, baseline: st.Baseline, resting_hr: float | None = None,
            gate: GateResult | None = None,
            bp_recheck: tuple[float, float] | None = None) -> list[Advice]:
    """Act-now advice for a check-in, most urgent first.

    resting_hr: sustained resting heart rate from the wearable (e.g. median of the last
    10 minutes); None when no reading is available, then heart rate is not judged.
    bp_recheck: (systolic, diastolic) of the 5-minute recheck after a severe reading.
    """
    if gate is not None and not gate.can_forecast:
        return [Advice("measure_again", gate.message, "self", "data_quality")]
    unusable = set(gate.missing) if gate is not None else set()
    items: list[Advice] = []

    g = checkin.blood_glucose
    if "blood_glucose" not in unusable and g < config.ACT_NOW_GLUCOSE_SEVERE_LOW:
        items.append(Advice(
            "severe_low_glucose",
            f"Glucose {g:.0f} mg/dL is severely low: take 15 g of fast-acting carbs now, have "
            f"a crewmate stay with you and recheck in 15 minutes; notify the flight surgeon.",
            "crew", "blood_glucose", notify_ground=True))
    elif "blood_glucose" not in unusable and g < config.GLUCOSE_LOW_MG_DL:
        items.append(Advice(
            "low_glucose",
            f"Glucose {g:.0f} mg/dL is low: take 15 g of fast-acting carbs and recheck "
            f"glucose in 15 minutes.", "self", "blood_glucose"))

    if resting_hr is not None and "heart_rate" not in unusable:
        above = resting_hr - baseline.resting_hr
        if above >= config.ACT_NOW_HR_ABOVE_BASELINE_BPM:
            items.append(Advice(
                "high_heart_rate",
                f"Resting heart rate {resting_hr:.0f} bpm is {above:.0f} bpm above your "
                f"baseline: rest for 15 minutes, then recheck.", "crew", "heart_rate"))

    sbp, dbp = checkin.systolic_bp, checkin.diastolic_bp
    if not {"systolic_bp", "diastolic_bp"} & unusable:
        (s_lo, s_hi), (d_lo, d_hi) = config.ACT_NOW_BP_RANGE["sbp"], config.ACT_NOW_BP_RANGE["dbp"]
        severe = config.ACT_NOW_BP_SEVERE
        is_severe = lambda s, d: s >= severe["sbp"] or d >= severe["dbp"]
        hold_text = "EVA and high-load tasks on hold until the blood pressure recheck"
        if is_severe(sbp, dbp) and bp_recheck is None:
            items.append(Advice(
                "severe_hypertension_recheck",
                f"Blood pressure {sbp:.0f}/{dbp:.0f} mmHg is in the severe hypertension range: "
                f"sit and rest, then recheck in {config.ACT_NOW_BP_RECHECK_MIN} minutes. Today's "
                f"EVA and high-load tasks are on hold until the recheck. {SEVERE_BP_SYMPTOMS}",
                "crew", "blood_pressure", hold=SEVERE_BP_HOLD,
                hold_status=HOLD_PENDING_REMEASUREMENT, restriction=hold_text))
        elif is_severe(sbp, dbp) and is_severe(*bp_recheck):
            rs, rd = bp_recheck
            items.append(Advice(
                "severe_hypertension",
                f"Blood pressure is still {rs:.0f}/{rd:.0f} mmHg on recheck (severe "
                f"hypertension): contact the flight surgeon now. EVA and high-load tasks stay "
                f"on hold. {SEVERE_BP_SYMPTOMS}",
                "ground", "blood_pressure", hold=SEVERE_BP_HOLD,
                hold_status=HOLD_PENDING_FLIGHT_SURGEON,
                restriction="EVA and high-load tasks on hold pending the flight surgeon"))
        else:
            if is_severe(sbp, dbp):          # recheck came back below the severe range
                sbp, dbp = bp_recheck
            if not (s_lo <= sbp <= s_hi and d_lo <= dbp <= d_hi):
                items.append(Advice(
                    "blood_pressure",
                    f"Blood pressure {sbp:.0f}/{dbp:.0f} mmHg is out of range: recheck twice; "
                    f"if it is still out of range, contact the flight surgeon.",
                    "ground", "blood_pressure"))

    debt = st.sleep_debt_h(checkin, baseline)
    last = checkin.sleep_last_night_h
    if "sleep_last_night_h" not in unusable and (
            last < config.ACT_NOW_SLEEP_SHORT_H or debt >= config.ACT_NOW_SLEEP_DEBT_H):
        items.append(Advice(
            "short_sleep",
            f"Short sleep ({last:.1f} h last night, {debt:.1f} h sleep debt over 3 nights): "
            f"schedule a 20-minute nap before demanding tasks and go to bed early tonight.",
            "crew", "sleep"))

    if checkin.mood <= config.ACT_NOW_MOOD_LOW or checkin.stress >= config.ACT_NOW_STRESS_HIGH:
        items.append(Advice(
            "mood_stress",
            f"Mood {checkin.mood}/5, stress {checkin.stress}/5: talk with a crewmate or "
            f"schedule a psychological support session today.", "crew", "mood_stress"))

    if unusable:
        names = ", ".join(sorted(unusable))
        items.append(Advice("remeasure", f"Measure again: {names} could not be used.",
                            "self", "data_quality"))

    return sorted(items, key=lambda a: ESCALATION_ORDER[a.effective_escalation])  # stable


def overall_escalation(items: list[Advice]) -> str:
    """Highest escalation among the items, counting ground notifications ("self" if none)."""
    return min((a.effective_escalation for a in items), key=ESCALATION_ORDER.get, default="self")


def advice_now(items: list[Advice]) -> list[str]:
    """The `advice_now` output field (plan 3.10)."""
    return [a.text for a in items]


def holds(items: list[Advice]) -> dict[PlanType, str]:
    """Held plan task type -> HOLD_* status code (the engine / UI decide on these codes).

    If several items hold the same type, the later stage wins (pending_flight_surgeon
    over pending_remeasurement)."""
    rank = {HOLD_PENDING_REMEASUREMENT: 0, HOLD_PENDING_FLIGHT_SURGEON: 1}
    out: dict[PlanType, str] = {}
    for a in items:
        for t in a.hold:
            if t not in out or rank.get(a.hold_status, -1) > rank.get(out[t], -1):
                out[t] = a.hold_status
    return out


def action_restrictions(items: list[Advice]) -> list[str]:
    """The `action_restrictions` output field (plan 3.10)."""
    return [a.restriction for a in items if a.restriction]


# ---------------------------------------------------------------- prevent today (plan 3.11)

RATING_ESCALATION = {"LOW": "self", "MEDIUM": "crew", "HIGH": "ground", "CRITICAL": "ground",
                     "UNKNOWN": "self"}


def prevent_today(by_rule: dict, rating: str, checkin: st.CheckIn,
                  baseline: st.Baseline) -> list[Advice]:
    """Advice that lowers today's forecast risk. by_rule: crashed samples per rule from the
    probability track (plan or adjusted plan); escalation follows the risk rating."""
    level = RATING_ESCALATION[rating]
    hr_limit = config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(baseline)
    wake_window = checkin.wake_time_min + int(config.RULE1_WINDOW_H * 60)
    texts = {
        "cardiovascular": ("eva_heat", "heart_rate",
                           f"Drink water and take carbs before EVA; keep the planned rest breaks "
                           f"and return to cool down if heart rate stays above {hr_limit:.0f} bpm."),
        "cognitive": ("cognitive_load", "cognitive_fatigue",
                      "Keep demanding precision work out of the late afternoon; take a short "
                      "break every 2 hours and ask a crewmate to double-check critical steps."),
        "metabolic": ("eva_fuel", "blood_glucose",
                      "Eat a carbohydrate snack before EVA and bring fast carbs in the suit."),
        "radiation_dose": ("eva_dose", "radiation",
                           "Keep EVA time outside the habitat short today and check the dose "
                           "meter during the EVA."),
        "rule1_sleep_eva": ("eva_after_short_sleep", "sleep",
                            f"After short sleep, do not start an EVA before {pl.clock(wake_window)} "
                            f"(12 h after waking)."),
        "rule2_exercise_eva": ("training_order", "exercise",
                               "Do strength or treadmill training after the EVA, not in the "
                               "2 hours before it."),
        "rule3_co2_precision": ("co2_break", "co2",
                                "Take a rest break in fresh air before docking or precision "
                                "manual work after long work in high CO2."),
        "rule4_spe_eva": ("spe_check", "radiation",
                          "Check the solar particle event forecast before the EVA; do not start "
                          "an EVA during a warning and stay in the shielded area."),
    }
    items = [Advice(code, text, level, indicator)
             for rule, (code, indicator, text) in texts.items() if by_rule.get(rule)]
    if st.sleep_debt_h(checkin, baseline) >= config.ACT_NOW_SLEEP_DEBT_H and items:
        items.append(Advice("sleep_tonight",
                            "Go to bed 1 hour earlier tonight to start repaying the sleep debt.",
                            level, "sleep"))
    if not items:
        items.append(Advice("keep_going", "No added risk forecast today: keep the plan, "
                            "drink regularly and keep the usual breaks.", "self", "none"))
    return items


def advice_today(items: list[Advice]) -> list[str]:
    """The `advice_today` output field (plan 3.10)."""
    return [a.text for a in items]


MEASURE_TEXT = {
    "sleep_last_night_h": "Confirm last night's sleep with the wearable record",
    "blood_glucose": "Recheck blood glucose first at the midday check",
    "systolic_bp": "Recheck blood pressure first at the midday check",
    "diastolic_bp": "Recheck blood pressure first at the midday check",
    "reaction_time_ms": "Repeat the reaction time test first at the midday check",
}


def measurement_prompts(sensitivity, top: int = 2) -> list[str]:
    """The `measurement_prompts` field. sensitivity: [(field, risk change within its
    measurement error)], largest first; entries that do not move the risk are dropped."""
    out, seen = [], set()
    for name, delta in sorted(sensitivity, key=lambda x: -x[1]):
        text = MEASURE_TEXT.get(name)
        if delta <= 0 or text is None or text in seen:
            continue
        seen.add(text)
        out.append(f"{text} (within its measurement error the risk moves by "
                   f"{delta * 100:.0f} points).")
        if len(out) == top:
            break
    return out
