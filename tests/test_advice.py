"""Tests for advice.act_now (plan 3.11, act-now part)."""
import dataclasses

import pytest

from spacefuzz.fuzzing import advice as adv
from spacefuzz import config
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType

BASE = st.make_baseline()


def checkin(name="safe_day", **changes):
    return dataclasses.replace(st.load_checkin(name), **changes)


def codes(items):
    return [a.code for a in items]


def only(items, code):
    (a,) = [a for a in items if a.code == code]
    return a


# ---- acceptance: the four cases

def test_low_glucose_self_care():
    a = only(adv.act_now(checkin(blood_glucose=62), BASE), "low_glucose")
    assert a.escalation == "self"
    assert "carbs" in a.text and "15 minutes" in a.text and "62" in a.text


def test_high_heart_rate_alerts_crew():
    hr = BASE.resting_hr + config.ACT_NOW_HR_ABOVE_BASELINE_BPM + 5
    a = only(adv.act_now(checkin(), BASE, resting_hr=hr), "high_heart_rate")
    assert a.escalation == "crew" and "rest" in a.text.lower() and "recheck" in a.text


def test_high_blood_pressure_goes_to_ground():
    a = only(adv.act_now(checkin(systolic_bp=152, diastolic_bp=96), BASE), "blood_pressure")
    assert a.escalation == "ground"
    assert "recheck twice" in a.text and "flight surgeon" in a.text and "152/96" in a.text


def test_short_sleep_alerts_crew():
    a = only(adv.act_now(checkin(sleep_last_night_h=5.0, sleep_last_3_nights_h=(7.0, 7.0, 5.0)),
                         BASE), "short_sleep")
    assert a.escalation == "crew" and "nap" in a.text


# ---- edges

def test_sleep_debt_alone_triggers_for_the_main_plan():
    items = adv.act_now(st.load_checkin("main_sleep_debt_eva"), BASE)
    assert "short_sleep" in codes(items)               # 6.2 h last night, 6.9 h debt


def test_low_blood_pressure_also_out_of_range():
    assert "blood_pressure" in codes(adv.act_now(checkin(systolic_bp=84, diastolic_bp=55), BASE))


def test_poor_mood_or_stress_suggests_support():
    a = only(adv.act_now(checkin(mood=2, stress=4), BASE), "mood_stress")
    assert a.escalation == "crew" and "crewmate" in a.text


def test_healthy_checkin_gives_no_act_now_advice():
    assert adv.act_now(checkin(), BASE, resting_hr=BASE.resting_hr + 2) == []


def test_heart_rate_not_judged_without_a_reading():
    assert "high_heart_rate" not in codes(adv.act_now(checkin(), BASE))


def test_most_urgent_first_and_overall_escalation():
    items = adv.act_now(checkin(blood_glucose=62, systolic_bp=150, sleep_last_night_h=5.0,
                                sleep_last_3_nights_h=(6.0, 6.0, 5.0)), BASE)
    assert [a.escalation for a in items] == ["ground", "crew", "self"]
    assert adv.overall_escalation(items) == "ground"
    assert adv.overall_escalation([]) == "self"


def test_output_field_is_plain_english_text():
    items = adv.act_now(checkin(blood_glucose=62), BASE)
    texts = adv.advice_now(items)
    assert texts == [items[0].text] and texts[0].isascii()


# ---- data quality gate

def test_untrusted_data_only_asks_to_measure_again():
    gate = qg.GateResult(status="untrusted", missing=["heart_rate", "spo2"])
    items = adv.act_now(checkin(blood_glucose=62), BASE, gate=gate)
    assert codes(items) == ["measure_again"]
    assert items[0].text == qg.UNTRUSTED_MESSAGE


def test_unusable_checkin_value_is_not_advised_on_but_flagged():
    # an implausible value the rules read stops the forecast; the one item names what to redo
    gate = qg.check(checkin=checkin(blood_glucose=900))
    items = adv.act_now(checkin(blood_glucose=900), BASE, gate=gate)
    assert "low_glucose" not in codes(items)
    a = only(items, "measure_again")
    assert "Re-measure: blood glucose (900 outside 20-600)" in a.text and a.escalation == "self"
    # the optional reaction test left out: still forecast, named in a remeasure item
    gate = qg.check(checkin=checkin(reaction_time_ms=None))
    a = only(adv.act_now(checkin(reaction_time_ms=None), BASE, gate=gate), "remeasure")
    assert "reaction_time_ms" in a.text


# ---- most urgent tiers

def test_severe_low_glucose_alerts_crew_and_notifies_ground():
    items = adv.act_now(checkin(blood_glucose=48), BASE)
    a = only(items, "severe_low_glucose")
    assert a.escalation == "crew" and a.notify_ground
    assert "crewmate" in a.text and "flight surgeon" in a.text and "48" in a.text
    assert "low_glucose" not in codes(items)                  # one glucose item, the urgent one
    assert adv.overall_escalation(items) == "ground"
    assert items[0] is a                                      # ranks with ground items


def test_glucose_between_54_and_70_stays_self_care():
    a = only(adv.act_now(checkin(blood_glucose=config.ACT_NOW_GLUCOSE_SEVERE_LOW), BASE),
             "low_glucose")
    assert a.escalation == "self" and not a.notify_ground


@pytest.mark.parametrize("sbp, dbp", [(185, 100), (150, 122)])
def test_severe_hypertension_first_reading_rechecks_and_holds_eva(sbp, dbp):
    """AHA: recheck after 5 min; spaceflight: hold today's EVA / high-load tasks meanwhile."""
    items = adv.act_now(checkin(systolic_bp=sbp, diastolic_bp=dbp), BASE)
    a = only(items, "severe_hypertension_recheck")
    assert a.escalation == "crew" and not a.notify_ground
    assert "severe hypertension" in a.text and "5 minutes" in a.text
    assert "chest pain, severe headache, or vision changes" in a.text   # symptoms -> call now
    held = {PlanType.EVA, PlanType.DOCKING, PlanType.PRECISION_MANUAL, PlanType.GENERIC_HIGH_LOAD}
    assert a.hold_status == adv.HOLD_PENDING_REMEASUREMENT == "pending_remeasurement"
    assert adv.holds(items) == {t: adv.HOLD_PENDING_REMEASUREMENT for t in held}
    assert len(adv.action_restrictions(items)) == 1       # wording is free; logic uses the code
    assert "blood_pressure" not in codes(items) and adv.overall_escalation(items) == "crew"


def test_severe_hypertension_confirmed_on_recheck_goes_to_ground():
    items = adv.act_now(checkin(systolic_bp=186, diastolic_bp=104), BASE, bp_recheck=(184, 102))
    a = only(items, "severe_hypertension")
    assert a.escalation == "ground" and "184/102" in a.text and "flight surgeon" in a.text
    assert a.hold_status == adv.HOLD_PENDING_FLIGHT_SURGEON
    assert adv.holds(items)[PlanType.EVA] == adv.HOLD_PENDING_FLIGHT_SURGEON  # still held


def test_recheck_below_severe_lifts_the_hold():
    normal = adv.act_now(checkin(systolic_bp=186, diastolic_bp=104), BASE, bp_recheck=(128, 82))
    assert not {"severe_hypertension", "severe_hypertension_recheck", "blood_pressure"} & set(codes(normal))
    assert adv.holds(normal) == {} and adv.action_restrictions(normal) == []
    high = adv.act_now(checkin(systolic_bp=186, diastolic_bp=104), BASE, bp_recheck=(162, 96))
    assert "recheck twice" in only(high, "blood_pressure").text and "162/96" in only(high, "blood_pressure").text


def test_high_but_not_crisis_blood_pressure_rechecks_first():
    a = only(adv.act_now(checkin(systolic_bp=170, diastolic_bp=100), BASE), "blood_pressure")
    assert "recheck twice" in a.text


# ---- prevent today (driven by the forecast) and measurement prompts (plan 3.11)

def test_prevent_today_follows_the_rules_that_crashed():
    c = checkin("main_sleep_debt_eva")
    items = adv.prevent_today({"cardiovascular": 400, "cognitive": 50}, "HIGH", c, BASE)
    got = codes(items)
    assert "eva_heat" in got and "cognitive_load" in got and "sleep_tonight" in got
    assert all(a.escalation == "ground" for a in items)          # red -> ground
    heat = only(items, "eva_heat")
    assert "140" in heat.text or f"{config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(BASE):.0f}" in heat.text


def test_prevent_today_escalation_follows_the_rating():
    c = checkin("safe_day")
    assert all(a.escalation == "crew" for a in adv.prevent_today({"rule4_spe_eva": 3}, "MEDIUM", c, BASE))
    low = adv.prevent_today({}, "LOW", c, BASE)
    assert codes(low) == ["keep_going"] and low[0].escalation == "self"


def test_prevent_today_rule_texts():
    c = checkin("main_sleep_debt_eva")
    items = adv.prevent_today({"rule1_sleep_eva": 5, "rule2_exercise_eva": 5, "rule3_co2_precision": 5,
                               "rule4_spe_eva": 5, "metabolic": 5, "radiation_dose": 5}, "HIGH", c, BASE)
    text = " ".join(a.text for a in items)
    assert "18:00" in text                                       # 12 h after waking at 06:00
    for code in ("eva_after_short_sleep", "training_order", "co2_break", "spe_check",
                 "eva_fuel", "eva_dose"):
        assert code in codes(items), code
    assert all(t.isascii() for t in (a.text for a in items))


def test_no_sleep_advice_without_debt():
    c = checkin("safe_day")
    assert "sleep_tonight" not in codes(adv.prevent_today({"cardiovascular": 3}, "MEDIUM", c, BASE))


def test_measurement_prompt_text():
    texts = adv.measurement_prompts([("sleep_last_night_h", 0.21), ("blood_glucose", 0.01),
                                     ("systolic_bp", 0.0)])
    assert len(texts) == 2                                      # zero change is not worth a prompt
    assert texts[0].startswith("Confirm last night's sleep")
    assert "21" in texts[0]
    assert "glucose" in texts[1].lower()
