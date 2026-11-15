"""Tests for page_files/1_Morning_check-in.py (self-report with "how to measure" + reaction test)."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import checkin_store as cs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "page_files" / "1_Morning_check-in.py"
MAIN = "main_sleep_debt_eva"


def page(ss=None):
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    for k, v in (ss or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def num(at, label):
    return next(n for n in at.number_input if n.label == label)


def text(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                     + [h.proto.body for h in at.get("html")])


def test_prefilled_with_the_scenario_example_and_how_to_measure():
    at = page()
    ex = cs.example(MAIN)
    assert num(at, "Systolic (mmHg)").value == ex["systolic_bp"]
    assert num(at, "Diastolic (mmHg)").value == ex["diastolic_bp"]
    assert num(at, "Blood glucose (mg/dL)").value == ex["blood_glucose"]
    assert num(at, "Sleep last night (h)").value == ex["sleep_last_night_h"]
    t = text(at)
    for how in ("2 readings", "Fingerstick", "wristband", "Samn-Perelli"):
        assert how in t, how
    assert "example</span>" in t and "synthetic" in t            # tag + top bar
    assert "not used by the forecast" in t                     # food / water: recorded only


def test_save_edits_runs_the_gate_and_previews_act_now():
    at = page()
    num(at, "Systolic (mmHg)").set_value(150.0)
    next(b for b in at.button if b.label == "Save check-in").click().run()
    assert not at.exception
    edits = at.session_state["checkin_edits"]
    assert edits[MAIN]["systolic_bp"] == 150.0
    t = text(at)
    assert "Data quality: ok" in t
    assert "Blood pressure 150/82 mmHg is out of range" in t        # act-now preview
    assert any("local profile" in (c.help or "") for c in at.caption)   # edited -> re-forecast
    assert "few minutes" in t


def test_implausible_entry_is_flagged_by_the_gate():
    at = page()
    num(at, "Blood glucose (mg/dL)").set_value(5.0)
    next(b for b in at.button if b.label == "Save check-in").click().run()
    t = text(at)
    assert "blood_glucose" in t and "outside" in t
    assert "Data quality: untrusted" in t                 # a rule input: never simulated
    assert "Re-measure: blood glucose (5 outside 20-600)" in t


def test_reset_back_to_the_example():
    at = page({"checkin_edits": {MAIN: {**cs.example(MAIN), "systolic_bp": 150}}})
    next(b for b in at.button if b.label == "Reset to example values").click().run()
    assert MAIN not in at.session_state["checkin_edits"]


def test_follows_the_scenario_chosen_on_the_dashboard():
    at = page({"active_scenario": "backup_high_co2"})
    assert num(at, "Systolic (mmHg)").value == cs.example("backup_high_co2")["systolic_bp"]


def _with_reaction(monkeypatch, result):
    from ui import reaction
    monkeypatch.setattr(reaction, "run", lambda duration_s, key: result)


def test_reset_clears_a_check_in_that_has_a_reaction_result(monkeypatch):
    _with_reaction(monkeypatch, {"rts": [300, 320, 340], "false_starts": 0, "duration_s": 60})
    at = page()
    assert at.session_state["checkin_edits"][MAIN]["reaction_time_ms"] == 320.0
    next(b for b in at.button if b.label == "Reset to example values").click().run()
    assert not at.exception
    assert MAIN not in at.session_state["checkin_edits"]       # the old result is not re-saved
    assert not [b for b in at.button if b.label == "Reset to example values"]


def test_a_test_without_plausible_responses_changes_nothing(monkeypatch):
    _with_reaction(monkeypatch, {"rts": [5, 9000], "false_starts": 4, "duration_s": 60})
    at = page()
    edits = at.session_state["checkin_edits"] if "checkin_edits" in at.session_state else {}
    assert MAIN not in edits
    assert "Mean -- ms" in text(at)


def test_number_formats_integers_and_one_decimal_for_sleep():
    """User review 2026-11-14: BP, glucose and sleep efficiency as integers; sleep 1 decimal."""
    at = page()
    fmt = {n.label: n.proto.format for n in at.number_input}
    for label in ("Systolic (mmHg)", "Diastolic (mmHg)", "Blood glucose (mg/dL)",
                  "Sleep efficiency (%)"):
        assert fmt[label] == "%.0f", label
    for label in ("Sleep last night (h)", "Night before (h)", "Two nights before (h)",
                  "Water (L, 24 h)"):
        assert fmt[label] == "%.1f", label


def test_act_now_preview_says_the_dashboard_has_the_final_advice():
    t = text(page())
    i = t.index("Act now (preview)")
    assert "Final advice follows the adjusted plan on the dashboard." in t[i:]


def save(at):
    return next(b for b in at.button if b.label == "Save check-in")


def idle(at):
    """Save shown as disabled: greyed out and a no-op, but still clickable, so a value typed
    and then Save pressed straight away (the field commits on that click) is not lost."""
    return 'data-testid="save-idle"' in text(at)


def test_save_looks_disabled_until_a_value_changes():
    """Nothing to save, nothing to press (user review 11/15)."""
    at = page()
    assert idle(at)
    save(at).click().run()                                     # a no-op: nothing is written
    assert "checkin_edits" not in at.session_state or MAIN not in at.session_state["checkin_edits"]
    num(at, "Water (L, 24 h)").set_value(3.0).run()
    assert not idle(at)
    save(at).click().run()
    assert at.session_state["checkin_edits"][MAIN]["food_water_intake"]["water_l_24h"] == 3.0
    assert idle(at)                                            # saved: nothing new again


def test_back_to_the_saved_value_looks_disabled_again():
    at = page()
    old = num(at, "Systolic (mmHg)").value
    num(at, "Systolic (mmHg)").set_value(150.0).run()
    num(at, "Systolic (mmHg)").set_value(old).run()
    assert idle(at)


def test_reset_puts_the_example_back_in_the_fields():
    at = page({"checkin_edits": {MAIN: {**cs.example(MAIN), "systolic_bp": 150}}})
    assert num(at, "Systolic (mmHg)").value == 150
    next(b for b in at.button if b.label == "Reset to example values").click().run()
    assert num(at, "Systolic (mmHg)").value == cs.example(MAIN)["systolic_bp"]
    assert idle(at)


def test_reset_button_stands_out():
    at = page({"checkin_edits": {MAIN: {**cs.example(MAIN), "systolic_bp": 150}}})
    b = next(b for b in at.button if b.label == "Reset to example values")
    assert b.key == "reset_checkin"
    css = (ROOT / "ui" / "styles.css").read_text(encoding="utf-8")
    assert ".st-key-reset_checkin button" in css


def test_idle_save_has_no_hover_or_press_effect():
    at = page()
    t = text(at)
    for state in (":hover", ":active", ":focus-visible"):
        assert f".st-key-save_checkin button{state}" in t
    assert "box-shadow:none !important" in t and "transform:none !important" in t


def test_no_reset_while_a_forecast_runs():
    import time
    edited = {"checkin_edits": {MAIN: {**cs.example(MAIN), "systolic_bp": 150}}}
    at = page({**edited, "_sf_job": {"name": "forecast", "t": time.time(), "t0": time.time()}})
    assert not [b for b in at.button if b.label == "Reset to example values"]
    assert any("forecast is running" in c.value for c in at.caption)
    at = page(edited)
    assert [b for b in at.button if b.label == "Reset to example values"]
