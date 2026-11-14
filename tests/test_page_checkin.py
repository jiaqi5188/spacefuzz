"""Tests for pages/1_Morning_check-in.py (self-report with "how to measure" + reaction test)."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import checkin_store as cs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "1_Morning_check-in.py"
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
    assert "example data" in t and "synthetic" in t
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
    assert "local profile" in t                                     # edited -> re-forecast


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
