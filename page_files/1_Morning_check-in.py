"""Morning check-in (plan 4.1 annex 1): self-report with "how to measure" + reaction test.

Pre-filled with the selected scenario's example check-in (synthetic example data). Saving
runs the data-quality gate and previews the act-now advice; the edited check-in is what the
dashboard forecasts next (with the local profile, since the cached full-profile results
belong to the example check-in). The reaction test writes reaction_time_ms by itself.
"""

import hashlib
import json

import streamlit as st

from spacefuzz.fuzzing import advice
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state
import ui_config
from ui import checkin_store as cs
from ui import actions, assist, nav, reaction, theme
from ui.common import brief_html, esc, illustrative_badge, neutral_nap, tip

SAMN_PERELLI = {1: "1 fully alert, wide awake", 2: "2 very lively, responsive",
                3: "3 okay, somewhat fresh", 4: "4 a little tired, less than fresh",
                5: "5 moderately tired, let down", 6: "6 extremely tired, hard to concentrate",
                7: "7 completely exhausted"}
MEALS = ("normal", "reduced", "skipped a meal")

# How to measure (example procedure shown to the crew; illustrative, not a clinical protocol).
HOW = {
    "bp": "Seated, back supported, after 5 min rest; cuff on the bare upper arm at heart "
          "level; take 2 readings 1 min apart and enter the average.",
    "glucose": "Fingerstick glucometer before breakfast.",
    "sleep": "From the wristband sleep record; without one, your own estimate (less precise).",
    "efficiency": "Time asleep / time in bed, from the wristband.",
    "mood": "Your own rating right now: mood 1 very low to 5 very good; stress 1 none to "
            "5 very high.",
    "fatigue": "Samn-Perelli 7-point fatigue scale: pick the line that fits you now.",
    "food": "Meals in the last 24 h and water from your drink log. Recorded; not used by the "
            "forecast yet.",
}



def gate_html(ci: state.CheckIn) -> str:
    gate = qg.check(checkin=ci)
    color = {"ok": "green", "degraded": "yellow", "untrusted": "red"}[gate.status]
    issues = "".join(f"<div>• {esc(i.field)}: {esc(i.detail)}</div>" for i in gate.issues)
    items = advice.act_now(ci, state.make_baseline(), gate=gate)
    act = "".join(f"<div>• {brief_html(neutral_nap(a.text))}</div>" for a in items) or \
        "<div>Nothing to act on now.</div>"
    msg = (f'<div style="margin-top:6px">{esc(gate.message)}</div>'
           if not gate.can_forecast else "")
    return (f'<div class="sf-panel" data-testid="checkin-gate"><div class="sf-h">Data quality'
            f'</div><div style="color:var(--sf-{color});font-weight:700">Data quality: '
            f'{gate.status}</div>{issues}{msg}<div class="sf-h" style="margin-top:10px">'
            f'Act now (preview)</div>{act}<div class="sf-small" data-testid="act-now-note" '
            f'style="margin-top:6px">{esc(FINAL_ADVICE)}</div></div>')


def ai_hint(ci: state.CheckIn) -> None:
    """Optional, with an OPENAI_API_KEY only: a plain-language hint for a rejected check-in
    (ui/assist.py). The gate's decision above stays as it is."""
    gate = qg.check(checkin=ci)
    if gate.can_forecast or not assist.available():
        return
    if st.button(assist.BUTTON, key="assist_hint", icon=":material/help:"):
        with st.spinner("Asking..."):
            st.session_state["assist_hint_result"] = (gate.message, assist.hint(gate))
    asked, text = st.session_state.get("assist_hint_result", (None, None))
    if asked != gate.message:              # an answer for other values is not shown
        return
    if text is None:
        st.caption(assist.UNAVAILABLE)
    else:
        st.html(f'<div class="sf-panel" data-testid="assist-hint"><div class="sf-h">'
                f'{esc(assist.LABEL)}</div><div>{esc(text)}</div></div>')


FINAL_ADVICE = "Final advice follows the adjusted plan on the dashboard."
PREFILLED = ("Pre-filled with this scenario's synthetic example values: edit, save, then press "
             "Run Forecast on the dashboard.")


# One short visible line per section (plan 4.1: "how to measure" stays visible); the full
# procedure is in the heading's (i).
HOW_SHORT = {
    "bp": "Seated, 5 min rest, average of 2 readings.",
    "glucose": "Fingerstick before breakfast.",
    "sleep": "Wristband record; otherwise your own estimate.",
    "mood": "Your rating right now.",
    "fatigue": "Pick the line that fits you now.",
    "food": "Recorded only; not used by the forecast yet.",
}


def section(title: str, how: str, example: bool = False, short: str = "") -> None:
    """A form section heading with the full "how to measure" note in its (i), and a short
    visible version of it underneath."""
    tag = " <span class='sf-tag'>example</span>" if example else ""
    st.markdown(f"**{title}**{tip(how)}{tag}", unsafe_allow_html=True)
    if short:
        st.caption(short)


# Save with nothing to save: grey, and no hover glow, press shift or focus ring either
# (!important: the primary button's own hover rule is more specific)
IDLE_SAVE_CSS = (
    ".st-key-save_checkin button, .st-key-save_checkin button:hover, "
    ".st-key-save_checkin button:active, .st-key-save_checkin button:focus, "
    ".st-key-save_checkin button:focus-visible { background:var(--sf-panel-2) !important; "
    "border-color:var(--sf-line) !important; box-shadow:none !important; "
    "transform:none !important; outline:none !important; cursor:default !important; } "
    ".st-key-save_checkin button * { color:var(--sf-muted) !important; }")

FIELDS = ("systolic_bp", "diastolic_bp", "blood_glucose", "sleep_last_night_h",
          "sleep_last_3_nights_h", "sleep_efficiency_pct", "mood", "stress", "fatigue_score",
          "food_water_intake")


def _same(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        a, b = a or {}, b or {}
        return all(_same(a.get(x), b.get(x)) for x in set(a) | set(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def changed(data: dict, cur: dict) -> bool:
    """Whether the fields differ from the saved check-in (Save is disabled when not)."""
    return not all(_same(data.get(f), cur.get(f)) for f in FIELDS)


def field_key(name: str, cur: dict):
    """Widget keys tied to the saved values: after Save or Reset the fields start from them."""
    sig = hashlib.sha1(json.dumps([cur.get(f) for f in FIELDS], sort_keys=True,
                                  default=str).encode()).hexdigest()[:8]
    return lambda field: f"ci_{name}_{field}_{sig}"


def main() -> None:
    st.set_page_config(page_title="SpaceFuzz · Morning check-in", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(theme.page_style())
    name = st.session_state.get("active_scenario", "main_sleep_debt_eva")
    cur = cs.current(name)
    st.html(f'<div class="sf-topbar" data-testid="checkin-topbar"><span>Morning check-in'
            f'</span><span>{esc(ui_config.CREW_MEMBER_NAME)}</span>{illustrative_badge()}'
            f'<span class="sub">{esc(ui_config.SCENARIOS.get(name, name))} · synthetic example '
            f'values{tip(PREFILLED, "l")}</span></div>')
    nav.link_dashboard(label="Next: Dashboard, press Run Forecast there")

    left, right = st.columns([3, 2], gap="small")
    with left:
        # plain widgets, not a form: Save can tell whether anything differs from the saved
        # check-in. Keys carry the saved values, so Save and Reset refill the fields.
        k = field_key(name, cur)
        with st.container():
            section("Blood pressure", HOW["bp"], example=True, short=HOW_SHORT["bp"])
            c1, c2 = st.columns(2)
            sbp = c1.number_input("Systolic (mmHg)", 40.0, 300.0, float(cur["systolic_bp"]), 1.0,
                                  format="%.0f", key=k("sbp"))
            dbp = c2.number_input("Diastolic (mmHg)", 20.0, 200.0, float(cur["diastolic_bp"]),
                                  1.0, format="%.0f", key=k("dbp"))
            section("Blood glucose", HOW["glucose"], example=True, short=HOW_SHORT["glucose"])
            glu = st.number_input("Blood glucose (mg/dL)", 1.0, 800.0, float(cur["blood_glucose"]),
                                  1.0, format="%.0f", key=k("glu"))
            section("Sleep", HOW["sleep"], short=HOW_SHORT["sleep"])
            n3 = list(cur["sleep_last_3_nights_h"])
            c1, c2, c3 = st.columns(3)
            s_last = c1.number_input("Sleep last night (h)", 0.0, 24.0,
                                     float(cur["sleep_last_night_h"]), 0.1, format="%.1f",
                                     key=k("s_last"))
            s_prev = c2.number_input("Night before (h)", 0.0, 24.0, float(n3[1]), 0.1,
                                     format="%.1f", key=k("s_prev"))
            s_prev2 = c3.number_input("Two nights before (h)", 0.0, 24.0, float(n3[0]), 0.1,
                                      format="%.1f", key=k("s_prev2"))
            eff = st.number_input("Sleep efficiency (%)", 0.0, 100.0,
                                  float(cur["sleep_efficiency_pct"]), 1.0, format="%.0f",
                                  help=HOW["efficiency"], key=k("eff"))
            section("Mood and stress", HOW["mood"], short=HOW_SHORT["mood"])
            c1, c2 = st.columns(2)
            mood = c1.slider("Mood (1-5)", 1, 5, int(cur["mood"]), key=k("mood"))
            stress = c2.slider("Stress (1-5)", 1, 5, int(cur["stress"]), key=k("stress"))
            section("Fatigue", HOW["fatigue"], short=HOW_SHORT["fatigue"])
            fat = st.select_slider("Fatigue (Samn-Perelli 1-7)", list(SAMN_PERELLI),
                                   value=int(cur["fatigue_score"]), format_func=SAMN_PERELLI.get, key=k("fat"))
            section("Food and water", HOW["food"], short=HOW_SHORT["food"])
            food = cur.get("food_water_intake", {})
            c1, c2 = st.columns(2)
            meals = c1.selectbox("Meals (24 h)", MEALS,
                                 index=MEALS.index(food.get("meals", "normal"))
                                 if food.get("meals", "normal") in MEALS else 0, key=k("meals"))
            water = c2.number_input("Water (L, 24 h)", 0.0, 10.0,
                                    float(food.get("water_l_24h", 2.0)), 0.1, format="%.1f",
                                    key=k("water"))
            data = {**cur, "systolic_bp": sbp, "diastolic_bp": dbp, "blood_glucose": glu,
                    "sleep_last_night_h": s_last,
                    "sleep_last_3_nights_h": [s_prev2, s_prev, s_last],
                    "sleep_efficiency_pct": eff, "mood": mood, "stress": stress,
                    "fatigue_score": fat,
                    "food_water_intake": {**food, "meals": meals, "water_l_24h": water}}
            # Nothing changed: Save looks disabled and does nothing. It stays clickable, since a
            # typed value commits only when the field loses focus, i.e. on the click on Save
            # itself; a truly disabled button would swallow that click.
            dirty = changed(data, cur)
            if not dirty:
                st.html('<div data-testid="save-idle" style="display:none"></div><style>'
                        + IDLE_SAVE_CSS + '</style>')
            # stays at the bottom of the screen while the form is in view (ui/styles.css)
            if st.button("Save check-in", type="primary", key="save_checkin",
                         icon=":material/save:") and dirty:
                cs.save(name, data)
                st.rerun()                 # fields re-keyed to the saved values

    with right:
        section("Reaction time", "The mean of your responses is written to the check-in "
                "automatically. In a mission this value comes from the dedicated test device; "
                "this is its simulation.")
        st.caption("Tap or press space when the counter appears.")
        label = st.radio("Test length", list(ui_config.REACTION_TEST_S), horizontal=True,
                         label_visibility="collapsed")
        result = reaction.run(ui_config.REACTION_TEST_S[label], key=f"reaction_{name}")
        s = reaction.summarise(result)
        # Write a new result once; a test without plausible responses (mean None) changes nothing.
        if s and result != st.session_state.get(f"reaction_used_{name}"):
            st.session_state[f"reaction_used_{name}"] = result
            if s["mean_ms"] is not None:
                cs.save(name, {**cs.current(name), "reaction_time_ms": s["mean_ms"]})
        if s:
            detail = (f"{s['n_valid']} responses. Lapses (>= {ui_config.REACTION_LAPSE_MS} ms "
                      f"or missed): {s['lapses']}, of which missed {s['misses']}. False starts: "
                      f"{s['false_starts']}. Implausible (outside the quality-gate range): "
                      f"{s['suspicious']}.")
            st.html(f'<div class="sf-panel" data-testid="reaction-summary">Mean '
                    f'{s["mean_ms"] if s["mean_ms"] is not None else "--"} ms · '
                    f'{s["lapses"]} lapses{tip(detail)}</div>')
        else:
            rt = cs.current(name).get("reaction_time_ms")
            st.caption(f"Not taken yet: the check-in uses the example value "
                       f"{rt:.0f} ms." if rt is not None else "Not taken yet.")

        ci = cs.checkin(name)
        st.html(gate_html(ci))
        ai_hint(ci)
        if cs.edited(name):
            st.caption("Edited: the next forecast takes a few minutes.",
                       help="This check-in differs from the scenario example: the dashboard "
                            "forecasts it with the local profile instead of the cached "
                            "full-profile results.")
            if actions.job("forecast", actions.COMPUTE_STALE_S):   # Run Forecast is working
                st.caption("A forecast is running: Reset is back when it ends.")
            elif st.button("Reset to example values", key="reset_checkin",
                           icon=":material/restart_alt:"):
                cs.reset(name)
                # the component keeps returning its last result: mark it as used so the
                # rerun does not write it back
                st.session_state[f"reaction_used_{name}"] = result
                st.rerun()


main()
