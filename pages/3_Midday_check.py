"""Midday check (plan 1.2, manual 4.3): re-measure one value at 14:00, re-forecast the afternoon.

The morning measurement prompt said to confirm last night's sleep with the wristband first.
The crew enters the wristband value (5.9 h is the demo's example input; the self-report was
6.2 h); midday.py re-simulates the day from the morning with that value and counts only what
can still happen after 14:00. What already happened is shown as a record, not a forecast, and
the adjustment may only change the afternoon. Results for the example input come from the
full-profile cache; any other value is computed with the local profile.
"""
import json
import time
from pathlib import Path

import streamlit as st

from spacefuzz.fuzzing import midday
from spacefuzz.core import plan as pl
import ui_config
from ui import actions, body_map, checkin_store, live_frame, plan_board, theme
from ui.common import brief_html, esc, illustrative_badge, no_change_text, tip

CUT = ui_config.MIDDAY_CHECK_MIN
RUN = "midday"                     # ui/actions.py job name
PLANS_DIR = Path(__file__).resolve().parents[1] / "data" / "plans"



def plan_header(name: str) -> dict:
    """Date and Earth delay from the scenario's plan file (the dashboard shows the same)."""
    return json.loads((PLANS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def morning_prompt(name: str) -> str:
    """The morning's first measurement prompt, read from the cache (never computed here)."""
    a = midday.cached_morning(name, ui_config.app_profile())
    if a is None:
        return ("The morning forecast is not in the cache yet: run it on the dashboard first. "
                "The demo re-measures last night's sleep.")
    prompts = a.get("measurement_prompts") or []
    return prompts[0] if prompts else "No morning measurement prompt."


def result_for(name: str, value: float) -> dict:
    store = st.session_state.setdefault("midday_results", {})
    key = f"{name}|{value:g}"
    if key not in store:
        example = ui_config.MIDDAY_EXAMPLE[name]["sleep_last_night_h"]
        profile = ui_config.app_profile() if value == example else ui_config.EDITED_PROFILE
        store[key] = midday.forecast_midday(name, {"sleep_last_night_h": value}, CUT,
                                            profile_name=profile)
    return store[key]


def record_html(r: dict) -> str:
    rows = "".join(
        f'<div>• {esc(v["clock"])} {esc(body_map.rule_text(v["rule"]))}'
        f'{tip("Recorded after the fact and reported to the flight surgeon.")}</div>'
        for v in r["morning_violations"])
    body = rows or "<div>Nothing before 14:00 breaks a rule.</div>"
    return (f'<div class="sf-panel" data-testid="midday-record" style="opacity:.85">'
            f'<div class="sf-h">Already happened · before {esc(r["cutoff"])}'
            f'{tip("A record, not a forecast: only what can still happen is forecast.")}</div>'
            f'{body}</div>')


def risk_html(r: dict) -> str:
    rs = plan_board.midday_risk_states(r)
    b, a = rs["before"], rs["after"]
    rest = ", ".join(f"{body_map.rule_text(k)} ({100 * v / max(1, sum(r['by_rule_after'].values())):.0f}% "
                     f"of the remaining crashes)" for k, v in r["by_rule_after"].items())
    still_red = ui_config.RATING_COLOR[r["rating_after"]] == "red"
    changes = "".join(f"<div>• {esc(c)}</div>" for c in r["changes"]) or \
        f"<div>{esc(no_change_text(still_red, 'afternoon'))}</div>"
    more = (f"{r['candidates']} afternoon candidates screened; results: {r['profile']} profile."
            + (f" Remaining afternoon risk: {rest}." if rest else ""))
    return (f'<div class="sf-panel" data-testid="midday-risk"><div class="sf-h">Afternoon risk '
            f'· from {esc(r["cutoff"])}{tip(more)}</div><div style="font-size:36px;font-weight:700">'
            f'<span style="color:var(--sf-{b["status"]})">{esc(b["text"])}</span> → '
            f'<span style="color:var(--sf-{a["status"]})">{esc(a["text"])} '
            f'{esc(r["rating_after"])}</span></div><div>{esc(b["chip"])}</div>'
            f'<div>{esc(a["chip"])}</div><div class="sf-h" style="margin-top:8px">Changes'
            f'{tip("Afternoon only: the morning cannot change any more.")}</div>{changes}</div>')


def main() -> None:
    st.set_page_config(page_title="SpaceFuzz · Midday check", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(theme.page_style())
    name = st.session_state.get("active_scenario", "main_sleep_debt_eva")
    p = plan_header(name)
    st.html(f'<div class="sf-topbar" data-testid="midday-topbar"><span>Mars Base</span>'
            f'<span>{esc(ui_config.CREW_MEMBER_NAME)}</span><span>{esc(p.get("date", ""))}</span>'
            f'<span>Earth delay {esc(p.get("comm_delay_min", "?"))} min</span><span>Midday check {pl.clock(CUT)}'
            f'</span>{illustrative_badge()}<span class="sub">{esc(ui_config.SCENARIOS.get(name, name))}'
            f'</span></div>')
    if name not in ui_config.MIDDAY_EXAMPLE:
        st.info("The midday demo is set up for the main scenario (sleep debt + EVA): choose it "
                "on the dashboard.")
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        prompt = morning_prompt(name)
        st.html(f'<div class="sf-panel"><div class="sf-h">Re-measure first</div>'
                f'<div>{brief_html(prompt)}</div></div>')
        morning = checkin_store.example(name)["sleep_last_night_h"]
        example = ui_config.MIDDAY_EXAMPLE[name]["sleep_last_night_h"]
        value = st.number_input("Last night's sleep, wristband record (h)", 0.0, 16.0,
                                float(example), 0.1, key="midday_sleep", format="%.1f",
                                help=f"{example:g} h is an example input for the demo. This "
                                     f"morning's self-report was {morning:g} h; the wristband "
                                     f"record replaces it.")
        if checkin_store.edited(name):
            st.caption("Check-in edits are not used here.",
                       help="The midday check starts from the example morning check-in and "
                            "its morning plan.")
        job = actions.job(RUN)
        actions.button(st, "Run midday forecast", "Running midday forecast...", RUN,
                       job is not None, "run_midday",       # value read at click time
                       lambda: {"run": f"{name}|{st.session_state['midday_sleep']:g}"})
        status = ("Re-forecasting the afternoon..." + tip("Cached for the example input; "
                                                           "another value takes a few minutes.")
                  if job else "Enter the wristband value, then run.")
        st.html(f'<div class="sf-status" data-testid="run-status">{status}</div>')
        actions.show_error(st)             # only after a failed run
        record_slot = st.empty()           # "Already happened" goes here, under the button
    with right:
        if job:
            st.session_state["midday_shown"] = job["run"]
        ran = st.session_state.get("midday_shown")
        if not ran or not ran.startswith(name + "|"):
            st.caption("The afternoon forecast appears here.")
            return
        try:
            r = result_for(name, float(ran.split("|")[1]))
        except midday.MorningMissing:
            st.session_state.pop("midday_shown", None)
            msg = ("The midday check starts from this morning's forecast, which is not in the "
                   "cache yet: open the Dashboard, press Run Forecast, then come back.")
            if job:
                actions.fail(msg)          # unlocks the button; shown under it on the rerun
            st.info(msg)
            return
        record_slot.html(record_html(r))   # left column: balances the two columns
        st.html(risk_html(r))
        blocks = plan_board.match_tasks(r["plan_before"], r["plan_after"])
        risk = plan_board.midday_risk_states(r)
        f_s, a_s = 0.3 * ui_config.ANIM_SCALE, 1.2 * ui_config.ANIM_SCALE
        doc = plan_board.html_doc("play" if job else "final", blocks, risk,
                                  r["plan_before"].get("events", []), forecast_s=f_s,
                                  adjust_s=a_s, cutoff_min=CUT,
                                  height=ui_config.BOARD_HEIGHT["wide"])
        live_frame.show(doc, ui_config.BOARD_HEIGHT["wide"], key="midday_board")
        st.html('<div data-testid="midday-done" style="display:none">midday-done</div>')
    if job:                       # let the board finish its slide, then unlock the button
        time.sleep(a_s + ui_config.BOARD_S * ui_config.ANIM_SCALE)
        actions.finish()

main()
