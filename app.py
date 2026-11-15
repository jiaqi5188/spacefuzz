"""SpaceFuzz main screen (plan 4.1): layout only, every number comes from engine / adjuster.

Run with:  streamlit run app.py   (optional: ?scenario=radiation_spe in the URL)

Compute first, play back after: Run Forecast computes (or loads from cache/) the forecast
and the plan adjustment, stores them in st.session_state, then replays them on a
timeline. Key elements and the end of every animation carry a data-testid
(forecast-done, adjust-done, playback-done) so Playwright can wait explicitly.
"""
import dataclasses
import re
import time
from pathlib import Path

import streamlit as st

from spacefuzz.fuzzing import adjuster
from spacefuzz.fuzzing import advice
from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import quality_gate as qg
from spacefuzz.core import state
import ui_config
from ui import actions, body_map, charts, common, fuzz_log, guard, hud, nav, plan_board, viewport
from ui import checkin_store as cs
from ui import theme
from ui.common import (brief, brief_html, clock, esc, illustrative_badge, no_change_text,
                       loading_html, pct as fmt_pct, tip)

ROOT = Path(__file__).resolve().parent
PLANS_DIR = ROOT / "data" / "plans"
SCENARIOS = ui_config.SCENARIOS

FRAME_S = ui_config.FRAME_S * ui_config.ANIM_SCALE
REVEAL_S = ui_config.REVEAL_S * ui_config.ANIM_SCALE
BOARD_S = ui_config.BOARD_S * ui_config.ANIM_SCALE
COLOR, RATING_COLOR = ui_config.COLOR, ui_config.RATING_COLOR
ESCALATION_TEXT = {
    "self": "self-care: keep the plan, maintenance advice",
    "crew": "crew reminder: recheck, adjust the schedule, ask a crewmate to confirm",
    "ground": "reported to the ground flight surgeon",
}
ESCALATION_SHORT = {"self": "self-care", "crew": "crew reminder", "ground": "ground surgeon"}
CUTOFFS = (f"green < {config.RISK_GREEN_BELOW:.0%}, yellow {config.RISK_GREEN_BELOW:.0%}-"
           f"{config.RISK_RED_ABOVE:.0%}, red > {config.RISK_RED_ABOVE:.0%}")


# ---------------------------------------------------------------- formatting


def time_to_crash_text(ttc_min, start_min: int) -> str:
    """Visible part of the crash time line (the minutes after check-in go into its tip)."""
    if ttc_min is None:
        return f"No crash time (not shown at or below {config.RISK_RED_ABOVE:.0%})"
    return f"First crash ≈ {clock(start_min + ttc_min)}"


def time_to_crash_detail(ttc_min, start_min: int) -> str:
    if ttc_min is None:
        return ""
    return (f"{ttc_min // 60} h {ttc_min % 60:02d} min after the {clock(start_min)} check-in: "
            f"the median time of the first rule or system crash in the simulations that crash.")



# ---------------------------------------------------------------- data

def _indicator_rows(checkin: state.CheckIn, baseline: state.Baseline, now: dict,
                    gate: qg.GateResult) -> list[dict]:
    """The six example indicators (plan 2.2): current value, personal baseline, status."""
    items = advice.act_now(checkin, baseline, resting_hr=now["heart_rate"], gate=gate)
    by_ind = {a.indicator: a for a in items}
    missing = set(gate.missing)

    def status(indicator: str, fields: set) -> str:
        if not gate.can_forecast or missing & fields:
            return "grey"
        a = by_ind.get(indicator)
        if a is None:
            return "green"
        return "red" if a.effective_escalation == "ground" else "yellow"

    (s_lo, s_hi), (d_lo, d_hi) = config.ACT_NOW_BP_RANGE["sbp"], config.ACT_NOW_BP_RANGE["dbp"]
    nights = " / ".join(f"{h:.1f}" for h in checkin.sleep_last_3_nights_h)
    # name and value stay visible; the threshold (and any extra context) goes into the tip
    return [
        {"key": "heart_rate", "name": "Resting HR",
         "value": f"{now['heart_rate']:.0f} bpm", "baseline": f"{baseline.resting_hr:.0f} bpm",
         "threshold": f"Resting heart rate. Act at baseline + "
                      f"{config.ACT_NOW_HR_ABOVE_BASELINE_BPM:.0f} bpm.",
         "status": status("heart_rate", {"heart_rate"}), "example_data": False},
        {"key": "blood_pressure", "name": "Blood pressure",
         "value": f"{checkin.systolic_bp:.0f}/{checkin.diastolic_bp:.0f} mmHg",
         "baseline": f"{baseline.sbp:.0f}/{baseline.dbp:.0f}",
         "threshold": f"Act outside {s_lo:.0f}-{s_hi:.0f} / {d_lo:.0f}-{d_hi:.0f} mmHg. "
                      f"Synthetic example value.",
         "status": status("blood_pressure", {"systolic_bp", "diastolic_bp"}),
         "example_data": True},
        {"key": "blood_glucose", "name": "Glucose",
         "value": f"{checkin.blood_glucose:.0f} mg/dL",
         "baseline": f"{baseline.glucose:.0f} mg/dL",
         "threshold": f"Blood glucose. Low below {config.GLUCOSE_LOW_MG_DL:.0f} mg/dL. "
                      f"Synthetic example value.",
         "status": status("blood_glucose", {"blood_glucose"}), "example_data": True},
        {"key": "body_temperature", "name": "Core temp",
         "value": f"{now['core_temp']:.1f} °C", "baseline": f"{baseline.core_temp:.1f} °C",
         "threshold": f"Core body temperature from the model. Heat strain above "
                      f"+{config.HEAT_STRAIN_ONSET_C:.1f} °C during EVA.",
         "status": status("core_temp", {"core_temp"}), "example_data": False},
        {"key": "sleep", "name": "Sleep",
         "value": f"{checkin.sleep_last_night_h:.1f} h",
         "baseline": f"{baseline.sleep_h:.1f} h",
         "threshold": f"Last night. Last 3 nights: {nights} h. Short below "
                      f"{config.ACT_NOW_SLEEP_SHORT_H:.1f} h.",
         "status": status("sleep", {"sleep_last_night_h"}), "example_data": False},
        {"key": "mood", "name": "Mood · stress",
         "value": f"{checkin.mood}/5 · {checkin.stress}/5",
         "baseline": "",
         "threshold": f"Self-rating, 1-5. Act at mood <= {config.ACT_NOW_MOOD_LOW} or stress "
                      f">= {config.ACT_NOW_STRESS_HIGH}. No personal baseline.",
         "status": status("mood_stress", {"mood", "stress"}), "example_data": False},
    ]


def load_static(name: str, ss=None) -> dict:
    """Inputs shown before any forecast: plan, check-in (the crew's edits or the example
    file, ui/checkin_store.py), the six indicators (fast)."""
    scenario = mu.Scenario.load(name)
    scenario = dataclasses.replace(scenario, checkin=cs.checkin(name, ss))
    baseline = state.make_baseline()
    out = md.simulate(scenario.plan, scenario.trace, scenario.checkin, baseline)
    now = out.iloc[0].to_dict()
    gate = qg.check(checkin=scenario.checkin)
    return {"name": name, "key": cs.key(name, ss), "checkin_json": cs.checkin_json(name, ss),
            "plan": pl.to_dict(scenario.plan),
            "check_window": scenario.checkin.check_window,
            "start_min": en._start_min(scenario.checkin),
            "data_quality": gate.status,
            "indicators": _indicator_rows(scenario.checkin, baseline, now, gate)}


def load_results(name: str, profile_name: str | None = None,
                 checkin_json: str | None = None) -> dict:
    """Forecast + adjustment for one scenario (engine cache when fresh, else computed). An
    edited check-in has no cached full-profile run, so it is forecast with the local profile."""
    path = PLANS_DIR / f"{name}.json"
    scenario = mu.Scenario.load(name)
    if checkin_json:
        scenario = dataclasses.replace(scenario, checkin=cs.from_json(name, checkin_json))
        profile_name = ui_config.EDITED_PROFILE
    baseline = state.make_baseline()
    fc = en.forecast(path, scenario=scenario, profile_name=profile_name, baseline=baseline)
    adj = None
    if fc["risk_rating"] != "UNKNOWN":
        adj = adjuster.adjust(path, scenario=scenario, profile_name=profile_name,
                              baseline=baseline)
    return {"name": name, "profile": en.profile(profile_name)["name"], "forecast": fc,
            "adjust": adj, "edited": bool(checkin_json), "checkin_json": checkin_json}


# ---------------------------------------------------------------- timeline

def timeline(res: dict) -> list[dict]:
    """Playback frames built from the real results; `done` marks the end of a stage."""
    fc, adj = res["forecast"], res.get("adjust")
    if fc["risk_rating"] == "UNKNOWN":
        return [{"stage": "forecast", "text": fc.get("message") or qg.UNTRUSTED_MESSAGE,
                 "done": "forecast"},
                {"stage": "adjust", "text": "No forecast, so no plan adjustment.",
                 "done": "adjust"}]
    prob, guided, heat = fc["probability_track"], fc["guided_track"], fc["heatmap"]
    cells = len(heat["sleep_cut_h"]) * len(heat["eva_extend_h"])
    frames = [
        {"stage": "forecast", "text": f"Probability track: {prob['samples']:,} simulations, "
                                      f"{prob['crashes']:,} crashed"},
        {"stage": "forecast", "text": f"Guided track: {guided['rounds']:,} rounds, "
                                      f"{len(guided['crash_types'])} crash types found"},
        {"stage": "forecast", "text": f"Crash boundary map: {cells} cells "
                                      f"(sleep cut x EVA extension)"},
        {"stage": "forecast", "done": "forecast",
         "text": f"Forecast: risk {fmt_pct(fc['risk_probability'])} → {fc['risk_rating']}"},
    ]
    pa = (adj or {}).get("plan_adjustment")
    if not pa or not pa["changes"]:
        frames.append({"stage": "adjust", "done": "adjust",
                       "text": no_change_text(bool(pa and pa.get("escalate_to_ground")))})
        return frames
    det = adj.get("details", {})
    frames.append({"stage": "adjust", "text": f"Adjusting: {len(det.get('candidates', []))} "
                                              f"candidate plans screened"})
    for s in det.get("selection", []):
        frames.append({"stage": "adjust",
                       "text": f"Re-test {'; '.join(s['changes'])}: "
                               f"{fmt_pct(s.get('confirm', s.get('screen')))} ({s['result']})"})
    frames.append({"stage": "adjust", "done": "adjust",
                   "text": f"Adjusted: {fmt_pct(pa['risk_before'])} → "
                           f"{fmt_pct(pa['risk_after'])} {pa['rating_after']}"})
    return frames


def layers(adj: dict) -> dict:
    """Two layers (stage 4 checklist): original plan escalation vs adjusted plan level, as a
    short line each plus the full sentence (for the tip)."""
    pa = adj["plan_adjustment"]
    original = (f"Original plan: {fmt_pct(adj['risk_probability'])} {adj['risk_rating']}, "
                f"{ESCALATION_TEXT[adj['escalation']]}.")
    short_o = f"Original → {ESCALATION_SHORT[adj['escalation']]}"
    if pa["escalate_to_ground"]:
        adjusted = (f"Adjusted plan: {fmt_pct(pa['risk_after'])} {pa['rating_after']}, still "
                    f"red: report to the ground flight surgeon.")
        short_a = "Adjusted → still red, ground surgeon"
    else:
        level = advice.RATING_ESCALATION[pa["rating_after"]]
        adjusted = (f"Adjusted plan: {fmt_pct(pa['risk_after'])} {pa['rating_after']}, "
                    f"{ESCALATION_TEXT[level]}.")
        short_a = f"Adjusted → {ESCALATION_SHORT[level]}"
    return {"original": original, "adjusted": adjusted, "original_short": short_o,
            "adjusted_short": short_a}


# ---------------------------------------------------------------- HTML pieces

CSS = """
<style>
.sf-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
  margin-right:8px; vertical-align:middle; }
.sf-ind { display:flex; justify-content:space-between; gap:8px; padding:6px 0;
  border-bottom:1px solid var(--sf-grid); font-size:14px; }
.sf-ind:last-child { border-bottom:none; }
.sf-small { font-size:12px; color:var(--sf-muted); }
.sf-warn { display:inline-block; position:relative; top:1px; }   /* the sign sat high */
.sf-table { width:100%; border-collapse:collapse; font-size:13px; }
.sf-table td { padding:3px 6px; border-bottom:1px solid var(--sf-grid); }
.sf-table tr.new td { color:var(--sf-green); }
.sf-table tr.env td { color:var(--sf-yellow); font-style:italic; }
.sf-risk { font-size:44px; font-weight:700; line-height:1.1; }
/* the risk row is 48px before and after the reveal (idle text vs number): no 1px jump */
.sf-risk[data-testid="risk-probability"] { height:48px; }
.sf-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:8px; }
.sf-card { background:var(--sf-panel); border:1px solid var(--sf-line); border-left:4px solid var(--c);
  border-radius:8px; padding:8px 12px; font-size:13px; line-height:1.45; }
.sf-chain { color:var(--sf-ink); }
.sf-fold { margin-top:8px; }
.sf-fold summary { cursor:pointer; font-size:12px; color:var(--sf-muted); }
/* the folded tables open and close smoothly, like the expanders (Chrome 131+; elsewhere
   they simply open) */
.sf-fold { interpolate-size:allow-keywords; }
.sf-fold::details-content { block-size:0; overflow:hidden;
  transition:block-size .3s ease, content-visibility .3s allow-discrete; }
.sf-fold[open]::details-content { block-size:auto; }
/* Fixed heights: a panel keeps its size when the forecast fills it, so nothing below moves
   and the eye goes to what changed. Long content scrolls inside the panel. */
.sf-fix { overflow-y:auto; }
.sf-act { height:130px; }
/* before the forecast: one grey line in the risk number's row height (no "--" glyphs, whose
   dashes sit above the text baseline and looked misaligned) */
.sf-risk-idle { height:48px; display:flex; align-items:center; font-size:22px; font-weight:600; }
.sf-summary { height:96px; }
.sf-summary .rows { font-size:13px; line-height:1.5; }
.sf-compare { min-height:360px; container-type:inline-size; }
/* adjusted plan table opened at the foot of the panel: the panel scrolls to its end by itself
   (a column-reverse scroller starts at its end), so the whole table shows without scrolling;
   margin-bottom:auto keeps short content at the top */
.sf-compare:has(details[open]) { display:flex; flex-direction:column-reverse; }
.sf-compare-in { margin-bottom:auto; }
.sf-compare-risk { font-size:34px; white-space:nowrap; }
@container (max-width: 360px) { .sf-compare-risk { font-size:24px; } }   /* 4-column layout */
[data-testid="advice-cards"] { min-height:300px; align-items:start; }
</style>
"""


def dot(status: str) -> str:
    return f'<span class="sf-dot" style="background:{COLOR[status]}"></span>'


def marker(test_id: str) -> str:
    return f'<div data-testid="{test_id}" style="display:none">{test_id}</div>'


def topbar_html(static: dict, res: dict | None) -> str:
    p = static["plan"]
    window = static["check_window"].capitalize()
    if res:
        prob = res["forecast"].get("probability_track") or {}
        n = f" ({prob['samples']:,} simulated days)" if prob.get("samples") else ""
        src = f"seed {res['forecast'].get('seed', '--')}"
        src_tip = tip(f"Results from the {res['profile']} sampling profile{n}. Same inputs and "
                      f"seed give the same result.", "l")
    else:
        src, src_tip = "not run yet", ""
    if static.get("checkin_json"):
        src += " · edited check-in"
    return (f'<div class="sf-topbar" data-testid="topbar"><span>Mars Base</span>'
            f'<span data-testid="crew-member">{esc(ui_config.CREW_MEMBER_NAME)}</span>'
            f'<span>{esc(p["date"])}</span><span>Earth delay {esc(p["comm_delay_min"])} min</span>'
            f'<span data-testid="check-window">{esc(window)} check</span>'
            f'{illustrative_badge()}'
            f'<span class="sub">{esc(SCENARIOS[static["name"]])} · {esc(src)}{src_tip}</span>'
            f'</div>')


def task_label(t: dict) -> str:
    """A planned sleep up to RULE1_NAP_MAX_H is a nap (same rule as the evaluator)."""
    if t["type"] == "sleep" and t["duration_h"] <= config.RULE1_NAP_MAX_H:
        return "nap"
    return t["type"]


def _nap(adj: dict | None) -> dict | None:
    plan = ((adj or {}).get("details") or {}).get("adjusted_plan") or {}
    return next((t for t in plan.get("tasks", []) if task_label(t) == "nap"), None)


def _hm(hours: float) -> str:
    m = round(hours * 60)
    return f"{m} min" if m < 60 else (f"{m // 60} h" if m % 60 == 0 else f"{m // 60} h {m % 60} min")


def plan_table_html(plan: dict, test_id: str, original: dict | None = None) -> str:
    before = {(t["start"], t["type"], t["duration_h"]) for t in (original or plan)["tasks"]}
    rows = []
    for t in plan["tasks"]:
        new = original is not None and (t["start"], t["type"], t["duration_h"]) not in before
        rows.append(f'<tr class="{"new" if new else ""}"><td>{esc(t["start"])}</td>'
                    f'<td>{t["duration_h"]:g} h</td><td>{esc(task_label(t))}</td></tr>')
    for e in plan.get("events", []):
        extra = f', {e["co2_ppm"]:,} ppm' if e.get("co2_ppm") else ""
        rows.append(f'<tr class="env"><td>{esc(e["start"])}</td><td>to {esc(e["end"])}</td>'
                    f'<td>{esc(e["type"])}{extra} (environment)</td></tr>')
    return (f'<table class="sf-table" data-testid="{test_id}">'
            f'<tr><td class="sf-small">Start</td><td class="sf-small">Duration</td>'
            f'<td class="sf-small">Task</td></tr>{"".join(rows)}</table>')


def indicators_html(rows: list[dict]) -> str:
    items = []
    for r in rows:
        tag = '<span class="sf-tag">example</span>' if r["example_data"] else ""
        base = (f' <span class="base sf-small">base {esc(r["baseline"])}</span>'
                if r["baseline"] else "")
        items.append(
            f'<div class="sf-ind" data-testid="indicator-{r["key"]}" data-status="{r["status"]}">'
            f'<div class="sf-ind-name">{dot(r["status"])}{esc(r["name"])}{tip(r["threshold"])}'
            f'{tag}</div><div class="sf-ind-val"><b>{esc(r["value"])}</b>{base}</div></div>')
    return (f'<div class="sf-panel" data-testid="indicators"><div class="sf-h">Check-in'
            f'{tip("This morning: wearable readings and the crew member self-report.")}'
            f'</div>{"".join(items)}</div>')


NAP_ADVICE = common.NAP_ADVICE


def act_now_view(res: dict | None, frames: list, shown: int, busy: bool = False) -> str:
    """Act now at playback frame `shown`: the engine's advice once the forecast is revealed,
    pointing to the adjusted plan's nap only once the adjustment is revealed."""
    done = {f.get("done") for f in frames[:shown]}
    fc = res["forecast"] if res and "forecast" in done else None
    adj = res.get("adjust") if res and "adjust" in done else None
    # forecast shown, adjustment not yet: an adjusted plan with its own nap is coming, so do
    # not show the engine's generic "20-minute nap" only to replace it a few seconds later
    pending = fc is not None and adj is None and _nap((res or {}).get("adjust")) is not None
    return act_now_html(fc, adj, nap_pending=pending, busy=busy)


def act_now_html(fc: dict | None, adj: dict | None = None, nap_pending: bool = False,
                 busy: bool = False) -> str:
    """Act-now advice. When the adjusted plan schedules a nap, the generic nap advice from
    the engine ("a 20-minute nap") points to that nap instead, so the two never disagree."""
    if fc is None:
        body = loading_html() if busy else '<div class="sf-small">Shown with the forecast.</div>'
    else:
        lines = fc["advice_now"] + fc.get("action_restrictions", [])
        nap = _nap(adj)
        if nap_pending and not nap:
            lines = [common.neutral_nap(x) for x in lines]
        if nap:
            lines = [NAP_ADVICE.sub(f"take the {_hm(nap['duration_h'])} nap at {nap['start']} "
                                    f"(scheduled by the adjusted plan) before demanding tasks",
                                    x)
                     for x in lines]
        body = "".join(f"<div>• {brief_html(x)}</div>" for x in lines) or \
            '<div class="sf-small">Nothing to act on now.</div>'
    return (f'<div class="sf-panel sf-fix sf-act" data-testid="act-now"><div class="sf-h">Act now</div>'
            f'{body}</div>')


def coverage_summary(warnings: list[str]) -> str:
    """One small grey line; the full warnings stay in the fuzz log's [cover] lines."""
    parts = []
    for w in warnings:
        m = re.match(r"Not covered by the guided track: (.*)", w)
        if m:
            counts = []
            for kind, items in re.findall(r"(rules|systems) ([^;]+)", m[1]):
                n = len([x for x in items.split(",") if x.strip()])
                counts.append(f"{n} {kind[:-1] if n == 1 else kind}")
            parts.append("gaps: " + ", ".join(counts))
    ooe = [w for w in warnings if not w.startswith("Not covered")]
    if ooe:
        parts.append(f"{len(ooe)} outside experience")
    if not parts:
        return ""
    return (f'<div class="sf-small" style="color:var(--sf-muted)"><span class="sf-warn">⚠</span> '
            f'{esc(" · ".join(parts))}'
            f'{tip(" ".join(warnings) + " Details: the [cover] lines of the fuzz log.")}</div>')


def risk_html(fc: dict | None, start_min: int, revealed: bool, busy: bool = False) -> str:
    idle = not revealed or fc is None
    if idle:
        risk, rating, color = "", "Not forecast yet", "grey"
        ttc = "&nbsp;"                       # keeps the row: nothing moves at the reveal
    else:
        rating = fc["risk_rating"]
        color = RATING_COLOR[rating]
        risk = fmt_pct(fc["risk_probability"])
        if rating != "UNKNOWN":
            ttc = (esc(time_to_crash_text(fc["time_to_crash_minutes"], start_min))
                   + tip(time_to_crash_detail(fc["time_to_crash_minutes"], start_min)))
        else:
            ttc = esc(fc.get("message") or qg.UNTRUSTED_MESSAGE)
    n = ((fc or {}).get("probability_track") or {}).get("samples")
    what = (f"Risk = share of the {n:,} simulated days that crash. " if n else
            "Risk = share of the simulated days that crash. ") + f"Cut-offs: {CUTOFFS}."
    warn = (coverage_summary((fc or {}).get("coverage_warnings", [])) if revealed else "") or \
        '<div class="sf-small">&nbsp;</div>'
    return (f'<div class="sf-panel"><div class="sf-h">Today\'s forecast{tip(what)}</div>'
            + (f'<div class="sf-risk sf-risk-idle" data-testid="risk-probability" '
               f'data-rating="grey" style="color:{COLOR["grey"]}">'
               f'{loading_html("Forecasting", 1) if busy else esc(rating)}</div>' if idle else
               f'<div class="sf-risk" data-testid="risk-probability" data-rating="{color}" '
               f'style="color:{COLOR[color]}">{risk} <span style="font-size:20px">{esc(rating)}'
               f'</span></div>')
            + f'<div style="margin-top:6px" data-testid="time-to-crash">{ttc}</div>'
            f'<div data-testid="coverage-warnings">{warn}</div></div>')


def retest_text(retest: dict) -> tuple[str, str]:
    """(visible line, tip) for the re-test of the chosen plan. A plan that left the red zone
    had both checks; a fallback that is still red had the guided re-test only: the 30-min
    schedule-slip check asks whether a fixed plan keeps a margin, and a red plan has none."""
    types = retest.get("guided_crash_types") or []
    rounds = retest.get("rounds")
    guided = (f"Guided track re-test ({rounds:,} rounds): "
              + (", ".join(types) if types else "no crash type found") + "."
              if rounds else "")
    if retest["slip_margin"].startswith("not run"):
        return ("Re-tested: guided track only (still red)",
                f"{guided} The 30-min schedule-slip check is not run: it only applies to a plan "
                f"that left the red zone. The plan is reported to the ground.".strip())
    return "Re-tested ✓", f"Schedule slip check: {retest['slip_margin']}. {guided}".strip()


def comparison_html(adj: dict | None, original_plan: dict, revealed: bool,
                    busy: bool = False) -> str:
    head = '<div class="sf-h">Before / after</div>'
    if not revealed or adj is None or not adj.get("plan_adjustment"):
        body = (loading_html("Adjusting and re-testing", 3) if busy and not revealed else
                '<div class="sf-small">Shown after the adjustment is re-tested.</div>')
        return f'<div class="sf-panel sf-compare" data-testid="risk-comparison">{head}{body}</div>'
    pa = adj["plan_adjustment"]
    c_before = COLOR[RATING_COLOR[adj["risk_rating"]]]
    c_after = COLOR[RATING_COLOR[pa["rating_after"]]]
    lay = layers(adj)
    parts = [
        f'<div class="sf-risk sf-compare-risk"><span style="color:{c_before}">'
        f'{fmt_pct(pa["risk_before"])}</span> → <span style="color:{c_after}" '
        f'data-testid="risk-after" data-rating="{RATING_COLOR[pa["rating_after"]]}">'
        f'{fmt_pct(pa["risk_after"])} {esc(pa["rating_after"])}</span></div>',
        f'<div data-testid="layer-original">{esc(lay["original_short"])}'
        f'{tip(lay["original"])}</div>',
        f'<div data-testid="layer-adjusted">{esc(lay["adjusted_short"])}'
        f'{tip(lay["adjusted"])}</div>',
    ]
    if pa["changes"]:
        parts.append('<div class="sf-small" style="margin-top:6px">Changes</div>'
                     + "".join(f"<div>• {esc(c)}</div>" for c in pa["changes"]))
    for key, title in (("safer_option", "Safer"), ("same_day_alternative", "Same day")):
        o = pa.get(key)
        if o:
            scope = o.get("scope") or pa.get("scope") or ""
            parts.append(f'<div data-testid="{key.replace("_", "-")}" style="margin-top:6px">'
                         f'<b>{title}:</b> {esc("; ".join(o["changes"]))} → {fmt_pct(o["risk"])} '
                         f'{esc(o["rating"])}{tip(scope)}</div>')
    retest = pa.get("retest")
    if retest:
        label, detail = retest_text(retest)
        parts.append(f'<div class="sf-small" style="margin-top:6px">{esc(label)}{tip(detail)}'
                     f'</div>')
    elif pa.get("scope"):
        parts.append(f'<div class="sf-small">Scope{tip(pa["scope"])}</div>')
    adjusted = adj.get("details", {}).get("adjusted_plan")
    if adjusted and pa["changes"]:
        parts.append('<details class="sf-fold"><summary>Adjusted plan table'
                     f'{tip("New or moved tasks in green.")}</summary>'
                     + plan_table_html(adjusted, "plan-table-adjusted", original_plan)
                     + '</details>')
    # one inner block: with the plan table open the panel anchors at its end (CSS)
    return (f'<div class="sf-panel sf-compare" data-testid="risk-comparison">'
            f'<div class="sf-compare-in">{head}{"".join(parts)}</div></div>')


def measurement_lines(adj: dict | None) -> list[str]:
    """Midday measurement prompts, only for indicators whose measurement error moves the
    risk by at least ui_config.MEASURE_MIN_RISK_PP percentage points."""
    a = adj or {}
    sens = (a.get("details") or {}).get("measurement_sensitivity")
    if sens is None:
        return list(a.get("measurement_prompts", []))
    keep = [(n, d) for n, d in sens if 100 * d >= ui_config.MEASURE_MIN_RISK_PP]
    return advice.measurement_prompts(keep)


def advice_panel(inner: str) -> str:
    """The advice cards in a panel like the others (fixed height: nothing below moves)."""
    return (f'<div class="sf-panel sf-advice sf-fix" data-testid="advice-cards">'
            f'<div class="sf-h">Advice</div>{inner}</div>')


def cards_html(fc: dict | None, adj: dict | None, revealed: bool, busy: bool = False) -> str:
    if not revealed or fc is None:
        return advice_panel(loading_html("Forecasting", 3) if busy else
                            '<div class="sf-small">Advice cards appear after the forecast.</div>')
    rating = fc["risk_rating"]
    c = COLOR[RATING_COLOR[rating]]
    if rating == "UNKNOWN":
        msg = fc.get("message") or qg.UNTRUSTED_MESSAGE
        cards = [f'<div class="sf-card" style="--c:{c}">{esc(msg)}</div>']
        return advice_panel(f'<div class="sf-cards">{"".join(cards)}</div>')
    a = adj or {}
    links = [brief(x) for x in fc["causal_chain"]]
    chain = " <b>→</b> ".join(f"<span>{esc(short)}</span>" for short, _ in links)
    chain_tip = " → ".join(fc["causal_chain"])
    level = a.get("escalation", fc["escalation"])
    system = fc["primary_vulnerable_system"]
    summary = (f"Your plan today is {rating} risk ({fmt_pct(fc['risk_probability'])}), mainly "
               f"{system}. Level: {ESCALATION_TEXT[level]}.")
    nothing = (f"No single measurement moves the risk by {ui_config.MEASURE_MIN_RISK_PP:g} point "
               f"or more within its error.")
    cards = [
        f'<div class="sf-card" data-testid="card-risk" style="--c:{c}"><b>{esc(rating)} · '
        f'{esc(system)}</b>{tip(summary)}'
        f'<div class="sf-chain" style="margin-top:6px">{chain}'
        f'{tip(chain_tip) if any(d for _, d in links) else ""}</div></div>',
        f'<div class="sf-card" data-testid="card-today" style="--c:{COLOR["green"]}">'
        f'<b>Prevent today</b>'
        + "".join(f"<div>• {brief_html(x)}</div>" for x in a.get("advice_today", []))
        + "</div>",
        f'<div class="sf-card" data-testid="card-measure" style="--c:{COLOR["yellow"]}">'
        f'<b>Measure first at midday</b>'
        + ("".join(f"<div>• {brief_html(x)}</div>" for x in measurement_lines(a))
           or f"<div>Nothing to re-measure{tip(nothing)}</div>")
        + "</div>",
    ]
    if a.get("min_crash_perturbation"):
        cards.append(f'<div class="sf-card" data-testid="card-boundary" '
                     f'style="--c:{COLOR["grey"]}"><b>Closest crash</b><div>'
                     f'{brief_html(a["min_crash_perturbation"])}</div></div>')
    return advice_panel(f'<div class="sf-cards">{"".join(cards)}</div>')


# ---------------------------------------------------------------- page

RUN = "forecast"                    # ui/actions.py job name of Run Forecast


@dataclasses.dataclass
class View:
    """What one script run shows: the results, the playback frame, and the HTML slots placed
    so far (slots update in place per frame; the animated panels are drawn once per run)."""
    static: dict
    res: dict | None
    shown: int
    busy: bool = False
    ph: dict = dataclasses.field(default_factory=dict)
    sent: dict = dataclasses.field(default_factory=dict)

    @property
    def frames(self) -> list:
        return timeline(self.res) if self.res else []

    @property
    def ended(self) -> bool:
        return bool(self.frames) and self.shown >= len(self.frames)


def status_html(v: View) -> str:
    """One line under Run Forecast: what to do, or what is happening (fixed height)."""
    if v.busy and v.res is None:
        text, more = "Running", ("Fuzzing today's plan: read from the engine cache when it is "
                                 "fresh, otherwise a full run that takes minutes.")
    elif v.busy:
        text, more = "Replaying", "Replaying the fuzzing run."
    elif v.ended:
        text, more = "Done", "Press again to replay the run."
    else:
        text, more = "Start", "Fuzz today's plan, then adjust it."
    call = "" if (v.busy or v.ended) else '<style>.st-key-run_forecast button{animation:sf-call 2.4s ease-in-out infinite}</style>'
    # no visible text (the panels' loading states show what is happening); the state stays as
    # a hidden marker for the tests and the video recorder
    state = text.lower()
    return (f'<div data-testid="run-status" data-state="{state}" title="{esc(more)}" '
            f'style="display:none"></div>{call}'
            f'{call}')


def run_summary_html(fc: dict | None, revealed: bool, busy: bool = False) -> str:
    """Engine screen, under Run Forecast: what the run did, in three short lines (fixed
    height, so the charts below never move)."""
    head = '<div class="sf-h">Run summary</div>'
    if not revealed or fc is None or fc.get("risk_rating") == "UNKNOWN":
        rows = (loading_html() if busy and not revealed else
                '<div class="sf-small">Shown with the forecast.</div>')
    else:
        prob, guided = fc.get("probability_track") or {}, fc.get("guided_track") or {}
        perf = fc.get("performance") or (fc.get("details") or {}).get("performance_before") or {}
        n, crashed = prob.get("samples", 0), prob.get("crashes", 0)
        lines = [f"{n:,} simulated days · {crashed:,} crashed ({fmt_pct(crashed / n if n else None)})",
                 f"{guided.get('rounds', 0):,} guided rounds · "
                 f"{len(guided.get('crash_types', []))} crash types"]
        if perf.get("simulations"):
            lines.append(f"{perf['simulations']:,} simulations · "
                         f"{perf.get('simulations_per_second', 0):.0f} per second")
        rows = "".join(f"<div>{esc(x)}</div>" for x in lines)
    return (f'<div class="sf-panel sf-summary" data-testid="run-summary">{head}'
            f'<div class="rows">{rows}</div></div>')


def regions(v: View) -> dict:
    """HTML of every slot for the current frame."""
    s, res, frames, shown = v.static, v.res, v.frames, v.shown
    done = {f.get("done") for f in frames[:shown]}
    fc = res["forecast"] if res else None
    adj = res.get("adjust") if res else None
    marks = (marker("forecast-done") if "forecast" in done else "") + \
        (marker("adjust-done") if "adjust" in done else "") + \
        (marker("playback-done") if v.ended else "")
    return {
        "topbar": lambda: topbar_html(s, res),
        "indicators": lambda: indicators_html(s["indicators"]),
        "act_now": lambda: act_now_view(res, frames, shown, busy=v.busy),
        "risk": lambda: risk_html(fc, s["start_min"], "forecast" in done, busy=v.busy),
        "status": lambda: status_html(v),
        "compare": lambda: comparison_html(adj, s["plan"], "adjust" in done, busy=v.busy),
        "cards": lambda: cards_html(fc, adj, "forecast" in done, busy=v.busy),
        "summary": lambda: run_summary_html(fc, "forecast" in done, busy=v.busy),
        "markers": lambda: marks or "<div></div>",
    }


def put(v: View, name: str) -> None:
    """Place slot `name` here and fill it at once (an empty slot would show as a gap)."""
    v.ph[name] = st.empty()
    v.sent[name] = regions(v)[name]()
    v.ph[name].html(v.sent[name])


def refresh(v: View) -> None:
    """Redraw the slots whose HTML changed for the current frame."""
    draw = regions(v)
    for name, ph in v.ph.items():
        h = draw[name]()
        if h != v.sent[name]:
            ph.html(h)
            v.sent[name] = h


def run_progress_css(compute_s: float, replay_s: float, phase: str, elapsed: float,
                     compute_elapsed: float | None = None) -> str:
    """A rough % on the busy Run Forecast button, counted up by the browser (ui/actions.py),
    so it moves while Python is busy computing. Compute takes the first share (by its
    estimate) and keeps creeping up past the estimate; the replay runs on to 100 %."""
    compute_s, replay_s = max(compute_s, 0.1), max(replay_s, 0.1)   # tests: no animation time
    share = min(95.0, 100.0 * compute_s / (compute_s + replay_s))
    if phase == "compute":
        return actions.compute_css("run_forecast", compute_s, share, elapsed)
    start = actions.eased_pct(compute_s if compute_elapsed is None else compute_elapsed,
                              compute_s, share)
    return actions.finish_css("run_forecast", start, replay_s, elapsed)


def run_button(v: View, target=st) -> None:
    actions.button(target, "Run Forecast", "Running forecast...", RUN, v.busy, "run_forecast",
                   {"key": v.static["key"]})
    j = st.session_state.get(actions.JOB)
    if v.busy and j:
        est = ui_config.RUN_ESTIMATE_S
        compute = est["edited"] if v.static.get("checkin_json") else est["cached"]
        if v.res is None:                         # step 1: computing (or reading the cache)
            css = run_progress_css(compute, est["replay"], "compute",
                                   time.time() - j.get("t0", j["t"]))
        else:                                     # step 2: the replay, its real length
            replay = sum(frame_wait(f, k == len(v.frames) - 1) for k, f in enumerate(v.frames))
            css = run_progress_css(compute, replay, "replay", time.time() - j["t"],
                                   compute_elapsed=j["t"] - j.get("t0", j["t"]))
        target.html(css)
    put(v, "status")
    actions.show_error(target)                    # only after a failed run


def frame_wait(frame: dict, last: bool) -> float:
    """Seconds to hold a playback frame; the last one also lets the plan board finish its
    slide and risk roll (BOARD_S), so the final picture never cuts the roll short."""
    wait = REVEAL_S if frame.get("done") else FRAME_S
    return max(wait, BOARD_S) if last else wait


def panels(v: View, *names: str) -> None:
    """The animated panels (once per run): body map, plan board, fuzz log."""
    mods = {"body": body_map, "board": plan_board, "progress": fuzz_log}
    for n in names:
        mods[n].show(v.static, v.res, v.shown, v.frames)


def choose_scenario(disabled: bool) -> str:
    """The scenario picker, in the sidebar of every page (drawn by the router, one shared
    key): the check-in, midday and reference pages follow the same scenario."""
    q = st.query_params.get("scenario")
    names = list(SCENARIOS)
    default = names.index(q) if q in SCENARIOS else 0
    return st.sidebar.selectbox("Scenario", names, index=default, format_func=SCENARIOS.get,
                                disabled=disabled, key="scenario_select")


def router() -> None:
    """Every page: the scenario picker in the sidebar, then the page the menu points to."""
    name = choose_scenario(disabled=actions.job(RUN, actions.COMPUTE_STALE_S) is not None)
    st.session_state["active_scenario"] = name
    st.navigation(nav.pages()).run()


def fixed_heights_css(h: dict) -> str:
    """Fixed panel heights for one layout (the same before, during and after the run, so the
    columns end level and nothing jumps); only the keys given are set."""
    rules = {"act": ".sf-panel.sf-act {{ height:{}px; }}",
             "compare": ".sf-compare {{ height:{}px; min-height:0; overflow-y:auto; }}",
             "advice": ".sf-advice {{ height:{}px; overflow-y:auto; }}",
             "summary": ".sf-summary {{ height:{}px; }}",
             "indicators": '.sf-panel[data-testid="indicators"] {{ height:{}px; }}'}
    return "<style>" + " ".join(rules[k].format(v) for k, v in h.items() if k in rules) + "</style>"


def four_body_css(column_px: int) -> str:
    """Column B (Run Forecast, plan table, body map) at a fixed height, the body map taking
    what is left: opening the plan table squeezes the space around the figure instead of
    lengthening the column (user, 11/15)."""
    return ("<style>"
            # doubled class: Streamlit's own block class sets height:auto after this
            f".st-key-four_body.st-key-four_body {{ height:{column_px}px; flex:none; flex-wrap:nowrap; }} "
            ".st-key-four_body > .st-key-body_map { flex:1 1 0; min-height:0; } "
            ".st-key-four_body > .st-key-body_map > div { height:100%; } "
            ".st-key-four_body > .st-key-body_map iframe { height:100% !important; "
            "display:block; }</style>")


def four_layout(v: View, wide: bool) -> None:
    """The dashboard in four columns (ui_config.DASHBOARD_COLUMNS = 4): what the crew reported |
    the body | the decision (risk, Run Forecast, Act now, before / after) | the plan and the
    engine (plan board, advice cards, fuzz log). On a 13-14 inch screen the adjusted risk and
    the advice are on the first screen. From WIDE_MIN_PX the charts get a fifth column."""
    put(v, "topbar")
    h = ui_config.FOUR_HEIGHTS                # fixed: the same before, during and after the run
    st.html(fixed_heights_css(h) + four_body_css(h["column"]) + marker("layout-four"))
    cols = st.columns(ui_config.COLUMNS_FOUR_WIDE if wide else ui_config.COLUMNS_FOUR,
                      gap="small")
    a, b, c, d = cols[:4]
    with a:
        put(v, "indicators")
        charts.show_trends(v.static, height=h["trend"])
    with b, st.container(key="four_body"):    # Run Forecast on top of the body map
        run_button(v)
        # opened, the plan table takes room from the body map (four_body_css), not the page
        with st.expander("Today's plan as a table", icon=":material/table_rows:",
                         key="plan_table_exp"):
            st.html(plan_table_html(v.static["plan"], "plan-table"))
        body_map.show(v.static, v.res, v.shown, v.frames, height=h["body"])
    with c:
        put(v, "risk")
        put(v, "act_now")
        put(v, "compare")
    with d:
        plan_board.show(v.static, v.res, v.shown, v.frames,
                        height=ui_config.BOARD_HEIGHT["four"])
        put(v, "cards")
    panels(v, "progress")                     # full width under the columns: easier to read
    if wide:
        with cols[4]:
            st.html(marker("layout-wide"))
            charts.show(v.static, v.res, v.shown, v.frames, stacked=True, busy=v.busy)
    hud.show(v.static, ready=v.ended and not v.busy, busy=v.busy)
    if not wide:
        charts.show(v.static, v.res, v.shown, v.frames, busy=v.busy)


def full_layout(v: View, wide: bool) -> None:
    """The whole dashboard. Above the fold at 1440x800 (13-inch laptop) and up: risk number,
    Run Forecast, plan board, body map. From WIDE_MIN_PX the fuzz log and the charts move to a
    fourth column, so the page needs (almost) no scrolling."""
    if ui_config.DASHBOARD_COLUMNS == 4:
        return four_layout(v, wide)
    put(v, "topbar")
    cols = st.columns(ui_config.COLUMNS_WIDE if wide else ui_config.COLUMNS, gap="small")
    left, middle, right = cols[:3]
    # Three columns of about equal length (no empty block mid-page): the before / after panel
    # under the body map ("where the risk is -> what the plan changes"), the plan table under
    # the trends. Panels that grow at the reveal sit last in their column.
    with left:
        put(v, "indicators")
        put(v, "act_now")
        charts.show_trends(v.static)
        with st.expander("Today's plan as a table", expanded=True):
            st.html(f'<div class="sf-panel">{plan_table_html(v.static["plan"], "plan-table")}'
                    f'</div>')
    with middle:
        panels(v, "body")
        put(v, "compare")
    with right:
        put(v, "risk")
        run_button(v)
        panels(v, "board")
        if not wide:
            panels(v, "progress")
        put(v, "cards")
    if wide:
        with cols[3]:
            st.html(marker("layout-wide"))
            panels(v, "progress")
            charts.show(v.static, v.res, v.shown, v.frames, stacked=True)
    hud.show(v.static, ready=v.ended and not v.busy)
    if not wide:
        charts.show(v.static, v.res, v.shown, v.frames)


def role_css(view: str) -> str:
    """Per role screen: under a panel (10 px bottom margin) Run Forecast had more room above
    than below; the button moves up by that margin."""
    if view in ("surgeon", "engine"):
        return "<style>.st-key-run_forecast { margin-top:-10px; }</style>"
    return ""


def view_layout(v: View, view: str) -> None:
    """One screen per role, all reading the same results (multi-screen demo, video shots)."""
    put(v, "topbar")
    st.html(marker(f"view-{view}") + role_css(view))
    if view == "crew":                          # few, large items: the visor, then the day's plan
        hud.show_visor(v.static, v.res if v.ended else None, busy=v.busy)   # full width
        run_button(v)
        plan_board.show(v.static, v.res, v.shown, v.frames,
                        height=ui_config.BOARD_HEIGHT["wide"])     # full width: fewer lines
    elif view == "surgeon":                     # decide: risk, what to do now, before/after
        st.html(fixed_heights_css(ui_config.SURGEON_HEIGHTS))
        risk_col, act_col, cards_col = st.columns(3, gap="small")
        with risk_col:                          # panels that grow at the reveal sit last
            put(v, "risk")
            run_button(v)
            put(v, "indicators")
            theme.page_link(st, "page_files/2_Indicators.py", "Why these indicators", "📖")
        with act_col:
            put(v, "act_now")
            put(v, "compare")
        with cards_col:
            put(v, "cards")
    else:                                       # engine: how the fuzzer got there
        st.html(fixed_heights_css(ui_config.ENGINE_HEIGHTS))
        left, right = st.columns([2, 3], gap="small")   # same split as the two charts below
        with left:
            put(v, "risk")
            run_button(v)
            put(v, "summary")
        with right:                             # log as tall as the left column
            fuzz_log.show(v.static, v.res, v.shown, v.frames,
                          height=ui_config.ENGINE_HEIGHTS["log"])
        charts.show(v.static, v.res, v.shown, v.frames, busy=v.busy,
                    height=ui_config.ENGINE_HEIGHTS["charts"])   # the footer on a 14" screen


def footer_html() -> str:
    """One grey line naming the data sources (ui_config.DATA_FOOTER)."""
    return (f'<div class="sf-small sf-footer" data-testid="data-footer">'
            f'{esc(ui_config.DATA_FOOTER)}</div>')


def main(view: str | None = None) -> None:
    st.set_page_config(page_title="SpaceFuzz", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(CSS)                                  # style-only: takes no room on the page
    st.html(f"<style>{(ROOT / 'ui' / 'styles.css').read_text()}</style>")
    ss = st.session_state
    name = ss.get("active_scenario") or list(SCENARIOS)[0]   # picked in the router's sidebar
    static = load_static(name)
    key = static["key"]                           # scenario + check-in (edited or example)
    results = ss.setdefault("results", {})
    shown_by = ss.setdefault("shown", {})
    view = view or st.query_params.get("view")
    view = view if view in ui_config.VIEWS else "full"
    wide = viewport.is_wide() if view == "full" else False   # None until the browser answers

    job = actions.job(RUN, actions.COMPUTE_STALE_S if key not in results else actions.STALE_S)
    if job and job.get("key") != key:             # scenario or check-in changed mid-run
        ss.pop(actions.JOB, None)
        job = None
    # ?autoplay=1: play once per session without a click (multi-screen demo, recording); the
    # full page waits for the width probe, whose answer would otherwise rerun mid-playback.
    flag = f"autoplayed|{view}|{key}"
    if (not job and st.query_params.get("autoplay") == "1" and wide is not None
            and not shown_by.get(key) and not ss.get(flag)):
        ss[flag] = True
        actions.start(RUN, key=key)
        job = actions.job(RUN)

    playing = bool(job) and key in results
    if playing:
        res, shown = results[key], 1
    else:
        res = results.get(key) if shown_by.get(key) else None
        shown = shown_by.get(key, 0) if res else 0
    v = View(static, res, shown, busy=bool(job))
    if view == "full":
        full_layout(v, bool(wide))
    else:
        view_layout(v, view)
    st.html(footer_html())                        # data sources, last on every screen
    put(v, "markers")

    if job and not playing:                       # step 1: compute (or read the cache)
        try:
            results[key] = load_results(name, ui_config.app_profile(), static["checkin_json"])
        except Exception as e:  # noqa: BLE001 - shown to the user, button unlocked
            actions.fail(f"Forecast failed: {e}. Press Run Forecast to try again.")
        actions.advance()
    if playing:                                   # step 2: replay the run frame by frame
        frames = v.frames
        for k in range(1, len(frames) + 1):
            if k > 1:
                v.shown = k
                refresh(v)
            time.sleep(frame_wait(frames[k - 1], last=k == len(frames)))
        shown_by[key] = len(frames)
        actions.finish()                          # step 3: unlock the button, final picture


if __name__ == "__main__":     # streamlit run / AppTest; not on import (tests, ui/nav.py)
    guard.run(router)
