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
from ui import actions, body_map, charts, fuzz_log, guard, hud, nav, plan_board, viewport
from ui import checkin_store as cs
from ui import theme
from ui.common import clock, esc, no_change_text, pct as fmt_pct

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
CUTOFFS = (f"green < {config.RISK_GREEN_BELOW:.0%}, yellow {config.RISK_GREEN_BELOW:.0%}-"
           f"{config.RISK_RED_ABOVE:.0%}, red > {config.RISK_RED_ABOVE:.0%} (illustrative)")


# ---------------------------------------------------------------- formatting



def time_to_crash_text(ttc_min, start_min: int) -> str:
    if ttc_min is None:
        return f"Not shown (risk at or below {config.RISK_RED_ABOVE:.0%})"
    return (f"around {clock(start_min + ttc_min)} "
            f"({ttc_min // 60} h {ttc_min % 60:02d} min after check-in)")



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
    return [
        {"key": "heart_rate", "name": "Heart rate (resting)",
         "value": f"{now['heart_rate']:.0f} bpm", "baseline": f"{baseline.resting_hr:.0f} bpm",
         "threshold": f"act at baseline + {config.ACT_NOW_HR_ABOVE_BASELINE_BPM:.0f} bpm "
                      f"(illustrative)",
         "status": status("heart_rate", {"heart_rate"}), "example_data": False},
        {"key": "blood_pressure", "name": "Blood pressure",
         "value": f"{checkin.systolic_bp:.0f}/{checkin.diastolic_bp:.0f} mmHg",
         "baseline": f"{baseline.sbp:.0f}/{baseline.dbp:.0f} mmHg",
         "threshold": f"range {s_lo:.0f}-{s_hi:.0f} / {d_lo:.0f}-{d_hi:.0f} mmHg (illustrative)",
         "status": status("blood_pressure", {"systolic_bp", "diastolic_bp"}),
         "example_data": True},
        {"key": "blood_glucose", "name": "Blood glucose",
         "value": f"{checkin.blood_glucose:.0f} mg/dL",
         "baseline": f"{baseline.glucose:.0f} mg/dL",
         "threshold": f"low below {config.GLUCOSE_LOW_MG_DL:.0f} mg/dL (illustrative)",
         "status": status("blood_glucose", {"blood_glucose"}), "example_data": True},
        {"key": "body_temperature", "name": "Body temperature (core, model)",
         "value": f"{now['core_temp']:.1f} °C", "baseline": f"{baseline.core_temp:.1f} °C",
         "threshold": f"heat strain above +{config.HEAT_STRAIN_ONSET_C:.1f} °C during EVA "
                      f"(illustrative)",
         "status": status("core_temp", {"core_temp"}), "example_data": False},
        {"key": "sleep", "name": "Sleep",
         "value": f"{checkin.sleep_last_night_h:.1f} h last night (3 nights: {nights} h)",
         "baseline": f"{baseline.sleep_h:.1f} h",
         "threshold": f"short below {config.ACT_NOW_SLEEP_SHORT_H:.1f} h (illustrative)",
         "status": status("sleep", {"sleep_last_night_h"}), "example_data": False},
        {"key": "mood", "name": "Mood / stress (self-rating)",
         "value": f"mood {checkin.mood}/5, stress {checkin.stress}/5",
         "baseline": "not tracked",
         "threshold": f"act at mood <= {config.ACT_NOW_MOOD_LOW}/5 or stress >= "
                      f"{config.ACT_NOW_STRESS_HIGH}/5 (illustrative)",
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
         "text": f"Forecast: risk {fmt_pct(fc['risk_probability'])} -> {fc['risk_rating']}"},
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
                   "text": f"Adjusted: {fmt_pct(pa['risk_before'])} -> "
                           f"{fmt_pct(pa['risk_after'])} {pa['rating_after']}"})
    return frames


def layers(adj: dict) -> dict:
    """Two layers (stage 4 checklist): original plan escalation vs adjusted plan level."""
    pa = adj["plan_adjustment"]
    original = (f"Original plan: {fmt_pct(adj['risk_probability'])} {adj['risk_rating']}, "
                f"{ESCALATION_TEXT[adj['escalation']]}.")
    if pa["escalate_to_ground"]:
        adjusted = (f"Adjusted plan: {fmt_pct(pa['risk_after'])} {pa['rating_after']}, still "
                    f"red: report to the ground flight surgeon.")
    else:
        level = advice.RATING_ESCALATION[pa["rating_after"]]
        adjusted = (f"Adjusted plan: {fmt_pct(pa['risk_after'])} {pa['rating_after']}, "
                    f"{ESCALATION_TEXT[level]}.")
    return {"original": original, "adjusted": adjusted}


# ---------------------------------------------------------------- HTML pieces

CSS = """
<style>
.sf-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
  margin-right:8px; vertical-align:middle; }
.sf-ind { display:flex; justify-content:space-between; gap:8px; padding:6px 0;
  border-bottom:1px solid var(--sf-grid); font-size:14px; }
.sf-ind:last-child { border-bottom:none; }
.sf-small { font-size:12px; color:var(--sf-muted); }
.sf-table { width:100%; border-collapse:collapse; font-size:13px; }
.sf-table td { padding:3px 6px; border-bottom:1px solid var(--sf-grid); }
.sf-table tr.new td { color:var(--sf-green); }
.sf-table tr.env td { color:var(--sf-yellow); font-style:italic; }
.sf-risk { font-size:44px; font-weight:700; line-height:1.1; }
.sf-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; }
.sf-card { background:var(--sf-panel); border:1px solid var(--sf-line); border-left:4px solid var(--c);
  border-radius:8px; padding:10px 14px; font-size:14px; }
.sf-chain { color:var(--sf-ink); }
.sf-fold { margin-top:8px; }
.sf-fold summary { cursor:pointer; font-size:12px; color:var(--sf-muted); }
/* Fixed heights: a panel keeps its size when the forecast fills it, so nothing below moves
   and the eye goes to what changed. Long content scrolls inside the panel. */
.sf-fix { overflow-y:auto; }
.sf-act { height:172px; }
.sf-compare { min-height:440px; }
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
    src = (f"Results: {res['profile']} profile, seed {res['forecast'].get('seed', '--')}"
           if res else "Results: not computed yet")
    if static.get("checkin_json"):
        src += " · edited check-in"
    return (f'<div class="sf-topbar" data-testid="topbar"><span>Mars Base</span>'
            f'<span data-testid="crew-member">{esc(ui_config.CREW_MEMBER_NAME)}</span>'
            f'<span>{esc(p["date"])}</span><span>Earth delay {esc(p["comm_delay_min"])} min</span>'
            f'<span data-testid="check-window">Check window: {esc(window)}</span>'
            f'<span class="sub">{esc(SCENARIOS[static["name"]])} · {esc(src)}</span></div>')


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
        tag = '<span class="sf-tag">example data</span>' if r["example_data"] else ""
        items.append(
            f'<div class="sf-ind" data-testid="indicator-{r["key"]}" data-status="{r["status"]}">'
            f'<div>{dot(r["status"])}{esc(r["name"])}{tag}<div class="sf-small">'
            f'{esc(r["threshold"])}</div></div><div style="text-align:right">'
            f'<b>{esc(r["value"])}</b><div class="sf-small">baseline {esc(r["baseline"])}'
            f'</div></div></div>')
    return (f'<div class="sf-panel" data-testid="indicators"><div class="sf-h">Now: '
            f'morning check-in</div>{"".join(items)}</div>')


NAP_ADVICE = re.compile(r"schedule a [\w-]+ nap before demanding tasks")


def act_now_view(res: dict | None, frames: list, shown: int) -> str:
    """Act now at playback frame `shown`: the engine's advice once the forecast is revealed,
    pointing to the adjusted plan's nap only once the adjustment is revealed."""
    done = {f.get("done") for f in frames[:shown]}
    fc = res["forecast"] if res and "forecast" in done else None
    adj = res.get("adjust") if res and "adjust" in done else None
    return act_now_html(fc, adj)


def act_now_html(fc: dict | None, adj: dict | None = None) -> str:
    """Act-now advice. When the adjusted plan schedules a nap, the generic nap advice from
    the engine ("a 20-minute nap") points to that nap instead, so the two never disagree."""
    if fc is None:
        body = '<div class="sf-small">Shown with the forecast.</div>'
    else:
        lines = fc["advice_now"] + fc.get("action_restrictions", [])
        nap = _nap(adj)
        if nap:
            lines = [NAP_ADVICE.sub(f"take the {_hm(nap['duration_h'])} nap the adjusted plan "
                                    f"schedules at {nap['start']} before demanding tasks", x)
                     for x in lines]
        body = "".join(f"<div>• {esc(x)}</div>" for x in lines) or \
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
            parts.append("coverage gaps: " + ", ".join(counts))
    ooe = [w for w in warnings if not w.startswith("Not covered")]
    if ooe:
        parts.append(f"{len(ooe)} outside experience")
    if not parts:
        return ""
    return (f'<div class="sf-small" style="color:var(--sf-muted)">⚠ {esc(" · ".join(parts))} '
            f'(see log)</div>')


def risk_html(fc: dict | None, start_min: int, revealed: bool) -> str:
    if not revealed or fc is None:
        risk, rating, color = "--", "not forecast yet", "grey"
        ttc = "--"
    else:
        rating = fc["risk_rating"]
        color = RATING_COLOR[rating]
        risk = fmt_pct(fc["risk_probability"])
        ttc = time_to_crash_text(fc["time_to_crash_minutes"], start_min) \
            if rating != "UNKNOWN" else fc.get("message") or qg.UNTRUSTED_MESSAGE
    warn = (coverage_summary((fc or {}).get("coverage_warnings", [])) if revealed else "") or \
        '<div class="sf-small">&nbsp;</div>'
    return (f'<div class="sf-panel"><div class="sf-h">Today\'s forecast</div>'
            f'<div class="sf-risk" data-testid="risk-probability" data-rating="{color}" '
            f'style="color:{COLOR[color]}">{risk} <span style="font-size:20px">{esc(rating)}'
            f'</span></div><div class="sf-small">Risk probability = share of simulations that '
            f'crash. Cut-offs: {CUTOFFS}</div>'
            f'<div style="margin-top:6px" data-testid="time-to-crash">Forecast risk time: '
            f'{esc(ttc)}</div><div data-testid="coverage-warnings">{warn}</div></div>')


def comparison_html(adj: dict | None, original_plan: dict, revealed: bool) -> str:
    head = '<div class="sf-h">Before / after adjustment</div>'
    if not revealed or adj is None or not adj.get("plan_adjustment"):
        body = '<div class="sf-small">Shown after the adjustment is re-tested.</div>'
        return f'<div class="sf-panel sf-compare" data-testid="risk-comparison">{head}{body}</div>'
    pa = adj["plan_adjustment"]
    c_before = COLOR[RATING_COLOR[adj["risk_rating"]]]
    c_after = COLOR[RATING_COLOR[pa["rating_after"]]]
    lay = layers(adj)
    parts = [
        f'<div class="sf-risk" style="font-size:34px"><span style="color:{c_before}">'
        f'{fmt_pct(pa["risk_before"])}</span> → <span style="color:{c_after}" '
        f'data-testid="risk-after" data-rating="{RATING_COLOR[pa["rating_after"]]}">'
        f'{fmt_pct(pa["risk_after"])} {esc(pa["rating_after"])}</span></div>',
        f'<div data-testid="layer-original">{esc(lay["original"])}</div>',
        f'<div data-testid="layer-adjusted">{esc(lay["adjusted"])}</div>',
    ]
    if pa["changes"]:
        parts.append('<div class="sf-small" style="margin-top:6px">Changes</div>'
                     + "".join(f"<div>• {esc(c)}</div>" for c in pa["changes"]))
    for key, title in (("safer_option", "Safer option"),
                       ("same_day_alternative", "Same-day alternative")):
        o = pa.get(key)
        if o:
            scope = o.get("scope") or pa.get("scope") or ""
            parts.append(f'<div data-testid="{key.replace("_", "-")}" style="margin-top:6px">'
                         f'<b>{title}:</b> {esc("; ".join(o["changes"]))}: {fmt_pct(o["risk"])} '
                         f'{esc(o["rating"])}. <span class="sf-small">{esc(scope)}</span></div>')
    if pa.get("scope"):
        parts.append(f'<div class="sf-small">{esc(pa["scope"])}</div>')
    retest = pa.get("retest")
    if retest:
        parts.append(f'<div class="sf-small" style="margin-top:6px">Re-test: '
                     f'{esc(retest["slip_margin"])}</div>')
    adjusted = adj.get("details", {}).get("adjusted_plan")
    if adjusted and pa["changes"]:
        parts.append('<details class="sf-fold"><summary>Adjusted plan as a table (new or '
                     'moved tasks in green)</summary>'
                     + plan_table_html(adjusted, "plan-table-adjusted", original_plan)
                     + '</details>')
    return (f'<div class="sf-panel sf-compare" data-testid="risk-comparison">{head}'
            f'{"".join(parts)}</div>')


def measurement_lines(adj: dict | None) -> list[str]:
    """Midday measurement prompts, only for indicators whose measurement error moves the
    risk by at least ui_config.MEASURE_MIN_RISK_PP percentage points."""
    a = adj or {}
    sens = (a.get("details") or {}).get("measurement_sensitivity")
    if sens is None:
        return list(a.get("measurement_prompts", []))
    keep = [(n, d) for n, d in sens if 100 * d >= ui_config.MEASURE_MIN_RISK_PP]
    return advice.measurement_prompts(keep)


def cards_html(fc: dict | None, adj: dict | None, revealed: bool) -> str:
    if not revealed or fc is None:
        body = '<div class="sf-small">Advice cards appear after the forecast.</div>'
        return f'<div data-testid="advice-cards">{body}</div>'
    rating = fc["risk_rating"]
    c = COLOR[RATING_COLOR[rating]]
    if rating == "UNKNOWN":
        msg = fc.get("message") or qg.UNTRUSTED_MESSAGE
        cards = [f'<div class="sf-card" style="--c:{c}">{esc(msg)}</div>']
        return f'<div class="sf-cards" data-testid="advice-cards">{"".join(cards)}</div>'
    a = adj or {}
    chain = " <b>→</b> ".join(f"<span>{esc(x)}</span>" for x in fc["causal_chain"])
    level = a.get("escalation", fc["escalation"])
    cards = [
        f'<div class="sf-card" data-testid="card-risk" style="--c:{c}"><b>Your plan today is '
        f'{esc(rating)} risk ({fmt_pct(fc["risk_probability"])}), mainly '
        f'{esc(fc["primary_vulnerable_system"])}.</b><div class="sf-chain" '
        f'style="margin-top:6px">{chain}</div><div class="sf-small" style="margin-top:6px">'
        f'Level: {esc(ESCALATION_TEXT[level])}</div></div>',
        f'<div class="sf-card" data-testid="card-today" style="--c:{COLOR["green"]}">'
        f'<b>Prevent today</b>'
        + "".join(f"<div>• {esc(x)}</div>" for x in a.get("advice_today", [])) + "</div>",
        f'<div class="sf-card" data-testid="card-measure" style="--c:{COLOR["yellow"]}">'
        f'<b>Measure first at midday</b>'
        + ("".join(f"<div>• {esc(x)}</div>" for x in measurement_lines(a))
           or f"<div>No single measurement moves the risk by {ui_config.MEASURE_MIN_RISK_PP:g} "
              f"point or more within its error.</div>")
        + "</div>",
    ]
    if a.get("min_crash_perturbation"):
        cards.append(f'<div class="sf-card" data-testid="card-boundary" '
                     f'style="--c:{COLOR["grey"]}"><b>Closest crash</b><div>'
                     f'{esc(a["min_crash_perturbation"])}</div></div>')
    return f'<div class="sf-cards" data-testid="advice-cards">{"".join(cards)}</div>'


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
        text = ("Fuzzing today's plan: read from the engine cache when it is fresh, otherwise "
                "a full run that takes minutes...")
    elif v.busy:
        text = "Replaying the fuzzing run..."
    elif v.ended:
        text = "Done. Press again to replay the run."
    else:
        text = "Start here: fuzz today's plan, then adjust it."
    call = "" if (v.busy or v.ended) else '<style>.st-key-run_forecast button{animation:sf-call 2.4s ease-in-out infinite}</style>'
    return f'<div class="sf-status" data-testid="run-status">{esc(text)}</div>{call}'


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
        "act_now": lambda: act_now_view(res, frames, shown),
        "risk": lambda: risk_html(fc, s["start_min"], "forecast" in done),
        "status": lambda: status_html(v),
        "compare": lambda: comparison_html(adj, s["plan"], "adjust" in done),
        "cards": lambda: cards_html(fc, adj, "forecast" in done),
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


def run_button(v: View, target=st) -> None:
    actions.button(target, "Run Forecast", "Running forecast...", RUN, v.busy, "run_forecast",
                   {"key": v.static["key"]})
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
    q = st.query_params.get("scenario")
    names = list(SCENARIOS)
    default = names.index(q) if q in SCENARIOS else 0
    return st.sidebar.selectbox("Scenario", names, index=default, format_func=SCENARIOS.get,
                                disabled=disabled)


def full_layout(v: View, wide: bool) -> None:
    """The whole dashboard. Above the fold at 1440x800 (13-inch laptop) and up: risk number,
    Run Forecast, plan board, body map. From WIDE_MIN_PX the fuzz log and the charts move to a
    fourth column, so the page needs (almost) no scrolling."""
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


def view_layout(v: View, view: str) -> None:
    """One screen per role, all reading the same results (multi-screen demo, video shots)."""
    put(v, "topbar")
    st.html(marker(f"view-{view}"))
    if view == "crew":                          # few, large items: the visor, then the day's plan
        hud.show_visor(v.static, v.res if v.ended else None)   # full width: laid out for it
        run_button(v)
        panels(v, "board")
    elif view == "surgeon":                     # decide: risk, what to do now, before/after
        risk_col, act_col, cards_col = st.columns(3, gap="small")
        with risk_col:                          # panels that grow at the reveal sit last
            put(v, "risk")
            run_button(v)
            put(v, "indicators")
            theme.page_link(st, "pages/2_Indicators.py", "Why these indicators", "📖")
        with act_col:
            put(v, "act_now")
            put(v, "compare")
        with cards_col:
            put(v, "cards")
    else:                                       # engine: how the fuzzer got there
        left, right = st.columns([2, 3], gap="small")   # same split as the two charts below
        with left:
            put(v, "risk")
            run_button(v)
        with right:
            panels(v, "progress")
        charts.show(v.static, v.res, v.shown, v.frames)


def main(view: str | None = None) -> None:
    st.set_page_config(page_title="SpaceFuzz", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(CSS)                                  # style-only: takes no room on the page
    st.html(f"<style>{(ROOT / 'ui' / 'styles.css').read_text()}</style>")
    ss = st.session_state
    name = choose_scenario(disabled=actions.job(RUN, actions.COMPUTE_STALE_S) is not None)
    ss["active_scenario"] = name                  # the check-in page follows this scenario
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
    guard.run(lambda: st.navigation(nav.pages()).run())
