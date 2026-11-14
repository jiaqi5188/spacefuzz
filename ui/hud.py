"""In-mission head-up display (plan 1.2 / 4.2): visor style, large high-contrast text, one line.

During a task nobody reads a dashboard, so the HUD shows one instruction (from the engine's
hud_directives, picked for the task: the SPE instruction only if the task overlaps an SPE
warning), four large readings against their illustrative thresholds and, when the adjusted
plan has a forecast risk time, a countdown to it. The task is the first high-load task of
the adjusted plan (EVA, docking, precision work).

The readings replay the adjusted plan's forecast over the task window (median of the
probability-track runs, status from the median and the upper / lower 5%): this is the
"short-horizon forecast" the base runs during a task, shown as a replay - not live
telemetry, and the HUD says so.

Opened with the "In-mission HUD" toggle (inside st.fragment: no page rerun); the replay adds
data-testid="hud-done" inside the iframe when it ends
Playwright: the panel sits inside ui/live_frame's component iframe, so reach it with
page.frame_locator('iframe[title*="live_frame"]') (several panels: pick by order) and then
FRAME_SELECTOR, or wait for the top-level markers (forecast-done, adjust-done,
playback-done) instead.
"""
import json

import streamlit as st

from spacefuzz import config
from spacefuzz.core import state
import ui_config
from ui import live_frame, theme
from spacefuzz.core.plan_types import PlanType
from ui.common import clock, esc as _esc, tip

FRAME_SELECTOR = 'iframe[srcdoc*="sf-hud"]'
HEIGHT_PX = 250
MISSION_TYPES = (PlanType.EVA.value, PlanType.DOCKING.value, PlanType.PRECISION_MANUAL.value)
SPE_DIRECTIVE = "SPE warning"
FALLBACK = "Follow the adjusted plan"


def _min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m



def mission_task(plan: dict | None) -> dict | None:
    """First high-load task of the plan: type, start, end (minutes after midnight)."""
    for t in (plan or {}).get("tasks", []):
        if t["type"] in MISSION_TYPES:
            s = _min(t["start"])
            return {"type": t["type"], "start": s, "end": s + round(t["duration_h"] * 60)}
    return None


def _plan(adj: dict) -> dict | None:
    return ((adj or {}).get("details") or {}).get("adjusted_plan")


def directive(adj: dict, events: list | None = None) -> str:
    """One line for the visor."""
    lines = [d for d in adj.get("hud_directives", []) if not d.startswith("Time to risk")]
    plan = _plan(adj) or {}
    task = mission_task(plan)
    evs = plan.get("events") or events or []
    in_spe = task is not None and any(
        e["type"] == PlanType.SPE_WARNING.value and _min(e["start"]) < task["end"]
        and task["start"] < _min(e["end"]) for e in evs)
    spe = [d for d in lines if d.startswith(SPE_DIRECTIVE)]
    other = [d for d in lines if not d.startswith(SPE_DIRECTIVE)]
    if in_spe and spe:
        return spe[0]
    return other[0] if other else FALLBACK


def risk_clock(adj: dict, start_min: int) -> int | None:
    ttc = (adj.get("plan_adjustment") or {}).get("time_to_crash_after")
    return None if ttc is None else start_min + ttc


def countdown(risk_min: int, now_min: int) -> str:
    s = max(0, (risk_min - now_min) * 60)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


GAUGES = {   # key: label, unit, side the threshold guards
    "heart_rate": ("Heart rate", "bpm", "high"),
    "core_temp": ("Core temp", "°C", "high"),
    "spo2": ("SpO2", "%", "low"),
    "cognitive_fatigue": ("Fatigue", "/100", "high"),
}


def _thresholds(baseline: state.Baseline) -> dict:
    return {"heart_rate": config.CARDIO_HR_MAX_FRACTION * state.max_heart_rate(baseline),
            "core_temp": baseline.core_temp + config.HEAT_STRAIN_ONSET_C,
            "spo2": config.COG_SPO2_BELOW_PCT,
            "cognitive_fatigue": config.COG_FATIGUE_ABOVE}


def mission_series(adj: dict, baseline: state.Baseline) -> dict | None:
    """Forecast readings over the task window with a status per step."""
    task = mission_task(_plan(adj))
    det = (adj or {}).get("details") or {}
    traj, mins = det.get("trajectory_after"), det.get("trajectory_minutes_after")
    if task is None or not traj or not mins:
        return None
    idx = [k for k, m in enumerate(mins) if task["start"] <= m <= task["end"]]
    if not idx:
        return None
    thr = _thresholds(baseline)
    gauges = {}
    for key, (label, unit, side) in GAUGES.items():
        q = traj[key]
        p50 = [round(q["p50"][k], 1) for k in idx]
        tail = [round(q["p95" if side == "high" else "p5"][k], 1) for k in idx]
        beyond = (lambda v, t: v >= t) if side == "high" else (lambda v, t: v <= t)
        status = ["red" if beyond(m, thr[key]) else "yellow" if beyond(e, thr[key]) else "green"
                  for m, e in zip(p50, tail)]
        gauges[key] = {"label": label, "unit": unit, "side": side, "threshold": thr[key],
                       "p50": p50, "tail": tail, "status": status}
    return {"task": task, "minutes": [mins[k] for k in idx], "gauges": gauges,
            "note": "Median of all forecast runs of the adjusted plan, not live telemetry; "
                    "the colour also counts the worst 5% of runs; thresholds illustrative.",
            "note_short": "Forecast median · not live telemetry"}


# ---------------------------------------------------------------- HTML

_CSS = """
* { box-sizing:border-box; }
body { margin:0; background:var(--sf-bg); color:var(--sf-ink);
  font:14px/1.3 "SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace; }
#sf-hud { position:relative; height:%(h)dpx; border-radius:60px / 40px; overflow:hidden;
  padding:18px 34px; background:radial-gradient(ellipse at center, color-mix(in srgb, var(--sf-green) 6%%, var(--sf-bg)) 0%%, var(--sf-bg) 75%%);
  border:2px solid %(green)s; box-shadow:0 0 18px %(green)s55 inset; }
#sf-hud::after { content:""; position:absolute; inset:0; pointer-events:none;
  background:repeating-linear-gradient(0deg, color-mix(in srgb, var(--sf-ink) 3%%, transparent) 0 1px, transparent 1px 3px); }
.row { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
.mode { color:%(green)s; letter-spacing:.12em; font-size:13px; }
.clock { font-size:20px; font-weight:700; }
.dir { margin:14px 0 12px; font-size:26px; font-weight:800; letter-spacing:.01em;
  text-transform:uppercase; color:var(--sf-ink); text-shadow:0 0 8px color-mix(in srgb, var(--sf-ink) 33%%, transparent); }
.dir.alert { color:%(red)s; text-shadow:0 0 10px %(red)s88; }
.gauges { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; }
.g { border-top:3px solid var(--c, %(grey)s); padding-top:6px; }
.g .v { font-size:30px; font-weight:800; color:var(--c, var(--sf-ink)); }
.g .u { font-size:13px; color:var(--sf-ink); margin-left:3px; }
.g .l { font-size:12px; color:var(--sf-ink); letter-spacing:.06em; text-transform:uppercase; }
.g .t { font-size:11px; color:var(--sf-muted); }
.cd { text-align:right; }
.cd .v { font-size:22px; font-weight:800; color:%(yellow)s; }
.note { position:absolute; left:34px; right:34px; bottom:12px; font-size:10px; color:var(--sf-muted); }
.none { color:var(--sf-muted); font-size:16px; margin-top:40px; }
@media (prefers-reduced-motion: reduce) { .dir { text-shadow:none; } }
"""

_JS = """
const root = document.getElementById('sf-hud');
const S = %(series)s, RISK = %(risk)s, MS = %(ms)d, COLORS = %(colors)s;
function done() {
  if (document.querySelector('[data-testid="hud-done"]')) return;
  const m = document.createElement('div'); m.setAttribute('data-testid', 'hud-done');
  m.style.display = 'none'; m.textContent = 'hud-done'; root.appendChild(m);
}
function pad(n) { return String(n).padStart(2, '0'); }
function hhmm(m) { m = ((m %% 1440) + 1440) %% 1440; return pad(Math.floor(m / 60)) + ':' + pad(m %% 60); }
function cd(now) { const s = Math.max(0, (RISK - now) * 60);
  return pad(Math.floor(s / 3600)) + ':' + pad(Math.floor(s %% 3600 / 60)) + ':' + pad(s %% 60); }
function show(k) {
  const now = S.minutes[k]; let red = false;
  document.getElementById('clock').textContent = hhmm(now);
  for (const key in S.gauges) {
    const g = S.gauges[key], el = document.getElementById('g-' + key);
    el.style.setProperty('--c', COLORS[g.status[k]]);
    el.querySelector('.v').firstChild.textContent = g.p50[k].toFixed(key === 'core_temp' ? 1 : 0);
    el.querySelector('.tail').textContent = g.tail[k].toFixed(key === 'core_temp' ? 1 : 0);
    red = red || g.status[k] === 'red';
  }
  document.getElementById('dir').classList.toggle('alert', red);
  if (RISK !== null) document.getElementById('cd').textContent = cd(now);
}
if (S && S.minutes.length) {
  const n = S.minutes.length, step = n > 1 ? MS / (n - 1) : 0;
  let k = 0; show(0);
  (function next() { if (k >= n - 1) { done(); return; }
    setTimeout(() => { k++; show(k); next(); }, step); })();
} else { done(); }
"""


def _sep(unit: str) -> str:
    return "" if unit in ("%", "/100") else " "



def _no_hud_reason(adj: dict | None, forecast_ran: bool) -> str:
    if not adj:
        return ("No adjusted plan: the forecast is UNKNOWN (data quality), so there is no HUD."
                if forecast_ran else "Available after the forecast: run it first.")
    if mission_task(_plan(adj)) is None:
        return "No EVA, docking or precision task in the adjusted plan."
    return "No forecast readings for the task window."


def html_doc(adj: dict | None, baseline: state.Baseline, start_min: int,
             mission_name: str | None = None, events: list | None = None,
             forecast_ran: bool = False) -> str:
    pal = ui_config.COLOR
    css = _CSS % {**pal, "h": HEIGHT_PX - 4}
    series = mission_series(adj, baseline) if adj else None
    if not series:
        body = ('<div class="mode">IN-MISSION HUD</div><div class="none">'
                + _no_hud_reason(adj, forecast_ran) + "</div>")
        js_series, risk = "null", None
    else:
        task = series["task"]
        risk = risk_clock(adj, start_min)
        line = directive(adj, events)
        g_html = "".join(
            f'<div class="g" id="g-{k}"><div class="l">{_esc(g["label"])}</div>'
            f'<div class="v">{g["p50"][0]:.{1 if k == "core_temp" else 0}f}'
            f'<span class="u">{_esc(g["unit"])}</span></div>'
            f'<div class="t">worst 5% <span class="tail">'
            f'{g["tail"][0]:.{1 if k == "core_temp" else 0}f}</span> · limit '
            f'{g["threshold"]:.{1 if k == "core_temp" else 0}f}</div></div>'
            for k, g in series["gauges"].items())
        cd_html = (f'<div class="cd"><div class="l" style="font-size:11px;color:var(--sf-ink)">'
                   f'TIME TO RISK</div><div class="v" id="cd">'
                   f'{countdown(risk, task["start"])}</div></div>' if risk is not None else
                   '<div class="cd"><div style="font-size:11px;color:var(--sf-ink)">TIME TO RISK'
                   '</div><div style="font-size:14px;color:var(--sf-muted)">none'
                   f'{tip(f"Risk at or below {config.RISK_RED_ABOVE:.0%}: no crash time.", "l")}'
                   '</div></div>')
        pa = adj.get("plan_adjustment") or {}
        ground = ('<div class="mode" style="color:%s">PLAN STILL RED · REPORTED TO GROUND</div>'
                  % pal["red"] if pa.get("escalate_to_ground") else "")
        name = mission_name or task["type"].replace("_", " ").upper()
        body = (f'<div class="row"><div class="mode">IN MISSION · {_esc(name)} '
                f'{clock(task["start"])}-{clock(task["end"])}</div>'
                f'<div class="clock" id="clock">{clock(task["start"])}</div></div>{ground}'
                f'<div class="row"><div class="dir" id="dir">{_esc(line)}</div>{cd_html}</div>'
                f'<div class="gauges">{g_html}</div>'
                f'<div class="note">{_esc(series["note_short"])}{tip(series["note"], "u")}'
                f'</div>')
        js_series = json.dumps({"minutes": series["minutes"],
                                "gauges": {k: {"p50": g["p50"], "tail": g["tail"],
                                               "status": g["status"]}
                                           for k, g in series["gauges"].items()}})
    js = _JS % {"series": js_series, "risk": "null" if risk is None else risk,
                "ms": int(ui_config.HUD_S * ui_config.ANIM_SCALE * 1000),
                "colors": json.dumps(pal)}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{theme.root_css()}{theme.tip_css()}{css}</style></head>
<body><div id="sf-hud" data-testid="hud-figure" data-state="play">{body}</div>
<script>{js}</script></body></html>"""


# ---------------------------------------------------------------- Streamlit

def show_visor(static: dict, res: dict | None) -> None:
    """The visor at a fixed height: the replay once there are results, a note before."""
    if res is None:
        st.html(f'<div class="sf-pending" style="height:{HEIGHT_PX}px">The in-mission HUD '
                f'appears after the forecast.</div>')
        return
    live_frame.show(html_doc(res.get("adjust"), state.make_baseline(), static["start_min"],
                             events=static["plan"].get("events"), forecast_ran=True),
                    HEIGHT_PX, key="hud")


@st.fragment
def show(static: dict, ready: bool = True) -> None:
    """Toggle + HUD (fragment: switching never reruns the page or reloads other iframes).
    ready=False while a forecast is computed or replayed: the HUD waits for the replay."""
    on = st.toggle("In-mission HUD", key="hud_on", value=True,   # on: judges should see it
                   help="Visor view for the first EVA / docking / precision task of the "
                        "adjusted plan")
    st.html('<div data-testid="hud" style="display:none">hud</div>')
    if not on:
        return
    key = static.get("key", static["name"])
    res = st.session_state.get("results", {}).get(key)
    if not res or not st.session_state.get("shown", {}).get(key):
        st.caption("The HUD is available after the forecast: press Run Forecast first.")
        return
    if not ready:
        st.caption("The HUD appears when the forecast replay ends.")
        return
    show_visor(static, res)
