"""Plan board (plan 4.2): the day's task blocks slide to the adjusted plan, risk number rolls.

The original plan is drawn as blocks on a 06:00-06:00 axis (environment events such as an
SPE warning as a band above). At the adjustment reveal the blocks animate to the adjusted
plan the engine chose: matched blocks slide / resize, new blocks (nap, rest) grow in place,
an extra block of an existing type (a split EVA) slides out of the original block, removed
blocks fade out. Blocks that really changed get an outline and a number explained in the
legend; knock-on moves (a meal pushed 15 min) only slide, so the change stays readable.
At the same time the risk number rolls from the forecast to the adjusted risk and, as it
lands, the label switches to the adjusted plan's level (e.g. "MEDIUM · crew reminder")
while the original plan's line ("reported to the ground flight surgeon") stays visible.

One iframe per playback (no reload); data-testid="plan-board-done" is added inside the
iframe when the animation ends 
Playwright: the panel sits inside ui/live_frame's component iframe, so reach it with
page.frame_locator('iframe[title*="live_frame"]') (several panels: pick by order) and then
FRAME_SELECTOR, or wait for the top-level markers (forecast-done, adjust-done,
playback-done) instead.
"""
import json

import streamlit as st

from spacefuzz.fuzzing import advice
from spacefuzz import config
import ui_config
from ui import live_frame, theme
from spacefuzz.core.plan_types import PlanType
from ui.common import CIRCLED, clock, esc as _esc, no_change_text, pct as _pct, stage_times, tip

FRAME_SELECTOR = 'iframe[srcdoc*="sf-plan-board"]'
HEIGHT_PX = ui_config.BOARD_HEIGHT["column"]      # dashboard column (default)
AXIS = (6 * 60, 30 * 60)                     # 06:00 today to 06:00 tomorrow
HIGH_LOAD = {PlanType.EVA.value, PlanType.DOCKING.value, PlanType.PRECISION_MANUAL.value}
SHORT = {"exercise_ARED": "ARED", "exercise_T2": "T2", "comm_session": "comm",
         "science_experiment": "science", "precision_manual": "precision",
         "personal_time": "personal", "maintenance": "maint.", "generic_high_load": "high load"}
CATEGORY = {"EVA": "eva", "docking": "focus", "precision_manual": "focus", "sleep": "rest",
            "rest": "rest", "exercise_ARED": "exercise", "exercise_T2": "exercise"}
LEVEL = {"self": "self-care", "crew": "crew reminder", "ground": "ground surgeon"}


# A narrow block shows the short form instead of an ellipsis ("sci", not "scie...").
# the header's height with the number, chip and original line all shown: kept before the run
# too, so the timeline never moves when the result arrives
TOP_PX = 78
PB = '<span class="pb">Plan board · </span>'     # title prefix, dropped when the board is narrow
ABBR = {"science": "sci", "personal": "pers", "maint.": "mnt", "precision": "prec",
        "docking": "dock", "high load": "load", "comm": "com"}


def label_html(label: str) -> str:
    """Full label, plus a short form the CSS shows when the block is narrow."""
    short = ABBR.get(label)
    if not short:
        return f'<span class="full keep">{_esc(label)}</span>'
    return f'<span class="full">{_esc(label)}</span><span class="abbr">{_esc(short)}</span>'


def minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    t = h * 60 + m
    return t + 24 * 60 if t < AXIS[0] else t



def dur(m: float) -> str:
    m = int(round(m))
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h" if m % 60 == 0 else f"{m // 60} h {m % 60} min"


def geometry(start: int, length: float) -> tuple[float, float]:
    """Left and width in % of the day axis (clipped to it)."""
    a, b = AXIS
    lo, hi = max(a, start), min(b, start + length)
    return 100 * (lo - a) / (b - a), 100 * max(0.0, hi - lo) / (b - a)


def _task(t: dict) -> dict:
    return {"start": minutes(t["start"]), "len": round(t["duration_h"] * 60)}


def _label(typ: str, length: int) -> str:
    if typ == PlanType.SLEEP.value and length <= config.RULE1_NAP_MAX_H * 60:
        return "nap"
    return SHORT.get(typ, typ)


def match_tasks(original: dict, adjusted: dict | None) -> list[dict]:
    """Blocks with `from` / `to` geometry and a status: same, moved, resized, new, split,
    removed. Same-type tasks are paired greedily by the smallest start difference."""
    orig = [dict(type=t["type"], **_task(t)) for t in original["tasks"]]
    adj = [dict(type=t["type"], **_task(t)) for t in (adjusted or original)["tasks"]]
    pairs = sorted(((abs(o["start"] - a["start"]), i, j) for i, o in enumerate(orig)
                    for j, a in enumerate(adj) if o["type"] == a["type"]))
    used_o, used_a, match = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_o and j not in used_a:
            used_o.add(i)
            used_a.add(j)
            match[j] = i
    blocks = []
    for j, a in enumerate(adj):
        if j in match:
            o = orig[match[j]]
            status = ("same" if (o["start"], o["len"]) == (a["start"], a["len"])
                      else "resized" if o["len"] != a["len"] else "moved")
            src = o
        elif any(o["type"] == a["type"] and _label(o["type"], o["len"]) ==
                 _label(a["type"], a["len"]) for o in orig):
            status = "split"            # another part of a task the plan already had (not a nap)
            src = min((o for o in orig if o["type"] == a["type"]),
                      key=lambda o: abs(o["start"] - a["start"]))
        else:
            status, src = "new", None
        blocks.append({"type": a["type"], "label": _label(a["type"], a["len"]), "status": status,
                       "from": {"start": src["start"], "len": src["len"]} if src else None,
                       "to": {"start": a["start"], "len": a["len"]}})
    for i, o in enumerate(orig):
        if i not in used_o and not any(b["status"] == "split" and b["from"]["start"] == o["start"]
                                       for b in blocks):
            blocks.append({"type": o["type"], "label": _label(o["type"], o["len"]),
                           "status": "removed", "from": {"start": o["start"], "len": o["len"]},
                           "to": None})
    part: dict = {}
    for b in sorted((b for b in blocks if b["to"]), key=lambda b: b["to"]["start"]):
        part[b["type"]] = part.get(b["type"], 0) + 1
        b["part"] = part[b["type"]]
    n = 0
    for b in sorted(blocks, key=lambda b: (b["to"] or b["from"])["start"]):
        if b["status"] in ("new", "split", "resized", "removed") or (
                b["status"] == "moved" and b["type"] in HIGH_LOAD):
            n += 1
            b["n"] = n
    return blocks


def legend(blocks: list[dict]) -> list[str]:
    out = []
    for b in sorted((b for b in blocks if b.get("n")), key=lambda b: b["n"]):
        mark = CIRCLED[b["n"] - 1] if b["n"] <= len(CIRCLED) else f"{b['n']}."
        f, t = b["from"], b["to"]
        if b["status"] == "new":
            text = f"+ {b['label']} {clock(t['start'])}, {dur(t['len'])}"
        elif b["status"] == "split":
            text = f"{b['label']} part {b['part']} {clock(t['start'])}, {dur(t['len'])} (split)"
        elif b["status"] == "resized":
            text = (f"{b['label']} {clock(f['start'])}, {dur(f['len'])} → "
                    f"{clock(t['start'])}, {dur(t['len'])}")
        elif b["status"] == "removed":
            text = f"{b['label']} {clock(f['start'])} removed"
        else:
            delta = t["start"] - f["start"]
            text = (f"{b['label']} {clock(f['start'])} → {clock(t['start'])} "
                    f"(moved {'+' if delta >= 0 else '-'}{dur(abs(delta))})")
        out.append(f"{mark} {text}")
    return out



def risk_states(adj: dict | None) -> dict | None:
    """Before / after risk text, colour and label (two layers), safer option line."""
    if not adj or not adj.get("plan_adjustment"):
        return None
    pa = adj["plan_adjustment"]
    rc = ui_config.RATING_COLOR
    before = {"p": pa["risk_before"], "text": _pct(pa["risk_before"]),
              "status": rc[adj["risk_rating"]],
              "chip": f"Original: {adj['risk_rating']} · {LEVEL[adj['escalation']]}"}
    if pa["escalate_to_ground"]:
        chip = f"{pa['rating_after']} · still red, ground surgeon"
    else:
        chip = f"{pa['rating_after']} · {LEVEL[advice.RATING_ESCALATION[pa['rating_after']]]}"
    after = {"p": pa["risk_after"], "text": _pct(pa["risk_after"]),
             "status": rc[pa["rating_after"]], "chip": chip}
    safer, safer_tip = None, ""
    for key, title in (("safer_option", "Safer"), ("same_day_alternative", "Same day")):
        o = pa.get(key)
        if o:
            safer_tip = ("Risk for today only; the EVA is re-forecast at tomorrow's morning "
                         "check." if key == "safer_option" or pa.get("scope") else "")
            safer = f"{title}: {'; '.join(o['changes'])} → {_pct(o['risk'])} {o['rating']}"
    return {"before": before, "after": after, "safer": safer, "safer_tip": safer_tip,
            "changed": bool(pa["changes"])}


def midday_risk_states(m: dict) -> dict:
    """Risk states for the midday check: the afternoon only (midday.forecast_midday output)."""
    rc = ui_config.RATING_COLOR
    def level(rating):
        return LEVEL[advice.RATING_ESCALATION[rating]]
    after_chip = (f"Afternoon: {m['rating_after']} · still red, ground surgeon"
                  if m["escalate_to_ground"] else
                  f"Afternoon: {m['rating_after']} · {level(m['rating_after'])}")
    return {"before": {"p": m["risk_before"], "text": _pct(m["risk_before"]),
                       "status": rc[m["rating_before"]],
                       "chip": f"As planned: {m['rating_before']} · "
                               f"{level(m['rating_before'])}"},
            "after": {"p": m["risk_after"], "text": _pct(m["risk_after"]),
                      "status": rc[m["rating_after"]], "chip": after_chip},
            "safer": None, "changed": bool(m["changes"])}


# ---------------------------------------------------------------- HTML

_CSS = """
* { box-sizing:border-box; }
body { margin:0; background:transparent; color:var(--sf-ink); overflow:hidden;
  font:13px/1.4 "Source Sans Pro","Segoe UI",system-ui,sans-serif; }
#sf-plan-board { border:1px solid var(--sf-line); border-radius:10px; padding:10px 14px;
  background:var(--sf-panel);
  height:%(h)dpx; overflow:hidden; }
.top { display:flex; justify-content:space-between; align-items:flex-start; gap:12px;
  height:%(top)dpx; flex:none; overflow:hidden; }
.h { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--sf-muted); }
.risk { text-align:right; flex:none; }
.num { font-size:34px; font-weight:700; line-height:1; transition: color .3s; }
.chip { display:inline-block; margin-top:4px; padding:2px 8px; border-radius:10px;
  font-size:12px; font-weight:600; border:1px solid currentColor; white-space:nowrap; }
.chip:empty { display:none; }
.orig { font-size:11px; color:var(--sf-muted); margin-top:2px; }
.track { position:relative; height:86px; margin-top:8px; }
.axis { position:absolute; left:0; right:0; top:72px; height:14px; font-size:10px;
  color:var(--sf-muted); }
.axis span { position:absolute; transform:translateX(-50%%); }
.grid { position:absolute; top:0; bottom:14px; width:1px; background:var(--sf-grid); }
.env { position:absolute; top:0; height:14px; border-radius:3px; font-size:10px;
  color:var(--sf-panel); padding-left:4px; overflow:hidden; white-space:nowrap;
  background:repeating-linear-gradient(45deg, %(yellow)s 0 6px, color-mix(in srgb, var(--sf-yellow) 65%%, var(--sf-bg)) 6px 12px); }
.blk { position:absolute; top:22px; height:44px; border-radius:5px; overflow:hidden;
  font-size:11px; padding:3px 4px; white-space:nowrap; text-overflow:ellipsis; color:var(--sf-ink);
  background:var(--sf-block-task); border:1px solid var(--sf-bg);
  transition: left %(t)dms ease-in-out, width %(t)dms ease-in-out, opacity %(t)dms ease,
              box-shadow .3s; }
.blk { container-type:inline-size; }
@media (max-width: 400px) { .pb { display:none; } }        /* narrow board: "ADJUSTED PLAN" */
.blk .abbr { display:none; }
@container (max-width: 52px) { .blk .full:not(.keep) { display:none; } .blk .abbr { display:inline; } }
@container (max-width: 28px) { .blk .abbr { display:none; } }   /* no "pe..." */
@container (max-width: 20px) { .blk .full { display:none !important; } }   /* no "p..." */
.blk.eva { background:var(--sf-block-eva); } .blk.focus { background:var(--sf-block-focus); }
.blk.rest { background:var(--sf-block-rest); color:var(--sf-muted); } .blk.exercise { background:var(--sf-block-exercise); }
.blk .n { position:absolute; right:3px; bottom:2px; font-weight:700; color:%(yellow)s;
  opacity:0; transition: opacity .3s; }
[data-state="final"] .blk.mark, .done .blk.mark { box-shadow:0 0 0 2px var(--sf-ink) inset; }
[data-state="final"] .blk .n, .done .blk .n { opacity:1; }
.legend { margin-top:6px; display:flex; flex-wrap:wrap; gap:2px 14px; font-size:12px;
  opacity:0; transition: opacity .4s; }
[data-state="final"] .legend, .done .legend { opacity:1; }
.safer { margin-top:4px; font-size:12px; color:var(--sf-muted); opacity:0; transition: opacity .4s; }
[data-state="final"] .safer, .done .safer { opacity:1; }
.idle { color:var(--sf-muted); font-size:12px; }
.past { position:absolute; left:0; top:0; bottom:14px; background:color-mix(in srgb,
  var(--sf-bg) 62%%, transparent); border-right:0; z-index:2; pointer-events:none; }
.past span { position:absolute; left:6px; top:2px; font-size:10px; color:var(--sf-muted); }
.now { position:absolute; top:-4px; bottom:10px; width:0; border-left:2px dashed var(--sf-ink);
  z-index:3; }
.now span { position:absolute; left:4px; top:-2px; font-size:10px; color:var(--sf-ink);
  white-space:nowrap; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""

_JS = """
const root = document.getElementById('sf-plan-board');
const R = %(risk)s, FC = %(fc_ms)d, AD = %(ad_ms)d, T = %(t_ms)d, COLORS = %(colors)s;
const num = document.getElementById('num'), chip = document.getElementById('chip'),
      orig = document.getElementById('orig');
function done() {
  root.classList.add('done');
  if (document.querySelector('[data-testid="plan-board-done"]')) return;
  const m = document.createElement('div'); m.setAttribute('data-testid', 'plan-board-done');
  m.style.display = 'none'; m.textContent = 'plan-board-done'; root.appendChild(m);
}
function setRisk(s) {
  num.textContent = s.text; num.style.color = COLORS[s.status];
  chip.textContent = s.chip; chip.style.color = COLORS[s.status];
}
function statusOf(p) { return p > %(red)s ? 'red' : (p >= %(green)s ? 'yellow' : 'green'); }
if (root.dataset.state === 'play' && R) {
  setTimeout(() => setRisk(R.before), FC);
  setTimeout(() => {
    document.querySelectorAll('.blk').forEach(b => {
      b.style.left = b.dataset.toLeft + '%%'; b.style.width = b.dataset.toWidth + '%%';
      b.style.opacity = b.dataset.toOpacity;
    });
    const t0 = performance.now(), a = R.before.p, z = R.after.p;
    (function roll() {
      const k = Math.min(1, (performance.now() - t0) / Math.max(T, 1));
      const e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      const p = a + (z - a) * e;
      num.textContent = (100 * p).toFixed(1) + '%%'; num.style.color = COLORS[statusOf(p)];
      if (k < 1) requestAnimationFrame(roll);
      else { setRisk(R.after); if (R.changed) orig.textContent = R.before.chip;
             document.getElementById('title').innerHTML = '<span class="pb">Plan board · </span>adjusted plan';
             done(); }
    })();
  }, AD);
} else if (root.dataset.state === 'play') { setTimeout(done, AD); }
"""



def html_doc(mode: str, blocks: list[dict], risk: dict | None, events: list,
             forecast_s: float = 0.0, adjust_s: float = 0.0,
             cutoff_min: int | None = None, height: int | None = None) -> str:
    """mode: idle (original plan) | play (animates at the reveals) | final (adjusted).
    cutoff_min (midday check): the part of the day before it is greyed as already happened,
    with a "now" line; only later tasks can change."""
    pal = ui_config.COLOR
    end = mode == "final"
    items = []
    for b in blocks:
        f, t = b["from"], b["to"]
        if b["status"] == "new":
            a_left, a_w = geometry(t["start"], 0)
            z_left, z_w = geometry(t["start"], t["len"])
            a_op, z_op = 0, 1
        elif b["status"] == "removed":
            a_left, a_w = geometry(f["start"], f["len"])
            z_left, z_w, a_op, z_op = a_left, a_w, 1, 0
        else:
            a_left, a_w = geometry(f["start"], f["len"])
            z_left, z_w = geometry(t["start"], t["len"])
            a_op = 0 if b["status"] == "split" else 1
            z_op = 1
        if mode == "idle":
            if b["status"] in ("new", "split"):
                continue
            z_left, z_w, z_op = a_left, a_w, 1
        left, width, op = (z_left, z_w, z_op) if end else (a_left, a_w, a_op)
        cls = " ".join(filter(None, ["blk", CATEGORY.get(b["type"], ""),
                                     "mark" if b.get("n") and mode != "idle" else ""]))
        n = b.get("n")
        mark = (f'<span class="n">{CIRCLED[n - 1] if n <= len(CIRCLED) else n}</span>'
                if n and mode != "idle" else "")
        span = t or f
        items.append(
            f'<div class="{cls}" data-status="{b["status"]}" title="{_esc(b["type"])} '
            f'{clock(span["start"])}, {dur(span["len"])}" data-to-left="{z_left:.3f}" '
            f'data-to-width="{z_w:.3f}" data-to-opacity="{z_op}" '
            f'style="left:{left:.3f}%;width:{width:.3f}%;opacity:{op}">'
            f'{label_html(b["label"]) if max(a_w, z_w) >= 5.5 else ""}{mark}</div>')
    for e in events:
        s, w = geometry(minutes(e["start"]), minutes(e["end"]) - minutes(e["start"]))
        extra = f", {e['co2_ppm']:,} ppm" if e.get("co2_ppm") else ""
        items.insert(0, f'<div class="env" style="left:{s:.3f}%;width:{w:.3f}%">'
                        f'{_esc(e["type"])}{extra}</div>')
    ticks = range(AXIS[0], AXIS[1] + 1, 180)
    grid = "".join(f'<div class="grid" style="left:{geometry(m, 0)[0]:.3f}%"></div>' for m in ticks)
    axis = "".join(f'<span style="left:{geometry(m, 0)[0]:.3f}%">{clock(m)}</span>'
                   for m in ticks)
    if risk and end:
        r_text, r_col, r_chip = (risk["after"]["text"], pal[risk["after"]["status"]],
                                 risk["after"]["chip"])
        r_orig = risk["before"]["chip"] if risk["changed"] else ""
    else:
        r_text, r_col, r_chip, r_orig = "--", pal["grey"], "", ""
    leg = "".join(f"<span>{_esc(x)}</span>" for x in legend(blocks)) if mode != "idle" else ""
    if risk and mode != "idle" and not risk["changed"]:
        leg = f"<span>{_esc(no_change_text(risk['after']['status'] == 'red'))}</span>"
    safer = (_esc(risk["safer"]) + tip(risk.get("safer_tip"), "u")
             if risk and risk.get("safer") and mode != "idle" else "")
    marker = ('<div data-testid="plan-board-done" style="display:none">plan-board-done</div>'
              if end else "")
    title = PB + ("adjusted plan" if end else "today")
    past = ""
    if cutoff_min is not None:
        w = geometry(AXIS[0], cutoff_min - AXIS[0])[1]
        past = (f'<div class="past" style="width:{w:.3f}%"><span>already happened</span></div>'
                f'<div class="now" data-testid="now-line" style="left:{w:.3f}%">'
                f'<span>now {clock(cutoff_min)}</span></div>')
    css = _CSS % {**pal, "top": TOP_PX, "h": (height or HEIGHT_PX) - 4, "t": int(ui_config.BOARD_S *
                                                        ui_config.ANIM_SCALE * 1000)}
    js = _JS % {"risk": theme.script_json(risk) if risk else "null",
                "fc_ms": int(round(forecast_s * 1000)), "ad_ms": int(round(adjust_s * 1000)),
                "t_ms": int(ui_config.BOARD_S * ui_config.ANIM_SCALE * 1000),
                "colors": json.dumps(pal), "red": config.RISK_RED_ABOVE,
                "green": config.RISK_GREEN_BELOW}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{theme.root_css()}{theme.tip_css()}{css}</style></head>
<body><div id="sf-plan-board" data-testid="plan-board-figure" data-state="{mode}">
<div class="top"><div><div class="h" id="title">{title}</div>
<div class="idle">{"" if mode != "idle" else "Adjusted after the forecast."}</div></div>
<div class="risk"><div class="num" id="num" style="color:{r_col}">{_esc(r_text)}</div>
<div class="chip" id="chip" style="color:{r_col}">{_esc(r_chip)}</div>
<div class="orig" id="orig">{_esc(r_orig)}</div></div></div>
<div class="track">{grid}{"".join(items)}{past}<div class="axis">{axis}</div></div>
<div class="legend">{leg}</div><div class="safer">{safer}</div>{marker}
</div><script>{js}</script></body></html>"""


# ---------------------------------------------------------------- Streamlit


def show(static: dict, res: dict | None, shown: int, frames: list,
         height: int | None = None) -> None:
    """Draw the board for app.py's playback frame `shown` (once per script run)."""
    playing = res is not None and 0 < shown < len(frames)
    mode = "idle" if res is None or shown == 0 else ("play" if playing else "final")
    adj = (res or {}).get("adjust")
    adjusted = ((adj or {}).get("details") or {}).get("adjusted_plan")
    blocks = match_tasks(static["plan"], adjusted if mode != "idle" else None)
    risk = risk_states(adj) if mode != "idle" else None
    f_s, a_s = stage_times(frames) if mode == "play" else (0.0, 0.0)
    doc = html_doc(mode, blocks, risk, static["plan"].get("events", []), f_s, a_s,
                   height=height)
    st.html('<div data-testid="plan-board" style="display:none">plan-board</div>')
    live_frame.show(doc, height or HEIGHT_PX, key="plan_board")
