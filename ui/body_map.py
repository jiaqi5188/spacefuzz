"""Six-system body map (plan 4.1 / 4.2 / 4.3): SVG + CSS inside one iframe.

- Cardiovascular, respiratory, CNS and metabolic are coloured by their share of crashed
  simulations (same cut-offs as the risk rating); musculoskeletal and immune change over
  weeks, are marked "weekly check" and are not forecast.
- While the forecast plays, the four forecast systems pulse (breathing animation); at the
  forecast reveal they fade to their colour. A radiation dose bar (operator D) sits below.
- Clicking a system opens a popup inside the figure (function, current state, space
  hazard, forecast); nothing is sent back to Python.

No flicker: the document is drawn once per script run with the reveal time built in and
plays its own animation; ui/live_frame swaps documents without a blank frame. When the
colour change ends the document adds data-testid="body-map-done".
Playwright: the panel sits inside ui/live_frame's component iframe, so reach it with
page.frame_locator('iframe[title*="live_frame"]') (several panels: pick by order) and then
FRAME_SELECTOR, or wait for the top-level markers (forecast-done, adjust-done,
playback-done) instead.
"""
import dataclasses
import functools

import streamlit as st

from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.core import evaluator as ev
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import state
import ui_config
from ui import checkin_store, live_frame, theme
from ui.common import esc as _esc, pct as _pct, stage_times

FRAME_SELECTOR = 'iframe[srcdoc*="sf-body-map"]'
HEIGHT_PX = 700
TRANSITION_S = 1.2           # colour fade at the reveal

SYSTEMS = [
    {"key": "cardiovascular", "label": "Cardiovascular", "engine": "Cardiovascular",
     "forecast": True,
     "function": "Heart and vessels deliver oxygen to working muscles and carry heat away.",
     "hazard": "Reduced gravity shifts body fluids and deconditions the heart; an EVA adds "
               "heat load, and sleep loss lowers heart rate variability."},
    {"key": "respiratory", "label": "Respiratory", "engine": "Respiratory", "forecast": True,
     "function": "Lungs exchange oxygen and carbon dioxide.",
     "hazard": "Cabin CO2 build-up raises breathing and heart rate and can lower blood "
               "oxygen saturation."},
    {"key": "cns", "label": "Nervous system (CNS)", "engine": "CNS", "forecast": True,
     "function": "Brain and nerves handle attention, reaction and decisions.",
     "hazard": "Sleep loss, workload and high CO2 slow reaction time and raise error risk "
               "in docking and precision work."},
    {"key": "metabolic", "label": "Metabolic", "engine": "Metabolic", "forecast": True,
     "function": "Keeps blood glucose and body fuel steady for the work ahead.",
     "hazard": "A long EVA burns glucose fast; low intake before it can leave too little "
               "fuel at the start."},
    {"key": "musculoskeletal", "label": "Musculoskeletal", "engine": None, "forecast": False,
     "function": "Bones and muscles carry loads and keep you moving.",
     "hazard": "Reduced gravity (Mars 0.38 g) slowly thins bone and shrinks muscle; daily "
               "exercise counters it."},
    {"key": "immune", "label": "Immune", "engine": None, "forecast": False,
     "function": "Defends the body against infection.",
     "hazard": "Stress, radiation and isolation change immune function over weeks."},
]
# Plain-language names for the composite rules (temporal rules use evaluator.RULE_TEXT).
def rule_text(rule: str | None) -> str:
    """Plain-language text of a temporal rule or a composite (system) check."""
    return COMPOSITE_TEXT.get(rule) or ev.RULE_TEXT.get(rule, rule or "")


COMPOSITE_TEXT = {
    "cardiovascular": "sustained high heart rate with low HRV, or resting blood pressure "
                      "out of range",
    "cognitive": "high cognitive fatigue together with low blood oxygen",
    "metabolic": "low blood glucose when an EVA starts",
    "radiation_dose": "EVA radiation dose above the single-EVA limit",
}
WEEKLY = "Slow-changing: checked at the weekly exam, not part of today's forecast."



# ---------------------------------------------------------------- data

def system_status(fc: dict | None) -> dict:
    """Per system: status colour, share of crashed simulations, most common rule."""
    prob = (fc or {}).get("probability_track")
    trusted = bool(prob) and (fc or {}).get("risk_rating") != "UNKNOWN"
    out = {}
    for s in SYSTEMS:
        if not s["forecast"]:
            out[s["key"]] = {"status": "weekly", "share": "weekly", "top_rule": ""}
            continue
        if not trusted:
            out[s["key"]] = {"status": "grey", "share": "--", "top_rule": ""}
            continue
        n = prob["samples"]
        p = prob["by_system"].get(s["engine"], 0) / n
        rules = {r: c for r, c in prob.get("by_rule", {}).items()
                 if ev.SYSTEM.get(r) == s["engine"]}
        top = max(rules, key=rules.get) if rules else None
        out[s["key"]] = {"status": ui_config.RATING_COLOR[en.risk_rating(p, None)],
                         "share": _pct(p),
                         "top_rule": rule_text(top)}
    rad = (prob["by_system"].get("Radiation", 0) / prob["samples"]) if trusted else None
    out["radiation"] = {"share": _pct(rad), "p": rad,
                        "status": ("grey" if rad is None else
                                   ui_config.RATING_COLOR[en.risk_rating(rad, None)])}
    return out


EXERCISE_NAME = {"exercise_ARED": "ARED", "exercise_T2": "T2"}


def _dur(minutes: int) -> str:
    return f"{minutes // 60} h" if minutes % 60 == 0 else \
        (f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min")


def _weekly_context(plan, checkin, dose: float) -> dict:
    """What today really holds for the two slow systems (never a forecast): the exercise
    countermeasure in the plan, and the check-in stress and planned EVA dose."""
    ex = [t for t in plan.tasks if t.source_type in EXERCISE_NAME]
    ex_short = ", ".join(f"{EXERCISE_NAME[t.source_type]} {_dur(t.duration_min)}" for t in ex)
    return {
        "musculoskeletal_today": f"{ex_short} today" if ex else "no exercise today",
        "musculoskeletal": (f"Today's countermeasure: {ex_short} planned (exercise slows bone "
                            f"and muscle loss). " if ex else "No exercise in today's plan. ")
                           + WEEKLY,
        "immune_today": f"stress {checkin.stress}/5 · {dose:.0f} µGy dose",
        "immune": f"Today: stress {checkin.stress}/5 (self-rating), sleep "
                  f"{checkin.sleep_last_night_h:.1f} h, planned EVA dose {dose:.1f} µGy. " + WEEKLY,
    }


@functools.lru_cache(maxsize=8)
def current_state(name: str, checkin_json: str | None = None) -> dict:
    """Check-in state per system (unmutated day) and the as-planned EVA dose; checkin_json is
    the crew's edited check-in (None = the scenario's example file)."""
    sc = mu.Scenario.load(name)
    if checkin_json:
        sc = dataclasses.replace(sc, checkin=checkin_store.from_json(name, checkin_json))
    out = md.simulate(sc.plan, sc.trace, sc.checkin, state.make_baseline())
    r, c = out.iloc[0], sc.checkin
    return {
        "cardiovascular": f"Resting heart rate {r.heart_rate:.0f} bpm, HRV {r.hrv_rmssd:.0f} ms, "
                          f"blood pressure {c.systolic_bp:.0f}/{c.diastolic_bp:.0f} mmHg "
                          f"(example data)",
        "respiratory": f"SpO2 {r.spo2:.1f}%, cabin CO2 {r.co2:,.0f} ppm",
        "cns": f"Reaction time {r.reaction_time_ms:.0f} ms, cognitive fatigue "
               f"{r.cognitive_fatigue:.0f}/100, sleep {c.sleep_last_night_h:.1f} h last night",
        "metabolic": f"Blood glucose {c.blood_glucose:.0f} mg/dL (example data), core "
                     f"temperature {r.core_temp:.1f} °C",
        **_weekly_context(sc.plan, c, float(out["eva_dose_ugy"].max())),
        "eva_dose_ugy": round(float(out["eva_dose_ugy"].max()), 1),
        "resting_hr": float(r.heart_rate),          # vitals readout; paces the heart pulse
        "spo2": float(r.spo2), "core_temp": float(r.core_temp),
    }



# ---------------------------------------------------------------- HTML

# Holographic bio-scan, one inline SVG (viewBox 0 0 400 450, figure centred on x = 200). The
# silhouette is one smooth closed path: the left half is written once as contour points and
# mirrored, so the figure is symmetric by construction; Catmull-Rom smoothing turns the points
# into curves. Medical-atlas view (the figure faces you, as in CT / X-ray): the body's right
# side is on the image left, so the heart sits right of centre; R / L marks make this clear.
W, H, CX = 400, 450, 200
_LEFT_CONTOUR = [
    (200, 18), (187, 21), (179, 31), (177, 46), (180, 60), (188, 71), (191, 82),
    (178, 89), (160, 94), (148, 102), (143, 118), (140, 146), (137, 176), (133, 206),
    (130, 236), (127, 257), (124, 270), (128, 280), (136, 276), (139, 258), (143, 232),
    (148, 204), (152, 178), (155, 150), (157, 136), (159, 162), (160, 186), (157, 212),
    (156, 238), (159, 276), (163, 316), (165, 350), (166, 388), (161, 410), (168, 420),
    (187, 419), (189, 396), (190, 352), (193, 304), (196, 270), (200, 262),
]


def silhouette_points() -> list[tuple[float, float]]:
    """Closed contour: the left half top -> crotch, then its mirror image back up."""
    left = _LEFT_CONTOUR
    return left + [(2 * CX - x, y) for x, y in reversed(left[1:-1])]


def _smooth(points: list) -> str:
    """Closed Catmull-Rom spline through `points` as cubic Bezier segments."""
    n = len(points)
    d = [f"M{points[0][0]:g} {points[0][1]:g}"]
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k) % n] for k in (-1, 0, 1, 2))
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:g} {p2[1]:g}")
    return " ".join(d) + " Z"


def _mirror_x(x: float) -> float:
    return 2 * CX - x


# Organs. Paired organs are written for the image-left side and mirrored; the heart sits
# right of centre on purpose (atlas view). Each organ has an anchor for its callout line.
_LUNG = "M{a} 104 C{b} 101 {c} 116 {d} 140 C{e} 160 {f} 174 {g} 174 C{h} 174 {a} 166 {a} 154 Z"


def _lung(side: int) -> str:
    xs = dict(a=196, b=182, c=168, d=166, e=164, f=171, g=183, h=193)
    if side > 0:
        xs = {k: _mirror_x(v) for k, v in xs.items()}
    return f'<path d="{_LUNG.format(**{k: f"{v:g}" for k, v in xs.items()})}"/>'


_ORGANS = {
    "cns": ('<ellipse cx="200" cy="40" rx="17" ry="14"/>'
            '<path class="line" d="M188 36q6-6 12 0t12 0M187 45q7-5 13 0t13 0"/>'
            '<path class="nerve" d="M200 55V250"/>'),
    "respiratory": _lung(-1) + _lung(1) + '<path class="line" d="M200 84V104M200 104l-10 8M200 104l10 8"/>',
    "cardiovascular": ('<path d="M212 128c4-9 18-7 18 4 0 9-10 16-18 22-8-6-18-13-18-22 '
                       '0-11 14-13 18-4z"/>'),
    "metabolic": ('<ellipse cx="200" cy="198" rx="27" ry="15"/>'
                  '<path class="line" d="M181 196q19 9 38 0M186 204q14 6 28 0"/>'),
    "musculoskeletal": ('<path class="bone" d="M200 88V252M176 252q24 14 48 0'
                        'M178 262L172 340M222 262L228 340M172 348L171 410M228 348L229 410'
                        'M156 110L146 196M244 110L254 196M145 204L134 266M255 204L266 266'
                        'M170 112q30 10 60 0M168 126q32 10 64 0M168 140q32 10 64 0"/>'),
    "immune": ('<circle cx="190" cy="80" r="3.5"/><circle cx="210" cy="80" r="3.5"/>'
               '<circle cx="160" cy="118" r="3.5"/><circle cx="240" cy="118" r="3.5"/>'
               '<circle cx="182" cy="256" r="3.5"/><circle cx="218" cy="256" r="3.5"/>'),
}
# Organs are drawn in this order, so later ones sit on top (the heart over the lungs).
DRAW_ORDER = ("musculoskeletal", "immune", "metabolic", "respiratory", "cns", "cardiovascular")
# Callouts: anchor on the organ -> label at the side ("l" image left, "r" image right), label y.
CALLOUT = {
    "cns": ((183, 40), "l", 40), "respiratory": ((172, 138), "l", 130),
    "immune": ((182, 256), "l", 300),
    "cardiovascular": ((222, 136), "r", 112), "metabolic": ((226, 198), "r", 196),
    "musculoskeletal": ((228, 330), "r", 330),
}
SHORT = {"cns": "Nervous (CNS)"}
# Rising particles and the heart rate of the pulse are fixed (no randomness: same page twice).
_SPARKS = [(176, 0.0), (214, 1.3), (192, 2.1), (226, 0.7), (168, 2.8), (206, 3.4), (186, 4.2),
           (232, 1.9)]


def _heat(key: str) -> str:
    """Per-system radial gradient: the risk colour at the centre, fading through the cool
    blue to a soft edge (stops take --c / --cool from the organ group, so CSS can fade them)."""
    return (f'<defs><radialGradient id="heat-{key}" r="65%"><stop offset="0" class="h0"/>'
            f'<stop offset=".55" class="h1"/><stop offset="1" class="h2"/></radialGradient></defs>')


def _defs() -> str:
    return (
        '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur in="SourceGraphic" stdDeviation="2.6" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '<filter id="soft" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="6"/></filter>'
        '<linearGradient id="skin" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" class="skin0"/><stop offset="1" class="skin1"/></linearGradient>'
        '<linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" class="beam0"/><stop offset="1" class="beam1"/></linearGradient>'
        '<linearGradient id="cone" x1="0" y1="1" x2="0" y2="0"><stop offset="0" class="cone0"/>'
        '<stop offset="1" class="cone1"/></linearGradient>'
        '<linearGradient id="beamline" x1="0" y1="0" x2="1" y2="0"><stop offset="0" class="bl0"/>'
        '<stop offset=".5" class="bl1"/><stop offset="1" class="bl0"/></linearGradient>'
        '<pattern id="slices" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<path d="M0 .5H6" class="slice"/></pattern>'
        f'<clipPath id="body-clip"><path d="{_smooth(silhouette_points())}"/></clipPath>')


def _figure_svg() -> str:
    body = _smooth(silhouette_points())
    sparks = "".join(f'<circle class="spark" cx="{x}" cy="420" r="1.6" '
                     f'style="animation-delay:{d}s"/>' for x, d in _SPARKS)
    return (
        # projector base: light cone, two rings (one turning), rising sparks
        f'<path d="M128 424L166 150H234L272 424Z" fill="url(#cone)"/>'
        f'<ellipse class="ring" cx="{CX}" cy="424" rx="74" ry="11"/>'
        f'<ellipse class="ring spin" cx="{CX}" cy="424" rx="92" ry="14"/>'
        f'<ellipse class="ring" cx="{CX}" cy="424" rx="50" ry="7"/>{sparks}'
        # body: soft halo, translucent fill, hologram slices, glowing outline
        f'<path class="halo" d="{body}"/><path class="body" d="{body}"/>'
        f'<rect x="100" y="10" width="200" height="420" fill="url(#slices)" '
        f'clip-path="url(#body-clip)" class="slices"/>'
        f'<path class="body-outline" d="{body}"/>'
        f'<text class="side" x="140" y="26" text-anchor="middle">R</text>'
        f'<text class="side" x="260" y="26" text-anchor="middle">L</text>')


_ECG = ("M14 428H40l4-3 4 3h6l3-22 4 34 3-12h8l5-4 5 4H112")


def _vitals_svg(current: dict) -> str:
    """Bottom corners of the stage: a running ECG trace and the check-in vitals."""
    hr, spo2, temp = current["resting_hr"], current["spo2"], current["core_temp"]
    return (f'<path class="ecg-base" d="{_ECG}"/><path class="ecg" d="{_ECG}"/>'
            f'<text class="vit" x="{W - 12}" y="414" text-anchor="end">HR {hr:.0f} bpm</text>'
            f'<text class="vit" x="{W - 12}" y="427" text-anchor="end">SpO2 {spo2:.1f}%</text>'
            f'<text class="vit" x="{W - 12}" y="440" text-anchor="end">CORE {temp:.1f} °C</text>')


def _callout(key: str, label: str, value: str, forecast: bool) -> str:
    (ax, ay), side, ly = CALLOUT[key]
    left = side == "l"
    x_text = 8 if left else W - 8
    x_end = 104 if left else W - 104
    anchor = "start" if left else "end"
    gx = x_text if left else x_text - 8
    return (f'<g class="callout{" fc" if forecast else ""}" data-system="{key}">'
            f'<polyline class="lead" points="{ax},{ay} {x_end + (10 if left else -10)},{ly} '
            f'{x_end},{ly}"/><circle class="tip" cx="{ax}" cy="{ay}" r="2.2"/>'
            f'<rect class="glyph" x="{gx}" y="{ly - 21}" width="8" height="8" rx="4"/>'
            f'<text class="name" x="{x_text + (12 if left else -12)}" y="{ly - 13}" '
            f'text-anchor="{anchor}">{label}</text>'
            + (f'<text class="val" x="{x_text}" y="{ly + 3}" text-anchor="{anchor}">{value}</text>'
               if forecast else "".join(          # slow systems: one short fact per line
                   f'<text class="val" x="{x_text}" y="{ly + 3 + 14 * i}" text-anchor="{anchor}">'
                   f'{part}</text>' for i, part in enumerate(value.split(" · "))))
            + ('' if forecast else
               f'<text class="note" x="{x_text}" y="{ly + 3 + 14 * value.count(" · ") + 14}" '
               f'text-anchor="{anchor}">'
               f'<tspan class="pre">weekly check</tspan>'
               f'<tspan class="post">scanned · not forecast</tspan></text>')
            + '</g>')


_CSS = """
:root { --green:%(green)s; --yellow:%(yellow)s; --red:%(red)s; --grey:%(grey)s;
  --cool:var(--sf-info); }
* { box-sizing:border-box; }
body { margin:0; background:var(--sf-panel); color:var(--sf-ink);
  font:13px/1.4 "Source Sans Pro","Segoe UI",system-ui,sans-serif; }
#sf-body-map { position:relative; border:1px solid var(--sf-line); border-radius:10px;
  padding:10px 12px; height:%(h)dpx; overflow:hidden; display:flex; flex-direction:column; }
.top { display:flex; justify-content:space-between; align-items:center; }
.h { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--sf-muted); }
.chip { font:600 10px/1 ui-monospace,Menlo,monospace; letter-spacing:.12em; padding:4px 7px;
  border-radius:4px; color:var(--cool); border:1px solid color-mix(in srgb, var(--cool) 40%%, transparent); }
.chip span { display:none; }
[data-state="idle"] .chip .s-idle, [data-state="play"] .chip .s-play,
[data-state="revealed"] .chip .s-done { display:inline; }
[data-state="play"] .chip { animation: blink 1s steps(2) infinite; }
[data-state="revealed"] .chip { color:var(--green); border-color:color-mix(in srgb, var(--green) 45%%, transparent); }
@keyframes blink { 50%% { opacity:.45; } }
/* holographic stage: fine grid, a centre glow, corner brackets, faint scan lines */
.stage { position:relative; flex:1; min-height:0; margin-top:8px; border-radius:8px; overflow:hidden;
  border:1px solid color-mix(in srgb, var(--cool) 18%%, transparent);
  background:
    radial-gradient(ellipse at 50%% 45%%, color-mix(in srgb, var(--cool) 14%%, transparent),
                    transparent 60%%),
    linear-gradient(color-mix(in srgb, var(--cool) 6%%, transparent) 1px, transparent 1px) 0 0 / 18px 18px,
    linear-gradient(90deg, color-mix(in srgb, var(--cool) 6%%, transparent) 1px, transparent 1px) 0 0 / 18px 18px,
    var(--sf-bg); }
.stage::before { content:""; position:absolute; inset:6px; pointer-events:none;
  --b:color-mix(in srgb, var(--cool) 55%%, transparent);
  background:
    linear-gradient(var(--b),var(--b)) 0 0/14px 1px no-repeat, linear-gradient(var(--b),var(--b)) 0 0/1px 14px no-repeat,
    linear-gradient(var(--b),var(--b)) 100%% 0/14px 1px no-repeat, linear-gradient(var(--b),var(--b)) 100%% 0/1px 14px no-repeat,
    linear-gradient(var(--b),var(--b)) 0 100%%/14px 1px no-repeat, linear-gradient(var(--b),var(--b)) 0 100%%/1px 14px no-repeat,
    linear-gradient(var(--b),var(--b)) 100%% 100%%/14px 1px no-repeat, linear-gradient(var(--b),var(--b)) 100%% 100%%/1px 14px no-repeat; }
.stage::after { content:""; position:absolute; inset:0; pointer-events:none;
  background:repeating-linear-gradient(0deg, color-mix(in srgb, var(--sf-ink) 4%%, transparent) 0 1px,
                                       transparent 1px 3px);
  animation: holo 3.6s ease-in-out infinite; }
@keyframes holo { 0%%,100%% { opacity:.35; } 50%% { opacity:.8; } }
svg { position:absolute; inset:0; width:100%%; height:100%%; }
/* figure */
.skin0 { stop-color:color-mix(in srgb, var(--cool) 22%%, var(--sf-panel)); stop-opacity:.75; }
.skin1 { stop-color:var(--sf-panel-2); stop-opacity:.55; }
.body { fill:url(#skin); }
.halo { fill:none; stroke:var(--cool); stroke-width:10; opacity:.18; filter:url(#soft); }
.body-outline { fill:none; stroke:var(--cool); stroke-width:1.6; opacity:.95; filter:url(#glow); }
.slice { stroke:var(--cool); stroke-width:.6; opacity:.22; }
.side { fill:var(--sf-muted); font-size:11px; font-weight:600; letter-spacing:.05em; }
.cone0 { stop-color:var(--cool); stop-opacity:.22; } .cone1 { stop-color:var(--cool); stop-opacity:0; }
.ring { fill:none; stroke:var(--cool); stroke-width:1; opacity:.5; }
.ring.spin { stroke-dasharray:4 9; opacity:.8; stroke-width:1.4; animation: spin 9s linear infinite; }
@keyframes spin { to { stroke-dashoffset:-260; } }
.spark { fill:var(--cool); opacity:0; animation: rise 5s linear infinite; }
@keyframes rise { 0%% { transform:translateY(0); opacity:0; } 15%% { opacity:.8; }
  100%% { transform:translateY(-300px); opacity:0; } }
/* scan beam: slow sweep at standby, fast while fuzzing, gone once the scan is complete */
.beam { opacity:0; }
.beam0 { stop-color:var(--cool); stop-opacity:0; } .beam1 { stop-color:var(--cool); stop-opacity:.45; }
.bl0 { stop-color:var(--cool); stop-opacity:0; } .bl1 { stop-color:var(--sf-ink); stop-opacity:.95; }
.beam-glow + rect { filter:url(#glow); }
[data-state="idle"] .beam { opacity:.55; animation: scan 6s ease-in-out infinite; }
[data-state="play"] .beam { opacity:1; animation: scan 1.4s linear infinite; }
@keyframes scan { 0%% { transform:translateY(-30px); } 100%% { transform:translateY(420px); } }
/* ECG trace and vitals readout */
.ecg-base { fill:none; stroke:color-mix(in srgb, var(--green) 18%%, transparent); stroke-width:1; }
.ecg { fill:none; stroke:var(--green); stroke-width:1.4; stroke-dasharray:40 200;
  animation: ecg %(beat).2fs linear infinite; filter:url(#glow); }
@keyframes ecg { from { stroke-dashoffset:200; } to { stroke-dashoffset:-40; } }
.vit { fill:color-mix(in srgb, var(--green) 85%%, var(--sf-ink)); font:600 10px ui-monospace,Menlo,monospace;
  letter-spacing:.06em; }
/* organs */
.organ { cursor:pointer; outline:none; }
.organ > :not(defs) { fill:var(--fill); stroke:color-mix(in srgb, var(--cool) 45%%, transparent);
  stroke-width:.8; opacity:.8; transition: opacity %(t).1fs ease; }
.organ .line, .organ .nerve, .organ .bone { fill:none !important; }
.organ .line { stroke:color-mix(in srgb, var(--cool) 55%%, transparent); stroke-width:.8; }
.organ .nerve { stroke:var(--cool); stroke-width:1.6; stroke-dasharray:2 3; }
.organ.slow .bone { stroke:var(--sf-muted); stroke-width:1.4; stroke-dasharray:3 3; opacity:.55; }
.organ.slow > circle { fill:color-mix(in srgb, var(--sf-muted) 35%%, transparent); stroke:var(--sf-muted); }
.organ stop { transition: stop-color %(t).1fs ease; }
.organ .h0 { stop-color:var(--cool); stop-opacity:.95; }
.organ .h1 { stop-color:var(--cool); stop-opacity:.5; }
.organ .h2 { stop-color:var(--cool); stop-opacity:.12; }
.organ:hover > :not(defs), .organ:focus > :not(defs) { stroke:var(--sf-ink); stroke-width:1.4; }
[data-system="cardiovascular"].organ > path { transform-box:fill-box; transform-origin:center;
  animation: beat %(beat).2fs ease-in-out infinite; }
@keyframes beat { 0%%,100%% { transform:scale(1); } 12%% { transform:scale(1.09); } 24%% { transform:scale(1); } }
[data-state="play"] .organ.fc > :not(defs) { animation: breathe 1.6s ease-in-out infinite; }
@keyframes breathe { 0%%,100%% { opacity:.3; } 50%% { opacity:1; } }
[data-state="revealed"] .organ.fc > :not(defs) { opacity:.95; }
[data-state="revealed"] .organ.fc .h0 { stop-color:var(--c); }
[data-state="revealed"] .organ.fc .h1 { stop-color:color-mix(in srgb, var(--c) 65%%, var(--cool)); }
[data-state="revealed"] .organ.fc .h2 { stop-color:color-mix(in srgb, var(--c) 30%%, var(--cool)); }
[data-state="revealed"] .organ.fc .nerve, [data-state="revealed"] .organ.fc .line { stroke:var(--c); }
[data-state="revealed"] .organ.fc[data-status="red"] > :not(defs) { filter:url(#glow); }
/* a red system sends out a slow ripple */
.ripple { fill:none; stroke:var(--red); stroke-width:1.5; opacity:0; transform-box:fill-box;
  transform-origin:center; }
[data-state="revealed"] .ripple { animation: ripple 2.2s ease-out infinite; animation-delay:%(t).1fs; }
@keyframes ripple { 0%% { transform:scale(.4); opacity:.9; } 100%% { transform:scale(2.4); opacity:0; } }
/* callouts: status glyph + name + share; the share only after the reveal */
.callout { cursor:pointer; }
.lead { fill:none; stroke:color-mix(in srgb, var(--cool) 45%%, transparent); stroke-width:.8; }
.tip { fill:var(--cool); }
.name { fill:var(--sf-muted); font-size:11px; letter-spacing:.02em; }
.val { fill:var(--sf-ink); font:700 15px ui-monospace,Menlo,monospace; }
.callout:not(.fc) .val { font:600 11px ui-monospace,Menlo,monospace; fill:var(--cool); }
.note { fill:var(--sf-muted); font-size:10px; letter-spacing:.03em; }
.note .post { display:none; }
[data-state="revealed"] .note .post { display:inline; }
[data-state="revealed"] .note .pre { display:none; }
[data-state="revealed"] .callout:not(.fc) .glyph { stroke-dasharray:2 2; }
.glyph { fill:transparent; stroke:var(--sf-muted); stroke-width:1.6; transform-box:fill-box;
  transform-origin:center; transition: fill %(t).1fs ease, rx %(t).1fs ease, transform %(t).1fs ease; }
.val .post { display:none; }
[data-state="revealed"] .callout.fc .val .post { display:inline; }
[data-state="revealed"] .callout.fc .val .pre { display:none; }
[data-state="revealed"] .callout.fc .lead { stroke:color-mix(in srgb, var(--c) 70%%, transparent); }
[data-state="revealed"] .callout.fc .tip { fill:var(--c); }
[data-state="revealed"] .callout.fc .val { fill:var(--c); }
[data-state="revealed"] .callout.fc .glyph { fill:var(--c); stroke:none; }
/* status is never colour alone (green vs yellow is dE 6.1 for protanopia): shapes at reveal */
[data-state="revealed"] .callout.fc[data-status="yellow"] .glyph { rx:1px; transform:rotate(45deg) scale(.85); }
[data-state="revealed"] .callout.fc[data-status="red"] .glyph { rx:1px; }
[data-state="revealed"] .callout.fc[data-status="grey"] .glyph { fill:transparent; stroke:var(--sf-muted); }
/* radiation dose (operator D), below the stage */
.dose { display:flex; align-items:center; gap:10px; margin-top:8px; }
.dose .h { flex:none; }
.tube { position:relative; flex:1; height:10px; background:var(--sf-grid); border-radius:5px;
  overflow:hidden; border:1px solid var(--sf-line-strong); }
.fill { position:absolute; top:0; bottom:0; left:0; background:var(--rad); width:0;
  transition: width %(t).1fs ease; box-shadow:0 0 8px var(--rad); }
[data-state="revealed"] .fill { width:var(--dose); }
.limit { position:absolute; top:-2px; bottom:-2px; right:0; border-left:2px dashed var(--red); }
.dose .v { flex:none; font:600 12px ui-monospace,Menlo,monospace; color:var(--sf-ink); }
.small { font-size:11px; color:var(--sf-muted); line-height:1.35; margin-top:4px; }
.pop { display:none; position:absolute; left:10px; right:10px; top:40px;
  background:var(--sf-panel-2); border:1px solid color-mix(in srgb, var(--cool) 45%%, transparent);
  border-radius:8px; padding:10px 12px; box-shadow:0 8px 30px rgba(0, 0, 0, .65); z-index:2; }
.pop.open { display:block; }
.pop b { font-size:15px; } .pop .x { float:right; cursor:pointer; color:var(--sf-muted); }
.pop p { margin:6px 0; }
@media (prefers-reduced-motion: reduce) {
  [data-state="play"] .organ.fc > :not(defs) { animation:none; opacity:.8; }
  .stage::after { animation:none; }
  .beam, .spark, .ring.spin, .ripple, .chip, .ecg, [data-system="cardiovascular"].organ > path { animation:none !important; }
  * { transition:none !important; }
}
"""

_JS = """
const root = document.getElementById('sf-body-map');
const info = %(info)s;
const pop = document.getElementById('pop');
function openPop(key) {
  const s = info[key];
  pop.innerHTML = '<span class="x" onclick="this.parentNode.classList.remove(\\'open\\')">✕</span>'
    + '<b>' + s.label + '</b><p><span class="small">Function</span><br>' + s.function + '</p>'
    + '<p><span class="small">Now</span><br>' + s.now + '</p>'
    + '<p><span class="small">Space hazard</span><br>' + s.hazard + '</p>'
    + '<p><span class="small">Forecast</span><br>' + (root.dataset.state === 'revealed'
        ? s.forecast : 'Run the forecast to see this system.') + '</p>';
  pop.classList.add('open');
}
document.querySelectorAll('[data-system]').forEach(el => {
  el.addEventListener('click', () => openPop(el.dataset.system));
  el.addEventListener('keydown', e => { if (e.key === 'Enter') openPop(el.dataset.system); });
});
function done() {
  if (document.querySelector('[data-testid="body-map-done"]')) return;
  const m = document.createElement('div');
  m.setAttribute('data-testid', 'body-map-done'); m.style.display = 'none';
  m.textContent = 'body-map-done'; root.appendChild(m);
}
if (root.dataset.state === 'play') {
  setTimeout(() => { root.dataset.state = 'revealed'; setTimeout(done, %(t_ms)d); },
             %(reveal_ms)d);
}
"""


def html_doc(mode: str, status: dict, current: dict, reveal_at_s: float = 0.0) -> str:
    """Full iframe document. mode: idle (standby scan) | play (fast scan, reveal later) |
    revealed (systems in their risk colour)."""
    pal = ui_config.COLOR
    organs, callouts, info = {}, [], {}
    for s in SYSTEMS:
        k, stt = s["key"], status[s["key"]]
        color = pal.get(stt["status"], pal["grey"])
        cls = "fc" if s["forecast"] else "slow"
        style = f'--c:{color};--fill:url(#heat-{k})'
        ripple = ""
        if s["forecast"] and stt["status"] == "red":
            (ax, ay), _, _ = CALLOUT[k]
            ripple = f'<circle class="ripple" cx="{ax}" cy="{ay}" r="14"/>'
        organs[k] = (f'<g class="organ {cls}" data-system="{k}" data-status="{stt["status"]}" '
                     f'tabindex="0" style="{style}" aria-label="{_esc(s["label"])}">'
                     f'<title>{_esc(s["label"])}</title>{_heat(k)}{_ORGANS[k]}{ripple}</g>')
        value = (f'<tspan class="pre">--</tspan><tspan class="post">{_esc(stt["share"])}</tspan>'
                 if s["forecast"] else _esc(current[f"{k}_today"]))
        callouts.append(
            _callout(k, _esc(SHORT.get(k, s["label"])), value, s["forecast"])
            .replace('<g class="callout', f'<g data-testid="organ-{k}" data-status="{stt["status"]}" '
                                          f'style="--c:{color}" class="callout', 1))
        if s["forecast"]:
            fc_text = (f"{stt['share']} of simulations crash in this system"
                       + (f" (most often: {stt['top_rule']})" if stt["top_rule"] else "") + ".")
            if stt["status"] == "grey":
                fc_text = "No forecast: data not trustworthy."
        else:
            fc_text = ("Not forecast: this system changes over weeks and is checked at the weekly "
                       "exam; the scan shows today's real context only.")
        info[k] = {"label": _esc(s["label"]), "function": _esc(s["function"]),
                   "now": _esc(current[k]), "hazard": _esc(s["hazard"]),
                   "forecast": _esc(fc_text)}
    limit = config.RAD_EVA_DOSE_LIMIT_UGY
    dose = current["eva_dose_ugy"]
    fill = min(100.0, 100.0 * dose / limit)
    rad = status["radiation"]["share"]
    marker = ('<div data-testid="body-map-done" style="display:none">body-map-done</div>'
              if mode == "revealed" else "")
    beat = 60.0 / max(40.0, current.get("resting_hr", 70.0))
    css = _CSS % {**pal, "h": HEIGHT_PX - 4, "t": TRANSITION_S, "beat": beat}
    js = _JS % {"info": theme.script_json(info), "t_ms": int(TRANSITION_S * 1000),
                "reveal_ms": int(round(reveal_at_s * 1000))}
    beam = (f'<g class="beam"><rect x="100" y="-38" width="200" height="40" fill="url(#beam)" '
            f'clip-path="url(#body-clip)" class="beam-glow"/>'
            f'<rect x="60" y="1.2" width="280" height="1.6" fill="url(#beamline)"/></g>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{theme.root_css()}{css}</style></head>
<body><div id="sf-body-map" data-testid="body-map-figure" data-state="{mode}">
<div class="top"><div class="h">Bio-scan · tap a system</div>
<div class="chip"><span class="s-idle">STANDBY</span><span class="s-play">SCANNING</span><span class="s-done">SCAN COMPLETE</span></div></div>
<div class="stage"><svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Body map"><defs>{_defs()}</defs>{_figure_svg()}{_vitals_svg(current)}{"".join(organs[k] for k in DRAW_ORDER)}{beam}{"".join(callouts)}</svg></div>
<div class="dose" data-testid="dose-bar" data-status="{status["radiation"]["status"]}"
 style="--dose:{fill:.0f}%;--rad:{pal.get(status["radiation"]["status"], pal["grey"])}">
<span class="h">EVA dose</span><div class="tube"><div class="fill"></div><div class="limit" title="limit"></div></div>
<span class="v">{dose:.0f} / {limit:.0f} µGy</span></div>
<div class="small">Radiation (operator D): {dose:.1f} µGy planned, single-EVA limit {limit:.0f} µGy
(illustrative); radiation crashes in {_esc(rad)} of simulations. Sources: MSL-RAD (Mars surface),
GOES flare timing (Earth-orbit proxy). Musculoskeletal, immune: weekly exam, not forecast.</div>
<div class="pop" id="pop" data-testid="body-map-popup"></div>
{marker}
</div><script>{js}</script></body></html>"""


# ---------------------------------------------------------------- Streamlit

def show(static: dict, res: dict | None, shown: int, frames: list) -> None:
    """Draw the body map for app.py's playback frame `shown` (once per script run: the
    frame then animates by itself; ui/live_frame.py swaps documents without a blank flash)."""
    playing = res is not None and 0 < shown < len(frames)
    mode = "idle" if res is None or shown == 0 else ("play" if playing else "revealed")
    fc = res["forecast"] if res else None
    doc = html_doc(mode, system_status(fc if mode != "idle" else None),
                   current_state(static["name"], static.get("checkin_json")),
                   stage_times(frames)[0] if mode == "play" else 0.0)
    st.html('<div data-testid="body-map" style="display:none">body-map</div>')
    live_frame.show(doc, HEIGHT_PX, key="body_map")
