"""Fuzz log (plan 4.2): typewriter replay of the real fuzzing run.

Every line comes from the engine output in st.session_state (forecast + adjustment):
probability track totals; each new lowest-fitness sample of the guided track (mutation,
steps from violation, nearest rule, seed pool size); each new crash type at the round it
was first found (highlighted); plateau round; final seed pool and lowest fitness;
causal-chain discards of both tracks (labelled: the guided track discards more because its
samples stack 1-3 operators); "not covered" warnings (highlighted); the closest crash; the
result and the adjustment re-tests. The header replays the guided round counter on a log
scale (early finds spread out), the seed pool size per round and the measured simulations
per second of each timing segment. Nothing is invented.

Like ui/body_map.py the log is one iframe rendered once per playback with the stage times
built in; it types itself and adds data-testid="fuzz-log-done" when the last line is out
Playwright: the panel sits inside ui/live_frame's component iframe, so reach it with
page.frame_locator('iframe[title*="live_frame"]') (several panels: pick by order) and then
FRAME_SELECTOR, or wait for the top-level markers (forecast-done, adjust-done,
playback-done) instead.
"""
import dataclasses
import functools
import json
import math
import re

import streamlit as st

from spacefuzz import config
from spacefuzz.core import evaluator as ev
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import state
import ui_config
from ui import checkin_store, live_frame, theme
from ui.body_map import rule_text
from ui.common import (CIRCLED, clock as _clock, esc as _esc, no_change_text, pct as _pct,
                       stage_times)

FRAME_SELECTOR = 'iframe[srcdoc*="sf-fuzz-log"]'
HEIGHT_PX = 300
CHAR_MS = 9                  # typing speed when the schedule allows it

# Forecast window layout (fractions of the time until the forecast reveal).
_GUIDED_FROM, _GUIDED_SPAN = 0.12, 0.70


def _dur(minutes: float) -> str:
    m = int(round(abs(minutes)))
    return f"{m // 60} h {m % 60:02d} min"




def describe_ops(ops: list[dict]) -> str:
    """Plain-language mutation, e.g. 'sleep -2.9 h + EVA +1 h 30 min, heat 0.29 °C/h'."""
    parts = []
    for o in ops:
        op, p = o["op"], o.get("params", {})
        if op == "A":
            d = p.get("duration_delta_min", 0)
            s = f"EVA {'+' if d >= 0 else '-'}{_dur(d)}" if d else "EVA"
            if "eva_heat_rate" in p:
                s += f", heat {p['eva_heat_rate']:.2f} °C/h"
            parts.append(s)
        elif op == "B":                 # one cause, its consequences after the arrow
            effects = []
            if p.get("work_extend_min"):
                effects.append(f"{p.get('work_task', 'work')} +{_dur(p['work_extend_min'])}")
            if p.get("spo2_drop"):
                effects.append(f"SpO2 -{p['spo2_drop']:.1f}%")
            parts.append(f"CO2 excursion +{100 * p.get('co2_excursion', 0):.0f}%"
                         + (" -> " + ", ".join(effects) if effects else ""))
        elif op == "C" and p.get("task") is None:
            parts.append("no free time left for an extra task")
        elif op == "C":
            parts.append(f"{p.get('task', 'task')} {_dur(p.get('duration_min', 0))} inserted, "
                         f"fatigue +{p.get('fatigue_jump', 0):.0f}")
        elif op == "D":
            w = p.get("warning")
            parts.append(f"SPE warning from {_clock(w[0])}" if w
                         else "solar particle event (no warning)")
        elif op == "E":
            parts.append(f"sleep -{p.get('sleep_cut_h', 0):.1f} h"
                         + (" (3 short nights)" if p.get("chronic") else ""))
        elif op == "F":
            parts.append("sensor fault: " + ", ".join(p.get("faults", [])))
        else:
            parts.append(op)
    return " + ".join(parts)


def round_to_t(rnd: int, rounds: int) -> float:
    """Replay position (0-1) of a guided-track round, log scale."""
    return math.log1p(max(rnd, 0)) / math.log1p(max(rounds, 1))


def thin_events(events: list, keep: int = 6) -> list:
    """At most `keep` events, evenly spread, always the first and the last."""
    if len(events) <= keep:
        return list(events)
    idx = sorted({round(i * (len(events) - 1) / (keep - 1)) for i in range(keep)})
    return [events[i] for i in idx]


def header_series(guided: dict) -> dict:
    """Seed pool (<= 101 points) and guided simulations/s per timing segment, for the header."""
    pool = (guided.get("history") or {}).get("pool") or []
    step = max(1, len(pool) // 100)
    pts = pool[::step] + ([pool[-1]] if pool and (len(pool) - 1) % step else [])
    rate, last_r, last_t = [], 0, 0.0
    for r, t in guided.get("timing") or []:
        if t > last_t:
            rate.append([r, round((r - last_r) / (t - last_t), 1)])
        last_r, last_t = r, t
    return {"pool": pts, "pool_step": step, "rate": rate}


@functools.lru_cache(maxsize=8)
def _plan_eval(name: str, checkin_json: str | None = None) -> tuple[dict, bool, float]:
    """Temporal-rule steps to violation of the plan as written (deterministic, no mutation),
    for the current check-in (checkin_json = the crew's edits, None = the example file)."""
    sc = mu.Scenario.load(name)
    if checkin_json:
        sc = dataclasses.replace(sc, checkin=checkin_store.from_json(name, checkin_json))
    b = state.make_baseline()
    e = ev.evaluate(md.simulate(sc.plan, sc.trace, sc.checkin, b), sc.plan, sc.checkin, b)
    return dict(e.steps_to_violation), e.crashed, e.fitness


def _rule_label(key: str) -> str:
    m = re.match(r"rule(\d+)_", key)
    num = f"rule {m[1]}" if m else key
    return f"{num} ({ev.RULE_TEXT.get(key, key)})"


def _change(steps: int) -> str:
    m = steps * config.STEP_MINUTES
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def plan_line(steps: dict, crashed: bool, best_steps: list) -> str:
    """Where the plan as written sits relative to the time-based rules."""
    known = {k: v for k, v in steps.items() if v is not None}
    hit = [k for k, v in known.items() if v == 0]
    if hit:
        return "[plan ] plan as written already breaks " + " and ".join(map(_rule_label, hit))
    if not known:
        return "[plan ] no time-based rule applies to this plan" + (
            " (it crashes on a composite rule)" if crashed else "")
    m = min(known.values())
    near = [k for k, v in known.items() if v == m]
    line = (f"[plan ] plan as written is {m} step{'s' if m != 1 else ''} ({_change(m)} of "
            f"change) from " + " and ".join(map(_rule_label, near)))
    if best_steps and all(x == m for x in best_steps):
        line += (f": it sits on these lines, so every near miss below is {m} step"
                 f"{'s' if m != 1 else ''} away (same finding as the closest crash further down)")
    return line


def _count_kind(lin: list, op: str) -> list:
    return [o for o in lin if o["op"] == op]


def _kind_text(lin: list, op: str) -> str:
    """One change in full; repeats of the same kind counted."""
    ops = _count_kind(lin, op)
    if len(ops) == 1:
        return describe_ops(ops)
    if op == "C":
        tasks = {}
        for o in ops:
            t = o["params"].get("task", "task")
            tasks[t] = tasks.get(t, 0) + 1
        mix = ", ".join(f"{t} x{n}" for t, n in sorted(tasks.items(), key=lambda kv: -kv[1]))
        return f"{len(ops)} tasks inserted ({mix})"
    if op == "B":
        top = max(o["params"].get("co2_excursion", 0) for o in ops)
        return f"{len(ops)} CO2 excursions (up to +{100 * top:.0f}%)"
    warned = sum(1 for o in ops if o["params"].get("warning"))
    return f"{len(ops)} solar particle events ({warned} with a warning)"


def change_items(b: dict, hist: dict) -> dict:
    """The sample's whole change vs the plan, grouped: `head` = sleep / EVA duration (from the
    per-round history) and EVA heat; `items` = other physical changes by kind, in the order
    they first occurred; `faults` = any sensor fault (readings only, not the body)."""
    i = b["round"] - 1
    head = []
    if i < len(hist.get("sleep_cut_h", [])):
        cut, ext = hist["sleep_cut_h"][i], hist["eva_extend_h"][i]
        if abs(cut) >= 0.05 or abs(ext) >= 0.05:
            head.append(f"sleep -{cut:.1f} h, EVA {ext:+.1f} h")
    lin = b.get("lineage")
    if lin is None:
        return {"head": head, "items": [], "faults": False, "lineage": False,
                "unchanged": False}
    heats = [o["params"]["eva_heat_rate"] for o in _count_kind(lin, "A")
             if "eva_heat_rate" in o.get("params", {})]
    if heats:
        head.append(f"EVA heat {heats[-1]:.2f} °C/h")
    kinds = [k for k in dict.fromkeys(o["op"] for o in lin) if k in ("B", "C", "D")]
    items = [_kind_text(lin, k) for k in kinds]
    faults = bool(_count_kind(lin, "F"))
    unchanged = not head and not items and not any(o["op"] in ("A", "E") for o in lin)
    return {"head": head, "items": items, "faults": faults, "lineage": True,
            "unchanged": unchanged}


def whole_change(b: dict, hist: dict) -> str:
    """One-line form of change_items (used by the new-low lines)."""
    c = change_items(b, hist)
    if not c["lineage"]:
        return "change vs plan: " + (c["head"][0] if c["head"] else "sleep and EVA as planned")
    if c["unchanged"]:
        return "plan unchanged (sensor fault only)" if c["faults"] else "plan unchanged"
    text = ", ".join(c["head"]) if c["head"] else "sleep and EVA as planned"
    if c["items"]:
        text += " + " + " + ".join(c["items"])
    return "change vs plan: " + text + (" (+ sensor faults)" if c["faults"] else "")




def simplest_near_miss(events: list, hist: dict, plan_fitness: float | None = None,
                       plan_min_steps: int | None = None) -> str | None:
    """Among the recorded new lowest-fitness samples nearest to a time-based rule, the one
    with the fewest physical changes (sensor faults not counted; ties: lower fitness). Every
    recorded new low is closer to a crash than the plan as written (lower fitness), so this
    answers "the fewest changes that bring this day closer to a crash". Changes are numbered
    and grouped by cause (a CO2 excursion and what it causes are one change)."""
    cands = [e for e in events if e.get("lineage") is not None
             and e.get("steps_to_violation") is not None]
    if not cands:
        return None
    m = min(e["steps_to_violation"] for e in cands)
    physical = lambda e: sum(o["op"] != "F" for o in e["lineage"])
    e = min((e for e in cands if e["steps_to_violation"] == m),
            key=lambda e: (physical(e), e["fitness"]))
    c = change_items(e, hist)
    groups = c["head"][:1] + c["items"] + c["head"][1:]
    k = len(groups)
    rule = rule_text(e["nearest_rule"])
    fit = (f"fitness {plan_fitness:.2f} -> {e['fitness']:.2f}, " if plan_fitness is not None
           else f"fitness {e['fitness']:.2f}, ")
    lines = [f"[near ] fewest changes that bring this day closer to a crash than the plan as "
             f"written (among the {len(events)} recorded new lows): round {e['round']:,}, {fit}"
             f"{k} change{'s' if k != 1 else ''} vs plan:"]
    lines += [f"   {CIRCLED[j] if j < len(CIRCLED) else str(j + 1) + '.'} {g}"
              for j, g in enumerate(groups)]
    tail = [] if c["head"] and not c["head"][0].startswith("EVA heat") else \
        ["sleep and EVA as planned"]
    if c["faults"]:
        tail.append("plus sensor faults (readings only)")
    same = (", like the plan as written" if plan_min_steps is not None and plan_min_steps == m
            else "")
    lines.append(f"   {'; '.join(tail) + '; ' if tail else ''}still {m} step"
                 f"{'s' if m != 1 else ''} ({_change(m)} of change) from a time-based rule"
                 f"{same}; nearest rule: {rule}")
    return "\n".join(lines)


def _best_line(b: dict, hist: dict) -> str:
    """One new lowest-fitness sample: whole change vs the plan first, then the rule it is
    nearest to, then the last mutation step."""
    rule = rule_text(b["nearest_rule"])
    change = whole_change(b, hist)
    stacked = f", {len(b['lineage'])} mutations stacked" if b.get("lineage") else ""
    parts = [f"[best ] round {b['round']:<6,} new lowest fitness {b['fitness']:.2f} "
             f"(seed pool {b.get('pool')}{stacked}): {change}", f"nearest rule: {rule}"]
    n = b.get("steps_to_violation")
    if n is not None:
        parts.append(f"a time-based rule is {n} step{'s' if n != 1 else ''} "
                     f"({n * config.STEP_MINUTES} min of change) away")
    how = describe_ops(b.get("ops", []))
    if how and not change.startswith("plan unchanged"):
        parts.append(f"last step: {how}")
    return "; ".join(parts)


def _discards(summary: str | None) -> str | None:
    """'mutator: 10000 drawn, ..., 1127 discarded (discard rate 11.3%; chain1 264, ...)'."""
    m = re.search(r"(\d+) drawn, \d+ accepted, (\d+) discarded \(discard rate ([\d.]+)%; (.*)\)",
                  summary or "")
    if not m:
        return None
    drawn, disc, rate, why = int(m[1]), int(m[2]), m[3], m[4]
    why = "" if why == "none" else "; " + why.replace("chain", "chain ")
    return f"causal-chain check discarded {disc:,} of {drawn:,} ({rate}%{why})"


def log_lines(res: dict, start_min: int) -> list[dict]:
    """Log lines with kind (info / new / uncovered / warn / result), stage and time t."""
    fc, adj = res["forecast"], res.get("adjust")
    lines: list[dict] = []

    def add(stage, t, text, kind="info"):
        lines.append({"stage": stage, "t": round(t, 4), "text": text, "kind": kind})

    if fc["risk_rating"] == "UNKNOWN":
        add("forecast", 0.0, "[gate ] " + fc.get("message", "Data not trustworthy."), "warn")
        add("forecast", 1.0, "[stop ] no simulation run on untrusted data", "warn")
        return lines

    prob, guided = fc.get("probability_track", {}), fc.get("guided_track", {})
    rounds = guided.get("rounds", 0)
    add("forecast", 0.0, f"[setup] {res.get('name', '')}: check-in {_clock(start_min)}, "
                         f"profile {res.get('profile', '--')}, seed {fc.get('seed', '--')}")
    if prob:
        add("forecast", 0.03, f"[prob ] probability track: {prob['samples']:,} samples, "
                              f"operators drawn with their daily probabilities")
        add("forecast", 0.06, f"[prob ] {prob['crashes']:,} / {prob['samples']:,} crashed "
                              f"({_pct(prob['risk_probability'])})")
    if rounds:
        add("forecast", 0.09, f"[guide] guided track: {rounds:,} rounds, explore probability "
                              f"{config.GUIDED_EXPLORE_PROB:.0%}")
    if res.get("name"):
        steps, crashed, _ = _plan_eval(res["name"], res.get("checkin_json"))
        best_steps = [b.get("steps_to_violation") for b in guided.get("best_events", [])]
        add("forecast", 0.10, plan_line(steps, crashed, best_steps))
    events = [(c.get("first_round", 0), "new", c) for c in guided.get("crash_types", [])]
    events += [(b["round"], "best", b) for b in thin_events(guided.get("best_events", []))]
    hist = guided.get("history") or {}
    if rounds and not guided.get("best_events") and hist.get("crashed"):
        accepted = len(hist["crashed"]) - sum(hist.get("discarded", []))
        add("forecast", 0.11, f"[best ] all {accepted:,} accepted guided samples "
                              f"crashed, so there is no near miss to show")
    if guided.get("plateau_round"):
        events.append((guided["plateau_round"], "plateau", None))
    order = {"best": 0, "new": 1, "plateau": 2}
    for rnd, what, c in sorted(events, key=lambda e: (e[0], order[e[1]])):
        t = _GUIDED_FROM + _GUIDED_SPAN * round_to_t(rnd, rounds)
        if what == "best":
            add("forecast", t, _best_line(c, hist), "best")
        elif what == "new":
            how = describe_ops(c.get("last_mutation", []))
            add("forecast", t, f"[guide] round {rnd:<6,} NEW CRASH {c['system']}: "
                               f"{c.get('description', c['rule'])}"
                               + (f"  <- {how}" if how else ""), "new")
        else:
            add("forecast", t, f"[guide] round {rnd:<6,} plateau: no new crash type for "
                               f"{config.GUIDED_PLATEAU_ROUNDS:,} rounds")
    if rounds:
        add("forecast", 0.84, f"[guide] round {rounds:,} done: seed pool {guided.get('seeds')}, "
                              f"lowest fitness {guided.get('best_fitness', 0):.2f} "
                              f"(0 = crash), {guided.get('explore_rounds', 0):,} explore rounds")
    plan_fit = plan_min = None
    if res.get("name"):
        steps0, _, plan_fit = _plan_eval(res["name"], res.get("checkin_json"))
        known = [v for v in steps0.values() if v is not None]
        plan_min = min(known) if known else None
    near = simplest_near_miss(guided.get("best_events", []), hist, plan_fit, plan_min)
    if near:
        add("forecast", 0.845, near, "near")
    disc_p, disc_g = _discards(prob.get("mutator")), _discards(guided.get("mutator"))
    if disc_p:
        add("forecast", 0.86, f"[chain] probability track: {disc_p}")
    if disc_g:
        add("forecast", 0.87, f"[chain] guided track: {disc_g} - guided samples stack 1-3 "
                              f"operators, so more of them break a causal chain")
    for w in fc.get("coverage_warnings", []):
        add("forecast", 0.90, f"[cover] {w}", "uncovered")
    heat = fc.get("heatmap")
    if heat:
        add("forecast", 0.93, f"[map  ] crash boundary map: {len(heat['sleep_cut_h'])} x "
                              f"{len(heat['eva_extend_h'])} cells (sleep cut x EVA extension)")
    minp = fc.get("min_crash_perturbation")
    if isinstance(minp, dict) and minp.get("text"):
        add("forecast", 0.95, f"[edge ] closest crash: {minp['text']}")
    perf = fc.get("performance")
    if perf:
        add("forecast", 0.97, f"[perf ] {perf['simulations']:,} simulations in "
                              f"{perf['seconds']:.0f} s, "
                              f"{perf['simulations_per_second']:.0f} simulations/s average")
    ttc = fc.get("time_to_crash_minutes")
    add("forecast", 1.0, f"[risk ] {_pct(fc['risk_probability'])} {fc['risk_rating']}"
                         + (f", time to crash {_dur(ttc)}" if ttc else ""), "result")

    pa = (adj or {}).get("plan_adjustment")
    if not pa or not pa.get("changes"):
        msg = no_change_text(bool(pa and pa.get("escalate_to_ground")))
        add("adjust", 0.5, "[adjust] " + msg[0].lower() + msg[1:])
        return lines
    det = adj.get("details", {})
    add("adjust", 0.1, f"[adjust] {len(det.get('candidates', [])):,} candidate plans screened")
    sel = det.get("selection", [])
    for i, s in enumerate(sel):
        add("adjust", 0.3 + 0.4 * i / max(len(sel), 1),
            f"[retest] {'; '.join(s['changes'])}: {_pct(s.get('confirm', s.get('screen')))} "
            f"({s['result']})", "new" if s["result"] != "chosen" else "info")
    rt = pa.get("retest") or {}
    if rt.get("slip_margin"):
        add("adjust", 0.8, f"[retest] schedule slip: {rt['slip_margin']}")
    add("adjust", 1.0, f"[adjust] {_pct(pa['risk_before'])} -> {_pct(pa['risk_after'])} "
                       f"{pa['rating_after']}", "result")
    return lines



# ---------------------------------------------------------------- HTML

_CSS = """
* { box-sizing:border-box; }
body { margin:0; background:var(--sf-bg); color:var(--sf-log);
  font:12px/1.5 Menlo,Consolas,"DejaVu Sans Mono",monospace; }
#sf-fuzz-log { border:1px solid var(--sf-line); border-radius:10px; height:%(h)dpx;
  display:flex; flex-direction:column; overflow:hidden; }
.head { display:flex; justify-content:space-between; gap:8px; padding:6px 10px;
  white-space:nowrap; overflow:hidden; font-size:11px;
  border-bottom:1px solid var(--sf-line); color:var(--sf-muted); background:var(--sf-panel); }
.head b { color:var(--sf-ink); font-weight:600; }
.body { flex:1; overflow-y:auto; padding:6px 10px; scrollbar-width:thin; }
.ln { white-space:pre-wrap; word-break:break-word; }
.ln.new { color:%(red)s; font-weight:700; }
.ln.new::before { content:"\\25B2 "; }
.ln.best { color:var(--sf-info); }
.ln.near { color:var(--sf-ink); font-weight:700; }
.ln.near::before { content:"\\25C6 "; color:%(green)s; }
.ln.uncovered { color:%(yellow)s; }
.ln.uncovered::before { content:"\\26A0 "; }
.ln.warn { color:%(yellow)s; }
.ln.result { color:var(--sf-ink); font-weight:700; }
.cursor { display:inline-block; width:7px; height:12px; background:%(green)s;
  vertical-align:-1px; animation: blink 1s steps(1) infinite; }
@keyframes blink { 50%% { opacity:0; } }
.idle { color:var(--sf-muted); }
"""

_JS = """
const root = document.getElementById('sf-fuzz-log');
const body = root.querySelector('.body');
const lines = %(lines)s, R = %(rounds)d, FC = %(fc_ms)d, AD = %(ad_ms)d, CH = %(char_ms)d;
const G0 = %(g0)s, GS = %(gs)s;
const rnd = document.getElementById('round');
const H = %(series)s, poolEl = document.getElementById('pool'), rateEl = document.getElementById('rate');
function headerAt(r) {
  if (H.pool.length) poolEl.textContent = H.pool[Math.min(H.pool.length - 1, Math.floor(r / H.pool_step))];
  const seg = H.rate.find(s => r <= s[0]) || H.rate[H.rate.length - 1];
  if (seg) rateEl.textContent = Math.round(seg[1]).toLocaleString('en-US') + ' sim/s now';
}
function at(l) { return l.stage === 'forecast' ? FC * l.t : FC + (AD - FC) * l.t; }
function done() {
  if (document.querySelector('[data-testid="fuzz-log-done"]')) return;
  const m = document.createElement('div'); m.setAttribute('data-testid', 'fuzz-log-done');
  m.style.display = 'none'; m.textContent = 'fuzz-log-done'; root.appendChild(m);
}
function counter(ms) {
  if (!R) return;
  const t = FC ? Math.min(1, ms / FC) : 1;
  const g = Math.max(0, Math.min(1, (t - G0) / GS));
  const r = t >= G0 + GS ? R : Math.round(Math.expm1(g * Math.log1p(R)));
  rnd.textContent = r.toLocaleString('en-US'); headerAt(r);
}
if (root.dataset.state === 'play') {
  const t0 = performance.now();
  let i = 0, busyUntil = 0;
  const cur = document.createElement('span'); cur.className = 'cursor';
  function next() {
    if (i >= lines.length) { body.appendChild(cur); done(); return; }
    const l = lines[i], start = Math.max(at(l), busyUntil);
    const wait = start - (performance.now() - t0);
    setTimeout(() => {
      const nxt = i + 1 < lines.length ? at(lines[i + 1]) : AD;
      const room = Math.max(0, nxt - (performance.now() - t0));
      const step = Math.min(CH, room / Math.max(1, l.text.length));
      const div = document.createElement('div'); div.className = 'ln ' + l.kind;
      const tn = document.createTextNode(''); div.appendChild(tn); div.appendChild(cur);
      body.appendChild(div);
      let k = 0;
      function type() {
        k = step < 1 ? l.text.length : k + 1;
        tn.textContent = l.text.slice(0, k); body.scrollTop = body.scrollHeight;
        if (k < l.text.length) setTimeout(type, step);
        else { busyUntil = performance.now() - t0; i++; next(); }
      }
      type();
    }, Math.max(0, wait));
  }
  (function tick() { const ms = performance.now() - t0; counter(ms);
    if (ms < FC + 50) requestAnimationFrame(tick); })();
  next();
}
"""



def html_doc(mode: str, lines: list[dict], rounds: int, forecast_s: float = 0.0,
             adjust_s: float = 0.0,
             series: dict | None = None) -> str:
    """Iframe document. mode: idle | play (types itself) | final (all lines, done marker)."""
    pal = ui_config.COLOR
    if mode == "final":
        body = "".join(f'<div class="ln {l["kind"]}">{_esc(l["text"])}</div>' for l in lines)
        body += '<span class="cursor"></span>'
        shown_round = f"{rounds:,}"
    elif mode == "play":
        body, shown_round = "", "0"
    else:
        body = '<div class="idle">Press Run Forecast to fuzz today\'s plan.</div>'
        shown_round = "--"
    series = series or {"pool": [], "pool_step": 1, "rate": []}
    # Header: the measured speed of the current guided segment ("now"); the [perf] line has the
    # whole-run average. After playback the header keeps the last segment's speed.
    last = series["rate"][-1][1] if series["rate"] else None
    speed = f"{last:,.0f} sim/s now" if last and mode == "final" else ""
    pool = series["pool"][-1] if mode == "final" and series["pool"] else "--"
    marker = ('<div data-testid="fuzz-log-done" style="display:none">fuzz-log-done</div>'
              if mode == "final" else "")
    css = _CSS % {**pal, "h": HEIGHT_PX - 4}
    js = _JS % {"lines": theme.script_json(lines), "rounds": rounds,
                "fc_ms": int(round(forecast_s * 1000)), "ad_ms": int(round(adjust_s * 1000)),
                "char_ms": CHAR_MS, "g0": _GUIDED_FROM, "gs": _GUIDED_SPAN,
                "series": json.dumps(series)}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{theme.root_css()}{css}</style></head>
<body><div id="sf-fuzz-log" data-testid="fuzz-log-figure" data-state="{mode}">
<div class="head"><span>FUZZ LOG · round <b id="round">{shown_round}</b>
{f"/ {rounds:,}" if rounds else ""} · seed pool <b id="pool">{pool}</b></span>
<span id="rate">{_esc(speed)}</span></div>
<div class="body">{body}</div>{marker}
</div><script>{js}</script></body></html>"""


# ---------------------------------------------------------------- Streamlit

def show(static: dict, res: dict | None, shown: int, frames: list) -> None:
    """Draw the log for app.py's playback frame `shown` (once per script run)."""
    playing = res is not None and 0 < shown < len(frames)
    mode = "idle" if res is None or shown == 0 else ("play" if playing else "final")
    lines = log_lines(res, static["start_min"]) if res else []
    fc = (res or {}).get("forecast", {})
    rounds = (fc.get("guided_track") or {}).get("rounds", 0)
    f_s, a_s = stage_times(frames) if mode == "play" else (0.0, 0.0)
    doc = html_doc(mode, lines, rounds if res else 0, f_s, a_s,
                   header_series(fc.get("guided_track") or {}) if res else None)
    st.html('<div data-testid="fuzz-log" style="display:none">fuzz-log</div>')
    live_frame.show(doc, HEIGHT_PX, key="fuzz_log")
