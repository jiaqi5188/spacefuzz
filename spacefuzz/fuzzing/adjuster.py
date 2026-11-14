"""Plan adjustment and re-test (plan 3.9), and the plan 3.10 output.

Sleep is protected: an edit that pushes later tasks back first shortens or cancels personal
time; if the pushed tasks would still reach a planned sleep, the candidate is not valid (an
adjustment must not create the sleep debt it is meant to fix). The fuzzer's perturbations
(mutator.py) still squeeze sleep on purpose: that is one of the what-ifs they test.

1. Candidates: move the EVA later, start it after an environment event it overlaps
   (e.g. an SPE warning: the EVA starts when the event ends; timing rules and sleep are
   then judged by the engine as usual), shorten it, split it with a rest, insert a rest or
   a nap before it, move training to after it, postpone it to another day; for days without
   an EVA, a rest before docking / precision work. Up to ADJ_MAX_COMBINED of these are
   combined (e.g. split + nap). Each has a "size of change" (config.ADJ_COST); postponing
   is by far the largest, so it is chosen only when nothing smaller works. Task types on
   hold (advice.holds, e.g. EVA pending a blood pressure recheck) leave only candidates
   without them.
2. Every candidate is screened with the probability track (ADJ samples, the same random
   numbers for all, so differences come from the plan). Candidates at green or yellow
   (risk <= 5%) are tried smallest change first: confirmed with the full probability
   track, then re-tested with the guided track. If none works, the lowest-risk candidate
   is kept, the plan stays red and is flagged for the ground.
   Three-change candidates come from a beam search of width ADJ_SAME_DAY_TOP: the best
   two-change candidates (by screened risk) extended by one more change; not every
   three-change combination is tried. Two options are always given when the plan has an
   EVA: if a same-day plan is chosen, postponing is added as the safer option; if
   postponing is chosen, the best same-day plan is added ("if the EVA must happen today",
   with ground reporting when still red). The crew and flight surgeon decide. The risk
   after postponing is today's risk only: the EVA is forecast again at tomorrow's
   morning check.
3. Guided re-test ("patch, then test again"): the plan must survive a normal schedule
   slip of ADJ_SLIP_MARGIN_MIN (training later, EVA overrun, EVA moved into an SPE
   warning, more high-CO2 work): the smallest crashing schedule change must be larger.
   The guided track's crash types on the chosen plan are reported alongside.

Only the original plan's rating sets the escalation level (red: adjust first, then report
to the flight surgeon, plan 1.3); plan_adjustment.escalate_to_ground says the adjustment
could not bring the plan out of red. No network is used: same inputs, same result.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from spacefuzz.fuzzing import advice
from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import EXERCISE_TYPES, PRECISION_TYPES, PlanType

log = logging.getLogger("adjuster")

CACHE_DIR = en.CACHE_DIR
STEP = config.STEP_MINUTES
ORDER = ("move_training", "shorten", "delay", "after_event", "rest", "nap", "split", "postpone")
MOVES_EVA = {"delay", "after_event"}           # never combined with each other


# ---------------------------------------------------------------- plan edits

_dur = en._fmt_minutes                 # "1 h 30 min"


class SleepConflict(ValueError):
    """Pushed tasks would reach a planned sleep: the adjustment is not valid."""


SQUEEZABLE = frozenset({PlanType.PERSONAL_TIME})       # gives way first when tasks are pushed


def resolve(tasks, anchor=None, protect_sleep: bool = True) -> list:
    """Sort tasks (anchor first on equal start) and push overlapping later tasks back in
    turn. Personal time keeps its end time and gets shorter (dropped if under one step).
    A planned sleep is never moved or shortened: reaching it raises SleepConflict.
    protect_sleep=False (descriptions only) squeezes sleep like personal time. Personal time
    that the anchor lands in is cut at the anchor (it gives way to the moved task)."""
    if anchor is not None:
        kept = []
        for t in tasks:
            if t is not anchor and t.type in SQUEEZABLE and \
                    t.start_min < anchor.start_min < t.end_min:
                if anchor.start_min - t.start_min >= STEP:
                    kept.append(dataclasses.replace(t, duration_min=anchor.start_min - t.start_min))
                continue
            kept.append(t)
        tasks = kept
    ts = sorted(tasks, key=lambda t: (t.start_min, 0 if t is anchor else 1))
    out, end = [], None
    for t in ts:
        if end is not None and t.start_min < end and t is not anchor:
            if t.type is PlanType.SLEEP and protect_sleep:
                raise SleepConflict(f"{pl.clock(t.start_min)} sleep would be pushed to "
                                    f"{pl.clock(end)}")
            if t.type in SQUEEZABLE or t.type is PlanType.SLEEP:
                if t.end_min - end < STEP:
                    continue
                t = dataclasses.replace(t, start_min=end, duration_min=t.end_min - end)
            else:
                t = dataclasses.replace(t, start_min=end)
        elif end is not None and t.start_min < end:
            t = dataclasses.replace(t, start_min=end)
        out.append(t)
        end = t.end_min if end is None else max(end, t.end_min)
    return out


def personal_time_note(before, after) -> str | None:
    """How an edit squeezed personal time (named in the changes), or None."""
    total = lambda ts: sum(t.duration_min for t in ts if t.type in SQUEEZABLE)   # noqa: E731
    a, b = total(before), total(after)
    if b >= a:
        return None
    return "Cancel personal time" if b == 0 else f"Shorten personal time to {_dur(b)}"


def _task(start, minutes, typ: PlanType) -> pl.Task:
    return pl.Task(start, minutes, typ, typ.value)


def _evas(tasks):
    return [t for t in tasks if t.type is PlanType.EVA]


def _apply(kind: str, value, tasks: list, protect: bool) -> tuple[list, str]:
    """Apply one adjustment; returns (tasks, English description)."""
    evas = _evas(tasks)
    eva = evas[0] if evas else None
    if kind == "delay":
        new = dataclasses.replace(eva, start_min=eva.start_min + value)
        return _res(protect, [new if t is eva else t for t in tasks], new), \
            f"Move the EVA {_dur(value)} later"
    if kind == "after_event":           # value: (event end minute, event type, minutes moved)
        end, name, _ = value
        new = dataclasses.replace(eva, start_min=max(eva.start_min, end))
        return _res(protect, [new if t is eva else t for t in tasks], new), \
            f"Start the EVA after the {name} ends ({pl.clock(end)})"
    if kind == "shorten":
        new = dataclasses.replace(eva, duration_min=max(config.ADJ_MIN_EVA_MIN,
                                                        eva.duration_min - int(value * 60)))
        return _res(protect, [new if t is eva else t for t in tasks]), \
            f"Shorten the EVA by {_dur(int(value * 60))}"
    if kind in ("rest", "nap") and eva is not None:
        new = _task(eva.start_min, value, PlanType.REST if kind == "rest" else PlanType.SLEEP)
        word = "rest" if kind == "rest" else "nap"
        return _res(protect, tasks + [new], new), f"Insert a {_dur(value)} {word} before the EVA"
    if kind == "rest":                                  # no EVA: rest before precision work
        precise = [t for t in tasks if t.type in PRECISION_TYPES]
        out = list(tasks)
        for p in precise:
            new = _task(p.start_min, value, PlanType.REST)
            out = _res(protect, out + [new], new)
        names = " / ".join(sorted({p.source_type for p in precise}))
        return out, f"Insert a {_dur(value)} rest before {names}"
    if kind == "split":
        first = (eva.duration_min // 2) // STEP * STEP
        rest = config.ADJ_SPLIT_REST_MIN
        a = dataclasses.replace(eva, duration_min=first)
        r = _task(eva.start_min + first, rest, PlanType.REST)
        b = dataclasses.replace(eva, start_min=eva.start_min + first + rest,
                                duration_min=eva.duration_min - first)
        others = [t for t in tasks if t is not eva]
        return _res(protect, others + [a, r, b], a), \
            f"Split the EVA into two parts with a {_dur(rest)} rest between"
    if kind == "move_training":
        train = [t for t in tasks if t.type in EXERCISE_TYPES and t.end_min <= eva.start_min]
        out = [t for t in tasks if t not in train]
        names = []
        for t in train:
            end = max(e.end_min for e in _evas(out))
            new = dataclasses.replace(t, start_min=end)
            out = _res(protect, out + [new], new)
            names.append(t.source_type)
        return out, f"Move {' and '.join(names)} to after the EVA"
    if kind == "postpone":
        held = value or {PlanType.EVA}
        names = sorted({t.source_type for t in tasks if t.type in held})
        return [t for t in tasks if t.type not in held], \
            f"Postpone {' and '.join(names)} to another day"
    raise ValueError(kind)


def _res(protect: bool, tasks, anchor=None) -> list:
    return resolve(tasks, anchor, protect_sleep=protect)


def _edit(kind: str, value, tasks: list) -> tuple[list | None, str]:
    """Apply one adjustment; returns (tasks, English description). tasks is None when the
    edit would push a planned sleep (not a valid plan); the description is still given."""
    try:
        return _apply(kind, value, tasks, protect=True)
    except SleepConflict:
        return None, _apply(kind, value, tasks, protect=False)[1]


def _cost(kind: str, value) -> float:
    c = config.ADJ_COST
    return {"delay": lambda: c["delay_per_h"] * value / 60,
            "after_event": lambda: c["delay_per_h"] * value[2] / 60,
            "shorten": lambda: c["shorten_per_h"] * value,
            "rest": lambda: c["rest_per_h"] * value / 60,
            "nap": lambda: c["nap_per_h"] * value / 60,
            "split": lambda: c["split"], "move_training": lambda: c["move_training"],
            "postpone": lambda: c["postpone"]}[kind]()


@dataclass(frozen=True, eq=False)       # identity: used as a dict key
class Candidate:
    plan: pl.Plan
    kinds: tuple
    changes: tuple            # English descriptions, in the order applied
    cost: float


def _valid(tasks) -> bool:
    ts = sorted(tasks, key=lambda t: t.start_min)
    return bool(ts) and all(t.duration_min > 0 for t in ts) and \
        all(a.end_min <= b.start_min for a, b in zip(ts, ts[1:]))


def _primitives(plan: pl.Plan) -> list[tuple[str, object]]:
    tasks = plan.tasks
    evas = _evas(tasks)
    if not evas:
        return [("rest", r) for r in config.ADJ_REST_BEFORE_MIN] \
            if any(t.type in PRECISION_TYPES for t in tasks) else []
    eva = evas[0]
    prims = [("delay", d) for d in config.ADJ_DELAY_MIN]
    for ev_ in plan.events:                      # an env event overlapping the EVA: start after it
        if ev_.start_min < eva.end_min and eva.start_min < ev_.end_min:
            prims.append(("after_event", (ev_.end_min, ev_.type.value, ev_.end_min - eva.start_min)))
    prims += [("shorten", h) for h in config.ADJ_SHORTEN_H
              if eva.duration_min - h * 60 >= config.ADJ_MIN_EVA_MIN]
    prims += [("split", None)]
    prims += [("rest", r) for r in config.ADJ_REST_BEFORE_MIN]
    prims += [("nap", n) for n in config.ADJ_NAP_BEFORE_MIN if n / 60 <= config.RULE1_NAP_MAX_H]
    if any(t.type in EXERCISE_TYPES and t.end_min <= eva.start_min for t in tasks):
        prims += [("move_training", None)]
    return prims


def _build(plan: pl.Plan, combo) -> Candidate | None:
    tasks, texts = list(plan.tasks), []
    for kind, value in sorted(combo, key=lambda kv: ORDER.index(kv[0])):
        tasks, text = _edit(kind, value, tasks)
        if tasks is None:                    # would push a planned sleep: not valid
            return None
        texts.append(text)
    if not _valid(tasks):
        return None
    note = personal_time_note(plan.tasks, tasks)
    if note:
        texts.append(note)
    evas = _evas(tasks)
    if evas and "postpone" not in [k for k, _ in combo]:
        texts.append("EVA now " + ", ".join(f"{pl.clock(e.start_min)}-{pl.clock(e.end_min)}" for e in evas))
    return Candidate(dataclasses.replace(plan, tasks=tasks), tuple(k for k, _ in combo),
                     tuple(texts), round(sum(_cost(k, v) for k, v in combo), 3))


def candidates(plan: pl.Plan, held=frozenset()) -> list[Candidate]:
    """All candidate plans, smallest change first."""
    held = set(held)
    if held & {t.type for t in plan.tasks}:
        c = _build(plan, [("postpone", frozenset(held))])
        return [c] if c else []
    prims = _primitives(plan)
    combos = [[p] for p in prims]
    for a, b in itertools.combinations(prims, 2):
        if a[0] != b[0] and {a[0], b[0]} != {"rest", "nap"} and not {a[0], b[0]} <= MOVES_EVA \
                and config.ADJ_MAX_COMBINED >= 2:
            combos.append([a, b])
    if _evas(plan.tasks):
        combos.append([("postpone", frozenset({PlanType.EVA}))])
    out = [c for c in (_build(plan, combo) for combo in combos) if c is not None]
    return sorted(out, key=lambda c: (c.cost, c.changes))


# ---------------------------------------------------------------- selection

@dataclass
class Decision:
    chosen: Candidate | None
    risk_after: float | None
    escalate_to_ground: bool
    log: list = field(default_factory=list)
    recheck: str = ""


def select(cands, screen, confirm, recheck) -> Decision:
    """Plan 3.9 step 2-3. screen / confirm: candidate -> risk; recheck: candidate -> (ok, why)."""
    if not cands:
        return Decision(None, None, True, [{"result": "no candidate"}])
    screened = [(c, screen(c)) for c in cands]
    log_, failed = [], set()
    for c, r in sorted(((c, r) for c, r in screened if r <= config.RISK_RED_ABOVE),
                       key=lambda cr: cr[0].cost):
        r2 = confirm(c)
        if r2 > config.RISK_RED_ABOVE:
            log_.append({"changes": list(c.changes), "screen": r, "confirm": r2,
                         "result": "confirmed red"})
            continue
        ok, why = recheck(c)
        if not ok:
            log_.append({"changes": list(c.changes), "screen": r, "confirm": r2,
                         "result": "re-test failed", "why": why})
            failed.add(c)
            continue
        log_.append({"changes": list(c.changes), "screen": r, "confirm": r2, "result": "chosen"})
        return Decision(c, r2, False, log_, why)
    # Nothing passed: report the lowest-risk plan to the ground. A plan that failed its
    # re-test is only used when nothing else is left, and is then still sent to the ground.
    pool = [(c, r) for c, r in screened if c not in failed] or screened
    best, r = min(pool, key=lambda cr: (cr[1], cr[0].cost))
    r2 = confirm(best)
    red, why = r2 > config.RISK_RED_ABOVE, ""
    retest_failed = best in failed
    if not red and not retest_failed:      # screened red but confirmed not red: every plan
        ok, why = recheck(best)            # reported as fixed has passed the re-test
        retest_failed = not ok
    log_.append({"changes": list(best.changes), "screen": r, "confirm": r2,
                 "result": "lowest risk, still red" if red
                 else "re-test failed, report to ground" if retest_failed else "chosen",
                 **({"why": why} if why and retest_failed else {})})
    return Decision(best, r2, red or retest_failed, log_, "" if retest_failed else why)


def needs_same_day_alternative(kinds, held) -> bool:
    """Postponing was chosen and the EVA is not on hold: also offer the best same-day plan."""
    return "postpone" in kinds and PlanType.EVA not in set(held)


def candidates_with_beam(plan: pl.Plan, screen, held=frozenset(),
                         top: int | None = None) -> list[Candidate]:
    """candidates() plus three-change candidates: a beam search of width `top` that
    extends the best two-change candidates (by screened risk) with one more change.
    Not every three-change combination is tried."""
    pool = candidates(plan, held)
    if set(held) & {t.type for t in plan.tasks}:
        return pool
    scored = [(c, screen(c)) for c in pool if "postpone" not in c.kinds]
    extra = [c for c, _ in _beam(plan, scored, screen, top)]
    return sorted(pool + extra, key=lambda c: (c.cost, c.changes))


def _beam(plan: pl.Plan, scored, screen, top: int | None = None) -> list:
    top = config.ADJ_SAME_DAY_TOP if top is None else top
    out = []
    prims = _primitives(plan)
    pairs = sorted((cr for cr in scored if len(cr[0].kinds) == 2), key=lambda cr: (cr[1], cr[0].cost))
    seen = {tuple(sorted(c.kinds)) + c.changes for c, _ in scored}
    for c, _ in pairs[:top]:
        combo = [p for p in prims if p[0] in c.kinds and _primitive_text(plan, p) in c.changes]
        for p in prims:
            kinds = {k for k, _ in combo} | {p[0]}
            if p[0] in {k for k, _ in combo} or {"rest", "nap"} <= kinds or MOVES_EVA <= kinds:
                continue
            c3 = _build(plan, combo + [p])
            if c3 is None or tuple(sorted(c3.kinds)) + c3.changes in seen:
                continue
            seen.add(tuple(sorted(c3.kinds)) + c3.changes)
            out.append((c3, screen(c3)))
    return out


def _primitive_text(plan: pl.Plan, prim) -> str:
    return _edit(prim[0], prim[1], list(plan.tasks))[1]


def slip_margin(scenario: mu.Scenario, baseline: st.Baseline) -> dict | None:
    """Smallest schedule slip that makes the plan crash (no "less sleep")."""
    return en.min_crash_perturbation(scenario, baseline, schedule_only=True)


def measurement_sensitivity(scenario: mu.Scenario, baseline: st.Baseline, n: int,
                            fluctuation: md.Fluctuation | None = None) -> list[tuple[str, float]]:
    """Risk change when each check-in entry moves by its measurement error (plan 3.11),
    largest first; the same random numbers for every run."""
    base = en.probability_track(scenario, baseline, n, fluctuation=fluctuation)["risk_probability"]
    out = []
    for name, err in config.MEASUREMENT_ERROR.items():
        c0 = scenario.checkin
        if getattr(c0, name) is None:
            continue
        deltas = []
        for sign in (-1, 1):
            v = max(0.0, getattr(c0, name) + sign * err)
            c = dataclasses.replace(c0, **{name: v})
            if name == "sleep_last_night_h":
                c = mu._with_sleep(c0, baseline, tuple(c0.sleep_last_3_nights_h[:-1]) + (v,))
            r = en.probability_track(dataclasses.replace(scenario, checkin=c), baseline, n,
                                     fluctuation=fluctuation)["risk_probability"]
            deltas.append(abs(r - base))
        out.append((name, round(max(deltas), 4)))
    return sorted(out, key=lambda x: -x[1])


# ---------------------------------------------------------------- output (plan 3.10)

def _hud(prob: dict, plan: pl.Plan, baseline: st.Baseline) -> list[str]:
    out = []
    ttc = prob.get("time_to_crash_minutes")
    if ttc is not None:
        out.append(f"Time to risk {ttc // 60:02d}:{ttc % 60:02d}:00")
    if _evas(plan.tasks):
        thr = config.CARDIO_HR_MAX_FRACTION * st.max_heart_rate(baseline)
        out.append(f"Return to the airlock if heart rate stays above {thr:.0f} bpm")
    if any(e.type is PlanType.SPE_WARNING for e in plan.events):
        out.append("SPE warning: stay in the shielded area")
    return out


def _concurrency(plan: pl.Plan) -> list[str]:
    if any(t.type in PRECISION_TYPES for t in plan.tasks):
        return ["No long voice reports during manual docking or precision work"]
    return []


def cache_key(name: str, scenario: mu.Scenario, prof: dict,
              baseline: st.Baseline | None = None) -> tuple[str, str]:
    """(file-name stem, key) of adjust()'s result (also used by midday.py and prune_cache.py)."""
    key = hashlib.sha256((en._cache_key(scenario, prof, baseline) + "adjust").encode()
                         + Path(__file__).read_bytes()).hexdigest()
    return f"{name}_{prof['name']}_adjust_", key


def adjust(plan_path, scenario: mu.Scenario | None = None, profile_name: str | None = None,
           use_cache: bool = True, baseline: st.Baseline | None = None) -> dict:
    """Forecast, adjust and re-test one plan; returns the plan 3.10 JSON (plus details)."""
    name = Path(plan_path).stem
    scenario = scenario or mu.Scenario.load(name)
    baseline = baseline or st.make_baseline()
    prof = en.profile(profile_name)
    read, path = en.cache_files(*cache_key(name, scenario, prof, baseline))
    if use_cache and read.exists():
        return {**json.loads(read.read_text()), "from_cache": True}

    f = en.forecast(plan_path, scenario=scenario, profile_name=profile_name,
                    use_cache=use_cache, baseline=baseline)
    common = {"plan": name, "data_quality": f["data_quality"], "advice_now": f["advice_now"],
              "action_restrictions": f["action_restrictions"],
              "coverage_warnings": f.get("coverage_warnings") or
              en.out_of_experience(scenario, baseline)}
    if f["risk_rating"] == "UNKNOWN":
        return {**common, "risk_rating": "UNKNOWN", "escalation": "self", "risk_probability": None,
                "time_to_crash_minutes": None, "primary_vulnerable_system": "None",
                "causal_chain": [], "min_crash_perturbation": None, "advice_today": [],
                "measurement_prompts": [], "plan_adjustment": None, "concurrency_control": [],
                "hud_directives": [], "details": {"message": f.get("message")},
                "from_cache": False}

    fl = en._wesad()
    prob = f["probability_track"]
    rating = f["risk_rating"]
    held = {PlanType(t) for t in f["holds"]}
    decision, cand_table, retest = None, [], None
    final_plan, final_prob = scenario.plan, prob
    if rating in ("HIGH", "CRITICAL") or held:
        screens: dict = {}

        def scen(c):
            return dataclasses.replace(scenario, plan=c.plan)

        def screen(c):
            if c not in screens:
                screens[c] = en.probability_track(scen(c), baseline, prof["adjust_samples"],
                                                  fluctuation=fl)["risk_probability"]
            return screens[c]

        confirmed: dict = {}

        def confirm(c):
            if c not in confirmed:
                confirmed[c] = en.probability_track(scen(c), baseline, prof["probability_samples"],
                                                    fluctuation=fl)
            return confirmed[c]["risk_probability"]

        def recheck(c):
            m = slip_margin(scen(c), baseline)
            ok = m is None or m["minutes"] > config.ADJ_SLIP_MARGIN_MIN
            return ok, ("no schedule slip within 3 h crashes" if m is None else m["text"])

        cands = candidates_with_beam(scenario.plan, screen, held)
        decision = select(cands, screen, confirm, recheck)
        cand_table = [{"changes": list(c.changes), "cost": c.cost, "screen_risk": r}
                      for c, r in sorted(screens.items(), key=lambda cr: cr[0].cost)]
        if decision.chosen is not None:
            final_plan = decision.chosen.plan
            final_prob = confirmed[decision.chosen]
            g = en.guided_track(scen(decision.chosen), baseline, prof["recheck_rounds"],
                                fluctuation=fl)
            retest = {"slip_margin": decision.recheck or "not run: no candidate left the red zone",
                      "guided_crash_types": [f"{c['system']}/{c['rule']}" for c in g["crash_types"]],
                      "rounds": g["rounds"]}

    alternative = None
    if decision and decision.chosen and needs_same_day_alternative(decision.chosen.kinds, held):
        alt = min((c for c in screens if "postpone" not in c.kinds),
                  key=lambda c: (screens[c], c.cost))
        r_alt = confirm(alt)
        m = slip_margin(scen(alt), baseline)
        alternative = {
            "changes": list(alt.changes), "risk": r_alt,
            "rating": en.risk_rating(r_alt, confirmed[alt]["time_to_crash_minutes"]),
            "escalate_to_ground": r_alt > config.RISK_RED_ABOVE,
            "slip_margin": "no schedule slip within 3 h crashes" if m is None else m["text"],
            "note": ("If the EVA must happen today, use this plan"
                     + (" and report to the ground flight surgeon." if r_alt > config.RISK_RED_ABOVE
                        else ".")),
        }
    risk_after = final_prob["risk_probability"]
    rating_after = en.experience_floor(
        en.risk_rating(risk_after, final_prob["time_to_crash_minutes"]),
        en.out_of_experience(dataclasses.replace(scenario, plan=final_plan), baseline))
    today = advice.prevent_today(final_prob["by_rule"], rating_after, scenario.checkin, baseline)
    sens = measurement_sensitivity(scenario, baseline, prof["adjust_samples"], fluctuation=fl)
    minp = f["min_crash_perturbation"]
    plan_adjustment = {
        "risk_before": prob["risk_probability"], "risk_after": risk_after,
        "rating_after": rating_after,
        "time_to_crash_after": final_prob["time_to_crash_minutes"],
        "changes": list(decision.chosen.changes) if decision and decision.chosen else [],
        "escalate_to_ground": bool(decision.escalate_to_ground) if decision else False,
        "retest": retest,
    }
    scope = ("Risk for today only, with the EVA not done today. The postponed EVA is forecast "
             "again at tomorrow's morning check (it depends on how well the crew sleeps tonight).")
    if decision and decision.chosen and "postpone" in decision.chosen.kinds:
        plan_adjustment["scope"] = scope
        plan_adjustment["same_day_alternative"] = alternative
    elif decision and decision.chosen and _evas(scenario.plan.tasks):
        post = next((c for c in cands if c.kinds == ("postpone",)), None)
        if post is not None:
            r_post = confirm(post)
            plan_adjustment["safer_option"] = {
                "changes": list(post.changes), "risk": r_post,
                "rating": en.risk_rating(r_post, confirmed[post]["time_to_crash_minutes"]),
                "scope": scope,
                "note": "Safer option: postpone the EVA; the crew and flight surgeon decide."}
    out = {
        **common, "risk_rating": rating, "escalation": f["escalation"],
        "risk_probability": prob["risk_probability"],
        "time_to_crash_minutes": prob["time_to_crash_minutes"],
        "primary_vulnerable_system": prob["primary_vulnerable_system"],
        "causal_chain": prob["causal_chain"],
        "min_crash_perturbation": minp["text"] if minp else None,
        "advice_today": advice.advice_today(today),
        "measurement_prompts": advice.measurement_prompts(sens),
        "plan_adjustment": plan_adjustment,
        "concurrency_control": _concurrency(final_plan),
        "hud_directives": _hud(final_prob, final_plan, baseline),
        "details": {"candidates": cand_table, "selection": decision.log if decision else [],
                    "risk_after_by_rule": final_prob["by_rule"],
                    "measurement_sensitivity": sens, "holds": f["holds"],
                    "adjusted_plan": pl.to_dict(final_plan) if all(
                        t.start_min < 24 * 60 for t in final_plan.tasks) else None,
                    "performance_before": f.get("performance"),
                    # stage 4 UI: trajectory fan / HUD for the plan the crew will fly
                    "traces_after": final_prob.get("traces"),
                    "trajectory_after": final_prob.get("trajectory"),
                    "trajectory_minutes_after": final_prob.get("trajectory_minutes")},
    }
    out = en._jsonable(out)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return {**out, "from_cache": False}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SpaceFuzz forecast + plan adjustment.")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--profile", choices=sorted(config.FUZZ_PROFILES), default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="flat environment, no raw data")
    ap.add_argument("--json", action="store_true", help="print the full plan 3.10 JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    name = Path(args.plan).stem
    r = adjust(args.plan, scenario=en._synthetic(name) if args.synthetic else None,
               profile_name=args.profile, use_cache=not args.no_cache)
    if args.json:
        print(json.dumps({k: v for k, v in r.items() if k != "details"}, indent=1))
        return
    print(f"plan {name} | data quality {r['data_quality']}")
    if r["risk_rating"] == "UNKNOWN":
        print("risk UNKNOWN: " + " ".join(r["advice_now"]))
        return
    pa = r["plan_adjustment"]
    print(f"before: risk {pa['risk_before']:.1%} -> {r['risk_rating']}")
    print(f"after:  risk {pa['risk_after']:.1%} -> {pa['rating_after']}"
          + ("  (still red: report to ground)" if pa["escalate_to_ground"] else ""))
    print("changes: " + ("; ".join(pa["changes"]) if pa["changes"] else "none"))
    if pa.get("scope"):
        print(f"scope: {pa['scope']}")
    safer = pa.get("safer_option")
    if safer:
        print(f"safer option: risk {safer['risk']:.1%} -> {safer['rating']}: "
              + "; ".join(safer["changes"]) + f". {safer['scope']}")
    alt = pa.get("same_day_alternative")
    if alt:
        print(f"same-day alternative: risk {alt['risk']:.1%} -> {alt['rating']}: "
              + "; ".join(alt["changes"]) + f". {alt['note']}")
    if pa.get("retest") and pa["escalate_to_ground"]:
        print("re-test not run: no candidate left the red zone; guided crash types: "
              + ", ".join(pa["retest"]["guided_crash_types"]))
    elif pa.get("retest"):
        print(f"re-test passed (a {config.ADJ_SLIP_MARGIN_MIN}-min slip is safe; smallest crashing "
              f"slip: {pa['retest']['slip_margin']}); guided crash types: "
              + ", ".join(pa["retest"]["guided_crash_types"]))
    print(f"escalation: {r['escalation']}")
    for key in ("advice_now", "advice_today", "measurement_prompts", "action_restrictions",
                "hud_directives"):
        for t in r[key]:
            print(f"  [{key}] {t}")
    if r.get("from_cache"):
        print("(cached result)")


if __name__ == "__main__":
    main()
