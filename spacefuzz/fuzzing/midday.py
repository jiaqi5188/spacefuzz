"""Midday check (plan 1.2, manual 4.3): re-measure one value, re-forecast the afternoon.

At the midday check (14:00 in the demo, during the rest between the two EVA parts) the crew
re-measures the indicator the morning measurement prompt named first. The day is simulated
again from the morning check-in with the corrected value, so the morning EVA's heat and
fatigue are kept, but only violations from the check time on count (evaluator count_from):
the question is "is the rest of today still safe?". What already happened before the check
is reported separately (e.g. the morning EVA broke rule 1 with the corrected sleep).

The adjustment may only change tasks that start after the check; adjuster.py is reused
unchanged: the morning EVA is masked while candidates are built (its edits target the first
EVA of the plan), every candidate that would change a morning task is dropped, and two
afternoon options are added: cancel the afternoon EVA, or start it once the rule-1 window
(12 h after waking) has passed. The same selection rule applies: smallest change at or below
5% (confirmed with the full sample count), otherwise the lowest risk, reported to the ground.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

from spacefuzz.fuzzing import adjuster
from spacefuzz import config
from spacefuzz.fuzzing import engine as en
from spacefuzz.core import evaluator as ev
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import plan as pl
from spacefuzz.core import state as st
from spacefuzz.core.plan_types import PlanType

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = en.CACHE_DIR
MASK = PlanType.GENERIC_HIGH_LOAD        # stands in for a morning EVA while candidates are built


def remeasured_checkin(checkin: st.CheckIn, values: dict) -> st.CheckIn:
    """The morning check-in with the re-measured values (last night also in the 3 nights)."""
    c = dataclasses.replace(checkin, **values)
    if "sleep_last_night_h" in values:
        nights = list(checkin.sleep_last_3_nights_h)
        nights[-1] = float(values["sleep_last_night_h"])
        c = dataclasses.replace(c, sleep_last_3_nights_h=tuple(nights))
    return c


def morning_record(plan: pl.Plan, trace, checkin: st.CheckIn, baseline: st.Baseline,
                   cutoff: int) -> list[dict]:
    """Violations of the plan as flown before the check, with the corrected value: these
    already happened and are recorded after the fact, not forecast."""
    out = md.simulate(plan, pl.apply_env_events(trace, plan), checkin, baseline)
    e = ev.evaluate(out, plan, checkin, baseline)
    return [{"rule": v.rule, "system": v.system, "minute": v.minute, "clock": pl.clock(v.minute),
             "text": v.detail} for v in e.violations if v.minute < cutoff]


def _key(t) -> tuple:
    return (t.start_min, t.duration_min, t.type)


class MorningMissing(RuntimeError):
    """The morning result the midday check starts from is not in the cache."""


def afternoon_candidates(plan: pl.Plan, checkin: st.CheckIn, cutoff: int) -> list:
    """Candidates that leave every task starting before the check untouched."""
    morning = {_key(t) for t in plan.tasks if t.start_min < cutoff}
    masked_evas = {(t.start_min, t.duration_min) for t in plan.tasks
                   if t.type is PlanType.EVA and t.start_min < cutoff}
    masked = dataclasses.replace(plan, tasks=[
        dataclasses.replace(t, type=MASK) if (t.type is PlanType.EVA and t.start_min < cutoff)
        else t for t in plan.tasks])
    out = []
    for c in adjuster.candidates(masked):
        tasks = [dataclasses.replace(t, type=PlanType.EVA)
                 if t.type is MASK and (t.start_min, t.duration_min) in masked_evas else t
                 for t in c.plan.tasks]
        if {_key(t) for t in tasks if t.start_min < cutoff} != morning:
            continue                                      # would change the past
        if "postpone" in c.kinds:
            continue                                      # replaced by "cancel the afternoon EVA"
        out.append(dataclasses.replace(c, plan=dataclasses.replace(plan, tasks=tasks)))
    pm_evas = [t for t in plan.tasks if t.type is PlanType.EVA and t.start_min >= cutoff]
    if pm_evas:
        keep = [t for t in plan.tasks if t not in pm_evas]
        out.append(adjuster.Candidate(
            dataclasses.replace(plan, tasks=keep), ("cancel_pm",),
            (f"Cancel the afternoon EVA ({pl.clock(pm_evas[0].start_min)}); finish it on "
             f"another day",), adjuster._cost("postpone", None)))
        step = config.STEP_MINUTES               # rule-1 window end, rounded up to the grid
        window_end = -(-(checkin.wake_time_min + int(config.RULE1_WINDOW_H * 60)) // step) * step
        first = pm_evas[0]
        if checkin.sleep_last_night_h < config.RULE1_MIN_SLEEP_H and first.start_min < window_end:
            moved = dataclasses.replace(first, start_min=window_end)
            try:                                 # sleep is protected: may not fit tonight
                tasks = adjuster.resolve([moved if t is first else t for t in plan.tasks],
                                         moved)
            except adjuster.SleepConflict:
                tasks = None
            if tasks is not None and {_key(t) for t in tasks if t.start_min < cutoff} == morning:
                note = adjuster.personal_time_note(plan.tasks, tasks)
                out.append(adjuster.Candidate(
                    dataclasses.replace(plan, tasks=tasks), ("after_window",),
                    (f"Start the afternoon EVA at {pl.clock(window_end)}, "
                     f"{config.RULE1_WINDOW_H:.0f} h after waking (rule 1)",)
                    + ((note,) if note else ()),
                    adjuster._cost("delay", window_end - first.start_min)))
    return sorted(out, key=lambda c: (c.cost, c.changes))


def cached_morning(name: str, profile: str | None = None) -> dict | None:
    """The morning adjustment result if it is in the cache, else None: never starts a run
    (reading a measurement prompt must not trigger a full-profile forecast)."""
    read, _ = en.cache_files(*adjuster.cache_key(name, mu.Scenario.load(name), en.profile(profile)))
    return json.loads(read.read_text()) if read.exists() else None


def _day_plan(name: str, profile: str | None) -> pl.Plan:
    """The plan the crew is flying: the morning adjusted plan (or the original if the morning
    changed nothing). Read from the cache only: the morning was decided before the check,
    with the profile the dashboard used (full); a missing morning result is an error, never
    a silent fall-back or a new run."""
    a = cached_morning(name, profile) or cached_morning(name, "full")
    if a is None:
        raise MorningMissing(f"the morning forecast for {name} is not in the cache: run it on "
                             f"the dashboard first")
    pa = a.get("plan_adjustment") or {}
    adjusted = (a.get("details") or {}).get("adjusted_plan")
    if pa.get("changes") and not adjusted:
        raise MorningMissing(f"the morning adjusted plan for {name} could not be stored "
                             f"(a task starts after midnight)")
    return pl.load_plan(adjusted) if adjusted else mu.Scenario.load(name).plan


def cache_key(name: str, sc: mu.Scenario, prof: dict, remeasured: dict,
              cutoff: int, baseline: st.Baseline | None = None) -> tuple[str, str]:
    """(file-name stem, key) of the midday result (also used by scripts/prune_cache.py)."""
    key = hashlib.sha256((en._cache_key(sc, prof, baseline) + json.dumps(remeasured, sort_keys=True)
                          + str(cutoff)).encode()
                         + Path(__file__).read_bytes() + Path(adjuster.__file__).read_bytes())
    return f"{name}_{prof['name']}_midday_", key.hexdigest()


def afternoon_rating(p: float, time_to_crash: int | None, cutoff: int, checkin: st.CheckIn,
                     warnings: list[str]) -> tuple[str, int | None]:
    """Rating and time to crash counted from the check, not from the morning: red with a crash
    within 2 h of the check is CRITICAL. Out of experience is never green (as in the morning)."""
    ttc = None if time_to_crash is None else time_to_crash - (cutoff - en._start_min(checkin))
    return en.experience_floor(en.risk_rating(p, ttc), warnings), ttc


def scenario_for(name: str, remeasured: dict, profile_name: str | None = None) -> mu.Scenario:
    """The day being flown (morning adjusted plan) with the re-measured check-in."""
    base = mu.Scenario.load(name)
    return dataclasses.replace(base, plan=_day_plan(name, profile_name),
                               checkin=remeasured_checkin(base.checkin, remeasured))


def forecast_midday(name: str, remeasured: dict, cutoff: int, profile_name: str | None = None,
                    use_cache: bool = True, baseline: st.Baseline | None = None) -> dict:
    """Afternoon risk with the re-measured value, the morning record and the adjustment."""
    baseline = baseline or st.make_baseline()
    sc = scenario_for(name, remeasured, profile_name)
    plan, checkin = sc.plan, sc.checkin
    prof = en.profile(profile_name)
    read, path = en.cache_files(*cache_key(name, sc, prof, remeasured, cutoff, baseline))
    if use_cache and read.exists():
        return {**json.loads(read.read_text()), "from_cache": True}

    fl = en._wesad()
    track = lambda s, n: en.probability_track(s, baseline, n, fluctuation=fl, count_from=cutoff)
    before = track(sc, prof["probability_samples"])
    cands = afternoon_candidates(plan, checkin, cutoff)
    screens, confirmed = {}, {}

    def screen(c):
        if c not in screens:
            screens[c] = track(dataclasses.replace(sc, plan=c.plan),
                               prof["adjust_samples"])["risk_probability"]
        return screens[c]

    def confirm(c):
        if c not in confirmed:
            confirmed[c] = track(dataclasses.replace(sc, plan=c.plan), prof["probability_samples"])
        return confirmed[c]["risk_probability"]

    decision = None
    if before["risk_probability"] > config.RISK_RED_ABOVE:
        decision = adjuster.select(
            cands, screen, confirm,
            lambda c: (True, "schedule-slip re-test not run at the midday check"))
    chosen = decision.chosen if decision else None
    after = confirmed[chosen] if chosen is not None else before
    ooe = en.out_of_experience(sc, baseline)
    rating_before, ttc_before = afternoon_rating(before["risk_probability"],
                                                 before["time_to_crash_minutes"], cutoff, checkin, ooe)
    rating_after, _ = afternoon_rating(after["risk_probability"], after["time_to_crash_minutes"],
                                       cutoff, checkin, ooe)
    out = en._jsonable({
        "plan": name, "profile": prof["name"], "seed": config.RANDOM_SEED,
        "cutoff_min": cutoff, "cutoff": pl.clock(cutoff), "remeasured": remeasured,
        "morning_violations": morning_record(plan, sc.trace, checkin, baseline, cutoff),
        "risk_before": before["risk_probability"],
        "rating_before": rating_before,
        "time_to_crash_before": ttc_before,            # minutes after the check
        "by_rule_before": before["by_rule"],
        "changes": list(chosen.changes) if chosen else [],
        "risk_after": after["risk_probability"],
        "rating_after": rating_after,
        "by_rule_after": after["by_rule"],
        "escalate_to_ground": bool(decision.escalate_to_ground) if decision else False,
        "plan_before": pl.to_dict(plan),
        "plan_after": pl.to_dict(chosen.plan if chosen else plan),
        "candidates": len(cands), "selection": decision.log if decision else [],
    })
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return {**out, "from_cache": False}
