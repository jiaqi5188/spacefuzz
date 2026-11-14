"""Load and validate a daily plan (format: plan 3.8), and pair it with an environment trace.

Plan JSON: date, environment, check_window, comm_delay_min, tasks, optional events.
Every item carries `kind`: "task" (crew activity in `tasks`; the adjuster may move,
shorten or drop it) or "env" (environment condition in `events`; the adjuster must
never move or shorten it; move_item raises PlanError). Types come from the single
PlanType enum.

Validation:
- Unknown task types map to GENERIC_HIGH_LOAD with a logged warning (also kept in
  Plan.warnings). Unknown env types are errors: a hazard must not be guessed.
- Overlapping tasks, zero or negative durations, a task longer than 24 hours, or
  a plan spanning more than 24 hours from first start to last end raise PlanError.
  Tasks may run past midnight (e.g. sleep 23:00 + 8 h).

CO2 rule: a `high_co2` event gives `co2_ppm` (never a percentage). When a plan
event and the replayed env_trace both give CO2 for the same step, the larger
value is used: the event is a known equipment condition (e.g. running on the
backup scrubber) and acts as a floor on top of the replayed curve
(see apply_env_events).

One line for a day's environment + plan:
    trace, plan = load_scenario("data/plans/main_sleep_debt_eva.json")
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from spacefuzz import config
from spacefuzz.readers import env_trace
from spacefuzz.core.plan_types import PlanType, kind_of, parse_type

log = logging.getLogger("plan")
ROOT = Path(__file__).resolve().parents[2]
TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
REQUIRED = ("date", "environment", "check_window", "comm_delay_min", "tasks")
DAY_MIN = 24 * 60
# Which recorded radiation environment a plan's `environment` replays.
RADIATION_FOR_ENVIRONMENT = {"mars_base": "mars", "iss": "iss"}


class PlanError(ValueError):
    """The plan cannot be used as given."""


@dataclass(frozen=True)
class Task:
    start_min: int          # minutes since 00:00 of the plan day
    duration_min: int
    type: PlanType
    source_type: str        # type string as written in the file
    kind: str = "task"

    @property
    def end_min(self) -> int:
        return self.start_min + self.duration_min


@dataclass(frozen=True)
class EnvEvent:
    start_min: int
    end_min: int
    type: PlanType
    kind: str = "env"
    co2_ppm: int | None = None


@dataclass
class Plan:
    date: str
    environment: str
    check_window: str
    comm_delay_min: int
    tasks: list[Task]
    events: list[EnvEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _minutes(value, where: str) -> int:
    m = TIME.match(str(value))
    if not m:
        raise PlanError(f"{where}: time {value!r} is not HH:MM (00:00-23:59)")
    return int(m.group(1)) * 60 + int(m.group(2))


def _check_kind(item: dict, expected: str, where: str) -> None:
    if not isinstance(item, dict):
        raise PlanError(f"{where}: must be an object with start, type and kind, "
                        f"got {type(item).__name__}")
    if "kind" not in item:
        raise PlanError(f"{where}: missing 'kind' (must be {expected!r})")
    if item["kind"] != expected:
        raise PlanError(f"{where}: kind {item['kind']!r}, expected {expected!r}")


def _on_grid(minutes: int, what: str, where: str) -> None:
    """The model sees the day in config.STEP_MINUTES steps: a start, end or duration between
    two steps would be invisible to the rules (they look for task starts on a step)."""
    if minutes % config.STEP_MINUTES:
        raise PlanError(f"{where}: {what} must be a whole number of {config.STEP_MINUTES}-minute "
                        f"steps (on the 15-minute grid), got {minutes} min")


def _parse_task(i: int, raw: dict, warnings: list[str]) -> Task:
    where = f"tasks[{i}]"
    _check_kind(raw, "task", where)
    for key in ("start", "duration_h", "type"):
        if key not in raw:
            raise PlanError(f"{where}: missing {key!r}")
    start = _minutes(raw["start"], where)
    hours = raw["duration_h"]
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or hours <= 0:
        raise PlanError(f"{where}: duration must be positive, got {hours!r}")
    if not math.isfinite(hours):
        raise PlanError(f"{where}: duration must be a finite number of hours, got {hours!r}")
    if hours > 24:
        raise PlanError(f"{where}: duration {hours} h exceeds 24 hours")
    _on_grid(start, "start", where)
    if abs(hours * 60 - round(hours * 60)) > 1e-6:
        raise PlanError(f"{where}: duration must be whole minutes on the 15-minute grid, "
                        f"got {hours} h")
    _on_grid(round(hours * 60), "duration", where)
    typ = parse_type(str(raw["type"]), kind="task") if str(raw["type"]) not in \
        {t.value for t in PlanType if kind_of(t) == "env"} else None
    if typ is None:
        raise PlanError(f"{where}: {raw['type']!r} is an env type; put it in 'events'")
    if typ is PlanType.GENERIC_HIGH_LOAD and raw["type"] != PlanType.GENERIC_HIGH_LOAD.value:
        msg = f"{where}: unknown task type {raw['type']!r} mapped to generic_high_load"
        log.warning(msg)
        warnings.append(msg)
    return Task(start, round(hours * 60), typ, str(raw["type"]))


def _parse_event(i: int, raw: dict) -> EnvEvent:
    where = f"events[{i}]"
    _check_kind(raw, "env", where)
    try:
        typ = parse_type(str(raw.get("type")), kind="env")
    except ValueError as exc:
        raise PlanError(f"{where}: {exc}") from None
    start, end = _minutes(raw.get("start"), where), _minutes(raw.get("end"), where)
    if end <= start:
        raise PlanError(f"{where}: end {raw['end']} must be after start {raw['start']}")
    _on_grid(start, "start", where)
    _on_grid(end, "end", where)
    co2 = None
    if typ is PlanType.HIGH_CO2:
        if "co2_pct" in raw:
            raise PlanError(f"{where}: use co2_ppm, not co2_pct (1% = 10000 ppm)")
        co2 = raw.get("co2_ppm")
        if not isinstance(co2, (int, float)) or co2 <= 0 or not math.isfinite(co2):
            raise PlanError(f"{where}: high_co2 needs a positive co2_ppm")
    return EnvEvent(start, end, typ, co2_ppm=int(co2) if co2 is not None else None)


def load_plan(source) -> Plan:
    """Load a plan from a JSON path or an already parsed dict, validating it fully."""
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.is_absolute():
            path = ROOT / path
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise PlanError(f"plan is missing {missing}")
    if not isinstance(data["tasks"], list) or not data["tasks"]:
        raise PlanError("plan needs a non-empty 'tasks' list")

    warnings: list[str] = []
    tasks = sorted((_parse_task(i, t, warnings) for i, t in enumerate(data["tasks"])),
                   key=lambda t: t.start_min)
    for a, b in zip(tasks, tasks[1:]):
        if a.end_min > b.start_min:
            raise PlanError(f"tasks overlap: {a.source_type} until {a.end_min // 60:02d}:"
                            f"{a.end_min % 60:02d} and {b.source_type} from "
                            f"{b.start_min // 60:02d}:{b.start_min % 60:02d}")
    span = max(t.end_min for t in tasks) - tasks[0].start_min
    if span > DAY_MIN:
        raise PlanError(f"plan spans {span / 60:.1f} h, more than 24 hours")
    raw_events = data.get("events") or []
    if not isinstance(raw_events, list):
        raise PlanError(f"'events' must be a list, got {type(raw_events).__name__}")
    events = [_parse_event(i, e) for i, e in enumerate(raw_events)]
    try:
        delay = int(data["comm_delay_min"])
    except (TypeError, ValueError, OverflowError):
        raise PlanError(f"comm_delay_min must be a whole number of minutes, "
                        f"got {data['comm_delay_min']!r}") from None

    return Plan(date=str(data["date"]), environment=str(data["environment"]),
                check_window=str(data["check_window"]), comm_delay_min=delay,
                tasks=tasks, events=events, warnings=warnings)


def _hhmm(minutes: int) -> str:
    """Plan-file time: no wrap past midnight, so load_plan rejects a task after 24:00."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def clock(minutes: int) -> str:
    """Clock time for messages (wraps past midnight)."""
    return f"{int(minutes) // 60 % 24:02d}:{int(minutes) % 60:02d}"


def to_dict(plan: Plan) -> dict:
    """Plan back to the JSON structure of plan 3.8 (original task type strings kept)."""
    data = {"date": plan.date, "environment": plan.environment,
            "check_window": plan.check_window, "comm_delay_min": plan.comm_delay_min,
            "tasks": [{"start": _hhmm(t.start_min), "duration_h": t.duration_min / 60,
                       "type": t.source_type, "kind": "task"} for t in plan.tasks]}
    if plan.events:
        data["events"] = [{"start": _hhmm(e.start_min), "end": _hhmm(e.end_min),
                           "type": e.type.value, "kind": "env",
                           **({"co2_ppm": e.co2_ppm} if e.co2_ppm is not None else {})}
                          for e in plan.events]
    return data


def move_item(plan: Plan, item: Task | EnvEvent, delta_min: int) -> Plan:
    """A new, fully revalidated plan with `item` moved by delta_min.

    Env events are environment conditions outside crew control: moving one raises.
    """
    if isinstance(item, EnvEvent) or getattr(item, "kind", None) == "env":
        raise PlanError(f"{item.type.value} is an env event and cannot be moved or shortened")
    idx = next((i for i, t in enumerate(plan.tasks) if t is item), None)
    if idx is None:
        raise PlanError("task is not part of this plan")
    start = item.start_min + delta_min
    if not 0 <= start < DAY_MIN:
        raise PlanError(f"moved start {start} min is outside 00:00-23:59")
    data = to_dict(plan)
    data["tasks"][idx]["start"] = _hhmm(start)
    return load_plan(data)


def apply_env_events(trace: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    """Apply plan env events to a trace: inside a high_co2 event, co2 = max(trace, event)."""
    out = trace.copy()
    for ev in plan.events:
        if ev.type is PlanType.HIGH_CO2:
            mask = (out.index >= ev.start_min) & (out.index < ev.end_min)
            out.loc[mask, "co2"] = out.loc[mask, "co2"].clip(lower=ev.co2_ppm)
    return out


def load_scenario(plan_path, cabin_start=config.ENV_DEFAULT_CABIN_START,
                  radiation_start=config.ENV_DEFAULT_RADIATION_START):
    """(trace, plan) for one day: trace minute 0 = 00:00 of the plan day."""
    p = load_plan(plan_path)
    radiation = RADIATION_FOR_ENVIRONMENT.get(p.environment)
    if radiation is None:
        raise PlanError(f"unknown environment {p.environment!r}; "
                        f"known: {sorted(RADIATION_FOR_ENVIRONMENT)}")
    trace = env_trace.build_trace(cabin_start, radiation_start, hours=24, radiation=radiation)
    merged = apply_env_events(trace, p)
    merged.attrs = dict(trace.attrs)
    return merged, p
