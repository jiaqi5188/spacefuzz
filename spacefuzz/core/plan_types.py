"""The single fixed enum for every `type` field in a daily plan.

Each type has a kind:
- "task": a crew activity in plan["tasks"]. The adjuster may move, shorten or
  drop it.
- "env": an environment condition in plan["events"] (radiation warning, high
  CO2). The crew cannot control it, so the adjuster must never move or shorten
  it; it can only move tasks around it.
Plan files repeat the kind on every item ("kind": "task" / "env") so the
distinction is visible in the data, and loaders check it against this enum.

HIGH_CO2 events carry `co2_ppm`. When a plan event and the replayed
environment trace (env_trace) both give CO2 for the same time, the engine uses
the larger value: the event is a known equipment condition (e.g. running on
the backup scrubber) and acts as a floor on top of the replayed curve.

Unknown task types map to GENERIC_HIGH_LOAD so a plan written by hand or by
another tool still runs, conservatively treated as demanding work.
"""
from enum import StrEnum


class PlanType(StrEnum):
    # Tasks (kind "task")
    SLEEP = "sleep"
    MEAL = "meal"
    REST = "rest"
    PERSONAL_TIME = "personal_time"
    HEALTH_CHECK = "health_check"
    EXERCISE_ARED = "exercise_ARED"           # strength training (Advanced Resistive Exercise Device)
    EXERCISE_T2 = "exercise_T2"               # treadmill
    EVA = "EVA"                               # extravehicular activity
    DOCKING = "docking"                       # spacecraft docking, high mental load
    PRECISION_MANUAL = "precision_manual"     # precise manual operation (robotic arm, repair)
    SCIENCE_EXPERIMENT = "science_experiment"
    MAINTENANCE = "maintenance"
    COMM_SESSION = "comm_session"             # ground communication
    GENERIC_HIGH_LOAD = "generic_high_load"   # fallback for unknown task types
    # Environment conditions (kind "env")
    SPE_WARNING = "spe_warning"               # solar particle event warning
    HIGH_CO2 = "high_co2"                     # elevated cabin CO2, field co2_ppm


ENV_TYPES = frozenset({PlanType.SPE_WARNING, PlanType.HIGH_CO2})
TASK_TYPES = frozenset(PlanType) - ENV_TYPES
EXERCISE_TYPES = frozenset({PlanType.EXERCISE_ARED, PlanType.EXERCISE_T2})
PRECISION_TYPES = frozenset({PlanType.DOCKING, PlanType.PRECISION_MANUAL})
RESTORATIVE_TYPES = frozenset({PlanType.SLEEP, PlanType.MEAL, PlanType.REST,
                               PlanType.PERSONAL_TIME})


def kind_of(plan_type: PlanType) -> str:
    """"task" (crew-controlled, adjustable) or "env" (environment, fixed)."""
    return "env" if plan_type in ENV_TYPES else "task"


def parse_type(value: str, kind: str = "task") -> PlanType:
    """Parse a plan `type` string for an item of the given kind ("task" or "env").

    Tasks: unknown values become GENERIC_HIGH_LOAD.
    Env: unknown values raise ValueError (an unknown hazard must not be guessed).
    A known type of the other kind always raises.
    """
    try:
        parsed = PlanType(value)
    except ValueError:
        if kind == "task":
            return PlanType.GENERIC_HIGH_LOAD
        raise ValueError(f"unknown env type: {value!r}") from None
    if kind_of(parsed) != kind:
        raise ValueError(f"{value!r} is a {kind_of(parsed)} type, not {kind}")
    return parsed
