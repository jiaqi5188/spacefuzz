"""The check-in the app works with: the scenario's example file, or the crew's edits.

The morning check-in page saves edits here (st.session_state); the dashboard, body map,
fuzz log and trend chart read the current check-in through it. A check-in identical to
the example file is not an edit, so the cached full-profile results stay in use.
"""
import hashlib
import json
from pathlib import Path

import streamlit as st

from spacefuzz.core import state

CHECKIN_DIR = Path(__file__).resolve().parents[1] / "data" / "checkins"
EDITS = "checkin_edits"


def _ss(ss):
    return st.session_state if ss is None else ss


def example(name: str) -> dict:
    return json.loads((CHECKIN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def current(name: str, ss=None) -> dict:
    return dict(_ss(ss).get(EDITS, {}).get(name) or example(name))


def edited(name: str, ss=None) -> bool:
    return current(name, ss) != example(name)


def save(name: str, data: dict, ss=None) -> None:
    edits = _ss(ss).setdefault(EDITS, {})
    if data == example(name):
        edits.pop(name, None)
    else:
        edits[name] = dict(data)


def reset(name: str, ss=None) -> None:
    _ss(ss).setdefault(EDITS, {}).pop(name, None)


def checkin_json(name: str, ss=None) -> str | None:
    """Canonical JSON of the edited check-in (None when it is the example file)."""
    return json.dumps(current(name, ss), sort_keys=True) if edited(name, ss) else None


def key(name: str, ss=None) -> str:
    """Results key: the scenario name, plus a hash of the check-in when it was edited."""
    j = checkin_json(name, ss)
    return name if j is None else f"{name}|{hashlib.sha1(j.encode()).hexdigest()[:10]}"


def to_checkin(data: dict) -> state.CheckIn:
    """The UI's way in for a check-in: any entry that cannot be read (empty text, None, NaN or
    infinity where a whole number is needed, a wrong shape) is a CheckInError with a message,
    never a crash. state.load_checkin does this itself now; this guard stays as the UI's
    last line of defence."""
    try:
        return state.load_checkin(data)
    except state.CheckInError:
        raise
    except (TypeError, ValueError, OverflowError, KeyError, AttributeError) as exc:
        raise state.CheckInError(f"check-in entry cannot be used: {exc}") from None


def checkin(name: str, ss=None) -> state.CheckIn:
    return to_checkin(current(name, ss))


def from_json(name: str, checkin_json: str | None) -> state.CheckIn:
    """For cached helpers keyed by (name, checkin_json)."""
    return to_checkin(json.loads(checkin_json)) if checkin_json else state.load_checkin(name)
