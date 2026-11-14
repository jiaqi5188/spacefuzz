"""Primary action buttons that lock while they work (one click = one run).

A Streamlit click during a running script restarts the script, so repeated clicks used to
restart the forecast playback. Here the click only records a job (on_click, before the
script runs); the script then draws the button disabled with a busy label, does the work,
and calls finish(), which clears the job and reruns once to unlock the button. A job left
over from a run that was cut short (page switch, reload) expires after STALE_S; a job still
computing (an edited check-in can take minutes) gets COMPUTE_STALE_S. fail() unlocks the
button after an error and shows the message once.
"""
import time

import streamlit as st

JOB = "_sf_job"
ERROR = "_sf_job_error"
STALE_S = 30.0
COMPUTE_STALE_S = 900.0


def start(name: str, **data) -> None:
    st.session_state[JOB] = {"name": name, "t": time.time(), **data}


def _start(name: str, data) -> None:
    """on_click: job data may be a function, read at click time (a value typed just before)."""
    start(name, **((data() if callable(data) else data) or {}))


def job(name: str, max_age: float = STALE_S) -> dict | None:
    """The job `name` that is running now, or None (dropped once older than max_age)."""
    j = st.session_state.get(JOB)
    if not j or j["name"] != name:
        return None
    if time.time() - j["t"] > max_age:
        st.session_state.pop(JOB, None)
        return None
    return j


def advance(**data) -> None:
    """Next step of the running job in a fresh script run (e.g. compute -> play)."""
    st.session_state[JOB].update(t=time.time(), **data)
    st.rerun()


def finish() -> None:
    st.session_state.pop(JOB, None)
    st.rerun()


def fail(message: str) -> None:
    """The job raised: unlock the button and show `message` on the next run."""
    st.session_state.pop(JOB, None)
    st.session_state[ERROR] = message
    st.rerun()


def show_error(target=st) -> None:
    msg = st.session_state.pop(ERROR, None)
    if msg:
        target.error(msg)


def button(target, label: str, busy_label: str, name: str, busy: bool, widget_key: str,
           job_data=None) -> None:
    """Primary button; disabled, with `busy_label`, while job `name` runs. job_data is a
    dict, or a function returning one at click time."""
    target.button(busy_label if busy else label, key=widget_key, type="primary", disabled=busy,
                  on_click=_start, args=(name, job_data),
                  icon=":material/hourglass_top:" if busy else ":material/play_arrow:")
