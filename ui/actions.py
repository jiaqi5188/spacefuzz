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
    now = time.time()        # t: this step's start (advance resets it); t0: the job's start
    st.session_state[JOB] = {"name": name, "t": now, "t0": now, **data}


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


def progress_css(widget_key: str, points: list[tuple[float, float]], elapsed: float = 0.0,
                 phase: str = "run") -> str:
    """A rough % on a busy button (widget_key), counted up by the browser: `points` are
    (seconds from the job's start, percent), the animation starts `elapsed` seconds in.
    Drawn once per run: Python may be busy for minutes meanwhile. Not exact."""
    total = max(points[-1][0], 0.1)
    frames = " ".join(f"{100 * t / total:.0f}% {{ --sf-pct: {round(p)}; }}" for t, p in points)
    name = f"sf-pct-{widget_key}-{phase}"         # a new name restarts the animation
    return (f'<div data-testid="{widget_key}-progress" style="display:none"></div><style>'
            # "<" escaped (\\3C): the HTML sanitiser would read <integer> as a tag
            "@property --sf-pct { syntax:'\\3C integer>'; inherits:true; initial-value:0; } "
            f"@keyframes {name} {{ {frames} }} "
            f".st-key-{widget_key} button {{ animation:{name} {total:.1f}s linear -{elapsed:.1f}s "
            "forwards; background:linear-gradient(90deg, color-mix(in srgb, var(--sf-green) 45%, "
            "var(--sf-panel)) calc(var(--sf-pct) * 1%), color-mix(in srgb, var(--sf-green) 18%, "
            "var(--sf-panel)) 0) !important; } "
            f'.st-key-{widget_key} button p::after {{ counter-reset:sf-pct var(--sf-pct); '
            'content:" " counter(sf-pct) "%"; }</style>')


# multiples of the estimate where the curve is drawn: it keeps rising past the estimate
_STEPS = (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32)


def eased_pct(elapsed: float, estimate_s: float, share: float) -> float:
    """Rough % after `elapsed` s of work estimated at `estimate_s`, towards `share` without
    reaching it: 80% of the share at the estimate, 89% at twice it, still creeping later."""
    k = max(elapsed, 0.0) / max(estimate_s, 0.1)
    return share * k / (k + 0.25)


def compute_css(widget_key: str, estimate_s: float, share: float, elapsed: float) -> str:
    """The % while Python computes (it cannot redraw meanwhile)."""
    est = max(estimate_s, 0.1)
    points = [(k * est, eased_pct(k * est, est, share)) for k in _STEPS]
    return progress_css(widget_key, points, elapsed, "compute")


def finish_css(widget_key: str, start: float, seconds: float, elapsed: float = 0.0) -> str:
    """The last stretch, from `start` % to 100 % over `seconds` (the replay)."""
    return progress_css(widget_key, [(0, start), (max(seconds, 0.1), 100)], elapsed, "finish")
