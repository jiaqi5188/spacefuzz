"""One computation per input at a time (user report 11/15: an edited check-in forecast ran 25x
slower). With fastReruns, a page switch during a long computation starts a new script run
while the old one cannot stop (pure Python, no Streamlit call to interrupt it); the new run
used to start the same computation again, and the copies shared one core. compute_once()
lets later callers wait for the running computation instead."""
import os
import threading
import time

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import app  # noqa: E402


def _slow(calls, delay=0.3, fail=False):
    def load(name, profile_name=None, checkin_json=None):
        calls.append((name, profile_name, checkin_json))
        time.sleep(delay)
        if fail:
            raise RuntimeError("engine failed")
        return {"name": name, "checkin_json": checkin_json}
    return load


def _run_parallel(n, target):
    out, threads = [None] * n, []
    for i in range(n):
        def run(i=i):
            try:
                out[i] = target()
            except Exception as e:  # noqa: BLE001 - collected for the assertion
                out[i] = e
        threads.append(threading.Thread(target=run))
    for t in threads:
        t.start()
        time.sleep(0.02)              # later page switches arrive while the first one runs
    for t in threads:
        t.join(5)
    return out


def test_concurrent_requests_for_the_same_input_compute_once(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "load_results", _slow(calls))
    out = _run_parallel(4, lambda: app.compute_once("radiation_spe", "full", '{"a": 1}'))
    assert len(calls) == 1                                    # one computation, not four
    assert all(r == out[0] for r in out) and out[0]["name"] == "radiation_spe"


def test_different_inputs_do_not_wait_for_each_other(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "load_results", _slow(calls))
    out = _run_parallel(2, lambda: app.compute_once(
        "radiation_spe", "full", f'{{"t": {threading.get_ident()}}}'))
    assert len(calls) == 2 and out[0] != out[1]


def test_a_failure_reaches_every_waiting_caller_and_is_not_kept(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "load_results", _slow(calls, fail=True))
    out = _run_parallel(3, lambda: app.compute_once("main_sleep_debt_eva", "full", None))
    assert len(calls) == 1 and all(isinstance(r, RuntimeError) for r in out)
    monkeypatch.setattr(app, "load_results", _slow(calls))
    assert app.compute_once("main_sleep_debt_eva", "full", None)["name"] == "main_sleep_debt_eva"
    assert len(calls) == 2                                    # tried again after the failure


def test_the_dashboard_computes_through_compute_once():
    src = (app.ROOT / "app.py").read_text()
    assert "results[key] = compute_once(name, ui_config.app_profile()" in src
    assert "results[key] = load_results(" not in src

