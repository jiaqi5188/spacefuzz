"""Locking action buttons (ui/actions.py), the sidebar menu (ui/nav.py) and the live frame
(ui/live_frame.py): one click = one run, every screen one click away, no blank flash."""
import os
import time
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_config  # noqa: E402
from tests.test_app import MAIN, button, click_and_hold, fake_results, page_html  # noqa: E402
from ui import actions, nav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dashboard():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.run()
    assert not at.exception, at.exception
    return at


def test_job_lifecycle_and_expiry(monkeypatch):
    st.session_state.clear()
    actions.start("forecast", key="k")
    assert actions.job("forecast")["key"] == "k" and actions.job("midday") is None
    later = time.time() + actions.STALE_S + 1
    monkeypatch.setattr(actions.time, "time", lambda: later)
    assert actions.job("forecast") is None          # a run cut short does not lock forever
    assert actions.JOB not in st.session_state


def test_run_forecast_locks_while_it_runs(monkeypatch):
    at = dashboard()
    b = button(at, "Run Forecast")
    assert not b.disabled
    assert 'data-state="start"' in page_html(at)                        # what to press first
    click_and_hold(at, "Run Forecast", monkeypatch)
    busy = button(at, "Running forecast...")
    assert busy.disabled                                     # a second click does nothing
    assert 'data-state="replaying"' in page_html(at)
    assert at.sidebar.selectbox[0].disabled                  # no scenario switch mid-run
    at.run()                                                 # the run finishes, unlocks
    assert not button(at, "Run Forecast").disabled
    assert "Press again to replay" in page_html(at)
    assert 'data-testid="playback-done"' in page_html(at)


def test_one_click_runs_the_whole_forecast():
    at = dashboard()
    button(at, "Run Forecast").click().run()
    assert not at.exception, at.exception
    h = page_html(at)
    assert 'data-testid="playback-done"' in h and "21.0%" in h
    assert not button(at, "Run Forecast").disabled and actions.JOB not in at.session_state


def test_fixed_height_slots():
    """Slots keep their size when the forecast fills them (no jump in the layout)."""
    at = dashboard()
    h = page_html(at)
    assert 'class="sf-panel sf-fix sf-act" data-testid="act-now"' in h
    assert 'class="sf-panel sf-compare" data-testid="risk-comparison"' in h
    assert '<div class="sf-small">&nbsp;</div>' in h        # coverage line reserved
    assert ".sf-act { height:" in h and "sf-status" in h


def test_menu_lists_every_screen_with_the_wiki_last():
    groups = [m[0] for m in nav.MENU]
    assert list(dict.fromkeys(groups)) == ["Mission", "Daily checks", "Role screens", "Reference"]
    assert nav.MENU[0][1] == "Dashboard" and nav.MENU[-1][1].startswith("Indicators")
    assert [m[3] for m in nav.MENU if m[0] == "Role screens"] == list(ui_config.VIEWS)
    assert {"Morning_check-in", "Midday_check", "Indicators"} <= {m[3] for m in nav.MENU}
    for *_, source in nav.MENU:                               # every page file exists
        assert source is None or source in ui_config.VIEWS or (ROOT / source).exists(), source


def test_app_registers_the_menu():
    at = dashboard()
    assert not at.exception                                   # st.navigation ran


def test_live_frame_swaps_after_load_and_skips_unchanged_documents():
    html = (ROOT / "ui" / "live_frame" / "index.html").read_text()
    assert 'addEventListener("load"' in html and ".next { visibility:hidden; }" in html
    assert "if (doc === current) return;" in html            # same document: left running
    assert "http://" not in html and "https://" not in html  # offline
