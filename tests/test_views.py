"""Role views (?view=crew|surgeon|engine), autoplay and the wide (>= 2400 px) layout of app.py.

All views read the same results (st.session_state, filled from the engine cache); the full
page without parameters is unchanged (tests/test_app.py).
"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import ui_config  # noqa: E402
from tests.test_app import MAIN, button, fake_results, frame_docs, page_html  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def app(**params):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def frames(at, marker):
    return frame_docs(at, marker)


def play(at):
    button(at, "Run Forecast").click().run()
    assert not at.exception, at.exception
    return page_html(at)


def test_crew_view_is_hud_and_adjusted_board_only():
    at = app(view="crew")
    assert 'data-testid="view-crew"' in page_html(at)
    assert "appears after the forecast" in page_html(at)          # fixed-height note first
    h = play(at)
    assert frames(at, "sf-hud") and frames(at, "sf-plan-board")
    assert not frames(at, "sf-body-map") and not frames(at, "sf-fuzz-log")
    assert 'data-testid="playback-done"' in h and '<div class="sf-panel" data-testid="indicators"' not in h


def test_surgeon_view_has_risk_comparison_measure_and_act_now():
    at = app(view="surgeon")
    h = play(at)
    for tid in ("view-surgeon", "risk-probability", "risk-comparison", "card-measure", "act-now",
                "indicators", "playback-done"):
        assert f'data-testid="{tid}"' in h, tid
    assert "21.0%" in h and "4.5%" in h
    assert not frames(at, "sf-body-map") and not frames(at, "sf-fuzz-log")


def test_engine_view_has_the_fuzz_log_and_the_charts():
    at = app(view="engine")
    h = play(at)
    assert 'data-testid="view-engine"' in h and frames(at, "sf-fuzz-log")
    assert 'data-testid="charts-done"' in h
    assert not frames(at, "sf-body-map") and not frames(at, "sf-hud")


def test_unknown_view_falls_back_to_the_full_page():
    h = page_html(app(view="nonsense"))
    assert 'data-testid="body-map"' in h and 'data-testid="view-' not in h


@pytest.mark.parametrize("view", ["crew", "surgeon", "engine"])
def test_autoplay_plays_once_without_a_click(view):
    at = app(view=view, autoplay="1")
    assert 'data-testid="playback-done"' in page_html(at)
    at.run()                                     # a later rerun does not replay
    assert not at.exception and 'data-testid="playback-done"' in page_html(at)


def test_full_page_autoplays_once_the_width_is_known():
    assert 'data-testid="playback-done"' not in page_html(app(autoplay="1"))   # width unknown
    assert 'data-testid="playback-done"' in page_html(app(autoplay="1", wide="0"))


def test_wide_layout_adds_a_fourth_column_for_log_and_charts():
    narrow, wide = app(wide="0"), app(wide="1")
    assert len(wide.columns) == len(narrow.columns) + 1
    assert 'data-testid="layout-wide"' in page_html(wide)
    assert 'data-testid="layout-wide"' not in page_html(narrow)
    play(wide)
    assert frames(wide, "sf-fuzz-log") and 'data-testid="charts-done"' in page_html(wide)


def test_wide_threshold_is_a_ui_setting():
    assert ui_config.WIDE_MIN_PX == 2400


def test_crew_hud_spans_the_full_width_above_the_board():
    at = app(view="crew")
    play(at)
    srcs = frame_docs(at)
    hud_i = next(i for i, s in enumerate(srcs) if "sf-hud" in s)
    board_i = next(i for i, s in enumerate(srcs) if "sf-plan-board" in s)
    assert hud_i < board_i and not at.columns          # one column: visor first, then the plan


def test_wide_layout_stacks_the_charts(monkeypatch):
    from ui import charts
    seen = []
    monkeypatch.setattr(charts, "_result_charts", lambda res, stacked=False: seen.append(stacked))
    play(app(wide="1"))
    play(app(wide="0"))
    assert seen == [True, False]


def test_surgeon_view_three_columns_growing_panels_last():
    """14-inch laptop without scrolling: risk / act now + comparison / cards side by side; the
    panels that grow at the reveal sit at the bottom of their column."""
    at = app(view="surgeon")
    assert len(at.columns) == 3
    h = page_html(at)
    assert h.index('data-testid="act-now"') < h.index('data-testid="risk-comparison"')


def test_engine_view_shows_the_risk_beside_the_log():
    at = app(view="engine")
    h = play(at)
    assert 'data-testid="risk-probability"' in h and frames(at, "sf-fuzz-log")


def test_hud_waits_for_the_replay(monkeypatch):
    """The full page's HUD shows a note while a run plays and appears once it has ended."""
    from tests.test_app import click_and_hold
    at = app()
    at.session_state["shown"] = {MAIN: 99}                     # an earlier run was shown
    at.run()
    click_and_hold(at, "Run Forecast", monkeypatch)
    assert not frames(at, "sf-hud")                             # replaying: no final HUD yet
    at.run()
    assert frames(at, "sf-hud")


@pytest.mark.parametrize("view", [None, "crew", "surgeon", "engine"])
def test_every_screen_ends_with_the_data_sources_line(view):
    """manual_v7 11/14 step 3 (optional footer): one grey line naming the data sources, at the
    bottom of the dashboard and of each role screen; the text lives in ui_config."""
    at = app(**({"view": view} if view else {}))
    h = page_html(at)
    assert ui_config.DATA_FOOTER == ("Data: NASA OSDR · NASA RadLab (MSL-RAD, DLR DosTel2) · "
                                     "NOAA GOES · WESAD")
    assert h.count('data-testid="data-footer"') == 1
    import re
    footer = re.search(r'<div class="([^"]*)" data-testid="data-footer">([^<]*)</div>', h)
    assert footer and "sf-small" in footer[1] and footer[2] == ui_config.DATA_FOOTER
    assert h.index('data-testid="data-footer"') > h.index('data-testid="topbar"')
