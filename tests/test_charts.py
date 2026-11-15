"""Tests for ui/charts.py: crash boundary heatmap, trajectory fan, left-column trend chart."""
import math
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from spacefuzz import config  # noqa: E402
from spacefuzz.core import state  # noqa: E402
import ui_config  # noqa: E402
from ui import charts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = "main_sleep_debt_eva"
STEPS = config.STEPS_PER_DAY


def results():
    """Small results dict shaped like the engine output (only what charts read)."""
    minutes = [390 + 15 * k for k in range(STEPS)]
    flat = lambda v: [float(v)] * STEPS
    traces = {"heart_rate": [flat(80), flat(150)], "cognitive_fatigue": [flat(30), flat(85)],
              "crashed": [0, 1]}
    traj = {f: {"p5": flat(70), "p50": flat(90), "p95": flat(140)}
            for f in config.ENGINE_TRAJECTORY_FIELDS}
    hist = {"sleep_cut_h": [0.0, 0.0, 0.6, 1.2, 0.0, 0.0],
            "eva_extend_h": [0.0, -1.0, 0.0, 0.5, 1.0, 0.0],
            "crashed": [0, 0, 1, 1, 1, 0], "discarded": [0, 0, 0, 0, 0, 1],
            "fitness": [3.0, 3.1, 0.0, 0.0, 0.0, None], "pool": [1] * 6}
    return {
        "name": MAIN, "profile": "full",
        "forecast": {
            "risk_rating": "HIGH", "risk_probability": 0.2096,
            "heatmap": {"sleep_cut_h": [0.0, 0.5, 1.0], "eva_extend_h": [0.0, 1.0],
                        "risk": [[0.21, 0.23], [1.0, 1.0], [1.0, 1.0]], "samples_per_cell": 2500},
            "guided_track": {"history": hist},
            "min_crash_perturbation": {"minutes": 15, "change": "less sleep",
                                       "text": "15 min less sleep turns this red"},
            "probability_track": {"samples": 2500, "traces": traces, "trajectory": traj,
                                  "trajectory_minutes": minutes},
        },
        "adjust": {"plan_adjustment": {"risk_after": 0.0448, "rating_after": "MEDIUM"},
                   "details": {"traces_after": {**traces, "crashed": [0, 0]},
                               "trajectory_after": traj,
                               "trajectory_minutes_after": minutes}},
    }


# ---------------------------------------------------------------- heatmap

def test_rating_colorscale_follows_the_cut_offs():
    cs = charts.rating_colorscale()
    pos = [p for p, _ in cs]
    assert pos[0] == 0 and pos[-1] == 1 and pos == sorted(pos)
    at = lambda x: next(c for p, c in reversed(cs) if p <= x)
    assert at(0.005) == charts.BAND["green"][0]
    assert at(0.03) == charts.BAND["yellow"][0]
    assert cs[-1][1] == ui_config.COLOR["red"]


def test_guided_boundary_between_closest_survivor_and_closest_crash():
    h = results()["forecast"]["guided_track"]["history"]
    b = charts.guided_boundary(h, bin_h=0.5)
    by = dict(zip(b["y"], b["x"]))
    assert by[0.0] == pytest.approx(0.3)      # survivor at cut 0, crash at cut 0.6 -> midpoint
    assert 1.0 not in by and 0.5 not in by    # nothing survived there: no boundary point
    assert by[-1.0] == 0.0                    # survivor only: edge at the furthest survivor


def test_heatmap_figure():
    fig = charts.heatmap_figure(results())
    hm = next(t for t in fig.data if t.type == "heatmap")
    assert [list(r) for r in hm.z] == [[0.21, 1.0, 1.0], [0.23, 1.0, 1.0]]   # rows = EVA
    assert list(hm.x) == [0.0, 0.5, 1.0] and list(hm.y) == [0.0, 1.0]
    assert "21%" in hm.text[0][0] and "100%" in hm.text[0][1]
    names = {t.name for t in fig.data}
    assert {"guided-track boundary", "15 min less sleep turns this red"} <= names
    assert not {"guided: no crash", "guided: crashed"} & names     # would cover the cell labels
    ring = next(t for t in fig.data if t.name == "guided-track boundary")
    assert list(ring.y) == [0.0] and ring.x[0] == pytest.approx(0.3)   # only rows on the chart
    notes = " ".join(a.text for a in fig.layout.annotations)
    tip = " ".join(a.hovertext or "" for a in fig.layout.annotations)
    assert "illustrative" not in notes                              # said once: the top bar
    assert "cell = risk of" in notes and "ⓘ" in notes               # short line + (i)
    assert "guided boundary 0.30 h · scan 0.25 h (15 min): same edge" in tip
    assert "guided survivors: EVA as planned or shorter only" in tip
    assert "Green < 1%" in tip
    lo, hi = fig.layout.yaxis.range
    assert lo == pytest.approx(-0.25) and hi == pytest.approx(1.25)   # notes moved to the tip
    assert "Sleep cut" in fig.layout.xaxis.title.text and "EVA" in fig.layout.yaxis.title.text


# ---------------------------------------------------------------- trajectory fan

def test_fan_figure_two_panels_crashed_red_threshold_illustrative():
    fig = charts.fan_figure(results(), "heart_rate", state.make_baseline())
    names = [t.name for t in fig.data]
    assert names.count("crashed simulation") == 2 and names.count("no crash") == 2
    crashed = next(t for t in fig.data if t.name == "crashed simulation")
    assert any(v is None or (isinstance(v, float) and math.isnan(v)) for v in crashed.y) \
        or len(crashed.y) == STEPS                # one trace per group, gaps between runs
    thr = [s for s in fig.layout.shapes if s.type == "line"]
    assert thr and thr[0].y0 == pytest.approx(0.8 * state.max_heart_rate(state.make_baseline()))
    titles = " ".join(a.text for a in fig.layout.annotations)
    assert "As planned" in titles and "21.0%" in titles
    assert "Adjusted" in titles and "4.5%" in titles and "illustrative" not in titles
    assert "1 / 2 crashed" in titles and "0 / 2 crashed" in titles
    assert "limit 140 bpm" in titles
    tip = " ".join(a.hovertext or "" for a in fig.layout.annotations)
    assert "80% of the maximum heart rate" in tip
    nocrash = next(t for t in fig.data if t.name == "no crash")
    assert nocrash.opacity >= 0.45                                   # visible on the dark panel


def test_fan_fatigue_threshold():
    fig = charts.fan_figure(results(), "cognitive_fatigue", state.make_baseline())
    thr = [s for s in fig.layout.shapes if s.type == "line"]
    assert thr[0].y0 == config.COG_FATIGUE_ABOVE


def test_fan_without_adjustment_has_one_panel():
    res = results()
    res["adjust"] = None
    names = [t.name for t in charts.fan_figure(res, "heart_rate", state.make_baseline()).data]
    assert names.count("no crash") == 1


# ---------------------------------------------------------------- trends (left column)

def test_trend_series_history_plus_today():
    for key in charts.TRENDS:
        s = charts.trend_series(MAIN, key)
        assert s["days"][-1] == 0 and len(s["days"]) == config.BASELINE_DAYS + 1
        assert s["unit"] and "illustrative" not in s.get("threshold_label", "")
    sleep = charts.trend_series(MAIN, "sleep")
    assert sleep["values"][-3:] == [4.3, 4.5, 6.2]      # last 3 nights from the check-in
    bp = charts.trend_series(MAIN, "blood_pressure")
    assert bp["values"][-1] == 126 and bp["values2"][-1] == 82


def test_trend_figure_marks_today_and_baseline():
    fig = charts.trend_figure(charts.trend_series(MAIN, "sleep"), today_status="yellow")
    names = {t.name for t in fig.data}
    today = next(t for t in fig.data if t.name == "today")
    assert today.marker.color == ui_config.COLOR["yellow"]      # indicator status colour
    assert {"synthetic example history", "today"} <= names
    tip = " ".join(a.hovertext or "" for a in fig.layout.annotations)
    assert "Synthetic example history" in tip                        # not a real astronaut
    assert any("baseline" in (a.text or "") for a in fig.layout.annotations)


# ---------------------------------------------------------------- in app.py

def test_app_charts_after_playback_trends_from_start():
    from tests.test_app import fake_results
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    res = fake_results()
    real = results()
    res["forecast"] = {**res["forecast"], "heatmap": real["forecast"]["heatmap"],
                       "guided_track": {**res["forecast"]["guided_track"],
                                        "history": real["forecast"]["guided_track"]["history"]},
                       "probability_track": {**res["forecast"]["probability_track"],
                                             **real["forecast"]["probability_track"],
                                             "samples": 2500}}
    res["adjust"]["details"] = {**res["adjust"]["details"], **real["adjust"]["details"]}
    at.session_state["results"] = {MAIN: res}
    at.run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1                       # trend chart only
    next(b for b in at.button if b.label == "Run Forecast").click().run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 3                       # + heatmap + fan
    html = "\n".join(h.proto.body for h in at.get("html"))
    assert 'data-testid="charts-done"' in html
    at.run()                                                       # later rerun: still there
    assert not at.exception and len(at.get("plotly_chart")) == 3


def test_missing_chart_data_does_not_crash_the_app():
    from tests.test_app import fake_results                       # no traces / history
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.run()
    next(b for b in at.button if b.label == "Run Forecast").click().run()
    assert not at.exception
    assert any("re-run the engine" in c.value for c in at.caption)


def test_heatmap_without_eva_says_so():
    res = results()
    res["forecast"]["heatmap"] = {"sleep_cut_h": [0.0, 0.5], "eva_extend_h": [0.0],
                                  "risk": [[1.0], [1.0]], "samples_per_cell": 2500}
    res["forecast"]["guided_track"]["history"] = {**res["forecast"]["guided_track"]["history"],
                                                  "eva_extend_h": [0.0] * 6}
    fig = charts.heatmap_figure(res)
    assert fig.layout.yaxis.title.text == "no EVA in this plan"


def test_trend_tabs_switch_to_short_labels_when_the_column_is_narrow():
    """Full names when there is room; whole short words (never "Heart ...") in a narrow
    column, e.g. with the menu open (container query on the tab row's own width)."""
    css = charts.trend_tab_css()
    assert set(charts.TRENDS_SHORT) == set(charts.TRENDS)
    assert "container-type:inline-size" in css and "@container" in css
    for i, k in enumerate(charts.TRENDS, 1):
        assert f'button:nth-of-type({i}) p::after {{ content:"{charts.TRENDS_SHORT[k]}"' in css
    assert "text-overflow" not in css


def test_trend_tabs_have_a_tighter_tier_for_a_very_narrow_column():
    """Menu open on a 14-inch screen: the tab row is about 260 px and even the short words
    were cut ("Hear", "Glucos"; user, 11/15). Below TABS_TINY_BELOW_PX: shorter words, a
    smaller font and less padding, so every label shows whole."""
    css = charts.trend_tab_css()
    assert set(charts.TRENDS_TINY) == set(charts.TRENDS)
    assert charts.TABS_TINY_BELOW_PX < charts.TABS_SHORT_BELOW_PX
    tiny = css[css.index(f"@container (max-width: {charts.TABS_TINY_BELOW_PX}px)"):]
    for i, k in enumerate(charts.TRENDS, 1):
        assert f'button:nth-of-type({i}) p::after {{ content:"{charts.TRENDS_TINY[k]}"' in tiny
    assert "padding" in tiny
