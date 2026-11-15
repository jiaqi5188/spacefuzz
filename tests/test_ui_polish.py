"""UI polish after the 2026-11-15 review (A: seen in the video, B: small, C: wording).

A1 midday page: "Already happened" sits in the left column, under the button.
A2 plan board: no empty band under the legend (column / wide heights from measurements).
B3 indicator rows stay tidy in a narrow column (menu open).
B4 plan-board task names: a short label when the block is narrow, never "p...".
B5 engine screen: a run summary fills the left column under Run Forecast.
B6 Streamlit's own "?" help icon looks like our (i).
C7 Act now never shows a nap that the adjusted plan contradicts.
C8 top bar: "seed 147" with the profile in a tip, not "full profile".
C9 check-in page: a short visible "how to measure" line under each heading, full text in (i).
"""
import os
import re
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

from streamlit.testing.v1 import AppTest  # noqa: E402

import app  # noqa: E402
import ui_config  # noqa: E402
from tests.test_app import fake_results, page_html  # noqa: E402
from tests.test_page_midday import MAIN, RESULT, page as midday_page  # noqa: E402
from ui import plan_board as pb  # noqa: E402
from ui import theme  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSS = theme.CSS_PATH.read_text()


# ---------------------------------------------------------------- A1

def test_midday_record_is_in_the_left_column_under_the_button():
    at = midday_page({"midday_results": {f"{MAIN}|5.9": RESULT}})
    next(b for b in at.button if b.label == "Run midday forecast").click().run()
    assert not at.exception, at.exception
    left, right = at.columns[0], at.columns[1]
    html = lambda c: " ".join(h.proto.body for h in c.get("html"))   # noqa: E731
    assert 'data-testid="midday-record"' in html(left)
    assert 'data-testid="midday-record"' not in html(right)
    assert 'data-testid="midday-risk"' in html(right)
    assert html(left).index('data-testid="run-status"') < html(left).index('data-testid="midday-record"')


# ---------------------------------------------------------------- A2

def test_board_heights_fit_the_measured_content():
    """Measured 2026-11-15 (content bottom + 10 px padding): a dashboard column needs up to
    274 px (narrow window, menu open), a full-width board (crew screen, midday page) 237 px."""
    assert ui_config.BOARD_HEIGHT["column"] == 280
    assert ui_config.BOARD_HEIGHT["wide"] == 245
    doc = pb.html_doc("idle", pb.match_tasks({"tasks": []}, None), None, [],
                      height=ui_config.BOARD_HEIGHT["wide"])
    assert f"height:{ui_config.BOARD_HEIGHT['wide'] - 4}px" in doc


def test_crew_screen_and_midday_use_the_wide_board():
    src = (ROOT / "app.py").read_text()
    assert 'BOARD_HEIGHT["wide"]' in src or "BOARD_HEIGHT['wide']" in src
    mid = (ROOT / "page_files" / "3_Midday_check.py").read_text()
    assert 'BOARD_HEIGHT["wide"]' in mid


# ---------------------------------------------------------------- B3

def test_indicator_rows_keep_name_and_value_on_one_line_each():
    h = app.indicators_html(app.load_static(MAIN)["indicators"])
    assert 'class="sf-ind-name"' in h and 'class="sf-ind-val"' in h and 'class="base' in h
    assert re.search(r"@container[^{]*\{[^}]*\.sf-ind", CSS)        # narrow: base on its own line
    assert re.search(r"\.sf-ind-name\s*\{[^}]*white-space:\s*nowrap", CSS)


# ---------------------------------------------------------------- B4

def test_narrow_blocks_show_a_short_label():
    blocks = pb.match_tasks({"tasks": [
        {"start": "18:00", "duration_h": 2, "type": "science_experiment", "kind": "task"},
        {"start": "20:30", "duration_h": 1.5, "type": "personal_time", "kind": "task"}]}, None)
    doc = pb.html_doc("idle", blocks, None, [])
    assert '<span class="full">science</span><span class="abbr">sci</span>' in doc
    assert '<span class="full">personal</span><span class="abbr">pers</span>' in doc
    assert "container-type:inline-size" in doc and "@container" in doc
    assert "@container (max-width: 20px)" in doc                          # too narrow: no text
    assert re.search(r"@container \(max-width: 340px\)[^}]*\.base", CSS)    # menu open only


# ---------------------------------------------------------------- B5

def test_engine_screen_has_a_run_summary_under_the_button():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.query_params["view"] = "engine"
    at.run()
    assert 'data-testid="run-summary"' in page_html(at)               # placeholder, same height
    next(b for b in at.button if b.label == "Run Forecast").click().run()
    h = page_html(at)
    s = h.split('data-testid="run-summary"')[1].split("</div></div>")[0]
    assert "2,500" in s and "crashed" in s and "crash types" in s


# ---------------------------------------------------------------- B6

def test_streamlit_help_icon_is_drawn_as_our_tip_mark():
    assert re.search(r'stTooltipIcon[^{]*svg[^{]*\{[^}]*display:\s*none', CSS)
    assert re.search(r'stTooltipIcon[^{]*::after\s*\{[^}]*content:\s*"ⓘ"', CSS)


# ---------------------------------------------------------------- C7

def test_act_now_before_the_adjust_reveal_does_not_promise_a_20_minute_nap():
    res = fake_results()
    frames = app.timeline(res)
    i_fc = next(i for i, f in enumerate(frames) if f.get("done") == "forecast") + 1
    before = app.act_now_view(res, frames, i_fc)
    assert "20-minute" not in before
    assert "Plan a nap before demanding tasks" in before
    assert "adjusted plan sets the time" in before                     # in the tip
    i_adj = next(i for i, f in enumerate(frames) if f.get("done") == "adjust") + 1
    assert "Take the 1 h 30 min nap at 10:00" in app.act_now_view(res, frames, i_adj)


def test_act_now_without_an_adjusted_nap_keeps_the_engine_advice():
    fc = {"advice_now": ["Short sleep (6.2 h): schedule a 20-minute nap before demanding tasks."],
          "action_restrictions": []}
    assert "20-minute nap" in app.act_now_html(fc, None)


# ---------------------------------------------------------------- C8

def test_top_bar_shows_the_seed_and_folds_the_profile():
    static = app.load_static(MAIN)
    h = app.topbar_html(static, fake_results())
    vis = re.sub(r'data-tip="[^"]*"|aria-label="[^"]*"', "", h)
    assert "seed 147" in vis and "profile" not in vis.split('class="sub"')[1]
    assert re.search(r'data-tip="[^"]*profile[^"]*same result', h)


# ---------------------------------------------------------------- C9

def test_check_in_page_shows_a_short_how_to_measure_line():
    from streamlit.testing.v1 import AppTest as T
    at = T.from_file(str(ROOT / "page_files" / "1_Morning_check-in.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    for short in ("Seated, 5 min rest, average of 2 readings.", "Fingerstick before breakfast.",
                  "Wristband record; otherwise your own estimate.",
                  "Your rating right now.", "Pick the line that fits you now.",
                  "Recorded only; not used by the forecast yet."):
        assert short in caps, short
    md = " ".join(m.value for m in at.markdown)
    assert "take 2 readings 1 min apart" in md                          # full text in the (i)


# ---------------------------------------------------------------- 2026-11-15, round 2

def test_check_in_preview_names_no_20_minute_nap():
    """The dashboard never shows the engine's generic "20-minute nap" any more (C7); the
    check-in preview says the same thing as the dashboard before the reveal."""
    at = AppTest.from_file(str(ROOT / "page_files" / "1_Morning_check-in.py"), default_timeout=30).run()
    h = " ".join(x.proto.body for x in at.get("html"))
    prev = h.split("Act now (preview)")[1]
    assert "20-minute" not in prev and "Plan a nap before demanding tasks" in prev
    assert "adjusted plan" in prev                                      # in the tip


def test_one_neutral_nap_wording_for_both_screens():
    from ui import common
    line = ("Short sleep (6.2 h last night): schedule a 20-minute nap before demanding tasks "
            "and go to bed early tonight.")
    assert common.neutral_nap(line) == ("Short sleep (6.2 h last night): plan a nap before "
                                        "demanding tasks (the adjusted plan sets the time) and "
                                        "go to bed early tonight.")
    assert common.neutral_nap("Drink water.") == "Drink water."


def test_top_bar_and_board_title_shrink_when_the_menu_is_open():
    assert re.search(r"\.sf-topbar\s*\{[^}]*container-type:\s*inline-size", CSS)
    assert re.search(r"@container[^{]*\{[^}]*\.sf-topbar", CSS)
    doc = pb.html_doc("final", pb.match_tasks({"tasks": []}, None), None, [])
    assert '<span class="pb">Plan board · </span>' in doc                # dropped when narrow
    assert re.search(r"@media \(max-width: 400px\)[^}]*\.pb", doc)


def test_the_narrow_top_bar_keeps_the_badge_size():
    rule = re.search(r"@container \(max-width: 1200px\) \{([^}]*)\}", CSS)[1]
    assert ":not(.sf-badge)" in rule


# ---------------------------------------------------------------- wiki review, 2026-11-15

def test_radiation_rule_is_described_as_the_days_eva_dose():
    from ui import body_map, indicator_doc
    lines = " ".join(indicator_doc.thresholds("`dose_rate`, `radiation_load`"))
    assert "single-EVA" not in lines and "day's EVA dose" in lines and "µGy" in lines
    assert "single-EVA" not in body_map.rule_text("radiation_dose")
    src = (ROOT / "ui" / "body_map.py").read_text()
    assert "single-EVA limit" not in src


def test_wiki_shows_site_units_and_arrows():
    at = AppTest.from_file(str(ROOT / "page_files" / "2_Indicators.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    h = " ".join(x.proto.body for x in at.get("html")) + " ".join(m.value for m in at.markdown)
    vis = re.sub(r'href="[^"]*"', "", h)
    assert "uGy" not in vis and "degC" not in vis and " -> " not in vis
    assert "µGy/h" in vis and "°C" in vis and "stress → HR" in vis


# ---------------------------------------------------------------- 14-inch review, 2026-11-15

def test_risk_panel_before_the_forecast_has_one_aligned_line():
    """"--" (44 px) next to "not forecast yet" (20 px) looked misaligned: one grey line in
    the same row height instead, and no lone "--" for the crash time."""
    h = app.risk_html(None, 390, False)
    vis = re.sub(r"<[^>]+>", " ", h)
    assert "--" not in vis and "Not forecast yet" in vis
    assert 'class="sf-risk sf-risk-idle"' in h
    assert re.search(r"\.sf-risk-idle\s*\{[^}]*height:", (ROOT / "app.py").read_text())


def test_an_abbreviation_that_does_not_fit_is_hidden_not_cut():
    doc = pb.html_doc("idle", pb.match_tasks({"tasks": []}, None), None, [])
    assert re.search(r"@container \(max-width: 28px\) \{ \.blk \.abbr \{ display:none", doc)
    assert re.search(r"@container \(max-width: 20px\) \{ \.blk \.full", doc)


# ---------------------------------------------------------------- trend default follows the scenario

def test_trend_default_is_the_flagged_indicator_else_heart_rate():
    from ui import charts
    ind = lambda **s: [{"key": k, "status": s.get(k, "green")}                # noqa: E731
                       for k in ("heart_rate", "blood_pressure", "blood_glucose",
                                 "body_temperature", "sleep", "mood")]
    assert charts.default_trend(ind(sleep="yellow")) == "sleep"            # main scenario
    assert charts.default_trend(ind()) == "heart_rate"                      # all green
    assert charts.default_trend(ind(sleep="yellow", blood_pressure="red")) == "blood_pressure"
    assert charts.default_trend(ind(mood="red")) == "heart_rate"            # no trend for mood
    for name in ("radiation_spe", "backup_high_co2"):
        assert charts.default_trend(app.load_static(name)["indicators"]) == "heart_rate"
    assert charts.default_trend(app.load_static(MAIN)["indicators"]) == "sleep"


def test_switching_scenario_resets_the_trend_tab_but_a_click_is_kept():
    from ui import charts
    ss = {}
    main = {"key": MAIN, "indicators": [{"key": "sleep", "status": "yellow"}]}
    rad = {"key": "radiation_spe", "indicators": [{"key": "sleep", "status": "green"}]}
    charts.reset_trend_default(ss, main)
    assert ss["trend_field"] == "sleep"
    ss["trend_field"] = "blood_pressure"                 # the user clicks another tab
    charts.reset_trend_default(ss, main)                  # same scenario: kept
    assert ss["trend_field"] == "blood_pressure"
    charts.reset_trend_default(ss, rad)                   # scenario switch: its own default
    assert ss["trend_field"] == "heart_rate"


# ---------------------------------------------------------------- 14-inch review: four columns, HUD slot

def _dash(**params):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _col_html(col):
    return " ".join(h.proto.body for h in col.get("html"))


def test_four_column_dashboard_puts_decisions_next_to_the_risk():
    assert ui_config.DASHBOARD_COLUMNS == 4
    at = _dash(wide="0")
    next(b for b in at.button if b.label == "Run Forecast").click().run()
    a, b, c, d = (_col_html(at.columns[i]) for i in range(4))
    assert 'data-testid="indicators"' in a and 'data-testid="act-now"' not in a
    assert 'data-testid="risk-probability"' in c and 'data-testid="act-now"' in c
    assert 'data-testid="risk-comparison"' in c
    assert c.index('data-testid="act-now"') < c.index('data-testid="risk-comparison"')
    assert 'data-testid="advice-cards"' in d and 'data-testid="plan-board"' in d
    assert d.index('data-testid="plan-board"') < d.index('data-testid="advice-cards"')
    assert 'data-testid="fuzz-log"' not in a + b + c + d               # full width below


def test_three_columns_are_one_setting_away(monkeypatch):
    monkeypatch.setattr(ui_config, "DASHBOARD_COLUMNS", 3)
    at = _dash(wide="0")
    assert 'data-testid="act-now"' in _col_html(at.columns[0])        # the previous layout


def test_hud_keeps_its_height_before_the_forecast():
    """The HUD toggle is on by default: before the forecast its slot is already reserved, so
    nothing below jumps when the visor appears."""
    from ui import hud
    at = _dash(wide="0")
    h = page_html(at)
    slot = re.search(r'<div class="sf-pending sf-hud-slot" style="height:(\d+)px"', h)
    assert slot and int(slot[1]) == hud.HEIGHT_PX


def test_body_map_title_fits_a_narrow_column():
    from ui import body_map
    src = (ROOT / "ui" / "body_map.py").read_text()
    assert '<span class="tap"> · tap a system</span>' in src
    assert "@media (max-width: 320px) { .tap { display:none; } }" in src


def test_hud_slot_looks_like_the_chart_slots():
    assert re.search(r"\.sf-hud-slot\s*\{[^}]*border:1px dashed", CSS)


# ---------------------------------------------------------------- loading states, balanced columns

def test_panels_show_a_loading_state_while_the_forecast_runs():
    assert 'class="sf-loading' in app.risk_html(None, 390, False, busy=True)
    assert 'class="sf-loading' in app.act_now_html(None, busy=True)
    assert 'class="sf-loading' in app.comparison_html(None, {"tasks": []}, False, busy=True)
    assert 'class="sf-loading' in app.cards_html(None, None, False, busy=True)
    assert 'class="sf-loading' in app.run_summary_html(None, False, busy=True)
    assert 'class="sf-loading' not in app.risk_html(None, 390, False)        # idle: no spinner
    assert re.search(r"\.sf-loading[^{]*\{", CSS) and "@keyframes sf-shimmer" in CSS


def test_run_status_is_a_hidden_state_not_text():
    """User 2026-11-15: the loading states say it all, no Start / Replaying / Done text; the
    state stays as a hidden marker for tests and the recorder."""
    def state(v):
        h = app.status_html(v)
        assert 'style="display:none"' in h and re.search(r'data-testid="run-status"[^>]*>\s*<', h)
        return re.search(r'data-state="(\w+)"', h)[1]
    v = app.View(app.load_static(MAIN), None, 0)
    assert state(v) == "start"
    v.busy = True
    assert state(v) == "running"
    res = fake_results()
    assert state(app.View(app.load_static(MAIN), res, 1, busy=True)) == "replaying"
    assert state(app.View(app.load_static(MAIN), res, len(app.timeline(res)))) == "done"


def test_advice_cards_sit_in_a_panel_like_the_others():
    for h in (app.cards_html(None, None, False), app.cards_html(None, None, False, busy=True)):
        assert 'class="sf-panel sf-advice' in h and ">Advice<" in h


def test_run_button_on_top_of_the_body_map_and_plan_table_folded():
    at = _dash(wide="0")
    b = at.columns[1]
    html = _col_html(b)
    assert 'data-testid="run-status"' in html
    assert any(x.label == "Run Forecast" for x in b.button)
    assert html.index('data-testid="run-status"') < html.index('data-testid="body-map"')
    exp = [e for e in b.expander if e.label == "Today's plan as a table"]
    assert exp and exp[0].proto.expanded is False
    assert 'data-testid="plan-table"' in _col_html(b)


def test_opening_the_plan_table_shrinks_the_body_map_not_the_page():
    """Column B has a fixed height and the body map takes what is left: opening the plan
    table squeezes the empty space around the figure instead of lengthening the column."""
    css = app.four_body_css(ui_config.FOUR_HEIGHTS["column"])
    assert f".st-key-four_body.st-key-four_body {{ height:{ui_config.FOUR_HEIGHTS['column']}px" in css
    assert ".st-key-four_body > .st-key-body_map { flex:1 1 0; min-height:0" in css
    assert "iframe { height:100% !important" in css
    frame = (ROOT / "ui" / "live_frame" / "index.html").read_text()
    assert "#stack { position:relative; width:100%; height:100%; }" in frame   # follows the frame
    from ui import body_map
    assert "height:100vh" in body_map._CSS                           # the figure follows too
    assert re.search(r"\.pop \{[^}]*overflow-y:auto", body_map._CSS)  # popup scrolls if short


def test_four_column_heights_are_fixed_and_from_ui_config():
    at = _dash(wide="0")
    h = page_html(at)
    hh = ui_config.FOUR_HEIGHTS
    assert f".sf-panel.sf-act {{ height:{hh['act']}px; }}" in h
    assert f".sf-compare {{ height:{hh['compare']}px;" in h and f".sf-advice {{ height:{hh['advice']}px;" in h
    assert 'data-testid="layout-four"' in h


def test_run_button_keeps_its_usual_width():
    h = page_html(_dash(wide="0"))
    assert ".st-key-run_forecast button { min-width:0" not in h          # no override any more


def _view(name):
    return page_html(_dash(view=name))


def test_role_screens_have_fixed_heights_and_loading():
    s = _view("surgeon")
    hh = ui_config.SURGEON_HEIGHTS
    assert f".sf-compare {{ height:{hh['compare']}px;" in s and f".sf-advice {{ height:{hh['advice']}px;" in s
    e = _view("engine")
    assert f".sf-summary {{ height:{ui_config.ENGINE_HEIGHTS['summary']}px;" in e
    from ui import hud
    assert "sf-loading" in hud.visor_pending_html(busy=True)
    assert "sf-loading" not in hud.visor_pending_html(busy=False)


def test_risk_row_has_one_height_before_and_after():
    assert re.search(r'\.sf-risk\[data-testid="risk-probability"\]\s*\{\s*height:48px', (ROOT / "app.py").read_text())


def test_engine_log_height_matches_the_left_column_setting():
    from ui import fuzz_log
    doc = fuzz_log.html_doc("idle", [], 0, height=ui_config.ENGINE_HEIGHTS["log"])
    assert f"height:{ui_config.ENGINE_HEIGHTS['log'] - 4}px" in doc


def test_run_button_counts_up_a_rough_percentage_while_busy():
    """A forecast can take minutes (edited check-in): the busy button shows a rough % that the
    browser counts up on its own (CSS), compute first, then the replay (user, 11/15). Past the
    estimate it keeps creeping up instead of stopping (user: "stuck at 95%")."""
    css = app.run_progress_css(compute_s=3, replay_s=7, phase="compute", elapsed=0)
    assert "@property --sf-pct" in css and 'data-testid="run_forecast-progress"' in css
    assert "{ --sf-pct: 24; }" in css                      # at the estimate: 80% of the share
    assert 'counter(sf-pct) "%"' in css and "<integer>" not in css
    late = app.run_progress_css(compute_s=180, replay_s=7, phase="compute", elapsed=400)
    pcts = [int(x) for x in re.findall(r"--sf-pct: (\d+);", late)]
    assert pcts == sorted(pcts) and len(set(pcts[-4:])) >= 3 and 90 < pcts[-1] < 100  # moving
    assert "-400.0s" in late                               # picks up where it is
    rep = app.run_progress_css(compute_s=3, replay_s=7, phase="replay", elapsed=2,
                               compute_elapsed=3)
    assert "0% { --sf-pct: 24; }" in rep and "100% { --sf-pct: 100; }" in rep
    assert "7.0s linear -2.0s" in rep
    at = _dash(wide="0")
    assert "run_forecast-progress" not in page_html(at)    # idle: plain button
    assert ui_config.RUN_ESTIMATE_S["cached"] < ui_config.RUN_ESTIMATE_S["edited"]


def test_round_eight_details():
    """User review 11/15, round 8: the body-map popup closes on a click anywhere else; no Reset
    while a forecast runs."""
    from ui import body_map
    doc = body_map._JS if hasattr(body_map, "_JS") else (ROOT / "ui" / "body_map.py").read_text()
    assert "!pop.contains(e.target)" in doc and "addEventListener('blur'" in doc


def test_round_six_details():
    """User review 11/15, round 6: the warning sign sits on the text line; an opened adjusted
    plan table shows itself (the panel stays scrolled to its end); frames fill their slot."""
    from tests.test_app import fake_results
    res = fake_results()
    h = app.comparison_html(res["adjust"], app.load_static(MAIN)["plan"], True)
    assert '<div class="sf-compare-in">' in h
    assert re.search(r"\.sf-compare:has\(details\[open\]\) \{[^}]*column-reverse", app.CSS)
    assert re.search(r"\.sf-compare-in \{[^}]*margin-bottom:auto", app.CSS)
    assert ".sf-warn {" in app.CSS
    from ui import body_map
    assert "height:100vh" in body_map._CSS and "background:transparent" in body_map._CSS
    assert "background:transparent" in pb._CSS
    for css in (pb._CSS, body_map._CSS):                    # the box keeps the panel colour
        assert re.search(r"#sf-(plan-board|body-map) \{[^}]*background:var\(--sf-panel\)", css)
    for view in ("surgeon", "engine"):
        assert ".st-key-run_forecast { margin-top:-10px; }" in app.role_css(view)
    html = (ROOT / "ui" / "reaction_test" / "index.html").read_text()
    assert "to begin.<br>When the counter appears" in html


def test_round_seven_details():
    """User review 11/15, round 7: a tab clicked twice stays on; the adjusted plan table opens
    and closes with a transition like the expander; the engine screen's charts are shorter
    so the data footer shows on a 14-inch screen; Run midday forecast shows a rough %."""
    src = (ROOT / "ui" / "charts.py").read_text()
    assert src.count("required=True") == 2                     # trend tabs and fan tabs
    assert re.search(r"\.sf-fold::details-content \{[^}]*transition", app.CSS)
    assert "interpolate-size:allow-keywords" in app.CSS
    assert ui_config.ENGINE_HEIGHTS["charts"] <= 330
    at = _dash(view="engine")
    assert f'min-height:{ui_config.ENGINE_HEIGHTS["charts"]}px' in page_html(at)
    from ui import actions
    css = actions.progress_css("run_midday", [(0, 0), (2.0, 60), (5.0, 99)], elapsed=1.0)
    assert ".st-key-run_midday button" in css and "40% { --sf-pct: 60; }" in css
    assert "5.0s linear -1.0s" in css and "<integer>" not in css   # starts 1 s in
