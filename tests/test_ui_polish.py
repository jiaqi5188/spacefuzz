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
    mid = (ROOT / "pages" / "3_Midday_check.py").read_text()
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
    at = T.from_file(str(ROOT / "pages" / "1_Morning_check-in.py"), default_timeout=30).run()
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
    at = AppTest.from_file(str(ROOT / "pages" / "1_Morning_check-in.py"), default_timeout=30).run()
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
    assert re.search(r"@media \(max-width: \d+px\)[^}]*\.pb", doc)


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
    at = AppTest.from_file(str(ROOT / "pages" / "2_Indicators.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    h = " ".join(x.proto.body for x in at.get("html")) + " ".join(m.value for m in at.markdown)
    vis = re.sub(r'href="[^"]*"', "", h)
    assert "uGy" not in vis and "degC" not in vis and " -> " not in vis
    assert "µGy/h" in vis and "°C" in vis and "stress → HR" in vis
