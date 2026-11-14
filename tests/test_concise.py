"""Concise UI (user review 2026-11-14: "the UI is very wordy").

Rules checked here:
- each dashboard panel shows its conclusion in few words (visible word budgets below);
- explanations move into an (i) tip (ui.common.tip): hover or keyboard focus shows them, so
  nothing is lost, only folded;
- "illustrative" is said once, by the (i) badge in the top bar, not on every threshold;
- engine sentences ("action; more detail", "action (reason)") are split by ui.common.brief
  into a short visible part and the rest in the tip, without changing any number.
"""
import os
import re
from html.parser import HTMLParser

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402

import app  # noqa: E402
import ui_config  # noqa: E402
from ui import common, theme  # noqa: E402

MAIN = "main_sleep_debt_eva"
SCENARIOS = list(ui_config.SCENARIOS)


class _Visible(HTMLParser):
    """Text a reader sees without hovering or opening anything: element text outside
    <style>/<script> and outside a closed <details> (its <summary> counts). Attribute values
    (data-tip, title) are not counted."""

    def __init__(self):
        super().__init__()
        self.out, self.skip, self.fold = [], 0, 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("style", "script")
        if tag == "details" and "open" not in dict(attrs):
            self.fold += 1
        if tag == "summary" and self.fold:
            self.fold -= 1                               # the summary text is visible
            self._summary = True

    def handle_endtag(self, tag):
        self.skip -= tag in ("style", "script")
        if tag == "summary" and getattr(self, "_summary", False):
            self._summary = False
            self.fold += 1
        if tag == "details" and self.fold:
            self.fold -= 1

    def handle_data(self, data):
        if not self.skip and not self.fold:
            self.out.append(data)


def visible(h: str) -> str:
    p = _Visible()
    p.feed(h)
    return " ".join(" ".join(p.out).split())


def words(h: str) -> int:
    return len([w for w in visible(h).split() if w not in (common.TIP_MARK, "•", "→", "·", "-")])


def tips(h: str) -> list[str]:
    import html
    return [html.unescape(t) for t in re.findall(r'data-tip="([^"]*)"', h)]


# ---------------------------------------------------------------- helpers

def test_tip_is_an_accessible_hover_and_focus_note():
    h = common.tip('limit < 6 h & "x"')
    assert common.TIP_MARK in visible(h)
    assert 'tabindex="0"' in h and 'role="note"' in h
    assert tips(h) == ['limit < 6 h & "x"']
    assert 'aria-label="limit &lt; 6 h &amp; &quot;x&quot;"' in h       # escaped
    assert common.tip("") == "" and common.tip(None) == ""


def test_tip_css_is_shared_with_the_iframes():
    css = theme.tip_css()
    assert ".sf-tip" in css and ":hover" in css and ":focus" in css
    assert css in theme.CSS_PATH.read_text()                          # one source
    # fixed would ignore the page's scroll offset in Chrome: notes drifted off screen once the
    # page was scrolled (found 2026-11-14); absolute without coordinates follows the (i)
    assert "position:absolute" in css and "position:fixed" not in css


@pytest.mark.parametrize("text,short,detail", [
    ("Drink water and take carbs before EVA; keep the planned rest breaks and return to cool "
     "down if heart rate stays above 140 bpm.",
     "Drink water and take carbs before EVA",
     "keep the planned rest breaks and return to cool down if heart rate stays above 140 bpm"),
    ("Confirm last night's sleep with the wearable record (within its measurement error the "
     "risk moves by 79 points).",
     "Confirm last night's sleep with the wearable record",
     "within its measurement error the risk moves by 79 points"),
    ("15 min less sleep turns this red (EVA started within 12 h of waking after < 6 h sleep)",
     "15 min less sleep turns this red",
     "EVA started within 12 h of waking after < 6 h sleep"),
    ("Short sleep (6.2 h last night, 6.9 h sleep debt over 3 nights): schedule a 20-minute nap "
     "before demanding tasks and go to bed early tonight.",
     "Schedule a 20-minute nap before demanding tasks and go to bed early tonight",
     "Short sleep (6.2 h last night, 6.9 h sleep debt over 3 nights)"),
    ("Go to bed 1 hour earlier tonight to start repaying the sleep debt.",
     "Go to bed 1 hour earlier tonight to start repaying the sleep debt", ""),
])
def test_brief_keeps_the_action_and_folds_the_rest(text, short, detail):
    assert common.brief(text) == (short, detail)


def test_brief_html_loses_no_number():
    text = ("Short sleep (6.2 h last night, 6.9 h sleep debt over 3 nights): take the 1 h nap "
            "at 10:00; go to bed early tonight.")
    h = common.brief_html(text)
    seen = visible(h) + " " + " ".join(tips(h))
    for n in re.findall(r"\d+(?:\.\d+)?", text):
        assert n in seen, n


# ---------------------------------------------------------------- the dashboard

@pytest.fixture(scope="module", params=SCENARIOS)
def final(request):
    """Every slot's HTML at the end of the playback, from the committed full-profile cache."""
    name = request.param
    static = app.load_static(name)
    try:
        res = app.load_results(name, "full")
    except Exception:                                   # noqa: BLE001
        pytest.skip("no full-profile cache")
    v = app.View(static, res, len(app.timeline(res)))
    return {k: f() for k, f in app.regions(v).items()}


BUDGET = {"topbar": 24, "indicators": 62, "act_now": 32, "risk": 18, "status": 6,
          "compare": 72, "cards": 90}


@pytest.mark.parametrize("slot", list(BUDGET))
def test_each_panel_stays_within_its_word_budget(final, slot):
    assert words(final[slot]) <= BUDGET[slot], visible(final[slot])


def test_illustrative_is_said_once_in_the_top_bar(final):
    for slot, h in final.items():
        if slot != "topbar":
            assert "illustrative" not in visible(h).lower(), slot
    assert "illustrative" in visible(final["topbar"]).lower()
    assert any("illustrative" in t.lower() for t in tips(final["topbar"]))


def test_the_details_are_folded_not_dropped():
    static = app.load_static(MAIN)
    res = app.load_results(MAIN, "full")
    v = app.View(static, res, len(app.timeline(res)))
    r = {k: f() for k, f in app.regions(v).items()}
    every = lambda h: visible(h) + " " + " ".join(tips(h))          # noqa: E731
    adj = res["adjust"]
    assert "79 points" in every(r["cards"])                          # measurement error
    assert "EVA started within 12 h of waking" in every(r["cards"])  # closest crash rule
    assert "1 h 15 min" in every(r["compare"])                       # re-test margin
    assert "tomorrow's morning check" in every(r["compare"])         # safer-option scope
    assert "share of" in every(r["risk"]) and "5%" in every(r["risk"])   # what the % means
    for c in adj["plan_adjustment"]["changes"]:
        assert c in every(r["compare"])                              # changes stay visible
    assert "6.9 h" in every(r["act_now"])


def test_indicator_thresholds_sit_in_tips():
    rows = app.load_static(MAIN)["indicators"]
    h = app.indicators_html(rows)
    assert len(tips(h)) >= len(rows)
    assert any("6.0 h" in t for t in tips(h))                         # sleep line
    assert "example" in visible(h)                                    # synthetic BP/glucose tag


def test_no_ascii_arrow_on_screen(final):
    """User review 2026-11-14: every arrow on screen is the same character (→)."""
    for slot, h in final.items():
        assert "->" not in visible(h) and not any("->" in t for t in tips(h)), slot


def test_no_ascii_arrow_in_the_animated_panels_and_log():
    from ui import fuzz_log, plan_board as pb
    static = app.load_static(MAIN)
    try:
        res = app.load_results(MAIN, "full")
    except Exception:                                   # noqa: BLE001
        pytest.skip("no full-profile cache")
    blocks = pb.match_tasks(static["plan"], res["adjust"]["details"]["adjusted_plan"])
    assert not any("->" in x for x in pb.legend(blocks))
    assert not any("->" in x["text"] for x in fuzz_log.log_lines(res, static["start_min"]))
    assert not any("->" in f["text"] for f in app.timeline(res))


def test_the_tip_never_wraps_away_from_its_word():
    """User review 2026-11-15: at 1920 px the (i) after the Act now sentence wrapped onto a line
    of its own. An inline-block allows a line break before it; a plain inline element glued to
    the previous word (no space) does not."""
    css = theme.tip_css()
    rule = re.search(r"\.sf-tip \{([^}]*)\}", css)[1]
    assert "display:inline;" in rule.replace(" ", "")
    h = common.brief_html("Take the 1 h nap at 10:00 before demanding tasks; and go to bed early.")
    assert re.search(r"tasks<span class=\"sf-tip", h)                 # glued, no space


def test_no_space_before_any_tip_on_the_dashboard(final):
    for slot, h in final.items():
        assert not re.search(r"\s<span class=\"sf-tip", h), slot


def test_a_tip_note_starts_with_a_capital():
    h = common.brief_html("Drink water before EVA; keep the planned rest breaks.")
    assert tips(h) == ["Keep the planned rest breaks"]
