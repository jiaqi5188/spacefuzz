"""Tests for ui/checkin_store.py (shared check-in edits) and ui/reaction.py (reaction test)."""
import json
from pathlib import Path

import pytest

from spacefuzz import config
from spacefuzz.core import quality_gate as qg
import ui_config
from ui import checkin_store as cs
from ui import reaction

ROOT = Path(__file__).resolve().parents[1]
MAIN = "main_sleep_debt_eva"


@pytest.fixture
def ss():
    return {}


def test_example_is_the_file(ss):
    ex = cs.example(MAIN)
    assert ex == json.loads((ROOT / "data" / "checkins" / f"{MAIN}.json").read_text())
    assert cs.current(MAIN, ss) == ex and not cs.edited(MAIN, ss)
    assert cs.key(MAIN, ss) == MAIN


def test_save_edit_changes_key_and_checkin(ss):
    data = {**cs.example(MAIN), "systolic_bp": 150}
    cs.save(MAIN, data, ss)
    assert cs.edited(MAIN, ss) and cs.current(MAIN, ss)["systolic_bp"] == 150
    assert cs.key(MAIN, ss).startswith(MAIN + "|") and cs.key(MAIN, ss) != MAIN
    assert cs.checkin(MAIN, ss).systolic_bp == 150
    assert json.loads(cs.checkin_json(MAIN, ss))["systolic_bp"] == 150
    cs.save(MAIN, cs.example(MAIN), ss)                 # same as the file: not an edit
    assert not cs.edited(MAIN, ss) and cs.key(MAIN, ss) == MAIN
    cs.save(MAIN, data, ss)
    cs.reset(MAIN, ss)
    assert cs.current(MAIN, ss) == cs.example(MAIN)
    assert cs.checkin_json(MAIN, ss) is None             # None = the example file


def test_reaction_summary_counts_lapses_and_screens_implausible_responses():
    lo, hi = config.QG_PLAUSIBLE_RANGES["reaction_time_ms"]
    s = reaction.summarise({"rts": [250, 300, 400, lo - 10, hi + 500], "false_starts": 1,
                            "duration_s": 60})
    assert s["n_valid"] == 3 and s["mean_ms"] == pytest.approx(316.7, abs=0.1)
    assert s["lapses"] == 1                              # >= 355 ms (Jones et al. 2022)
    assert s["suspicious"] == 2 and s["false_starts"] == 1
    assert ui_config.REACTION_LAPSE_MS == 355
    assert reaction.summarise({"rts": [], "false_starts": 3, "duration_s": 60})["mean_ms"] is None
    assert reaction.summarise(None) is None


def test_reaction_mean_goes_through_the_quality_gate():
    ex = {**cs.example(MAIN)}
    s = reaction.summarise({"rts": [260, 280, 300], "false_starts": 0, "duration_s": 60})
    ci = cs.to_checkin({**ex, "reaction_time_ms": s["mean_ms"]})
    assert qg.check(checkin=ci).status == "ok"
    bad = cs.to_checkin({**ex, "reaction_time_ms": 50})          # implausible mean
    assert "reaction_time_ms" in qg.check(checkin=bad).missing


def test_reaction_component_is_static_and_speaks_the_streamlit_protocol():
    html = (ROOT / "ui" / "reaction_test" / "index.html").read_text()
    for msg in ("streamlit:componentReady", "streamlit:setComponentValue",
                "streamlit:setFrameHeight", "streamlit:render"):
        assert msg in html, msg
    assert "http://" not in html and "https://" not in html      # offline, no CDN
    assert reaction.COMPONENT_DIR.name == "reaction_test"


def test_reaction_component_ignores_clicks_between_trials():
    html = (ROOT / "ui" / "reaction_test" / "index.html").read_text()
    assert "if (!running || gap) return;" in html          # one response per stimulus
    assert 'gap = true; pad.classList.remove("go");' in html


def test_reaction_component_ends_at_the_time_limit_even_with_a_stimulus_on_screen():
    html = (ROOT / "ui" / "reaction_test" / "index.html").read_text()
    assert "if (performance.now() >= endAt) { finish(); return; }" in html
    assert "if (performance.now() >= endAt && waiting)" not in html


def test_missed_stimuli_count_as_lapses_not_in_the_mean():
    """No response within the plausible window = a miss (PVT: a lapse), not a reaction time."""
    s = reaction.summarise({"rts": [260, 300], "false_starts": 0, "misses": 2, "duration_s": 60})
    assert s["mean_ms"] == 280.0 and s["misses"] == 2 and s["lapses"] == 2
    assert reaction.summarise({"rts": [260], "false_starts": 0, "duration_s": 60})["misses"] == 0


def test_component_resizes_and_times_out_a_missed_stimulus():
    html = (reaction.COMPONENT_DIR / "index.html").read_text()
    assert "ResizeObserver" in html                    # the frame follows its content
    assert "misses" in html and "plausible_ms[1]" in html
    assert "white-space: nowrap" in html                # "Run again" stays on one line


def test_reaction_pad_changes_border_and_background_in_every_state():
    """Go, false start and miss each recolour the pad itself, not only its text (11/15)."""
    import re
    html = (ROOT / "ui" / "reaction_test" / "index.html").read_text()
    for state in ("go", "early", "missed"):
        rule = re.search(r"#pad\." + state + r" \{([^}]*)\}", html)
        assert rule and "border-color" in rule.group(1) and "background" in rule.group(1), state
    assert 'classList.add("missed")' in html and 'classList.remove("go", "missed")' in html


def test_reaction_pad_starts_the_test_and_go_is_green():
    """Clicking "Ready" starts the test like Start does; go is green, early red, miss grey,
    like a traffic light; the stats line stays on one line (user, 11/15)."""
    import re
    html = (ROOT / "ui" / "reaction_test" / "index.html").read_text()
    assert 'startBtn.addEventListener("click", begin)' in html
    assert "if (!running) { begin(); return; }" in html            # pad: first click starts
    go = re.search(r"#pad\.go \{([^}]*)\}", html).group(1)
    assert "var(--green)" in go and "var(--yellow)" not in go
    stats = re.search(r"#stats \{([^}]*)\}", html).group(1)
    assert "nowrap" in stats and "ellipsis" in stats
