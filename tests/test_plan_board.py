"""Tests for ui/plan_board.py: task blocks slide to the adjusted plan, risk number rolls."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_config  # noqa: E402
from ui import plan_board as pb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = "main_sleep_debt_eva"

T = lambda start, h, typ: {"start": start, "duration_h": h, "type": typ, "kind": "task"}
ORIG = {"tasks": [T("07:00", 1.0, "exercise_ARED"), T("08:15", 0.5, "meal"),
                  T("10:00", 6.5, "EVA"), T("16:45", 0.5, "meal"), T("23:00", 6.0, "sleep")]}
ADJ = {"tasks": [T("07:00", 1.0, "exercise_ARED"), T("08:15", 0.5, "meal"),
                 T("10:00", 1.5, "sleep"), T("11:30", 2.25, "EVA"), T("13:45", 1.0, "rest"),
                 T("14:45", 2.25, "EVA"), T("17:00", 0.5, "meal"), T("23:00", 6.0, "sleep")]}


def by_to(blocks):
    return {(b["to"]["start"], b["type"]): b for b in blocks if b["to"]}


def test_match_pairs_same_type_by_nearest_start():
    blocks = pb.match_tasks(ORIG, ADJ)
    m = by_to(blocks)
    assert m[(23 * 60, "sleep")]["from"]["start"] == 23 * 60       # night sleep stays itself
    assert m[(10 * 60, "sleep")]["status"] == "new"                 # the nap is new
    assert m[(10 * 60, "sleep")]["label"] == "nap"
    assert m[(11 * 60 + 30, "EVA")]["from"]["start"] == 10 * 60     # EVA moved and shortened
    assert m[(11 * 60 + 30, "EVA")]["status"] == "resized"
    split = m[(14 * 60 + 45, "EVA")]
    assert split["status"] == "split" and split["from"]["start"] == 10 * 60   # grows out of EVA
    assert m[(13 * 60 + 45, "rest")]["status"] == "new"
    assert m[(17 * 60, "meal")]["status"] == "moved"                 # knock-on move
    assert m[(7 * 60, "exercise_ARED")]["status"] == "same"


def test_only_real_changes_are_numbered():
    blocks = pb.match_tasks(ORIG, ADJ)
    numbered = [b for b in blocks if b.get("n")]
    assert [b["n"] for b in numbered] == [1, 2, 3, 4]               # in time order
    assert [(b["type"], b["status"]) for b in numbered] == [
        ("sleep", "new"), ("EVA", "resized"), ("rest", "new"), ("EVA", "split")]
    legend = pb.legend(blocks)
    assert legend[0] == "① + nap 10:00, 1 h 30 min"
    assert legend[1] == "② EVA 10:00, 6 h 30 min -> 11:30, 2 h 15 min"
    assert legend[3] == "④ EVA part 2 14:45, 2 h 15 min (split)"


def test_high_load_move_is_numbered_knock_on_move_is_not():
    orig = {"tasks": [T("10:00", 5.0, "EVA"), T("15:30", 0.5, "meal")]}
    adj = {"tasks": [T("15:00", 0.5, "rest"), T("15:30", 5.0, "EVA"), T("20:30", 0.5, "meal")]}
    blocks = pb.match_tasks(orig, adj)
    eva = next(b for b in blocks if b["type"] == "EVA")
    meal = next(b for b in blocks if b["type"] == "meal")
    assert eva["status"] == "moved" and eva.get("n")
    assert meal["status"] == "moved" and not meal.get("n")
    assert "EVA 10:00 -> 15:30 (moved +5 h 30 min)" in " ".join(pb.legend(blocks))


def test_removed_task_fades_out():
    orig = {"tasks": [T("10:00", 6.5, "EVA"), T("23:00", 6.0, "sleep")]}
    adj = {"tasks": [T("23:00", 6.0, "sleep")]}
    blocks = pb.match_tasks(orig, adj)
    gone = next(b for b in blocks if b["type"] == "EVA")
    assert gone["status"] == "removed" and gone["to"] is None and gone.get("n")
    assert "EVA 10:00 removed" in " ".join(pb.legend(blocks))


def test_geometry_on_the_day_axis():
    assert pb.minutes("06:00") == 6 * 60 and pb.minutes("02:00") == 26 * 60  # after midnight
    left, width = pb.geometry(10 * 60, 6.5 * 60)
    assert left == pytest.approx(100 * (600 - 360) / 1440)
    assert width == pytest.approx(100 * 390 / 1440)
    left, width = pb.geometry(22 * 60 + 30, 8 * 60)                  # clipped at the axis end
    assert left + width == pytest.approx(100)


def test_risk_states_two_layers_and_safer_option():
    from tests.test_app import fake_adjust
    st = pb.risk_states(fake_adjust())
    assert st["before"]["text"] == "21.0%" and st["before"]["status"] == "red"
    assert "reported to the ground flight surgeon" in st["before"]["chip"]
    assert st["after"]["text"] == "4.5%" and st["after"]["status"] == "yellow"
    assert st["after"]["chip"] == "Adjusted plan: MEDIUM · crew reminder"
    assert "0.1%" in st["safer"] and "today only" in st["safer"]
    red = pb.risk_states(fake_adjust(after=0.3232, rating_after="HIGH", ground=True))
    assert red["after"]["chip"] == "Adjusted plan: HIGH · still red, report to ground"


def test_html_modes():
    from tests.test_app import fake_adjust
    blocks = pb.match_tasks(ORIG, ADJ)
    rs = pb.risk_states(fake_adjust())
    play = pb.html_doc("play", blocks, rs, events=[], forecast_s=2.0, adjust_s=4.0)
    final = pb.html_doc("final", blocks, rs, events=[])
    idle = pb.html_doc("idle", pb.match_tasks(ORIG, ORIG), None, events=[])
    mk = lambda h: h.split("<script")[0]
    assert 'data-state="play"' in play and "2000" in play and "4000" in play
    assert 'data-testid="plan-board-done"' in mk(final)
    assert 'data-testid="plan-board-done"' not in mk(play) + mk(idle)
    assert "Adjusted plan: MEDIUM · crew reminder" in final and "4.5%" in final
    assert "① + nap 10:00, 1 h 30 min" in final
    assert "http://" not in final and "https://" not in final
    assert str(int(ui_config.PACES[ui_config.PACE]["board_s"] * ui_config.ANIM_SCALE * 1000)) \
        in play


def test_env_band():
    blocks = pb.match_tasks(ORIG, ORIG)
    h = pb.html_doc("idle", blocks, None,
                    events=[{"start": "09:00", "end": "15:00", "type": "spe_warning"}])
    assert "spe_warning" in h and 'class="env"' in h


def test_app_board_plays_then_final(monkeypatch):
    from tests.test_app import click_and_hold, fake_results, frame_docs
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.run()
    assert not at.exception

    def doc():
        f = frame_docs(at, "sf-plan-board")
        assert len(f) == 1
        return f[0]

    assert 'data-state="idle"' in doc()
    click_and_hold(at, "Run Forecast", monkeypatch)       # one run: play, drawn once
    assert not at.exception and 'data-state="play"' in doc()
    at.run()                                               # playback ends: final picture
    assert 'data-state="final"' in doc()
    assert 'data-testid="plan-board-done"' in doc().split("<script")[0]


def test_new_block_keeps_its_name_while_growing_in():
    blocks = pb.match_tasks(ORIG, ADJ)
    play = pb.html_doc("play", blocks, None, events=[], forecast_s=0, adjust_s=0)
    nap = next(part for part in play.split('<div class="blk') if 'data-status="new"' in part
               and "nap" in part)
    assert ">nap<" in nap


# ---------------------------------------------------------------- midday (4.3)

def test_midday_cutoff_greys_the_past_and_marks_now():
    blocks = pb.match_tasks(ADJ, {"tasks": [t for t in ADJ["tasks"]
                                             if not (t["type"] == "EVA" and t["start"] == "14:45")]})
    h = pb.html_doc("final", blocks, None, events=[], cutoff_min=14 * 60)
    pre = h.split("<script")[0]
    assert 'class="past"' in pre and "already happened" in pre
    assert 'data-testid="now-line"' in pre and "now 14:00" in pre
    removed = next(b for b in blocks if b["status"] == "removed")
    assert removed["type"] == "EVA" and removed["from"]["start"] == 14 * 60 + 45


def test_midday_risk_states():
    rs = pb.midday_risk_states({"risk_before": 1.0, "rating_before": "HIGH", "risk_after": 0.0476,
                                "rating_after": "MEDIUM", "escalate_to_ground": False,
                                "changes": ["Cancel the afternoon EVA (14:45)"]})
    assert rs["before"]["text"] == "100.0%" and rs["before"]["status"] == "red"
    assert rs["before"]["chip"] == "Afternoon as planned: HIGH · report to ground"
    assert rs["after"]["chip"] == "Adjusted afternoon: MEDIUM · crew reminder"
    assert rs["safer"] is None and rs["changed"]


def test_risk_text_cannot_close_the_script_tag():
    from tests.test_app import fake_adjust
    rs = pb.risk_states(fake_adjust())
    rs["safer"] = "x</script><b>y"
    doc = pb.html_doc("final", pb.match_tasks(ORIG, ADJ), rs, events=[])
    script = doc.split("<script>", 1)[1]
    assert script.count("</script>") == 1
