"""Tests for ui/body_map.py: six-system body map (SVG + CSS) with dose bar and popups."""
import os
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from spacefuzz import config  # noqa: E402
from ui import body_map as bm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = "main_sleep_debt_eva"

FC = {"risk_rating": "HIGH", "risk_probability": 0.2096,
      "probability_track": {"samples": 2500,
                            "by_system": {"Cardiovascular": 472, "CNS": 64, "Radiation": 7},
                            "by_rule": {"cardiovascular": 472, "cognitive": 64,
                                        "rule4_spe_eva": 5, "radiation_dose": 4}}}


def test_six_systems_four_forecast_two_weekly():
    keys = [s["key"] for s in bm.SYSTEMS]
    assert keys == ["cardiovascular", "respiratory", "cns", "metabolic",
                    "musculoskeletal", "immune"]
    assert [s["key"] for s in bm.SYSTEMS if s["forecast"]] == keys[:4]
    for s in bm.SYSTEMS:
        assert s["function"] and s["hazard"]


def test_system_status_from_forecast_shares():
    st_ = bm.system_status(FC)
    assert st_["cardiovascular"]["status"] == "red" and st_["cardiovascular"]["share"] == "18.9%"
    assert st_["cns"]["status"] == "yellow" and st_["cns"]["share"] == "2.6%"
    assert st_["respiratory"]["status"] == "green" and st_["metabolic"]["status"] == "green"
    assert st_["musculoskeletal"]["status"] == "weekly" and st_["immune"]["status"] == "weekly"
    assert "heart rate" in st_["cardiovascular"]["top_rule"]      # plain text, not a rule id
    assert st_["radiation"]["share"] == "0.3%" and st_["radiation"]["status"] == "green"


def test_system_status_untrusted_is_grey():
    st_ = bm.system_status({"risk_rating": "UNKNOWN", "probability_track": None})
    assert {v["status"] for k, v in st_.items() if k != "radiation"} <= {"grey", "weekly"}


def test_current_state_has_units_and_planned_eva_dose():
    cur = bm.current_state(MAIN)
    text = " ".join(cur[k] for k in ("cardiovascular", "respiratory", "cns", "metabolic"))
    for unit in ("bpm", "mmHg", "ms", "%", "ppm", "mg/dL", "°C", "/100", " h"):
        assert unit in text, unit
    assert cur["eva_dose_ugy"] == pytest.approx(75.3, abs=0.1)
    assert "weekly" in cur["musculoskeletal"].lower()


def test_reveal_delay_matches_app_playback():
    frames = [{"stage": "forecast"}] * 3 + [{"stage": "forecast", "done": "forecast"},
                                             {"stage": "adjust", "done": "adjust"}]
    assert bm.stage_times(frames, frame_s=0.7, reveal_s=1.2)[0] == pytest.approx(2.1)
    assert bm.stage_times([], frame_s=0.7, reveal_s=1.2)[0] == 0.0


def doc(mode):
    return bm.html_doc(mode, bm.system_status(FC), bm.current_state(MAIN), reveal_at_s=2.1)


def test_html_idle_play_revealed():
    idle, play, done = doc("idle"), doc("play"), doc("revealed")
    assert 'data-state="idle"' in idle and "body-map-done" not in idle.split("<script")[0]
    assert 'data-state="play"' in play and "2100" in play          # reveal after 2.1 s
    assert 'data-testid="body-map-done"' in done                  # static: marker at once
    for s in bm.SYSTEMS:
        assert f'data-system="{s["key"]}"' in play
    assert "@keyframes breathe" in play and "prefers-reduced-motion" in play
    assert '<tspan class="pre">--</tspan><tspan class="post">18.9%</tspan>' in play  # no spoiler


def test_html_popup_dose_bar_and_labels():
    h = doc("revealed")
    assert 'data-testid="body-map-popup"' in h
    assert 'data-testid="dose-bar"' in h and "µGy" in h
    assert 'class="sf-tip u"' in h and ".sf-tip" in h          # the dose note sits in a tip
    assert f"{config.RAD_EVA_DOSE_LIMIT_UGY:.0f} µGy" in h
    assert "MSL-RAD" in h and "GOES" in h                         # data source + proxy note
    assert "weekly" in h.lower()
    assert "http://" not in h and "https://" not in h             # offline: nothing external


# ---------------------------------------------------------------- inside app.py

def iframe_doc(at):
    from tests.test_app import frame_docs
    frames = frame_docs(at, "sf-body-map")
    assert len(frames) == 1
    return frames[0]


def test_app_plays_then_reveals(monkeypatch):
    from tests.test_app import click_and_hold, fake_results
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {MAIN: fake_results()}
    at.run()
    assert not at.exception
    assert 'data-state="idle"' in iframe_doc(at)
    click_and_hold(at, "Run Forecast", monkeypatch)
    assert not at.exception
    assert 'data-state="play"' in iframe_doc(at)       # rendered once, not per frame
    at.run()                                             # playback ends: static result
    d = iframe_doc(at)
    assert 'data-state="revealed"' in d and 'data-testid="body-map-done"' in d


# ---------------------------------------------------------------- look (user review 4.2, 2026-11-13)

def test_figure_is_left_right_symmetric():
    """One smooth closed silhouette; every contour point has its mirror image (x = 400 - x)."""
    pts = bm.silhouette_points()
    assert len(pts) >= 60
    for x, y in pts:
        assert (2 * bm.CX - x, y) in pts, (x, y)
    d = bm._smooth(pts)
    assert d.startswith("M") and d.endswith("Z") and d.count("C") == len(pts)   # curves only


def test_heart_is_drawn_on_top_of_the_lungs():
    h = doc("revealed")
    order = [h.index(f'class="organ fc" data-system="{k}"') for k in
             ("respiratory", "metabolic", "cns", "cardiovascular")]
    assert order[-1] == max(order)              # later in the SVG = drawn on top


def test_organs_symmetric_except_heart():
    """Every circle / ellipse centre of the paired organs has a mirror partner."""
    import re
    for key, svg in bm._ORGANS.items():
        if key == "cardiovascular":             # the heart sits right of centre on purpose
            continue
        centres = [float(m.group(2)) for m in re.finditer(r'<(circle|ellipse) cx="([\d.]+)"', svg)]
        for c in centres:
            assert any(abs((2 * bm.CX - c) - d) < 1e-6 for d in centres), (key, c)
    lungs = re.findall(r'<path d="([^"]+)"/>', bm._ORGANS["respiratory"])[:2]
    xs = [sorted(float(v) for v in re.findall(r"([\d.]+) [\d.]+", p)) for p in lungs]
    assert xs[0] == sorted(2 * bm.CX - v for v in xs[1])                  # mirrored lungs


def test_r_l_labels_medical_atlas_view():
    """Facing-the-viewer view (atlas / CT convention): R on the image left, L on the right."""
    import re
    h = doc("revealed")
    r = re.search(r'<text class="side" x="([\d.]+)"[^>]*>R</text>', h)
    l_ = re.search(r'<text class="side" x="([\d.]+)"[^>]*>L</text>', h)
    assert r and l_ and float(r.group(1)) < bm.CX < float(l_.group(1))
    assert ".side {" in h and "var(--sf-muted)" in h.split(".side {")[1].split("}")[0]  # muted
    (hx, _), side, _ = bm.CALLOUT["cardiovascular"]
    assert hx > bm.CX and side == "r"                                    # heart: image right


def test_callouts_name_every_system_and_change_shape_at_the_reveal():
    """The shares sit on the figure (callouts), and status is never colour alone."""
    import re
    h = doc("revealed")
    css = h.split("<style>")[1].split("</style>")[0]
    for s in bm.SYSTEMS:
        assert f'data-testid="organ-{s["key"]}"' in h, s["key"]
    for status in ("yellow", "red"):
        assert re.search(rf'\[data-state="revealed"\] \.callout\.fc\[data-status="{status}"\] '
                         rf'\.glyph', css), status
    assert h.count("weekly check") >= 2                                  # slow systems: no %
    assert 'data-testid="organ-cardiovascular" data-status="red"' in h   # main: CV red
    assert 'data-testid="organ-cns" data-status="yellow"' in h           # CNS yellow


def test_scan_beam_standby_fast_while_fuzzing_none_when_done():
    import re
    css = doc("idle").split("<style>")[1].split("</style>")[0]
    idle = float(re.search(r'\[data-state="idle"\] \.beam \{[^}]*scan ([\d.]+)s', css).group(1))
    play = float(re.search(r'\[data-state="play"\] \.beam \{[^}]*scan ([\d.]+)s', css).group(1))
    assert play < idle and idle >= 4                                     # calm at standby
    assert not re.search(r'\[data-state="revealed"\] \.beam', css)       # still once complete
    for mode, chip in (("idle", "STANDBY"), ("play", "SCANNING"), ("revealed", "SCAN COMPLETE")):
        assert chip in doc(mode)


def test_vitals_readout_and_heart_pulse_follow_the_check_in():
    cur = bm.current_state(MAIN)
    h = doc("revealed")
    assert f"HR {cur['resting_hr']:.0f} bpm" in h and f"SpO2 {cur['spo2']:.1f}%" in h
    assert f"animation: beat {60 / cur['resting_hr']:.2f}s" in h        # one beat per heartbeat
    assert doc("revealed") == h                                          # no randomness


def test_holographic_look_is_inline_and_calm():
    import re
    h = doc("revealed")
    css = h.split("<style>")[1].split("</style>")[0]
    assert '<filter id="glow"' in h and "feGaussianBlur" in h          # glowing outline
    assert ".body-outline {" in css and "url(#glow)" in css.split(".body-outline {")[1].split("}")[0]
    for s in bm.SYSTEMS:                                                # one heat gradient each
        assert f'<radialGradient id="heat-{s["key"]}"' in h
        assert f"--fill:url(#heat-{s['key']})" in h
    assert re.search(r'\[data-state="revealed"\] \.organ\.fc \.h0 \{ stop-color:var\(--c\)', css)
    assert ".stage::after" in css and "@keyframes holo" in css          # grid + scan lines
    period = float(re.search(r"animation: holo ([\d.]+)s", css).group(1))
    assert 3.0 <= period <= 4.0                                         # slow breathing, no flicker
    reduced = css.split("prefers-reduced-motion")[1]
    assert ".stage::after { animation:none; }" in reduced
    for moving in (".beam", ".spark", ".ring.spin", ".ripple", ".ecg"):
        assert moving in reduced, moving
    assert 'data-status="red"' in h.split('class="organ fc" data-system="cardiovascular"')[1][:40]
    assert '<circle class="ripple"' in h                                # red system ripples
    assert "http://" not in h and "https://" not in h and "<img" not in h


def test_weekly_systems_show_todays_real_context_and_are_scanned_not_forecast():
    """Musculoskeletal / immune are not forecast (weeks, no daily model); they show what today
    really holds (plan countermeasure, check-in stress and planned dose) and, after the scan,
    say they were scanned but are not forecast. Never a percentage."""
    cur = bm.current_state(MAIN)
    assert cur["musculoskeletal_today"] == "ARED 1 h today"               # 07:00 exercise_ARED
    assert cur["immune_today"] == f"stress 3/5 · {cur['eva_dose_ugy']:.0f} µGy dose"
    assert "ARED" in cur["musculoskeletal"] and "weekly" in cur["musculoskeletal"]
    assert bm.current_state("radiation_spe")["musculoskeletal_today"] == "T2 45 min today"
    for mode in ("idle", "revealed"):
        h = doc(mode)
        assert "ARED 1 h today" in h and "stress 3/5" in h
        assert '<tspan class="post">scanned · not forecast</tspan>' in h
    seg = doc("revealed").split('data-testid="organ-musculoskeletal"')[1].split("</g>")[0]
    assert "%" not in seg
