"""Tests for video/build_v2.py: the timeline agreed on 11/15 (manual_v7 11/15 "video", aligned
to the real voice-over), its captions and the required labels. Rendering needs ffmpeg and a
browser, so the final file is checked by hand."""
from video import build_v2 as b


def test_segments_follow_the_agreed_timeline_without_gaps():
    names = [s.name for s in b.SEGMENTS]
    assert names == ["earth_from_mars", "dashboard_idle", "dashboard_forecast", "dashboard_hold",
                     "dashboard_adjust", "surgeon", "engine", "worker", "end_card"]
    assert b.SEGMENTS[0].start == 0.0
    for a, c in zip(b.SEGMENTS, b.SEGMENTS[1:]):
        assert abs(a.end - c.start) < 1e-9, (a.name, c.name)
    cuts = {s.name: (s.start, s.end) for s in b.SEGMENTS}
    assert cuts["earth_from_mars"] == (0.0, 2.2) and cuts["dashboard_hold"] == (14.5, 19.8)
    assert cuts["dashboard_adjust"] == (19.8, 22.2) and cuts["worker"] == (24.6, 27.8)
    assert b.TOTAL_S <= 30.0


def test_no_shot_runs_faster_than_one_and_a_half():
    for s in b.SEGMENTS:
        if s.kind == "clip":
            assert b.speed(s) <= 1.5 + 1e-9, s.name


def test_slow_push_in_on_the_held_dashboard():
    hold = next(s for s in b.SEGMENTS if s.name == "dashboard_hold")
    assert hold.zoom[0] == 1.0 and 1.05 <= hold.zoom[1] <= 1.1
    adjust = next(s for s in b.SEGMENTS if s.name == "dashboard_adjust")
    assert adjust.zoom == (hold.zoom[1], hold.zoom[1])          # no jump at the cut


def test_no_caption_cards_on_the_picture():
    """11/15 review: the caption labels sat on the dashboard's own text wherever they went;
    the final cut keeps only the subtitles, the watermark, the photo credit and the end card."""
    assert b.CAPTIONS == []


def test_subtitles_are_in_order_and_inside_the_voice_over():
    subs = b.SUBTITLES
    assert all(a.end <= c.start for a, c in zip(subs, subs[1:]))
    assert subs[0].start >= 0 and subs[-1].end <= b.TOTAL_S


def test_labels_watermark_credit_and_metadata():
    assert b.WATERMARK == "AI voice-over (ElevenLabs)"
    assert b.MARS_CREDIT == "Image: NASA/JPL-Caltech/MSSS/TAMU"
    assert "ElevenLabs" in b.METADATA["comment"] and "AI" in b.METADATA["comment"]
    assert b.END_CARD == ("See the risk before it hits.",
                          "An ounce of prevention is worth a pound of cure.",
                          "SpaceFuzz · Team Raccoon")
    assert b.OUTPUT.name == "spacefuzz_final.mp4"
