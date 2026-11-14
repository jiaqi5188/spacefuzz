"""Tests for the pure helpers in video/build.py."""
import pytest

from video import build


def test_format_srt_time():
    assert build.format_srt_time(0) == "00:00:00,000"
    assert build.format_srt_time(3.5) == "00:00:03,500"
    assert build.format_srt_time(61.25) == "00:01:01,250"


def test_srt_roundtrip():
    cues = [build.Cue(0.0, 3.0, "Hello."), build.Cue(3.0, 6.5, "Line one\nline two")]
    text = build.to_srt(cues)
    assert text.startswith("1\n00:00:00,000 --> 00:00:03,000\nHello.\n")
    assert build.parse_srt(text) == cues


def test_timeline_fits_in_30_seconds():
    total = sum(seg.duration for seg in build.TIMELINE)
    assert total <= 30.0
    assert build.TIMELINE[0].kind == "card" and build.TIMELINE[-1].kind == "card"


def test_default_cues_cover_timeline_without_overlap():
    cues = build.default_cues(name="Ada", team="Orbit")
    assert "Ada" in cues[0].text and "Orbit" in cues[0].text
    total = sum(seg.duration for seg in build.TIMELINE)
    for prev, cur in zip(cues, cues[1:]):
        assert prev.end <= cur.start
    assert cues[-1].end <= total


def test_fit_clip_speeds_up_when_slightly_long():
    plan = build.fit_clip(clip_duration=12.0, slot_duration=10.0, max_speed=2.0)
    assert plan.speed == pytest.approx(1.2)
    assert plan.trim_start == 0.0 and plan.pad == 0.0


def test_fit_clip_keeps_the_ending_when_far_too_long():
    # Cut from the start so the final state (heart colour change) survives.
    plan = build.fit_clip(clip_duration=30.0, slot_duration=10.0, max_speed=2.0)
    assert plan.speed == 2.0
    assert plan.trim_start == pytest.approx(10.0)
    assert plan.pad == 0.0


def test_fit_clip_pads_short_clips():
    plan = build.fit_clip(clip_duration=8.0, slot_duration=10.0, max_speed=2.0)
    assert plan.speed == 1.0 and plan.trim_start == 0.0
    assert plan.pad == pytest.approx(2.0)


def test_voiceover_tempo():
    assert build.voiceover_tempo(28.0, 29.8) == 1.0
    assert build.voiceover_tempo(31.0, 29.8) == pytest.approx(31.0 / 29.8)
    assert build.voiceover_tempo(40.0, 29.8) == 1.1


SILENCE_LOG = """
[silencedetect @ 0x1] silence_start: 0.75
[silencedetect @ 0x1] silence_end: 1.10 | silence_duration: 0.35
[silencedetect @ 0x1] silence_start: 4.0
[silencedetect @ 0x1] silence_end: 4.5 | silence_duration: 0.5
[silencedetect @ 0x1] silence_start: 9.0
"""


def test_parse_silencedetect_returns_speech_segments():
    segs = build.parse_silencedetect(SILENCE_LOG, total=10.0)
    assert segs == [(0.0, 0.75), (1.10, 4.0), (4.5, 9.0)]


def test_parse_silencedetect_handles_trailing_speech():
    segs = build.parse_silencedetect("silence_start: 1.0\nsilence_end: 2.0 |", total=5.0)
    assert segs == [(0.0, 1.0), (2.0, 5.0)]


def test_align_cues_snaps_boundaries_to_pauses():
    # Three phrases of similar length, one long pause between each.
    segs = [(0.0, 2.0), (2.5, 4.5), (5.0, 7.0)]
    cues = build.align_cues(["aaaa", "bbbb", "cccc"], segs, tempo=1.0)
    assert [(c.start, c.end) for c in cues] == [(0.0, 2.5), (2.5, 5.0), (5.0, 7.0)]
    assert [c.text for c in cues] == ["aaaa", "bbbb", "cccc"]


def test_align_cues_groups_extra_segments_and_applies_tempo():
    # Second phrase is spoken in two chunks; tempo 2 halves all times.
    segs = [(0.0, 2.0), (2.5, 3.5), (3.8, 4.8), (5.0, 7.0)]
    cues = build.align_cues(["aaaa", "bbbb", "cccc"], segs, tempo=2.0)
    assert [(c.start, c.end) for c in cues] == [(0.0, 1.25), (1.25, 2.5), (2.5, 3.5)]


def test_align_cues_needs_enough_segments():
    with pytest.raises(ValueError):
        build.align_cues(["a", "b", "c"], [(0.0, 1.0), (1.5, 2.0)], tempo=1.0)


def test_syllable_count():
    assert build.syllables("space time") == 2  # silent final e
    assert build.syllables("Each morning, stress-tests the plan") == 7


def test_align_cues_tolerates_a_slowly_spoken_first_cue():
    # Cue 1 is read slowly in three chunks; cue 2 is one normal chunk.
    segs = [(0.0, 0.8), (1.1, 2.3), (2.8, 4.3), (5.0, 7.6), (8.2, 11.4)]
    texts = ["I'm Ada, Team Orbit. This is SpaceFuzz.",
             "On Mars, help is twenty minutes away.",
             "Each morning, SpaceFuzz stress-tests the day's plan,"]
    cues = build.align_cues(texts, segs, tempo=1.0)
    assert [c.start for c in cues] == [0.0, 5.0, 8.2]


# ---- intro length follows the voiceover

SLOW_INTRO_SEGS = [(0.0, 0.8), (1.1, 2.3), (2.8, 4.3), (5.0, 7.6), (8.2, 11.4)]
SLOW_INTRO_TEXTS = ["I'm Ada, Team Orbit. This is SpaceFuzz.",
                    "On Mars, help is twenty minutes away.",
                    "Each morning, SpaceFuzz stress-tests the day's plan,"]


def test_first_cue_speech_end_uses_last_segment_of_first_cue():
    end = build.first_cue_speech_end(SLOW_INTRO_TEXTS, SLOW_INTRO_SEGS, tempo=1.0)
    assert end == pytest.approx(4.3)
    assert build.first_cue_speech_end(SLOW_INTRO_TEXTS, SLOW_INTRO_SEGS, tempo=2.0) \
        == pytest.approx(2.15)


def test_build_timeline_sets_intro_and_keeps_total():
    timeline = build.build_timeline(intro_duration=4.13)
    assert timeline[0].name == "intro" and timeline[0].duration == pytest.approx(4.13)
    assert sum(s.duration for s in timeline) == pytest.approx(build.TOTAL_S)
    flex = next(s for s in timeline if s.name == build.FLEX_SEGMENT)
    base = next(s for s in build.TIMELINE if s.name == build.FLEX_SEGMENT)
    assert flex.duration == pytest.approx(base.duration - 1.13)


def test_build_timeline_rejects_intro_that_eats_the_flex_shot():
    with pytest.raises(ValueError):
        build.build_timeline(intro_duration=20.0)


# ---- loudness normalisation

LOUDNORM_LOG = """
[Parsed_loudnorm_0 @ 0x1]
{
	"input_i" : "-20.71",
	"input_tp" : "-3.12",
	"input_lra" : "6.40",
	"input_thresh" : "-31.05",
	"output_i" : "-16.02",
	"target_offset" : "0.02"
}
"""


def test_parse_loudnorm_reads_measurement_json():
    m = build.parse_loudnorm(LOUDNORM_LOG)
    assert m["input_i"] == "-20.71" and m["target_offset"] == "0.02"


def test_loudnorm_filter_second_pass_is_linear_with_measurements():
    f = build.loudnorm_filter(build.parse_loudnorm(LOUDNORM_LOG))
    assert f.startswith(f"loudnorm=I={build.LOUDNESS_TARGET_LUFS}:")
    for part in ("measured_I=-20.71", "measured_TP=-3.12", "measured_LRA=6.40",
                 "measured_thresh=-31.05", "offset=0.02", "linear=true"):
        assert part in f


def test_loudnorm_filter_first_pass_prints_json():
    assert build.loudnorm_filter(None).endswith("print_format=json")
