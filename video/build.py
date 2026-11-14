"""Assemble the SpaceFuzz demo video from recorded shots.

Steps: title cards -> fit shots to the storyboard -> concat -> burn subtitles
-> mix voiceover (and optional music) -> export H.264 1080p30 -> ffprobe check.

Subtitles and title cards are rendered to PNG with Playwright (Chromium) and
overlaid with ffmpeg, so no libass/freetype build of ffmpeg is required.

Usage:  python video/build.py [--name NAME] [--team TEAM]
Inputs: video/build/clips/shot*.mp4 (from video/record.py),
        video/voiceover.mp3 (optional; a silent placeholder is generated),
        video/music.mp3 (optional background music).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
BUILD_DIR = VIDEO_DIR / "build"
CLIPS_DIR = BUILD_DIR / "clips"
SRT_PATH = VIDEO_DIR / "subtitles.srt"
VOICEOVER_PATH = VIDEO_DIR / "voiceover.mp3"
MUSIC_PATH = VIDEO_DIR / "music.mp3"
OUTPUT_PATH = VIDEO_DIR / "spacefuzz_demo.mp4"

WIDTH, HEIGHT, FPS = 1920, 1080, 30
MAX_SHOT_SPEED = 2.0          # speed-up limit before we start cutting the head
MAX_VOICE_TEMPO = 1.1         # atempo limit for the voiceover
MUSIC_VOLUME = 0.15           # background music relative to narration
PLACEHOLDER_VOICE_S = 28.0    # silent placeholder when voiceover.mp3 is missing
SILENCE_DB = -35              # silencedetect threshold for pauses in the voiceover
SILENCE_MIN_S = 0.25          # shortest pause treated as a phrase break
MAX_TOTAL_S = 30.0
LOUDNESS_TARGET_LUFS = -16    # integrated loudness target (loudnorm I)
LOUDNESS_TRUE_PEAK = -1.5     # loudnorm TP, dBTP
LOUDNESS_RANGE = 11           # loudnorm LRA
INTRO_MIN_S, INTRO_MAX_S = 1.5, 6.0  # clamp for the voice-driven intro card

# As spoken in the voice-over; used in subtitles and the intro card.
# Override per run with --name / --team.
# Team name Raccoon (2026-11-12). The current voiceover.mp3 still says the old name, so
# re-voice the first line together with the real-UI video (manual 4.5, plan 8.3).
PRESENTER_NAME = "Jiaqi"
TEAM_NAME = "Raccoon"

TAGLINE = "See the risk before it hits"


@dataclass(frozen=True)
class Segment:
    kind: str       # "card" or "shot"
    name: str
    duration: float


# Storyboard (plan 8.2), trimmed to the 3 stub shots. Ends at 29.8 s to leave
# headroom for container rounding below the 30 s limit.
TIMELINE = [
    Segment("card", "intro", 3.0),
    Segment("shot", "shot1", 3.0),    # 0:03-0:06 base monitoring
    Segment("shot", "shot2", 10.0),   # 0:06-0:16 forecast (+ heatmap slot)
    Segment("shot", "shot3", 11.0),   # 0:16-0:27 adjust (+ midday slot)
    Segment("card", "outro", 2.8),
]
TOTAL_S = sum(s.duration for s in TIMELINE)
FLEX_SEGMENT = "shot3"  # absorbs intro length changes (it ends on a long hold)


def build_timeline(intro_duration: float) -> list[Segment]:
    """Storyboard with the intro card resized; the flex shot keeps the total fixed."""
    delta = intro_duration - TIMELINE[0].duration
    out = []
    for seg in TIMELINE:
        if seg.name == "intro":
            seg = Segment(seg.kind, seg.name, intro_duration)
        elif seg.name == FLEX_SEGMENT:
            if seg.duration - delta < 1.0:
                raise ValueError(f"intro {intro_duration:.2f}s leaves no room for {seg.name}")
            seg = Segment(seg.kind, seg.name, seg.duration - delta)
        out.append(seg)
    return out


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def default_cues(name: str, team: str) -> list[Cue]:
    """English narration from plan 8.2, split into readable subtitle cues."""
    return [
        Cue(0.0, 3.0, f"I'm {name}, Team {team}.\nThis is SpaceFuzz."),
        Cue(3.0, 6.0, "On Mars, help is twenty minutes away."),
        Cue(6.0, 10.0, "Each morning, SpaceFuzz\nstress-tests the day's plan,"),
        Cue(10.0, 14.0, "running thousands of what-if scenarios\nto find where the body breaks first."),
        Cue(14.0, 16.0, "It maps the edge of failure."),
        Cue(16.0, 20.0, "Then it adjusts the plan,"),
        Cue(20.0, 24.0, "moving the spacewalk and adding rest,\nbefore anything goes wrong."),
        Cue(24.0, 27.0, "At midday, it checks again."),
        Cue(27.0, TOTAL_S, f"From Mars to the worksite:\n{TAGLINE.lower()}."),
    ]


# ---------------------------------------------------------------- SRT

def format_srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_time(text: str) -> float:
    h, m, rest = text.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def to_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{i}\n{format_srt_time(c.start)} --> {format_srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def parse_srt(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        start, end = (_parse_srt_time(t) for t in lines[1].split("-->"))
        cues.append(Cue(start, end, "\n".join(lines[2:])))
    return cues


# ---------------------------------------------------------------- timing

@dataclass(frozen=True)
class FitPlan:
    speed: float       # playback speed factor (>= 1)
    trim_start: float  # seconds cut from the head of the source clip
    pad: float         # seconds of last-frame hold appended at the end


def fit_clip(clip_duration: float, slot_duration: float, max_speed: float) -> FitPlan:
    """Fit a clip into its storyboard slot.

    Long clips are sped up (up to max_speed); if still too long, the head is
    cut so the ending (the visual payoff) is kept. Short clips hold the last frame.
    """
    if clip_duration <= slot_duration:
        return FitPlan(1.0, 0.0, slot_duration - clip_duration)
    speed = clip_duration / slot_duration
    if speed <= max_speed:
        return FitPlan(speed, 0.0, 0.0)
    return FitPlan(max_speed, clip_duration - slot_duration * max_speed, 0.0)


def voiceover_tempo(voice_duration: float, video_duration: float,
                    max_tempo: float = MAX_VOICE_TEMPO) -> float:
    if voice_duration <= video_duration:
        return 1.0
    return min(voice_duration / video_duration, max_tempo)


# ---------------------------------------------------------------- voiceover alignment

def parse_silencedetect(log: str, total: float) -> list[tuple[float, float]]:
    """Turn ffmpeg silencedetect output into (start, end) speech segments."""
    segments = []
    cursor: float | None = 0.0  # start of current speech run, None while silent
    for kind, value in re.findall(r"silence_(start|end): (-?[\d.]+)", log):
        t = max(float(value), 0.0)
        if kind == "start":
            if cursor is not None and t > cursor:
                segments.append((cursor, t))
            cursor = None
        else:
            cursor = t
    if cursor is not None and total > cursor:
        segments.append((cursor, total))
    return segments


def detect_speech(path: Path) -> list[tuple[float, float]]:
    log = subprocess.run(
        [require_tool("ffmpeg"), "-hide_banner", "-i", str(path), "-af",
         f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    return parse_silencedetect(log, probe_duration(path))


def syllables(text: str) -> int:
    """Rough English syllable count (vowel groups per word), a proxy for speaking time."""
    total = 0
    for word in re.findall(r"[A-Za-z']+|\d+", text.lower()):
        count = len(re.findall(r"[aeiouy]+", word))
        if count > 1 and word.endswith("e") and not word.endswith("le"):
            count -= 1  # silent final e: "space", "time"
        total += max(1, count)
    return total


def align_cues(texts: list[str], segments: list[tuple[float, float]], tempo: float) -> list[Cue]:
    """Subtitle cues timed to the voiceover (see group_segments)."""
    n = len(texts)
    starts = group_segments(texts, segments)
    cues = []
    for k, text in enumerate(texts):
        start = segments[starts[k]][0]
        end = segments[starts[k + 1]][0] if k + 1 < n else segments[-1][1]
        cues.append(Cue(round(start / tempo, 3), round(min(end / tempo, TOTAL_S), 3), text))
    return cues


def first_cue_speech_end(texts: list[str], segments: list[tuple[float, float]],
                         tempo: float) -> float:
    """When the first line stops being spoken, in final-video time."""
    starts = group_segments(texts, segments)
    last = starts[1] - 1 if len(starts) > 1 else len(segments) - 1
    return segments[last][1] / tempo


def group_segments(texts: list[str], segments: list[tuple[float, float]]) -> list[int]:
    """Index of the first speech segment of each cue, cutting only at pauses.

    Segments are split into consecutive groups, one per cue, minimising the
    squared gap between each group's speaking time and the time its text
    should take at the average speaking rate (by syllable count). Comparing
    each cue on its own keeps one slowly read line (e.g. names) from shifting
    every later cut.
    """
    n, m = len(texts), len(segments)
    if m < n:
        raise ValueError(f"{m} speech segments for {n} cues; voiceover pauses not found")
    weights = [syllables(t) or 1 for t in texts]
    durations = [e - s for s, e in segments]
    rate = sum(durations) / sum(weights)  # seconds per syllable
    prefix = [0.0]
    for d in durations:
        prefix.append(prefix[-1] + d)

    # best[k][j]: min cost for cues k.. when cue k starts at segment j.
    inf = float("inf")
    best = [[inf] * (m + 1) for _ in range(n + 1)]
    nxt = [[0] * (m + 1) for _ in range(n + 1)]
    best[n][m] = 0.0
    for k in range(n - 1, -1, -1):
        for j in range(k, m - (n - k) + 1):
            for e in range(j + 1, m - (n - k - 1) + 1):
                c = (prefix[e] - prefix[j] - rate * weights[k]) ** 2 + best[k + 1][e]
                if c < best[k][j]:
                    best[k][j], nxt[k][j] = c, e
    starts, j = [], 0
    for k in range(n):
        starts.append(j)
        j = nxt[k][j]
    return starts


# ---------------------------------------------------------------- loudness

def loudnorm_filter(measured: dict | None) -> str:
    """loudnorm for pass 1 (measure, measured=None) or pass 2 (linear, measured)."""
    base = f"loudnorm=I={LOUDNESS_TARGET_LUFS}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}"
    if measured is None:
        return base + ":print_format=json"
    return (f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")


def parse_loudnorm(log: str) -> dict:
    """Extract the JSON block loudnorm prints to stderr."""
    start, end = log.rfind("{"), log.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no loudnorm JSON in ffmpeg output")
    return json.loads(log[start:end + 1])


def audio_graph(voice_idx: int, music_idx: int | None, tempo: float, norm: str) -> list[str]:
    """Filter-graph lines producing [aout]: voice (+music) -> loudnorm -> pad to TOTAL_S."""
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    graph = [f"[{voice_idx}:a]atempo={tempo:.4f},{fmt}[voice]"]
    mixed = "[voice]"
    if music_idx is not None:
        graph.append(f"[{music_idx}:a]volume={MUSIC_VOLUME},{fmt}[music]")
        graph.append("[voice][music]amix=inputs=2:duration=first:normalize=0[mix]")
        mixed = "[mix]"
    graph.append(f"{mixed}{norm},aresample=48000,apad,atrim=0:{TOTAL_S}[aout]")
    return graph


def measure_loudness(inputs: list[Path], tempo: float) -> dict:
    """Pass 1: measure the narration mix exactly as it will be rendered."""
    args = []
    for p in inputs:
        args += ["-i", str(p)]
    graph = audio_graph(0, 1 if len(inputs) > 1 else None, tempo, loudnorm_filter(None))
    log = subprocess.run(
        [require_tool("ffmpeg"), "-hide_banner", *args, "-filter_complex", ";".join(graph),
         "-map", "[aout]", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    return parse_loudnorm(log)


def output_loudness(path: Path) -> float:
    log = subprocess.run(
        [require_tool("ffmpeg"), "-hide_banner", "-i", str(path), "-map", "0:a",
         "-af", loudnorm_filter(None), "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    return float(parse_loudnorm(log)["input_i"])


# ---------------------------------------------------------------- ffmpeg helpers

def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: '{name}' not found on PATH. Install ffmpeg first.")
    return path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffmpeg(*args: str) -> None:
    run([require_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *args])


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [require_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


VIDEO_OUT = ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", str(FPS)]
SCALE = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease," \
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x0b0f17,setsar=1"


def make_card_clip(png: Path, duration: float, out: Path) -> None:
    ffmpeg("-loop", "1", "-framerate", str(FPS), "-t", f"{duration}", "-i", str(png),
           "-vf", SCALE, *VIDEO_OUT, "-an", str(out))


def make_shot_clip(src: Path, slot: float, out: Path) -> FitPlan:
    plan = fit_clip(probe_duration(src), slot, MAX_SHOT_SPEED)
    vf = f"{SCALE},setpts=(PTS-STARTPTS)/{plan.speed:.6f}"
    if plan.pad > 0:
        vf += f",tpad=stop_mode=clone:stop_duration={plan.pad:.3f}"
    ffmpeg("-ss", f"{plan.trim_start:.3f}", "-i", str(src), "-vf", vf,
           "-t", f"{slot:.3f}", *VIDEO_OUT, "-an", str(out))
    return plan


def make_silent_voiceover(out: Path, duration: float = PLACEHOLDER_VOICE_S) -> None:
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{duration}",
           "-c:a", "pcm_s16le", str(out))


# ---------------------------------------------------------------- PNG rendering

PAGE_CSS = """
html, body { margin: 0; width: 1920px; height: 1080px; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.card { width: 100%; height: 100%; background: #0b0f17; color: #fff;
  display: flex; flex-direction: column; align-items: center; justify-content: center; }
.card .big { font-size: 150px; font-weight: 700; letter-spacing: 2px; }
.card .small { font-size: 60px; margin-top: 30px; color: #c9d4e3; }
.sub { position: absolute; left: 0; right: 0; bottom: 56px; text-align: center; }
.sub span { display: inline-block; background: rgba(0,0,0,.72); color: #fff;
  font-size: 50px; font-weight: 600; line-height: 1.3; padding: 12px 34px;
  border-radius: 12px; white-space: pre-line; }
"""


def _page(body: str, transparent: bool) -> str:
    bg = "background: transparent;" if transparent else ""
    return f"<html><head><style>{PAGE_CSS} body {{{bg}}}</style></head><body>{body}</body></html>"


def render_pngs(name: str, team: str, cues: list[Cue], out_dir: Path) -> tuple[dict, list[Path]]:
    from playwright.sync_api import sync_playwright

    cards = {
        "intro": (f'<div class="card"><div class="big">SpaceFuzz</div>'
                  f'<div class="small">{html.escape(name)} · Team {html.escape(team)}</div></div>'),
        "outro": (f'<div class="card"><div class="big" style="font-size:110px">{TAGLINE}</div>'
                  f'<div class="small">SpaceFuzz</div></div>'),
    }
    card_paths, sub_paths = {}, []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        for key, body in cards.items():
            card_paths[key] = out_dir / f"card_{key}.png"
            page.set_content(_page(body, transparent=False))
            page.screenshot(path=str(card_paths[key]))
        for i, cue in enumerate(cues):
            path = out_dir / f"sub_{i:02d}.png"
            page.set_content(_page(f'<div class="sub"><span>{html.escape(cue.text)}</span></div>',
                                   transparent=True))
            page.screenshot(path=str(path), omit_background=True)
            sub_paths.append(path)
        browser.close()
    return card_paths, sub_paths


# ---------------------------------------------------------------- final mix

def final_render(segments: list[Path], cues: list[Cue], sub_pngs: list[Path],
                 voice: Path, tempo: float, music: Path | None, out: Path) -> float:
    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]
    for png in sub_pngs:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{TOTAL_S}", "-i", str(png)]
    voice_idx = len(segments) + len(sub_pngs)
    inputs += ["-i", str(voice)]
    if music:
        inputs += ["-i", str(music)]

    n = len(segments)
    graph = ["".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v0]"]
    for k, cue in enumerate(cues):
        graph.append(f"[v{k}][{n + k}:v]overlay=0:0:"
                     f"enable='gte(t,{cue.start:.3f})*lt(t,{cue.end:.3f})'[v{k + 1}]")
    vlabel = f"[v{len(cues)}]"

    if tempo > 1.0:
        print(f"voiceover longer than video; applying atempo={tempo:.3f}")
        if tempo >= MAX_VOICE_TEMPO and probe_duration(voice) / tempo > TOTAL_S:
            print("warning: voiceover still too long at 1.1x; trim the script (plan 8.3).")
    measured = measure_loudness([voice] + ([music] if music else []), tempo)
    print(f"loudness before: {measured['input_i']} LUFS -> target {LOUDNESS_TARGET_LUFS} LUFS")
    graph += audio_graph(voice_idx, voice_idx + 1 if music else None, tempo,
                         loudnorm_filter(measured))

    ffmpeg(*inputs, "-filter_complex", ";".join(graph), "-map", vlabel, "-map", "[aout]",
           "-t", f"{TOTAL_S}", *VIDEO_OUT, "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", str(out))
    return probe_duration(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=PRESENTER_NAME, help="presenter name as spoken")
    parser.add_argument("--team", default=TEAM_NAME, help="team name as spoken")
    parser.add_argument("--regen-srt", action="store_true",
                        help="overwrite subtitles.srt with the default narration")
    args = parser.parse_args()
    require_tool("ffmpeg")
    require_tool("ffprobe")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    voice = VOICEOVER_PATH
    if not voice.exists():
        voice = BUILD_DIR / "voiceover_silent.wav"
        make_silent_voiceover(voice)
        print(f"no voiceover.mp3; using {PLACEHOLDER_VOICE_S:.0f}s silent placeholder")
    voice_duration = probe_duration(voice)
    tempo = voiceover_tempo(voice_duration, TOTAL_S)
    print(f"voiceover: {voice_duration:.3f}s, atempo {tempo:.4f}")

    texts = [c.text for c in default_cues(args.name, args.team)]
    speech = detect_speech(VOICEOVER_PATH) if VOICEOVER_PATH.exists() else None

    # Regenerate subtitles when asked, when missing, when the voiceover is newer,
    # or when the intro line no longer matches the name/team.
    stale = SRT_PATH.exists() and (
        (VOICEOVER_PATH.exists() and VOICEOVER_PATH.stat().st_mtime > SRT_PATH.stat().st_mtime)
        or [c.text for c in parse_srt(SRT_PATH.read_text(encoding="utf-8"))][:1] != texts[:1])
    if args.regen_srt or not SRT_PATH.exists() or stale:
        cues = default_cues(args.name, args.team)
        if speech:
            cues = align_cues(texts, speech, tempo)
            print("subtitles aligned to voiceover pauses")
        SRT_PATH.write_text(to_srt(cues), encoding="utf-8")
        print(f"wrote {SRT_PATH.relative_to(VIDEO_DIR.parent)}")
    cues = parse_srt(SRT_PATH.read_text(encoding="utf-8"))

    print("rendering title cards and subtitle overlays...")
    cards, sub_pngs = render_pngs(args.name, args.team, cues, BUILD_DIR)

    intro = TIMELINE[0].duration
    if speech:
        intro = first_cue_speech_end(texts, speech, tempo)
        intro = round(min(max(intro, INTRO_MIN_S), INTRO_MAX_S), 3)
    timeline = build_timeline(intro)
    print(f"intro card: {intro:.3f}s (first line ends), {FLEX_SEGMENT} slot "
          f"{next(s.duration for s in timeline if s.name == FLEX_SEGMENT):.3f}s")

    segments = []
    for seg in timeline:
        out = BUILD_DIR / f"seg_{seg.name}.mp4"
        if seg.kind == "card":
            make_card_clip(cards[seg.name], seg.duration, out)
        else:
            src = CLIPS_DIR / f"{seg.name}.mp4"
            if not src.exists():
                sys.exit(f"error: missing {src}. Run python video/record.py first.")
            plan = make_shot_clip(src, seg.duration, out)
            print(f"{seg.name}: speed x{plan.speed:.2f}, cut head {plan.trim_start:.2f}s, "
                  f"hold end {plan.pad:.2f}s")
        segments.append(out)

    music = MUSIC_PATH if MUSIC_PATH.exists() else None

    duration = final_render(segments, cues, sub_pngs, voice, tempo, music, OUTPUT_PATH)
    status = "OK" if duration <= MAX_TOTAL_S else "TOO LONG"
    print(f"exported {OUTPUT_PATH.relative_to(VIDEO_DIR.parent)}: "
          f"{duration:.3f}s (ffprobe) [{status}]")
    print(f"loudness after: {output_loudness(OUTPUT_PATH):.1f} LUFS")
    if duration > MAX_TOTAL_S:
        sys.exit(1)


if __name__ == "__main__":
    main()
