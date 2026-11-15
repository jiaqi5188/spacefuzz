"""Assemble the final demo video (manual_v7 11/15 "video", timeline agreed on 11/15).

The cuts follow the real voice-over (video/voiceover_v2.mp3, 29.9 s): each picture sits under
the sentence it illustrates. Website shots come from video/record.py (video/clips/), photos
from video/assets/. Captions, subtitles, the AI voice-over watermark, the photo credit and the
end card are rendered to PNG with Playwright and overlaid with ffmpeg (no libass needed).

Usage:  python video/build_v2.py
Output: video/spacefuzz_final.mp4 (1920x1080, 30 fps, H.264 + AAC, voice at -16 LUFS)
"""
from __future__ import annotations

import html
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
CLIPS = VIDEO_DIR / "clips"
ASSETS = VIDEO_DIR / "assets"
WORK = VIDEO_DIR / "build" / "v2"
VOICE = VIDEO_DIR / "voiceover_v2.mp3"
OUTPUT = VIDEO_DIR / "spacefuzz_final.mp4"
SRT = VIDEO_DIR / "subtitles_v2.srt"
WIDTH, HEIGHT, FPS = 1920, 1080, 30
LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"

WATERMARK = "AI voice-over (ElevenLabs)"
MARS_CREDIT = "Image: NASA/JPL-Caltech/MSSS/TAMU"
END_CARD = ("See the risk before it hits.",
            "An ounce of prevention is worth a pound of cure.",
            "SpaceFuzz · Team Raccoon")
METADATA = {"title": "SpaceFuzz – Team Raccoon – NASA Space Apps 2026",
            "comment": "Voice-over generated with ElevenLabs (AI). "
                       "App code written with AI assistance (Claude Code)."}


@dataclass(frozen=True)
class Segment:
    name: str
    kind: str                    # clip | contain (photo, whole) | cover (photo, filled) | card
    start: float                 # in the video, seconds
    end: float
    src: str = ""
    src_in: float = 0.0          # clips: source window; negative = seconds before its end
    src_out: float = 0.0
    zoom: tuple = (1.0, 1.0)     # slow push-in from .. to
    focus: tuple = (0.5, 0.5)    # where the push-in heads (0..1 of the frame)


# Agreed 11/15: cuts at the sentence boundaries measured in voiceover_v2.mp3.
SEGMENTS = [
    Segment("earth_from_mars", "contain", 0.0, 2.2, "assets/earth_from_mars.jpg"),
    Segment("dashboard_idle", "clip", 2.2, 6.2, "clips/dashboard_idle.mp4", 0.0, 4.0),
    Segment("dashboard_forecast", "clip", 6.2, 14.5, "clips/dashboard_forecast.mp4", 0.0, 8.3),
    # the dashboard holds on 21.0% HIGH while "mostly from sleep debt" is spoken: slow push-in
    # towards the 21.0% and the "Sleep debt 6.9 h" advice card (upper right of the screen)
    Segment("dashboard_hold", "clip", 14.5, 19.8, "clips/dashboard_adjust.mp4", 0.0, 5.0,
            zoom=(1.0, 1.08), focus=(1.0, 0.35)),
    # "reworks it to under five percent": the same dashboard reaches 4.9%, same framing
    Segment("dashboard_adjust", "clip", 19.8, 22.2, "clips/dashboard_adjust.mp4", 5.0, 8.5,
            zoom=(1.08, 1.08), focus=(1.0, 0.35)),
    Segment("surgeon", "clip", 22.2, 23.4, "clips/surgeon.mp4", -1.4, -0.2),
    Segment("engine", "clip", 23.4, 24.6, "clips/engine.mp4", -1.4, -0.2),
    Segment("worker", "cover", 24.6, 27.8, "assets/worker.jpg", zoom=(1.0, 1.04)),
    Segment("end_card", "card", 27.8, 29.9),
]
TOTAL_S = SEGMENTS[-1].end


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


# Caption cards: removed in the 11/15 review (they covered the dashboard's own text wherever
# they were placed); only the subtitles, watermark, photo credit and end card remain.
CAPTIONS: list = []

# The narration (manual_v7 11/15 "video"), timed by the pauses measured in voiceover_v2.mp3
# (silencedetect -38 dB, 0.18 s). The last line is the end card itself, so no subtitle there.
SUBTITLES = [
    Cue(0.00, 2.02, "On Mars, help is twenty minutes away."),
    Cue(2.47, 5.10, "Most monitors warn you only when something is already wrong."),
    Cue(5.47, 7.88, "SpaceFuzz predicts how today’s plan may harm your body."),
    Cue(8.25, 12.00, "Inspired by fuzzing in cybersecurity, it attacks an astronaut’s digital twin"),
    Cue(12.00, 14.29, "with thousands of what-if events from real NASA data,"),
    Cue(14.57, 16.34, "to find the weak points first and prevent them."),
    Cue(16.75, 19.80, "Today’s plan scores twenty-one percent risk, mostly from sleep debt."),
    Cue(20.25, 22.19, "SpaceFuzz reworks it to under five percent."),
    Cue(22.55, 24.50, "Most importantly, it’s not just a space tool:"),
    Cue(24.81, 28.09, "we can help the workers building Singapore under the tropical sun "
                      "with the same approach."),
]


def duration(s: Segment) -> float:
    return round(s.end - s.start, 3)


def speed(s: Segment) -> float:
    """Source seconds per video second (clips only)."""
    return round(abs(s.src_out - s.src_in) / duration(s), 3)


# ---------------------------------------------------------------- ffmpeg helpers

def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stderr


def probe(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return float(out.strip())


def push_in(z0: float, z1: float, frames: int, focus: tuple) -> str:
    """zoompan on an upscaled frame (less jitter), easing from z0 to z1 over the segment."""
    z = f"{z0}+({z1}-{z0})*on/{max(frames - 1, 1)}"
    fx, fy = focus
    return (f"scale={WIDTH * 2}:{HEIGHT * 2},zoompan=z='{z}':x='(iw-iw/zoom)*{fx}'"
            f":y='(ih-ih/zoom)*{fy}':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")


# every segment the same for concat: square pixels (the 1512x850 page scaled to 1080p carries
# a 1701:1700 aspect) and limited-range yuv420p (JPEG photos come in as yuvj420p)
UNIFORM = ",setsar=1,format=yuv420p"
ENCODE = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
          "-r", str(FPS), "-an"]


def render_segment(s: Segment, card_png: Path, out: Path) -> None:
    d = duration(s)
    frames = round(d * FPS)
    if s.kind == "clip":
        src = VIDEO_DIR / s.src
        a, b = s.src_in, s.src_out
        if a < 0:
            total = probe(src)
            a, b = total + a, total + b
        stretch = d / (b - a)
        vf = f"setpts=(PTS-STARTPTS)*{stretch:.5f},fps={FPS}"
        if s.zoom != (1.0, 1.0):
            vf += "," + push_in(*s.zoom, frames, s.focus)
        run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(src), "-vf", vf + UNIFORM,
             "-frames:v", str(frames), *ENCODE, str(out)])
        return
    if s.kind == "card":
        img, vf = card_png, f"fps={FPS}"
    elif s.kind == "contain":       # the whole photo on black (space)
        img = VIDEO_DIR / s.src
        vf = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
              f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,fps={FPS}")
    else:                           # cover: fill the frame, slow push-in
        img = VIDEO_DIR / s.src
        vf = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
              f"crop={WIDTH}:{HEIGHT},fps={FPS},{push_in(*s.zoom, frames, s.focus)}")
    run(["ffmpeg", "-y", "-loop", "1", "-t", f"{d:.3f}", "-i", str(img), "-vf", vf + UNIFORM,
         "-frames:v", str(frames), *ENCODE, str(out)])


# ---------------------------------------------------------------- overlays (Playwright)

CSS = """
html, body { margin:0; width:1920px; height:1080px; background:transparent;
  font-family:"Helvetica Neue", Helvetica, Arial, sans-serif; }
/* under the top bar (Mars Base, delay); small enough to sit on a panel heading and clear the
   role screens' 21.0% just below it */
.cap { position:absolute; left:28px; top:94px; }
.cap span { display:inline-block; background:#40d317; color:#05070a; font-size:24px;
  font-weight:700; padding:4px 14px; border-radius:7px; box-shadow:0 4px 16px rgba(0,0,0,.6); }
.sub { position:absolute; left:210px; right:210px; bottom:44px; text-align:center; }
.sub span { display:inline-block; background:rgba(0,0,0,.74); color:#fff; font-size:40px;
  font-weight:500; line-height:1.3; padding:8px 24px; border-radius:10px; }
.mark { position:absolute; right:24px; bottom:12px; color:rgba(230,237,243,.78);
  font-size:22px; text-shadow:0 0 6px #000; }
.credit { position:absolute; left:24px; bottom:12px; color:rgba(230,237,243,.82);
  font-size:22px; text-shadow:0 0 6px #000; }
.card { width:1920px; height:1080px; background:#05070a; color:#e6edf3; display:flex;
  flex-direction:column; align-items:center; justify-content:center; }
.card .big { font-size:104px; font-weight:700; color:#40d317; }
.card .small { font-size:44px; margin-top:34px; color:#c9d4e3; }
.card .team { font-size:36px; margin-top:22px; color:#8b949e; letter-spacing:2px; }
"""


def render_overlays(out_dir: Path) -> dict:
    from playwright.sync_api import sync_playwright
    e = html.escape
    pages = {"watermark": f'<div class="mark">{e(WATERMARK)}</div>',
             "credit": f'<div class="credit">{e(MARS_CREDIT)}</div>',
             "card": (f'<div class="card"><div class="big">{e(END_CARD[0])}</div>'
                      f'<div class="small">{e(END_CARD[1])}</div>'
                      f'<div class="team">{e(END_CARD[2])}</div></div>')}
    for i, c in enumerate(CAPTIONS):
        pages[f"cap{i}"] = f'<div class="cap"><span>{e(c.text)}</span></div>'
    for i, c in enumerate(SUBTITLES):
        pages[f"sub{i}"] = f'<div class="sub"><span>{e(c.text)}</span></div>'
    paths = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        for key, body in pages.items():
            page.set_content(f"<html><head><style>{CSS}</style></head><body>{body}</body></html>")
            paths[key] = out_dir / f"{key}.png"
            page.screenshot(path=str(paths[key]), omit_background=key != "card")
        browser.close()
    return paths


def srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt() -> None:
    cues = SUBTITLES + [Cue(28.43, TOTAL_S, END_CARD[0])]
    SRT.write_text("\n".join(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n"
                             for i, c in enumerate(cues, 1)), encoding="utf-8")


# ---------------------------------------------------------------- assemble

def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    png = render_overlays(WORK)
    parts = []
    for s in SEGMENTS:
        out = WORK / f"{s.name}.mp4"
        render_segment(s, png["card"], out)
        parts.append(out)
        print(f"{s.name:20s} {s.start:5.2f} -> {s.end:5.2f} s"
              + (f"  (x{speed(s):.2f})" if s.kind == "clip" else ""))

    overlays = ([("watermark", 0.0, TOTAL_S), ("credit", 0.0, SEGMENTS[0].end)]
                + [(f"cap{i}", c.start, c.end) for i, c in enumerate(CAPTIONS)]
                + [(f"sub{i}", c.start, c.end) for i, c in enumerate(SUBTITLES)])
    inputs = []
    for p in parts:
        inputs += ["-i", str(p)]
    for key, _, _ in overlays:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{TOTAL_S}", "-i", str(png[key])]
    inputs += ["-i", str(VOICE)]
    n = len(parts)
    graph = ["".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v0]"]
    for k, (_, a, b) in enumerate(overlays):
        graph.append(f"[v{k}][{n + k}:v]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'[v{k + 1}]")
    graph.append(f"[{n + len(overlays)}:a]{LOUDNESS},aresample=48000[aout]")
    meta = sum((["-metadata", f"{k}={v}"] for k, v in METADATA.items()), [])
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(graph),
         "-map", f"[v{len(overlays)}]", "-map", "[aout]", "-t", f"{TOTAL_S}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", *meta, "-movflags", "+faststart",
         str(OUTPUT)])
    write_srt()

    log = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(OUTPUT), "-af",
               "ebur128=peak=true", "-f", "null", "-"])
    lufs = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    peak = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
    info = {"duration_s": round(probe(OUTPUT), 2), "loudness_lufs": lufs[-1] if lufs else None,
            "true_peak_dbfs": peak[-1] if peak else None}
    print(json.dumps(info))


if __name__ == "__main__":
    main()
