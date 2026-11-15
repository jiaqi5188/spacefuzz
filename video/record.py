"""Record the website shots of the demo video with Playwright, from the running app (app.py).

Shots follow the storyboard rewritten on 11/15 (docs/manual_v7.md, 11/15 "video"); the two
photos and the end card are added when the video is assembled, not recorded here:
  dashboard_idle      0-8 s     idle dashboard (menu closed), then Run Forecast is clicked
  dashboard_forecast  8-16.5 s  fuzz log and body-map scan -> 21.0% HIGH, cardiovascular red
  dashboard_adjust    16.5-19 s  the same dashboard on to the adjusted plan, 21.0% -> 4.9%
  surgeon             16.5-21.5 s  flight surgeon screen, ?view=surgeon&autoplay=1, to the result
  engine              19.5-22 s    engine screen, ?view=engine&autoplay=1, fuzz log filled
Each shot waits on the app's top-level markers (forecast-done, playback-done) instead of fixed
sleeps and never reaches into the animated panels (one iframe deeper, inside ui/live_frame);
only the "camera hold" after a shot is timed. The session is recorded once, then cut into
video/clips/<shot>.mp4 (git-ignored) with the time windows in video/clips/shots.json.

Before recording, one unrecorded session warms the server up (the first forecast after a
server start loads the libraries and data, about 20 s).

Usage:  python video/record.py [--url http://localhost:8501] [--app app.py]
If nothing is listening at --url, the app is started automatically at the demo pace
(SPACEFUZZ_PACE=demo). The 11/15 clips were recorded at the everyday "fast" pace (an app
started without SPACEFUZZ_PACE): the forecast then fits the 8.5 s of its segment.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "video" / "build" / "raw"
CLIPS_DIR = ROOT / "video" / "clips"
DEFAULT_APP = "app.py"
SCENARIO = "main_sleep_debt_eva"          # the video tells the main story only
# the page at 1512x850 (the 14-inch layout the UI was tuned for, 16:9), 2x pixel density,
# scaled into the 1920x1080 video (manual_v7 11/15 video): fuller screens, larger text
PAGE_W, PAGE_H = 1512, 850
WIDTH, HEIGHT, FPS = 1920, 1080, 30
DEVICE_SCALE_FACTOR = 2
ACTION_TIMEOUT_MS = 120_000               # a replay at the demo pace takes about 30 s
SERVER_START_TIMEOUT_S = 60


@dataclass
class Shot:
    name: str
    action: Callable  # (page, url) -> None, runs while the shot is "rolling"


IDLE_HOLD_MS = 4000       # idle dashboard on screen before the click (the photo comes first)
RESULT_HOLD_MS = 2000     # camera hold on a result


def shot_dashboard_idle(page, url) -> None:
    """0-8 s (after the photo): the idle dashboard, menu closed, then Run Forecast."""
    page.wait_for_timeout(IDLE_HOLD_MS)
    page.get_by_role("button", name="Run Forecast").click()


def shot_dashboard_forecast(page, url) -> None:
    """8-16.5 s: fuzz log and body-map scan -> 21.0% HIGH, the heart turns red."""
    page.get_by_test_id("forecast-done").wait_for(state="attached")
    page.wait_for_timeout(RESULT_HOLD_MS)


def shot_dashboard_adjust(page, url) -> None:
    """16.5-19 s: the same dashboard goes on: plan board slides, risk rolls 21.0% -> 4.9%,
    the advice and before/after fill in; ends on the final picture (button unlocked)."""
    page.get_by_test_id("playback-done").wait_for(state="attached")
    page.get_by_role("button", name="Run Forecast", exact=True).wait_for(state="visible")
    page.wait_for_timeout(RESULT_HOLD_MS)


def _view(page, url: str, view: str) -> None:
    """A role screen on its own, playing once by itself (?autoplay=1), to its final picture."""
    page.goto(f"{url.rstrip('/')}/?scenario={SCENARIO}&view={view}&autoplay=1")
    page.get_by_test_id("playback-done").wait_for(state="attached")
    # the button unlocks (and the engine screen's charts appear) once the board has rolled:
    # end on that clean final picture, not on "Running forecast... 92%"
    page.get_by_role("button", name="Run Forecast", exact=True).wait_for(state="visible")
    page.wait_for_timeout(RESULT_HOLD_MS)


def shot_surgeon(page, url) -> None:
    """16.5-21.5 s: the flight surgeon screen, 21.0% -> 4.9% after re-planning, re-tested."""
    _view(page, url, "surgeon")      # view=surgeon&autoplay=1


def shot_engine(page, url) -> None:
    """19.5-22 s: the engine screen with the fuzz log filled (one engine, several screens)."""
    _view(page, url, "engine")       # view=engine&autoplay=1


SHOTS = [
    Shot("dashboard_idle", shot_dashboard_idle),
    Shot("dashboard_forecast", shot_dashboard_forecast),
    Shot("dashboard_adjust", shot_dashboard_adjust),
    Shot("surgeon", shot_surgeon),
    Shot("engine", shot_engine),
]


def server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/_stcore/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start_server(app: Path, port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    env = {**os.environ, "SPACEFUZZ_PACE": "demo"}
    return subprocess.Popen(cmd, cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cut_clip(src: Path, start: float, end: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-an", str(out)],
        check=True,
    )


def context_options(raw_dir: Path) -> dict:
    return {
        "viewport": {"width": PAGE_W, "height": PAGE_H},
        "device_scale_factor": DEVICE_SCALE_FACTOR,
        "record_video_dir": str(raw_dir),
        # Playwright records the page at its CSS size and pads (never scales) a larger frame,
        # so the video is the page size; cut_clip scales it to 1920x1080
        "record_video_size": {"width": PAGE_W, "height": PAGE_H},
        "color_scheme": "dark",
    }


def dashboard_url(url: str) -> str:
    return f"{url.rstrip('/')}/?scenario={SCENARIO}"


def warm_up(browser, url: str) -> None:
    """One unrecorded forecast, so the recorded one is as fast as on a warm server."""
    page = browser.new_page(viewport={"width": PAGE_W, "height": PAGE_H})
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.goto(dashboard_url(url))
    page.get_by_role("button", name="Run Forecast").click()
    page.get_by_test_id("forecast-done").wait_for(state="attached")
    page.close()


def record(url: str) -> list[dict]:
    from playwright.sync_api import sync_playwright

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        warm_up(browser, url)
        context = browser.new_context(**context_options(RAW_DIR))
        context.set_default_timeout(ACTION_TIMEOUT_MS)
        page = context.new_page()
        t0 = time.monotonic()  # video timeline starts with the page
        page.goto(dashboard_url(url))
        page.get_by_test_id("topbar").wait_for(state="visible")
        page.get_by_test_id("trends").wait_for(state="attached")
        page.wait_for_timeout(1500)  # let the panels finish their first paint
        for shot in SHOTS:
            start = time.monotonic() - t0
            shot.action(page, url)
            end = time.monotonic() - t0
            windows.append({"name": shot.name, "start": round(start, 3), "end": round(end, 3)})
            print(f"{shot.name}: {start:.2f}s -> {end:.2f}s ({end - start:.1f} s)")
        video = page.video
        context.close()  # finalises the .webm
        raw_path = Path(video.path())
        browser.close()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for w in windows:
        cut_clip(raw_path, w["start"], w["end"], CLIPS_DIR / f"{w['name']}.mp4")
    (CLIPS_DIR / "shots.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--app", default=DEFAULT_APP, help="app to start if none is running")
    args = parser.parse_args()

    server = None
    if not server_up(args.url):
        port = urllib.parse.urlparse(args.url).port or 8501
        print(f"starting {args.app} on port {port} at the demo pace...")
        server = start_server(ROOT / args.app, port)
        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        while not server_up(args.url):
            if time.monotonic() > deadline or server.poll() is not None:
                server.kill()
                sys.exit("error: Streamlit server did not start")
            time.sleep(0.5)
    else:
        print("using the app already running at", args.url,
              "(start it with SPACEFUZZ_PACE=demo for the video pace)")
    try:
        record(args.url)
        print(f"clips written to {CLIPS_DIR.relative_to(ROOT)}")
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
