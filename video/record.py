"""Record the demo shots from the running SpaceFuzz app (app.py) with Playwright.

Shots follow the 30-second storyboard (docs/manual_v7.md, 11/15 "video"): the top bar, the
forecast, the plan adjustment, the crash-boundary heatmap and the midday check. Each shot
waits on the app's top-level completion markers (data-testid forecast-done, playback-done,
charts-done, midday-done) instead of fixed sleeps, and never reaches into the animated
panels (they sit one iframe deeper, inside ui/live_frame); only the deliberate "camera
hold" after a shot is timed. The session is recorded once, then cut into
video/clips/<shot>.mp4 (git-ignored) with the time windows in video/clips/shots.json.

Before recording, one unrecorded session warms the server up (the first forecast after a
server start loads the libraries and data, about 20 s).

Usage:  python video/record.py [--url http://localhost:8501] [--app app.py]
If nothing is listening at --url, the app is started automatically at the demo pace
(SPACEFUZZ_PACE=demo). If you start it yourself, use:
    SPACEFUZZ_PACE=demo .venv/bin/streamlit run app.py --server.port 8501
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
WIDTH, HEIGHT, FPS = 1920, 1080, 30
DEVICE_SCALE_FACTOR = 2  # render at 2x, downscaled into the 1080p video for crisper text
ACTION_TIMEOUT_MS = 120_000               # a replay at the demo pace takes about 30 s
SERVER_START_TIMEOUT_S = 60


@dataclass
class Shot:
    name: str
    action: Callable  # (page, url) -> None, runs while the shot is "rolling"


def shot_topbar(page, url) -> None:
    """0-6 s: the idle dashboard (Mars Base, Cmdr. L. Tan, Earth delay 20 min)."""
    page.wait_for_timeout(6000)


def shot_forecast(page, url) -> None:
    """6-13 s: Run Forecast -> fuzz log, body map scanning -> 21.0% HIGH."""
    page.get_by_role("button", name="Run Forecast").click()
    page.get_by_test_id("forecast-done").wait_for(state="attached")
    page.wait_for_timeout(1500)  # camera hold on the risk number and the red heart


def shot_adjust(page, url) -> None:
    """16-22 s: the plan board slides to the adjusted plan, the risk rolls to 4.9%."""
    page.get_by_test_id("playback-done").wait_for(state="attached")
    page.wait_for_timeout(1500)  # camera hold on the adjusted plan


def shot_heatmap(page, url) -> None:
    """13-16 s: the crash-boundary heatmap ("15 min less sleep turns this red")."""
    page.get_by_test_id("charts-done").wait_for(state="attached")
    page.locator(".st-key-heatmap_chart").scroll_into_view_if_needed()
    page.wait_for_timeout(3000)


def shot_midday(page, url) -> None:
    """22-25 s: the midday check, 100% CRITICAL -> 4.9% (afternoon EVA moved to 18:00)."""
    page.goto(url.rstrip("/") + "/Midday_check")
    page.get_by_role("button", name="Run midday forecast").click()
    page.get_by_test_id("midday-done").wait_for(state="attached")
    page.get_by_role("button", name="Run midday forecast").wait_for(state="visible")  # unlocked
    page.wait_for_timeout(1500)


SHOTS = [
    Shot("topbar", shot_topbar),
    Shot("forecast", shot_forecast),
    Shot("adjust", shot_adjust),
    Shot("heatmap", shot_heatmap),
    Shot("midday", shot_midday),
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
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-vf", f"fps={FPS}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-an", str(out)],
        check=True,
    )


def context_options(raw_dir: Path) -> dict:
    return {
        "viewport": {"width": WIDTH, "height": HEIGHT},
        "device_scale_factor": DEVICE_SCALE_FACTOR,
        "record_video_dir": str(raw_dir),
        "record_video_size": {"width": WIDTH, "height": HEIGHT},
        "color_scheme": "dark",
    }


def dashboard_url(url: str) -> str:
    return f"{url.rstrip('/')}/?scenario={SCENARIO}"


def warm_up(browser, url: str) -> None:
    """One unrecorded forecast, so the recorded one is as fast as on a warm server."""
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
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
