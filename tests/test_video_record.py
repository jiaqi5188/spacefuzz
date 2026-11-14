"""Tests for video/record.py: browser settings, the storyboard shots and the server it starts.
The recording itself needs a browser and a running app, so it is checked by hand."""
import inspect
from pathlib import Path

from video import record


def test_context_options_are_1080p_at_2x():
    opts = record.context_options(Path("raw"))
    assert opts["viewport"] == {"width": 1920, "height": 1080}
    assert opts["device_scale_factor"] == 2
    assert opts["record_video_size"] == {"width": 1920, "height": 1080}
    assert opts["record_video_dir"] == "raw"


def test_shots_follow_the_storyboard_of_the_real_app():
    """docs/manual_v7.md, 11/15 video: top bar, forecast, adjustment, heatmap, midday."""
    assert [s.name for s in record.SHOTS] == ["topbar", "forecast", "adjust", "heatmap", "midday"]
    src = inspect.getsource(record)
    assert "demo_stub" not in src and "Adjust Plan" not in src and "data-heart" not in src
    # waits only on top-level markers, never inside the panels' nested iframes
    for marker in ("forecast-done", "playback-done", "charts-done", "midday-done"):
        assert marker in src
    assert "frame_locator" not in src


def test_server_defaults_to_the_app_at_demo_pace(monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, cmd, cwd=None, env=None, **kw):
            seen.update(cmd=cmd, env=env)
    monkeypatch.setattr(record.subprocess, "Popen", FakePopen)
    record.start_server(record.ROOT / record.DEFAULT_APP, 8599)
    assert record.DEFAULT_APP == "app.py" and seen["cmd"][-1] != "demo_stub.py"
    assert str(record.ROOT / "app.py") in seen["cmd"]
    assert seen["env"]["SPACEFUZZ_PACE"] == "demo"


def test_clips_go_to_a_git_ignored_folder():
    ignore = (record.ROOT / ".gitignore").read_text(encoding="utf-8")
    assert record.CLIPS_DIR == record.ROOT / "video" / "clips"
    assert "video/clips/" in ignore
