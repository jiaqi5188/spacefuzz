"""Tests for video/record.py: browser settings, the storyboard shots and the server it starts.
The recording itself needs a browser and a running app, so it is checked by hand."""
import inspect
from pathlib import Path

from video import record


def test_context_options_are_the_14_inch_page_scaled_to_1080p():
    """manual_v7 11/15 video: the page at 1512x850 (the 14-inch layout the UI was tuned for),
    2x pixel density, scaled into a 1920x1080 video."""
    opts = record.context_options(Path("raw"))
    assert opts["viewport"] == {"width": 1512, "height": 850}
    assert opts["device_scale_factor"] == 2
    # recorded at the page size (Playwright pads, never scales, a larger frame), scaled to
    # 1080p when cut
    assert opts["record_video_size"] == {"width": 1512, "height": 850}
    import inspect as _i
    assert (record.WIDTH, record.HEIGHT) == (1920, 1080)
    assert "scale={WIDTH}:{HEIGHT}" in _i.getsource(record.cut_clip)
    assert opts["record_video_dir"] == "raw"


def test_shots_follow_the_storyboard_of_the_real_app():
    """docs/manual_v7.md, 11/15 video (rewritten 11/15): the idle dashboard and Run Forecast,
    the forecast up to 21.0% HIGH, then the flight surgeon and engine screens, each recorded on
    its own with ?view=...&autoplay=1. Photos and the end card are added when the video is
    assembled, not recorded here."""
    assert [s.name for s in record.SHOTS] == ["dashboard_idle", "dashboard_forecast",
                                              "dashboard_adjust", "surgeon", "engine"]
    src = inspect.getsource(record)
    assert "demo_stub" not in src and "Adjust Plan" not in src and "data-heart" not in src
    assert "view=surgeon&autoplay=1" in src and "view=engine&autoplay=1" in src
    # waits only on top-level markers, never inside the panels' nested iframes
    for marker in ("forecast-done", "playback-done"):
        assert marker in src
    assert "frame_locator" not in src
    assert "stExpandSidebarButton" not in src            # the menu stays closed on camera


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
