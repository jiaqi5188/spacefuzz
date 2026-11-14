"""Tests for ui/styles.css: one palette, in sync with ui_config and the Streamlit theme."""
import os
import re
import tomllib
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "ui" / "styles.css").read_text(encoding="utf-8")


def tokens() -> dict:
    root = re.search(r":root\s*\{(.*?)\}", CSS, re.S).group(1)
    return {k: v.strip().lower() for k, v in re.findall(r"--(sf-[\w-]+)\s*:\s*([^;]+);", root)}


def test_palette_tokens_present():
    t = tokens()
    for name in ("bg", "panel", "panel-2", "line", "grid", "ink", "muted",
                 "green", "yellow", "red", "grey"):
        assert f"sf-{name}" in t, name
        assert re.fullmatch(r"#[0-9a-f]{6}", t[f"sf-{name}"]), name


def test_palette_matches_ui_config_and_streamlit_theme():
    t = tokens()
    for c in ("green", "yellow", "red", "grey"):
        assert t[f"sf-{c}"] == ui_config.COLOR[c].lower(), c
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    assert t["sf-bg"] == theme["backgroundColor"].lower()
    assert t["sf-panel-2"] == theme["secondaryBackgroundColor"].lower()
    assert t["sf-ink"] == theme["textColor"].lower()
    assert t["sf-green"] == theme["primaryColor"].lower()


def test_status_has_a_shape_not_only_a_colour():
    # green vs yellow is only dE 6.1 for protanopia: indicator dots need a second cue
    for status in ("yellow", "red", "grey"):
        assert re.search(rf'\[data-status="{status}"\]\s*\.sf-dot', CSS), status


def test_streamlit_controls_styled_and_offline():
    for sel in ('stBaseButton-primary', 'stCheckbox', 'stButtonGroup'):
        assert sel in CSS, sel
    assert "prefers-reduced-motion" in CSS
    assert "@import" not in CSS and "http" not in CSS        # no fonts / assets from the web


def test_app_loads_the_stylesheet():
    from tests.test_app import fake_results
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["results"] = {"main_sleep_debt_eva": fake_results()}
    at.run()
    assert not at.exception
    assert any("--sf-green" in h.proto.body for h in at.get("html"))   # style-only st.html


def test_every_named_colour_in_ui_config_is_a_css_token():
    t = tokens()
    for key, value in ui_config.SURFACE.items():
        assert t["sf-" + key.replace("_", "-")] == value.lower(), key


def test_no_hard_coded_colours_in_modules():
    """Colours come from the palette (styles.css tokens / ui_config), never literal hex."""
    offenders = []
    for f in [ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py"))]:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"#[0-9a-fA-F]{3,8}\b", line.split("  # ")[0]):
                offenders.append(f"{f.name}:{n}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)


def test_iframes_carry_the_palette():
    from ui import theme
    root = theme.root_css()
    assert root.startswith(":root") and "--sf-green" in root and "--sf-block-eva" in root
