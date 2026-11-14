"""The palette for iframes: the :root token block of ui/styles.css (single source)."""
import functools
import json
import re
from pathlib import Path

CSS_PATH = Path(__file__).with_name("styles.css")


@functools.lru_cache(maxsize=1)
def root_css() -> str:
    """':root { --sf-...; }' from styles.css, to prepend to an iframe's own CSS."""
    return re.search(r":root\s*\{.*?\}", CSS_PATH.read_text(), re.S).group(0)


@functools.lru_cache(maxsize=1)
def tip_css() -> str:
    """The (i) tip rules of styles.css (between tip:start and tip:end), for the iframes."""
    text = CSS_PATH.read_text()
    return text[text.index("/* tip:start"):text.index("/* tip:end */")]


def page_style() -> str:
    """The global stylesheet (shared panel classes included) for pages other than app.py,
    whose top bar sits directly above content (no navigation row)."""
    return f"<style>{CSS_PATH.read_text()}.sf-topbar {{ margin-bottom:12px; }}</style>"


def page_link(target, path: str, label: str, icon: str | None = None) -> None:
    """st.page_link that degrades to a caption when the multipage registry is absent
    (a page run on its own, e.g. in AppTest); in the app it is a normal link."""
    try:
        target.page_link(path, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - StreamlitAPIException / missing page registry
        target.caption(label)


def script_json(obj) -> str:
    """JSON for an inline <script>: "</" is escaped so no text can close the tag."""
    return json.dumps(obj).replace("</", "<\\/")
