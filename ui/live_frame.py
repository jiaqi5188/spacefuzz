"""Live frame: st.iframe for a whole HTML document, but without the blank flash on update.

st.iframe(srcdoc) reloads the frame whenever the document changes, so the panel is empty for
a frame or two; with three panels changing at once the whole page seemed to flash. This static
component (ui/live_frame/index.html) loads a new document into a hidden frame and swaps it in
after it has loaded; an unchanged document is not touched.

Playwright: the document sits one level down, so use
page.frame_locator(OUTER).frame_locator('iframe[srcdoc*="sf-body-map"]'), or frame_of().
"""
from pathlib import Path

import streamlit.components.v1 as components

OUTER = 'iframe[title*="live_frame"]'
_component = components.declare_component(
    "live_frame", path=str(Path(__file__).with_name("live_frame")))


def show(doc: str, height: int, key: str) -> None:
    """Draw `doc` at a fixed height. `key` keeps the same browser frame across reruns."""
    _component(doc=doc, height=height, key=key, default=None)
