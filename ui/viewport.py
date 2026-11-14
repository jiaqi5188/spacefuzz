"""Is the browser window wide (>= ui_config.WIDE_MIN_PX)? A static, zero-height component.

Returns None until the browser has answered (first run), then True / False. The page
switches to the four-column layout only for True; ?wide=1 / ?wide=0 in the URL overrides
the probe (screenshots, tests).
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import ui_config

_component = components.declare_component(
    "viewport", path=str(Path(__file__).with_name("viewport")))


def is_wide() -> bool | None:
    forced = st.query_params.get("wide")
    if forced in ("0", "1"):
        return forced == "1"
    return _component(min_px=ui_config.WIDE_MIN_PX, key="viewport", default=None)
