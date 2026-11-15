"""Sidebar menu (st.navigation): every screen one click away, in groups; the indicator
reference comes last, like a wiki. Pages link to each other with the same Page objects
(a page defined by a callable can only be linked through its Page)."""
import streamlit as st

import ui_config


def _dashboard(view: str | None = None):
    def page() -> None:
        import app                  # the entry script as a module: its main(), not its router
        app.main(view)
    page.__name__ = f"view_{view}" if view else "dashboard"
    return page


# (group, title, icon, url path, source): source is a page file, or a dashboard view (None =
# the full dashboard). Old page URLs (/Morning_check-in, /Midday_check, /Indicators) are kept.
MENU = [
    ("Mission", "Dashboard", "🛰️", "dashboard", None),
    ("Daily checks", "Morning check-in", "📝", "Morning_check-in", "page_files/1_Morning_check-in.py"),
    ("Daily checks", "Midday check", "🕑", "Midday_check", "page_files/3_Midday_check.py"),
    *[("Role screens", title, icon, v, v) for v, (title, icon) in ui_config.VIEW_PAGES.items()],
    ("Reference", "NASA Data", "🛰", "Data", "page_files/4_Data.py"),
    ("Reference", "Indicators (wiki)", "📖", "Indicators", "page_files/2_Indicators.py"),
]


def _page(title: str, icon: str, url_path: str, source: str | None) -> st.Page:
    if source is None or source in ui_config.VIEWS:
        return st.Page(_dashboard(source), title=title, icon=icon, url_path=url_path,
                       default=source is None)
    return st.Page(source, title=title, icon=icon, url_path=url_path)


def dashboard() -> st.Page:
    return _page(*MENU[0][1:])


def pages() -> dict:
    groups: dict = {}
    for group, *page in MENU:
        groups.setdefault(group, []).append(_page(*page))
    return groups


def link_dashboard(target=st, label: str = "Dashboard") -> None:
    """Link back to the dashboard (in-app: the session and check-in edits are kept)."""
    try:
        target.page_link(dashboard(), label=label, icon="🛰️")
    except Exception:  # noqa: BLE001 - no navigation registry (a page run on its own in AppTest)
        target.caption(label)
