"""NASA Data: how the real data enters the model (cabin air, radiation, stress, clinical).

Each block: what the data does in the model, the dataset with its link (data/SOURCES.md), a
chart in the site palette drawn from committed files only (ui/data_page.py), and the original
analysis figure folded underneath. Runs on a fresh clone without the data/raw downloads.
"""
import streamlit as st

import ui_config
from ui import charts, data_page as dp, theme
from ui.common import esc

CSS = """
<style>
.dat { font-size:14px; line-height:1.5; }
.dat .model { margin-top:4px; }
.dat .model b { color:var(--sf-green); font-weight:600; }
.dat .facts { margin-top:6px; font-size:12.5px; color:var(--sf-muted); }
.dat .src { margin-top:8px; font-size:12.5px; }
.dat .src a { color:var(--sf-info); }
.dat .k { font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--sf-muted); }
</style>
"""


def block_html(b: dict) -> str:
    links = " · ".join(f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(name)}</a>'
                       for name, url in b["links"])
    return (f'<div class="sf-panel dat" data-testid="data-{b["key"]}">'
            f'<div class="sf-h">{esc(b["title"])}</div>'
            f'<div class="model"><b>In the model:</b> {esc(b["model"])}</div>'
            f'<div class="facts">{esc(b["facts"])}</div>'
            f'<div class="src"><span class="k">Dataset</span> {links}</div></div>')


def main() -> None:
    st.set_page_config(page_title="SpaceFuzz · NASA Data", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(theme.page_style() + CSS)
    st.html(f'<div class="sf-topbar" data-testid="data-topbar"><span>NASA Data</span>'
            f'<span>{esc(ui_config.CREW_MEMBER_NAME)}</span><span class="sub">How real data '
            f'enters the model. Charts are drawn from the files committed to the repository; '
            f'all sources and licences: data/SOURCES.md.</span></div>')
    cols = st.columns(2, gap="medium")
    for i, b in enumerate(dp.blocks()):
        with cols[i % 2]:
            st.html(block_html(b))
            fig = b["figure"]()
            if fig is None:
                st.caption(f"The {b['title'].lower()} file is not in this copy of the "
                           f"repository; the numbers above come from the analysis results.")
            else:
                st.plotly_chart(fig, theme=None, config=charts.PLOT_CONFIG,
                                key=f"data_{b['key']}")
            with st.expander(f"Original analysis figure ({b['figure_file'].split('/')[-1]})"):
                st.image(str(dp.ROOT / b["figure_file"]), use_container_width=True)


main()
