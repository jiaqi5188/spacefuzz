"""Indicators (plan 4.1 annex 2): what each indicator is based on, from indicator_sources.md.

Every explanation, source and link is read from data/indicator_sources.md (ui/indicator_doc.py);
this page writes no second version. For each indicator: where the value comes from, the NASA
HRP risk, how well that risk supports the indicator (Direct / Indirect; "Not labelled" when the
file gives neither), the links, whether a team member has checked the source page, and the
thresholds SpaceFuzz acts on (from config.py, illustrative).
"""
import re

import streamlit as st

import ui_config
from ui import indicator_doc as doc
from ui import theme
from ui.common import esc

KIND_NOTE = {"Direct": "the risk is about this indicator",
             "Indirect": "closest available risk; the reason is in the text",
             "Not labelled": "the source file gives no Direct / Indirect label"}
KIND_MARK = {"Direct": "●", "Indirect": "◐", "Not labelled": "○"}   # shape, not colour alone



SITE_TEXT = ((" -> ", " → "), ("uGy", "µGy"), ("degC", "°C"))


def site_text(text: str) -> str:
    """The source file is plain ASCII; on screen use the site's arrow and unit symbols."""
    for a, b in SITE_TEXT:
        text = text.replace(a, b)
    return text


def md_inline(text: str) -> str:
    """The file's Markdown links / italics / code as HTML (the file is our own, trusted)."""
    t = esc(site_text(text))
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", t)


def card(r: dict) -> str:
    acts = "".join(f"<div>• {esc(a)}</div>" for a in doc.thresholds(r["code_field"]))
    checked = r["status"] == "checked"
    return (
        f'<div class="sf-panel ind" data-match="{r["match_kind"]}" data-status="{esc(r["status"])}">'
        f'<div class="row"><b class="name">{esc(r["indicator"])}</b>'
        f'<span class="unit">{esc(site_text(r["unit"]))}</span></div>'
        f'<div class="badges"><span class="badge {r["match_kind"].replace(" ", "-").lower()}">'
        f'{KIND_MARK[r["match_kind"]]} {r["match_kind"]}</span>'
        f'<span class="badge {"checked" if checked else "unchecked"}">source '
        f'{"checked" if checked else "unchecked"}</span></div>'
        f'<div class="k">Where it comes from</div><div>{md_inline(r["source"])}</div>'
        f'<div class="k">NASA HRP risk</div><div>{md_inline(r["risk"])}</div>'
        f'<div class="k">How well the risk supports it ({esc(KIND_NOTE[r["match_kind"]])})</div>'
        f'<div>{md_inline(r["match"])}</div>'
        f'<div class="k">Links</div><div>{md_inline(r["links"])}</div>'
        + (f'<div class="k">When SpaceFuzz acts</div>{acts}' if acts else "")
        + "</div>")


CSS = """
<style>
.ind { font-size:13px; line-height:1.45; }
.ind .row { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
.ind .name { font-size:16px; }
.ind .unit { color:var(--sf-muted); font-size:12px; }
.ind .k { margin-top:8px; font-size:11px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--sf-muted); }
.ind a { color:var(--sf-info); }
.ind code { font-size:12px; }
.badges { margin-top:4px; display:flex; gap:6px; flex-wrap:wrap; }
.badge { font-size:11px; border:1px solid var(--sf-line-strong); border-radius:10px;
  padding:1px 8px; color:var(--sf-ink); }
.badge.direct { border-color:var(--sf-green); }
.badge.indirect { border-color:var(--sf-yellow); }
.badge.unchecked { color:var(--sf-muted); }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="SpaceFuzz · Indicators", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(theme.page_style() + CSS)
    st.html(f'<div class="sf-topbar" data-testid="indicators-topbar"><span>Indicators</span>'
            f'<span>{esc(ui_config.CREW_MEMBER_NAME)}</span><span class="sub">What each '
            f'indicator is based on. Text, sources and links are read from '
            f'data/indicator_sources.md; thresholds from config.py are illustrative.</span></div>')
    rows = doc.rows()
    kinds = {k: sum(r["match_kind"] == k for r in rows) for k in KIND_NOTE}
    checked = sum(r["status"] == "checked" for r in rows)
    st.caption(f"{len(rows)} indicators: {kinds['Direct']} Direct, {kinds['Indirect']} Indirect, "
               f"{kinds['Not labelled']} Not labelled; sources checked by the team: {checked} "
               f"of {len(rows)}.")
    cols = st.columns(2, gap="small")
    for i, r in enumerate(rows):
        with cols[i % 2]:
            st.html(card(r))
    st.html('<div class="sf-panel ind"><div class="sf-h">Context</div>'
            + "".join(f"<div style='margin-top:6px'>• {md_inline(' '.join(n.split()))}</div>"
                      for n in doc.context()) + "</div>")
    with st.expander("Source excerpts (quotes behind the supplementary sources)"):
        st.html("".join(
            f'<div class="sf-panel ind"><b>{esc(e["rows"])}</b> · {md_inline(e["source"])}'
            f'<div style="margin-top:4px">{md_inline(e["text"])}</div>'
            f'<div>{md_inline(e["link"])}</div></div>' for e in doc.excerpts()))


main()
