"""Tests for pages/2_Indicators.py: every row of data/indicator_sources.md, verbatim."""
import html
import os
import re
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import indicator_doc as doc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "2_Indicators.py"


def page_text(at) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                     + [h.proto.body for h in at.get("html")])


def plain_md(text: str) -> str:
    """The file's Markdown as plain text (links -> their text, no * or `)."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return " ".join(text.replace("*", "").replace("`", "").split())


def plain_html(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", text)).split())


def test_page_shows_every_indicator_with_its_match_and_status():
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not at.exception, at.exception
    t = page_text(at)
    flat = plain_html(t)
    for r in doc.rows():
        assert r["indicator"] in t, r["indicator"]
        # the file's text, not a rewrite (only the site's arrow and unit symbols on screen)
        site = lambda x: plain_md(x).replace(" -> ", " → ").replace("uGy", "µGy").replace(  # noqa: E731
            "degC", "°C")
        assert site(r["match"]) in flat, r["indicator"]
        assert site(r["source"]) in flat and site(r["risk"]) in flat
        assert f'data-match="{r["match_kind"]}"' in t
        assert f'data-status="{r["status"]}"' in t
    assert "Not labelled" in t and "Indirect" in t and "Direct" in t
    assert "illustrative" in t                                  # thresholds from config
    assert "indicator_sources.md" in t                          # says where the text comes from


def test_page_shows_the_context_notes():
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    flat = plain_html(page_text(at))
    for n in doc.context():
        assert plain_md(n) in flat
