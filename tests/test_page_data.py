"""Tests for pages/4_Data.py (NASA Data): how the real data enters the model.

The page must render from committed files only (a fresh clone has no data/raw downloads
except the two committed NASA files), and every block names its dataset with a link."""
import os
import subprocess
from pathlib import Path

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import plotly.graph_objects as go  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import data_page as dp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "4_Data.py"
BLOCKS = ("cabin", "radiation", "stress", "clinical")


def page_html(at) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                     + [h.proto.body for h in at.get("html")])


def test_page_renders_four_blocks_each_with_a_dataset_link():
    at = AppTest.from_file(str(PAGE), default_timeout=60).run()
    assert not at.exception, at.exception
    h = page_html(at)
    for b in BLOCKS:
        block = h.split(f'data-testid="data-{b}"')[1].split('data-testid="data-')[0]
        assert 'href="http' in block, b                       # dataset name with its link
        assert "In the model:" in block, b                    # one line: what it does
    assert len(at.get("plotly_chart")) >= 4                  # a chart per block


def test_key_numbers_come_from_the_analysis():
    h = page_html(AppTest.from_file(str(PAGE), default_timeout=60).run())
    for text in ("3,997 ppm", "2.98", "47 h", "+21.7 bpm", "0.512", "X9.3", "X8.2",
                 "56-100 mg/dL", "11.5-17.0 g/dL", "operator B", "rule 3", "operator D",
                 "rule 4", "causal chain 1"):
        assert text in h, text
    assert "non-NASA" in h and "supplementary" in h          # WESAD is labelled as such


def test_page_reads_only_committed_files():
    """In the repository: every file is tracked by git. In a copy without .git (the clean-clone
    check copies the tracked files only): every file is there."""
    if (ROOT / ".git").exists():
        tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.split())
        missing = [f for f in dp.DATA_FILES if f not in tracked]
    else:
        missing = [f for f in dp.DATA_FILES if not (ROOT / f).exists()]
    assert not missing, f"{missing}: a fresh clone would not have them"


def test_a_missing_file_shows_a_note_not_an_error(monkeypatch):
    def gone(*a, **k):
        raise FileNotFoundError("not in this clone")
    monkeypatch.setattr(dp, "_cabin_frame", gone)
    monkeypatch.setattr(dp, "_mars_frame", gone)
    dp._cabin_frame_cached.cache_clear()
    dp._mars_frame_cached.cache_clear()
    for build in (dp.cabin_figure, dp.radiation_figure):
        assert build() is None                               # the page then shows a note
    dp._cabin_frame_cached.cache_clear()                     # no stale result for later tests
    dp._mars_frame_cached.cache_clear()


@pytest.mark.parametrize("build", [dp.cabin_figure, dp.radiation_figure, dp.stress_figure,
                                   dp.clinical_figure])
def test_figures_use_the_site_palette(build):
    fig = build()
    assert isinstance(fig, go.Figure)
    assert fig.layout.paper_bgcolor.lower() == dp.charts.BG.lower()


def test_menu_lists_the_data_page_before_the_wiki():
    from ui import nav
    titles = [m[1] for m in nav.MENU]
    assert "NASA Data" in titles and titles.index("NASA Data") < titles.index("Indicators (wiki)")
    assert titles[-1].startswith("Indicators")


def test_radiation_chart_shades_the_particle_event():
    shapes = dp.radiation_figure().layout.shapes
    assert any(sh.type == "rect" and str(sh.x0).startswith("2017-09-11") for sh in shapes)
