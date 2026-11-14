"""data/indicator_sources.md must stay usable by the indicator page."""
import dataclasses
import re
from pathlib import Path

from spacefuzz.readers import env_trace as et
from spacefuzz.core import model as md
from spacefuzz.core import state as st

DOC = Path(__file__).resolve().parents[1] / "data" / "indicator_sources.md"


def tables(text):
    """Each Markdown table as (header cells, list of row cells)."""
    out, block = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            out.append((block[0], [r for r in block[2:]]))
            block = []
    return out


def test_every_code_field_exists_in_the_code():
    header, rows = tables(DOC.read_text())[0]
    col = header.index("Code field")
    known = (set(md.OUTPUT_COLUMNS) | set(et.COLUMNS)
             | {f.name for f in dataclasses.fields(st.CheckIn)})
    for row in rows:
        for name in re.findall(r"`([^`]+)`", row[col]):
            assert name in known, f"{row[0]}: `{name}` is not a field the code produces"


def test_tables_are_not_split_by_blank_lines():
    """A blank line inside a table makes the next row render as a headerless line."""
    for header, rows in tables(DOC.read_text()):
        assert header[0] and set(rows and rows[0] or ["x"]) != {"---"}
    lines = DOC.read_text().splitlines()
    for i, line in enumerate(lines[2:], 2):
        if line.startswith("|") and not lines[i - 1].startswith("|"):
            assert lines[i + 1].startswith("| ---"), f"line {i + 1}: table row without a header"
