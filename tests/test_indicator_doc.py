"""Tests for ui/indicator_doc.py: the indicator page reads data/indicator_sources.md only."""
from pathlib import Path

from ui import indicator_doc as doc

ROOT = Path(__file__).resolve().parents[1]
MD = (ROOT / "data" / "indicator_sources.md").read_text(encoding="utf-8")


def test_every_table_row_is_parsed_with_its_own_text():
    rows = doc.rows()
    assert len(rows) == 14
    for r in rows:
        for k in ("indicator", "code_field", "unit", "source", "risk", "match", "links", "status"):
            assert r[k], (r["indicator"], k)
        assert r["match"] in MD and r["source"] in MD          # verbatim from the file
        assert "](" in r["links"]                               # Markdown links kept


def test_match_kind_direct_indirect_or_unlabelled():
    kind = {r["indicator"]: r["match_kind"] for r in doc.rows()}
    assert kind["Cabin CO2"] == "Direct"
    assert kind["Blood oxygen saturation"] == "Direct"          # "Direct (with note): ..."
    assert kind["Heart rate"] == "Indirect"
    assert kind["Core body temperature"] == "Indirect"
    assert kind["Reaction time (PVT-B)"] == "Indirect"          # user decision 2026-11-11
    assert set(kind.values()) <= {"Direct", "Indirect", "Not labelled"}


def test_status_is_shown_as_written_in_the_file():
    """The page shows each row's Status exactly as the file says (the team edits it)."""
    import re
    written = {m[1].strip(): m[2] for m in
               re.finditer(r"^\| ([^|]+?) \|.*\| (checked|unchecked) \|$", MD, re.M)}
    status = {r["indicator"]: r["status"] for r in doc.rows()}
    assert status and status == written


def test_context_notes_and_excerpts_come_from_the_file():
    notes = doc.context()
    assert len(notes) >= 3 and all(n in MD for n in notes)
    assert any("Shelter during a solar particle event" in n for n in notes)
    ex = doc.excerpts()
    assert ex and all(e["text"] in MD for e in ex)


def test_thresholds_come_from_config_and_say_illustrative():
    acts = doc.thresholds("`systolic_bp`, `diastolic_bp`")
    assert acts and all("illustrative" in a for a in acts)
    assert any("140" in a for a in acts)
    assert any(a.startswith("Crash rule: resting blood pressure") for a in acts)
    assert doc.thresholds("`food_water_intake`") == []
