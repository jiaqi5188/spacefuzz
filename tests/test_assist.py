"""Tests for ui/assist.py (optional AI hint when the quality gate rejects a check-in) and
ui/guard.py (friendly message instead of a stack trace).

The AI hint is a fallback for a rejected check-in only: it is off without OPENAI_API_KEY,
it never changes the gate decision or a number, and any failure of the call (timeout,
bad key, empty answer) falls back silently to the gate's own message. No test here goes
online: the client is always a fake."""
import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ["SPACEFUZZ_ANIM_SCALE"] = "0"

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_config  # noqa: E402
from spacefuzz.core import quality_gate as qg  # noqa: E402
from ui import assist, guard  # noqa: E402
from ui import checkin_store as cs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "page_files" / "1_Morning_check-in.py"
MAIN = "main_sleep_debt_eva"


def rejected_checkin() -> dict:
    d = cs.example(MAIN)
    d["sleep_last_night_h"] = 20.0          # outside the plausible 0-16 h
    return d


def rejected_gate():
    return qg.check(checkin=cs.to_checkin(rejected_checkin()))


class FakeClient:
    """Stands in for openai.OpenAI: records the request, returns `answer` or raises it."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.answer, BaseException):
            raise self.answer
        msg = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ---------------------------------------------------------------- availability

def test_off_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SPACEFUZZ_ASSIST", raising=False)
    monkeypatch.setattr(assist, "_load_dotenv", lambda: None)
    assert not assist.available()


def test_on_with_a_key_unless_switched_off(monkeypatch):
    monkeypatch.setattr(assist, "_load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("SPACEFUZZ_ASSIST", raising=False)
    assert assist.available()
    monkeypatch.setenv("SPACEFUZZ_ASSIST", "off")
    assert not assist.available()


def test_tests_run_with_the_hint_switched_off():
    """conftest switches it off, so a developer's .env never sends anything during tests."""
    assert os.environ.get("SPACEFUZZ_ASSIST") == "off"
    assert not assist.available()


def test_nothing_to_ask_when_the_gate_accepts():
    ok = qg.check(checkin=cs.to_checkin(cs.example(MAIN)))
    client = FakeClient("should not be asked")
    assert assist.hint(ok, client=client) is None and not client.calls


# ---------------------------------------------------------------- the request

def test_request_names_the_rejected_values_and_asks_for_no_medical_advice():
    client = FakeClient("Sleep of 20 h is above the 16 h limit; check the wristband record.")
    assert assist.hint(rejected_gate(), client=client).startswith("Sleep of 20 h")
    kw = client.calls[0]
    assert kw["model"] == ui_config.ASSIST_MODEL
    assert kw["timeout"] == ui_config.ASSIST_TIMEOUT_S
    text = " ".join(m["content"] for m in kw["messages"])
    assert "sleep last night" in text and "20 outside 0-16" in text
    assert "medical advice" in text and "risk" in text           # what it must not do
    assert "sk-" not in text                                     # never the key itself


def test_request_carries_only_the_rejected_fields():
    client = FakeClient("ok")
    assist.hint(rejected_gate(), client=client)
    text = " ".join(m["content"] for m in client.calls[0]["messages"])
    assert "systolic" not in text and "glucose" not in text


# ---------------------------------------------------------------- silent fallback

@pytest.mark.parametrize("answer", [TimeoutError("slow"), RuntimeError("401 bad key"),
                                    ValueError("quota"), "", "   ", None, 42])
def test_any_failure_falls_back_to_none(answer):
    assert assist.hint(rejected_gate(), client=FakeClient(answer)) is None


def test_a_broken_response_shape_falls_back_to_none():
    client = FakeClient("x")
    client.chat = SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[])))
    assert assist.hint(rejected_gate(), client=client) is None


def test_no_key_means_no_call_even_if_asked(monkeypatch):
    monkeypatch.setattr(assist, "available", lambda: False)
    assert assist.hint(rejected_gate()) is None                  # no client built, no import


def test_long_answers_are_cut_and_markup_is_not_trusted():
    long = "<b>" + "word " * 500
    out = assist.hint(rejected_gate(), client=FakeClient(long))
    assert len(out) <= ui_config.ASSIST_MAX_CHARS
    assert "<b>" in out                                          # plain text; the page escapes it


def test_the_hint_never_changes_the_gate():
    gate = rejected_gate()
    before = (gate.status, list(gate.rejected), gate.message)
    assist.hint(gate, client=FakeClient("Looks fine, go ahead."))
    assert (gate.status, list(gate.rejected), gate.message) == before


# ---------------------------------------------------------------- the check-in page

@pytest.fixture
def with_key(monkeypatch):
    """A key is set and the client answers `answer` (AppTest shares this process's modules)."""
    def setup(answer):
        monkeypatch.setattr(assist, "available", lambda: True)
        monkeypatch.setattr(assist, "_client", lambda: FakeClient(answer))
    return setup


def page():
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    at.session_state[cs.EDITS] = {MAIN: rejected_checkin()}
    at.run()
    assert not at.exception, at.exception
    return at


def text(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                     + [h.proto.body for h in at.get("html")] + [w.value for w in at.warning]
                     + [c.value for c in at.code])


def test_page_without_a_key_shows_the_gate_only():
    at = page()
    assert "Data quality: untrusted" in text(at)
    assert not [b for b in at.button if b.label == assist.BUTTON]


def test_page_with_an_accepted_checkin_offers_no_hint(with_key):
    with_key("unused")
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert "Data quality: ok" in text(at)
    assert not [b for b in at.button if b.label == assist.BUTTON]


def test_page_with_a_key_offers_the_hint_and_keeps_the_gate_decision(with_key):
    with_key("Sleep 20 h is above the 16 h limit; maybe 2.0 h was meant.")
    at = page()
    btn = next(b for b in at.button if b.label == assist.BUTTON)
    btn.click().run()
    assert not at.exception, at.exception
    t = text(at)
    assert "maybe 2.0 h was meant" in t
    assert "advisory" in t                                       # labelled as such
    assert "Data quality: untrusted" in t                        # the gate still decides
    assert at.session_state[cs.EDITS][MAIN]["sleep_last_night_h"] == 20.0   # nothing changed


def test_page_with_a_failing_call_shows_a_quiet_note(with_key):
    with_key(RuntimeError("down"))
    at = page()
    next(b for b in at.button if b.label == assist.BUTTON).click().run()
    assert not at.exception, at.exception
    t = text(at)
    assert assist.UNAVAILABLE in t and "Data quality: untrusted" in t


def test_page_hint_is_escaped(with_key):
    with_key("<script>alert(1)</script>")
    at = page()
    next(b for b in at.button if b.label == assist.BUTTON).click().run()
    assert "<script>alert(1)</script>" not in text(at)


# ---------------------------------------------------------------- offline by default

def test_openai_is_imported_only_lazily_inside_ui_assist():
    """The engine and every page stay offline: the client is imported inside a function of
    ui/assist.py, only after the user presses the button with a key set."""
    src = (ROOT / "ui" / "assist.py").read_text()
    assert not re.search(r"^(import|from)\s+openai\b", src, re.M)      # not at module level
    assert re.search(r"^\s+from openai import OpenAI\b", src, re.M)    # inside a function


# ---------------------------------------------------------------- guard

GUARDED = """
import streamlit as st
from ui import guard
def boom():
    st.write("before")
    raise ZeroDivisionError("division by zero in a page")
guard.run(boom)
"""

STOPPED = """
import streamlit as st
from ui import guard
def stop():
    st.write("stopped on purpose")
    st.stop()
guard.run(stop)
"""


def test_guard_shows_a_friendly_message_instead_of_a_stack_trace():
    at = AppTest.from_string(GUARDED, default_timeout=30).run()
    assert not at.exception, at.exception
    t = text(at)
    assert guard.MESSAGE in t
    assert "ZeroDivisionError" in t                        # in the folded details
    assert "Traceback" not in t                            # no raw stack on the page


def test_guard_lets_st_stop_and_rerun_through():
    at = AppTest.from_string(STOPPED, default_timeout=30).run()
    assert not at.exception
    assert guard.MESSAGE not in text(at)


def test_app_routes_through_the_guard():
    src = (ROOT / "app.py").read_text()
    assert "guard.run(" in src and "st.navigation(nav.pages()).run()" in src
