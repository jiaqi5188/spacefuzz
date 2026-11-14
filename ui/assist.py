"""Optional AI hint for a check-in the data-quality gate rejected.

The gate (spacefuzz/core/quality_gate.py) decides; this module only explains. When a
check-in value is rejected (for example sleep entered as 20 h), and only if an
OPENAI_API_KEY is set, the check-in page offers a button that asks a language model for a
short plain-language hint: what was probably mistyped and what to re-measure.

Rules:
- off without a key, and always off in tests (SPACEFUZZ_ASSIST=off);
- sends only the rejected fields, the entered values and the plausible ranges;
- never changes the gate decision, a check-in value or a forecast number;
- any failure (no client, timeout, bad key, empty or odd answer) returns None and the
  page keeps the gate's own message.
"""
from __future__ import annotations

import os

import ui_config
from spacefuzz import config
from spacefuzz.core import quality_gate as qg

BUTTON = "Ask AI what went wrong (optional)"
LABEL = "AI hint · advisory only, the data-quality decision above stands"
UNAVAILABLE = "AI hint unavailable right now; please follow the re-measure list above."

SYSTEM = (
    "You help an astronaut fix a daily health check-in that an automatic data-quality check "
    "rejected. In at most two short sentences, say which entry is most likely a typo, a unit "
    "mix-up or a misreading, and what to re-measure or re-enter. Do not give medical advice, "
    "do not judge health or risk, do not say the data is acceptable, and do not invent a "
    "value to enter without measuring. Plain text only."
)


def _load_dotenv() -> None:
    """Read .env without overriding variables already set (python-dotenv, if installed)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def available() -> bool:
    if os.environ.get("SPACEFUZZ_ASSIST", "").lower() == "off":
        return False
    _load_dotenv()
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _client():
    from openai import OpenAI          # imported only when the button is pressed with a key
    return OpenAI(max_retries=0)


def prompt(gate: qg.GateResult) -> str:
    lines = []
    for i in gate.rejected:
        label = qg.FIELD_LABELS.get(i.field, i.field)
        lo, hi = config.QG_PLAUSIBLE_RANGES.get(i.field, (None, None))
        rng = f"; plausible range {lo}-{hi}" if lo is not None else ""
        lines.append(f"- {label}: {i.detail}{rng}")
    return "Rejected check-in entries:\n" + "\n".join(lines)


def hint(gate: qg.GateResult, client=None) -> str | None:
    """A short hint for the rejected entries, or None (nothing rejected, off, or any failure)."""
    if not gate.rejected:
        return None
    try:
        if client is None:
            if not available():
                return None
            client = _client()
        resp = client.chat.completions.create(
            model=ui_config.ASSIST_MODEL, timeout=ui_config.ASSIST_TIMEOUT_S,
            max_completion_tokens=ui_config.ASSIST_MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt(gate)}])
        text = resp.choices[0].message.content
    except Exception:                   # noqa: BLE001 - any failure means "no hint"
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return " ".join(text.split())[:ui_config.ASSIST_MAX_CHARS]
