"""Small helpers shared by app.py, ui/ and pages/: escaping, number and time formats, and
the playback schedule every animated panel reads."""
import html
import re

import ui_config
from spacefuzz.core.plan import clock  # noqa: F401  (re-exported: HH:MM, wraps past midnight)


CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"   # numbered plan changes (plan board legend, fuzz log)
TIP_MARK = "ⓘ"


def esc(x) -> str:
    return html.escape(str(x))


def pct(p) -> str:
    """Risk share as a percentage with one decimal; "--" when there is none."""
    return "--" if p is None else f"{100 * p:.1f}%"


def no_change_text(still_red: bool, subject: str = "plan") -> str:
    """What to say when the adjustment changed nothing: either nothing was needed, or no
    plan change could bring the risk down (still red, so the ground decides)."""
    if still_red:
        return f"No safe adjustment: the {subject} stays red, report to the ground surgeon."
    return f"No change needed: the {subject} is not red."


def stage_times(frames: list[dict], frame_s: float | None = None,
                reveal_s: float | None = None) -> tuple[float, float]:
    """Seconds from the first playback frame to the forecast reveal and to the adjust reveal,
    as app.py plays them (the iframes schedule their own animation from these)."""
    scale = ui_config.ANIM_SCALE
    frame_s = ui_config.FRAME_S * scale if frame_s is None else frame_s
    reveal_s = ui_config.REVEAL_S * scale if reveal_s is None else reveal_s
    t, out = 0.0, {}
    for f in frames:
        if f.get("done"):
            out.setdefault(f["done"], t)
        t += reveal_s if f.get("done") else frame_s
    fc = out.get("forecast", 0.0)
    return fc, out.get("adjust", fc)


def tip(text, side: str = "") -> str:
    """An (i) that shows `text` on hover or keyboard focus (CSS in ui/styles.css, shared with
    the iframes). side: "" opens down-right, "l" left, "u" up, "l u" up-left. Empty text: ""."""
    if not text:
        return ""
    cls = f"sf-tip {side}".strip()
    return (f'<span class="{cls}" tabindex="0" role="note" aria-label="{esc(text)}" '
            f'data-tip="{esc(text)}">{TIP_MARK}</span>')


def _clean(s: str) -> str:
    return s.strip().rstrip(".").strip()


def brief(text: str) -> tuple[str, str]:
    """Split an engine sentence into (short visible part, detail for the tip), losing nothing:
    "Reason (...): action" -> action / reason; "action; more" -> action / more;
    "action (why)" -> action / why."""
    text = _clean(text)
    details = []
    m = re.match(r"^([^:;]*\([^)]*\)[^:;]*):\s+(.+)$", text)
    if m:                                    # "Short sleep (6.2 h ...): schedule a nap ..."
        details.append(_clean(m[1]))
        text = m[2]
    if "; " in text:
        text, rest = text.split("; ", 1)
        details.append(_clean(rest))
    parens = re.findall(r"\s*\(([^()]*)\)", text)
    if parens:
        text = re.sub(r"\s*\([^()]*\)", "", text)
        details = [_clean(x) for x in parens] + details
    text = _clean(text)
    return (text[:1].upper() + text[1:], "; ".join(d for d in details if d))


def illustrative_badge() -> str:
    """The one place per page that says "illustrative" (not a note on every threshold)."""
    return (f'<span class="sf-badge" data-testid="illustrative">Illustrative'
            f'{tip(ui_config.ILLUSTRATIVE_NOTE)}</span>')


def brief_html(text: str, side: str = "") -> str:
    """brief() as HTML: the short part, then an (i) with the rest."""
    short, detail = brief(text)
    return esc(short) + tip(detail[:1].upper() + detail[1:], side)


# The engine's generic nap advice ("schedule a 20-minute nap ...") before the adjusted plan is
# known: the adjusted plan sets the nap, so no screen promises a 20-minute nap it then replaces.
NAP_ADVICE = re.compile(r"schedule a [\w-]+ nap before demanding tasks")
NEUTRAL_NAP = "plan a nap before demanding tasks (the adjusted plan sets the time)"


def neutral_nap(text: str) -> str:
    return NAP_ADVICE.sub(NEUTRAL_NAP, text)
