"""Small helpers shared by app.py, ui/ and pages/: escaping, number and time formats, and
the playback schedule every animated panel reads."""
import html

import ui_config
from spacefuzz.core.plan import clock  # noqa: F401  (re-exported: HH:MM, wraps past midnight)


CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"   # numbered plan changes (plan board legend, fuzz log)


def esc(x) -> str:
    return html.escape(str(x))


def pct(p) -> str:
    """Risk share as a percentage with one decimal; "--" when there is none."""
    return "--" if p is None else f"{100 * p:.1f}%"


def no_change_text(still_red: bool, subject: str = "plan") -> str:
    """What to say when the adjustment changed nothing: either nothing was needed, or no
    plan change could bring the risk down (still red, so the ground decides)."""
    if still_red:
        return (f"No safe adjustment found: the {subject} is still red, report to the ground "
                f"flight surgeon.")
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
