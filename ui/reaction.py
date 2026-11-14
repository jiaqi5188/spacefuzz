"""Reaction time test (plan 2.2 / manual 4.3): a two-way custom component.

ui/reaction_test/index.html (static, no front-end build) times a PVT-B-style test in the
browser: a counter appears after a random 1-4 s wait and the crew member clicks (or presses
space) as fast as possible. The array of reaction times goes back to Python, the mean is
written to the check-in's reaction_time_ms automatically, and it passes the data-quality
gate like any other entry. A single response >= 355 ms counts as a lapse (Jones et al.,
Sleep 2022, ISS PVT-B). In a real mission this value comes from the dedicated test device;
the component simulates it, and the demo uses a 1-minute version of the 3-minute test.
"""
from pathlib import Path

import streamlit.components.v1 as components

from spacefuzz import config
import ui_config

COMPONENT_DIR = Path(__file__).with_name("reaction_test")
_component = components.declare_component("reaction_test", path=str(COMPONENT_DIR))


def run(duration_s: int, key: str):
    """Show the test; returns {"rts": [...], "false_starts": n, "misses": n, "duration_s": s}
    or None."""
    return _component(duration_s=duration_s, isi_min_s=ui_config.REACTION_ISI_S[0],
                      isi_max_s=ui_config.REACTION_ISI_S[1],
                      lapse_ms=ui_config.REACTION_LAPSE_MS,
                      plausible_ms=list(config.QG_PLAUSIBLE_RANGES["reaction_time_ms"]),
                      palette=ui_config.COLOR,
                      surface=ui_config.SURFACE, key=key, default=None)


def summarise(result: dict | None) -> dict | None:
    """Mean of the plausible responses, lapses, false starts, implausible responses. A missed
    stimulus (no response within the plausible window) is a lapse, not a reaction time."""
    if not result:
        return None
    lo, hi = config.QG_PLAUSIBLE_RANGES["reaction_time_ms"]
    rts = [float(r) for r in result.get("rts", [])]
    valid = [r for r in rts if lo <= r <= hi]
    misses = int(result.get("misses", 0))
    return {"n": len(rts), "n_valid": len(valid),
            "mean_ms": round(sum(valid) / len(valid), 1) if valid else None,
            "lapses": sum(r >= ui_config.REACTION_LAPSE_MS for r in valid) + misses,
            "misses": misses,
            "suspicious": len(rts) - len(valid),
            "false_starts": int(result.get("false_starts", 0)),
            "duration_s": result.get("duration_s")}
