"""data/indicator_sources.md as data for the indicator page (plan 4.1 annex 2).

The page shows each indicator's explanation and evidence exactly as the file states them
(no second copy of the text): where it comes from, the HRP risk, how well the risk supports
the indicator (Direct / Indirect, or "Not labelled" when the file gives neither), the links
and the check status. The only additions are the thresholds SpaceFuzz acts on, read from
config.py (illustrative).
"""
import functools
import re
from pathlib import Path

from spacefuzz import config
import ui_config

DOC = Path(__file__).resolve().parents[1] / "data" / "indicator_sources.md"
COLUMNS = {"Indicator": "indicator", "Plan variable (plan 2.2)": "plan_variable",
           "Code field": "code_field", "Unit / scale": "unit",
           "Where it comes from": "source", "HRP risk": "risk", "Match": "match",
           "Link": "links", "Status": "status"}


def _tables(text: str) -> list[tuple[list, list]]:
    out, block = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            out.append((block[0], block[2:]))
            block = []
    return out


def match_kind(match: str) -> str:
    if match.startswith("Indirect"):
        return "Indirect"
    if match.startswith("Direct"):
        return "Direct"
    return "Not labelled"


@functools.lru_cache(maxsize=1)
def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def rows() -> list[dict]:
    header, body = _tables(_text())[0]
    out = []
    for cells in body:
        r = {COLUMNS[h]: c for h, c in zip(header, cells) if h in COLUMNS}
        r["match_kind"] = match_kind(r["match"])
        out.append(r)
    return out


def context() -> list[str]:
    """Bullets of the 'Context for the indicator page' section, verbatim."""
    m = re.search(r"## Context for the indicator page\n(.*?)\n## ", _text(), re.S)
    bullets = re.split(r"\n- ", "\n" + m[1].strip()) if m else []
    return [b.strip() for b in bullets if b.strip()]


def excerpts() -> list[dict]:
    """The 'Supplementary source excerpts' table: rows, source, supporting text, link."""
    tables = _tables(_text())
    if len(tables) < 2:
        return []
    header, body = tables[1]
    return [{"rows": c[0], "source": c[1], "text": c[2], "link": c[3]} for c in body if len(c) >= 4]


def thresholds(code_field: str) -> list[str]:
    """What SpaceFuzz acts on for these code fields (config.py values, illustrative)."""
    fields = set(re.findall(r"`([^`]+)`", code_field))
    c = config
    lines = []
    if "heart_rate" in fields:
        lines.append(f"Act now: resting heart rate {c.ACT_NOW_HR_ABOVE_BASELINE_BPM:.0f} bpm "
                     f"above baseline; crash rule: >= {c.CARDIO_HR_MAX_FRACTION:.0%} of max "
                     f"heart rate (220 - age) with low HRV, sustained {c.CARDIO_SUSTAIN_MIN} min")
    if "hrv_rmssd" in fields:
        lines.append(f"Crash rule: RMSSD <= {c.CARDIO_HRV_RATIO:.0%} of baseline together with "
                     f"high heart rate")
    if "systolic_bp" in fields:
        (s_lo, s_hi), (d_lo, d_hi) = c.ACT_NOW_BP_RANGE["sbp"], c.ACT_NOW_BP_RANGE["dbp"]
        lines.append(f"Act now: outside {s_lo:.0f}-{s_hi:.0f} / {d_lo:.0f}-{d_hi:.0f} mmHg; "
                     f"severe from {c.ACT_NOW_BP_SEVERE['sbp']:.0f} / "
                     f"{c.ACT_NOW_BP_SEVERE['dbp']:.0f} mmHg")
        (cs_lo, cs_hi), (cd_lo, cd_hi) = c.CARDIO_BP_RANGE["sbp"], c.CARDIO_BP_RANGE["dbp"]
        lines.append(f"Crash rule: resting blood pressure outside {cs_lo:.0f}-{cs_hi:.0f} / "
                     f"{cd_lo:.0f}-{cd_hi:.0f} mmHg after {c.CARDIO_BP_SETTLE_MIN} min at rest "
                     f"(independent of the plan, so no plan change can clear it)")
    if "spo2" in fields:
        lines.append(f"Crash rule: SpO2 < {c.COG_SPO2_BELOW_PCT:.0f}% together with fatigue > "
                     f"{c.COG_FATIGUE_ABOVE:.0f}")
    if "co2" in fields:
        lines.append(f"Time rule: docking / precision work after > {c.RULE3_MAX_WORK_H:.0f} h of "
                     f"cabin work above {c.RULE3_CO2_PPM:,.0f} ppm")
    if "sleep_last_night_h" in fields:
        lines.append(f"Act now: < {c.ACT_NOW_SLEEP_SHORT_H:.0f} h last night or "
                     f">= {c.ACT_NOW_SLEEP_DEBT_H:.0f} h debt over 3 nights; time rule: no EVA "
                     f"within 12 h of waking after < {c.RULE1_MIN_SLEEP_H:.0f} h sleep")
    if "cognitive_fatigue" in fields:
        lines.append(f"Crash rule: fatigue > {c.COG_FATIGUE_ABOVE:.0f} with SpO2 < "
                     f"{c.COG_SPO2_BELOW_PCT:.0f}%")
    if "mood" in fields:
        lines.append(f"Act now: mood <= {c.ACT_NOW_MOOD_LOW}/5 or stress >= "
                     f"{c.ACT_NOW_STRESS_HIGH}/5")
    if "reaction_time_ms" in fields:
        lines.append(f"A single response >= {ui_config.REACTION_LAPSE_MS} ms counts as a lapse "
                     "(Jones et al. 2022); the "
                     "forecast uses the mean reaction time")
    if "eva_dose_ugy" in fields or "dose_rate" in fields:
        lines.append(f"Crash rule: the day's EVA dose (all EVAs) above "
                     f"{c.RAD_EVA_DOSE_LIMIT_UGY:.0f} µGy (a load signal, not a NASA limit); "
                     f"no EVA start during an SPE warning")
    if "blood_glucose" in fields:
        lines.append(f"Act now: < {c.GLUCOSE_LOW_MG_DL:.0f} mg/dL (severe < "
                     f"{c.ACT_NOW_GLUCOSE_SEVERE_LOW:.0f}); crash rule: < "
                     f"{c.METAB_PRE_EVA_GLUCOSE_MG_DL:.0f} mg/dL when an EVA starts")
    if "core_temp" in fields:
        lines.append(f"Heat strain from +{c.HEAT_STRAIN_ONSET_C:.1f} °C core rise during EVA")
    return [f"{x} (illustrative)" for x in lines]
