"""Data quality gate (plan 3.4): filter bad readings before the engine sees them.

Three checks per field:
- out_of_range: values outside physiological plausibility (config.QG_PLAUSIBLE_RANGES,
  e.g. heart rate 300) are removed from the cleaned data;
- stuck: identical consecutive readings for at least config.QG_STUCK_MINUTES[field]
  (a frozen sensor) make the whole field untrustworthy;
- missing: an absent field, or more than QG_MAX_MISSING_FRACTION of its readings
  empty or removed, makes it unusable.

Result:
- "ok": every field usable and nothing removed or empty;
- "untrusted": fewer than QG_MIN_USABLE_FRACTION of the fields usable, every vital sign
  series unusable, or a check-in value the model and rules read (QG_CHECKIN_FIELDS) out of
  range or not entered (only QG_OPTIONAL_CHECKIN_FIELDS may be left out) -> grey UI, no
  rating, UNTRUSTED_MESSAGE plus the items to re-measure. A rejected value is never
  simulated: sleep entered as 20 h would make the day look safe. The engine must not run
  on untrusted data (ensure_trusted raises), so no normal-looking risk is ever produced
  from the check-in values alone.
- "degraded": anything in between -> rated as usual, the card names `missing`.

Inputs: a time series of wearable readings (DataFrame; minutes from a `minute`
column or the index) and/or the check-in values (dict or state.CheckIn).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spacefuzz import config

UNTRUSTED_MESSAGE = "Data not trustworthy. Please measure again."
FIELD_LABELS = {"systolic_bp": "systolic blood pressure", "diastolic_bp": "diastolic blood "
                "pressure", "blood_glucose": "blood glucose", "sleep_last_night_h": "sleep last "
                "night", "sleep_last_3_nights_h": "sleep of the last 3 nights",
                "fatigue_score": "fatigue rating", "mood": "mood rating",
                "stress": "stress rating", "reaction_time_ms": "reaction time"}


class UntrustedDataError(RuntimeError):
    """Raised when a forecast is requested on untrusted data."""


@dataclass(frozen=True)
class Issue:
    field: str
    kind: str        # out_of_range | stuck | missing | gaps
    detail: str


@dataclass
class GateResult:
    status: str
    missing: list[str]                    # fields that cannot be used
    issues: list[Issue] = field(default_factory=list)
    clean: pd.DataFrame | None = None     # readings with bad values / unusable fields as NaN
    checkin: dict = field(default_factory=dict)  # check-in values, unusable ones as None
    rejected: list[Issue] = field(default_factory=list)  # check-in values that block the run

    @property
    def can_forecast(self) -> bool:
        return self.status != "untrusted"

    @property
    def message(self) -> str:
        """UNTRUSTED_MESSAGE, plus which rejected check-in values to re-measure."""
        if not self.rejected:
            return UNTRUSTED_MESSAGE
        items = ", ".join(f"{FIELD_LABELS.get(i.field, i.field)} ({i.detail})"
                          for i in self.rejected)
        return f"{UNTRUSTED_MESSAGE} Re-measure: {items}."

    def to_dict(self) -> dict:
        out = {"data_quality": self.status, "missing": list(self.missing),
               "issues": [dataclasses.asdict(i) for i in self.issues]}
        if not self.can_forecast:
            out["message"] = self.message
        return out


def ensure_trusted(result: GateResult) -> GateResult:
    """Call before forecasting: untrusted data raises instead of producing a risk value."""
    if not result.can_forecast:
        raise UntrustedDataError(f"{result.message} Unusable: {', '.join(result.missing)}")
    return result


def _constant_runs(values, minutes):
    """(start, end, duration [min]) of every run of identical consecutive non-empty readings."""
    v = pd.Series(values).to_numpy(dtype=float)
    m = np.asarray(minutes, dtype=float)
    step = float(np.median(np.diff(m))) if len(m) > 1 else 1.0
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or np.isnan(v[i]) or np.isnan(v[i - 1]) or v[i] != v[i - 1]:
            if not np.isnan(v[start]):
                yield start, i, m[i - 1] - m[start] + step
            start = i


def _longest_constant_run(values: pd.Series, minutes: np.ndarray) -> float:
    """Duration [min] of the longest run of identical consecutive non-empty readings."""
    if pd.Series(values).notna().sum() < 2:
        return 0.0
    return max((d for _, _, d in _constant_runs(values, minutes)), default=0.0)


def stuck_mask(values: pd.Series, minutes, limit_min: float) -> pd.Series:
    """True for readings inside a run of identical values lasting at least limit_min."""
    mask = np.zeros(len(values), dtype=bool)
    for a, b, d in _constant_runs(values, minutes):
        if d >= limit_min:
            mask[a:b] = True
    return pd.Series(mask, index=getattr(values, "index", None))


def _check_series(name: str, s: pd.Series, minutes: np.ndarray, issues: list[Issue]
                  ) -> tuple[pd.Series, bool]:
    """Return (cleaned series, usable)."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.notna().sum() == 0:
        issues.append(Issue(name, "missing", "no readings"))
        return s, False
    lo, hi = config.QG_PLAUSIBLE_RANGES.get(name, (-np.inf, np.inf))
    bad = s.notna() & ((s < lo) | (s > hi))
    if bad.any():
        sample = s[bad].iloc[0]
        issues.append(Issue(name, "out_of_range",
                            f"{int(bad.sum())} reading(s) outside {lo}-{hi}, e.g. {sample:g}"))
        s = s.mask(bad)
    limit = config.QG_STUCK_MINUTES.get(name)
    if limit is not None:
        run = _longest_constant_run(s, minutes)
        if run >= limit:
            issues.append(Issue(name, "stuck", f"unchanged for {run:.0f} min (limit {limit})"))
            return s * np.nan, False
    empty = s.isna().mean()
    if empty > config.QG_MAX_MISSING_FRACTION:
        issues.append(Issue(name, "missing", f"{empty:.0%} of readings empty or removed"))
        return s * np.nan, False
    gaps = int(s.isna().sum() - bad.sum())
    if gaps:
        issues.append(Issue(name, "gaps", f"{gaps} empty reading(s), still usable"))
    return s, True


def _number(value) -> float | None:
    """A check-in entry as a float (numbers written as text are accepted); None if it is not
    a number at all."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_value(name: str, value, issues: list[Issue]) -> bool:
    """One check-in entry (a sequence such as the last 3 nights is checked value by value)."""
    if value is None:
        issues.append(Issue(name, "missing", "not entered"))
        return False
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    nums = [_number(v) for v in items]
    if any(n is None for n in nums) or not items:
        issues.append(Issue(name, "invalid", f"{value!r} is not a number"))
        return False
    if any(np.isnan(n) for n in nums):
        issues.append(Issue(name, "missing", "not entered"))
        return False
    lo, hi = config.QG_PLAUSIBLE_RANGES.get(name, (-np.inf, np.inf))
    bad = [n for n in nums if not lo <= n <= hi]
    if bad:
        issues.append(Issue(name, "out_of_range", f"{bad[0]:g} outside {lo}-{hi}"))
        return False
    return True


def check(readings: pd.DataFrame | None = None, checkin=None,
          series_fields=config.QG_VITAL_SERIES,
          checkin_fields=config.QG_CHECKIN_FIELDS) -> GateResult:
    """Run the gate over wearable readings and/or check-in values."""
    issues: list[Issue] = []
    usable: dict[str, bool] = {}
    clean = None

    if readings is not None:
        clean = readings.copy()
        minutes = (readings["minute"].to_numpy(dtype=float) if "minute" in readings
                   else np.asarray(readings.index, dtype=float))
        for name in series_fields:
            if name not in readings.columns:
                issues.append(Issue(name, "missing", "no readings"))
                usable[name] = False
                continue
            clean[name], usable[name] = _check_series(name, readings[name], minutes, issues)

    values = {}
    if checkin is not None:
        raw = dataclasses.asdict(checkin) if dataclasses.is_dataclass(checkin) else dict(checkin)
        for name in checkin_fields:
            ok = _check_value(name, raw.get(name), issues)
            usable[name] = ok
            values[name] = raw.get(name) if ok else None

    missing = [f for f in usable if not usable[f]]
    vitals = [f for f in series_fields if f in usable]
    rejected = [i for i in issues if i.field in values and i.field in missing
                and (i.kind in ("out_of_range", "invalid")
                     or i.field not in config.QG_OPTIONAL_CHECKIN_FIELDS)]
    if not usable or rejected:
        status = "untrusted"
    elif (sum(usable.values()) / len(usable) < config.QG_MIN_USABLE_FRACTION
          or (vitals and not any(usable[f] for f in vitals))):
        status = "untrusted"
    elif issues:
        status = "degraded"
    else:
        status = "ok"
    return GateResult(status=status, missing=missing, issues=issues, clean=clean, checkin=values,
                      rejected=rejected)
