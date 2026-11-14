"""Environment scenario ("trace") for one simulated day.

A trace is a DataFrame indexed by `minute` (minutes since clip start, step 15)
with columns co2 [ppm], temp [degC], humidity [percent], dose_rate [uGy/h].

Sources (read through generic_reader + data/mappings/env_telemetry.yaml):
- Cabin air: EDA RR-1 telemetry, *_ISS columns only (2014), from config.EDA_ISS_START
  (earlier rows are launch / Dragon transit / transfer, not ISS cabin air).
- Radiation: RadLab DosTel2 for the ISS (primary ISS dose source) or MSL-RAD for
  the Mars surface (2017). DosTel2 spikes roughly every 90 min are South
  Atlantic Anomaly passes, not solar events.
Sources are never aligned by calendar date: each is clipped from its own start
time and expressed as minutes since that start, so 2014 cabin air and 2017
radiation can share one scenario day.

Also here: the EDA RR-1 daily dose (cross-check only) and the GOES-16 XRS-B
flux (flare timing labels only; flagged minutes, incl. the daily eclipse gap,
stay empty and are never interpolated).

Trace mutations (after the reference thesis, ch. 6) are applied by the mutator's operators B
and D (mutator.py), not here.
"""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from spacefuzz import config
from spacefuzz.readers import generic_reader as gr

log = logging.getLogger("env_trace")
ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "data" / "mappings" / "env_telemetry.yaml"
GOES_DIR = ROOT / "data" / "raw" / "goes"
COLUMNS = ["co2", "temp", "humidity", "dose_rate"]
MODEL_COLUMNS = ("co2", "dose_rate")      # read by the model at every step
RADIATION_SOURCES = {"iss": "radlab_dostel2", "mars": "radlab_msl_rad"}
CABIN_SOURCE = "eda_rr1"
DAILY_DOSE_SOURCE = "eda_rr1_radiation"

_cache: dict[str, pd.DataFrame] = {}


def load_source(source: str) -> pd.DataFrame:
    """A timestamped env-standard frame for one block of env_telemetry.yaml (cached)."""
    if source not in _cache:
        df = gr.read_source(MAPPING, source)
        iss_start = pd.Timestamp(config.EDA_ISS_START)
        if source == CABIN_SOURCE:
            # Launch, Dragon transit and transfer: not ISS cabin air (config.EDA_ISS_START)
            df = df[df["timestamp"] >= iss_start].reset_index(drop=True)
        elif source == DAILY_DOSE_SOURCE:
            # Same RR-1 hardware: keep whole days from the first full day on the station
            df = df[df["timestamp"] >= iss_start.ceil("D")].reset_index(drop=True)
        _cache[source] = df
    return _cache[source].copy()


def _fill_short_gaps(s: pd.Series, max_steps: int) -> pd.Series:
    """Linearly fill interior NaN runs of at most max_steps; longer runs stay NaN."""
    isna = s.isna()
    if not isna.any():
        return s
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    fill = isna & (run_len <= max_steps)
    return s.where(~fill, filled)


def clip(source: pd.DataFrame, start, hours: float = 24) -> pd.DataFrame:
    """Cut [start, start + hours) from a timestamped frame onto the 15-minute grid.

    Each bin is the time-weighted mean of its samples: a sample counts for the
    time until the next one (capped at one step). This matters for DosTel2,
    which samples ~5x faster inside SAA passes; a plain sample mean would
    double the dose rate. Runs of up to config.ENV_MAX_INTERP_STEPS empty bins are
    interpolated (a gap of up to about an hour between samples); longer gaps stay NaN.
    """
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    cols = [c for c in source.columns if c in COLUMNS]
    src = source.sort_values("timestamp")
    if not ((src["timestamp"] >= start) & (src["timestamp"] < end)).any():
        raise ValueError(f"no data between {start} and {end}")
    step = f"{config.STEP_MINUTES}min"
    step_s = config.STEP_MINUTES * 60
    # Bin a few steps of context on each side so edge bins can be interpolated too.
    pad = pd.Timedelta(minutes=config.STEP_MINUTES * config.ENV_MAX_INTERP_STEPS)
    window = src[(src["timestamp"] >= start - pad) & (src["timestamp"] < end + pad)]
    ext_end = end + pad
    grid = pd.date_range(start - pad, ext_end, freq=step, inclusive="left")
    start_bin = start - pad

    t = window["timestamp"]
    cadence = t.diff().median() if len(t) > 1 else pd.Timedelta(step)
    nxt = t.shift(-1).fillna(t.iloc[-1] + cadence).clip(upper=ext_end)
    weight = (nxt - t).dt.total_seconds().clip(upper=step_s)
    bins = start_bin + ((t - start_bin) // pd.Timedelta(step)) * pd.Timedelta(step)
    binned = pd.DataFrame(index=grid, columns=cols, dtype=float)
    for c in cols:
        w = weight.where(window[c].notna(), 0.0)
        num = (window[c].fillna(0.0) * w).groupby(bins).sum()
        den = w.groupby(bins).sum()
        binned[c] = (num / den.where(den > 0)).reindex(grid)
    for c in cols:
        binned[c] = _fill_short_gaps(binned[c], config.ENV_MAX_INTERP_STEPS)
    binned = binned[(binned.index >= start) & (binned.index < end)]
    grid = binned.index
    for c in cols:
        if binned[c].isna().any():
            log.warning("%s: %d of %d steps missing after %s", c, binned[c].isna().sum(),
                        len(binned), start)
    binned.index = pd.Index(((grid - start) / pd.Timedelta(minutes=1)).astype(int), name="minute")
    return binned


def build_trace(cabin_start=config.ENV_DEFAULT_CABIN_START,
                radiation_start=config.ENV_DEFAULT_RADIATION_START,
                hours: float = 24, radiation: str = "iss") -> pd.DataFrame:
    """One scenario: cabin air from EDA + dose rate from RadLab, each clipped by its own date."""
    rad_source = RADIATION_SOURCES[radiation]
    cabin = clip(load_source(CABIN_SOURCE), cabin_start, hours)
    dose = clip(load_source(rad_source), radiation_start, hours)
    trace = pd.concat([cabin, dose], axis=1)
    absent = [c for c in COLUMNS if c not in trace.columns]
    if absent:
        raise ValueError(f"environment sources give no {absent}: check data/mappings/"
                         f"env_telemetry.yaml (every source must map its fields)")
    trace = trace[COLUMNS]
    # The model reads CO2 and dose rate at every step: an empty step would silently turn off
    # the CO2 effect or make the EVA dose NaN (the radiation rule could never fire).
    holes = {c: int(trace[c].isna().sum()) for c in MODEL_COLUMNS if trace[c].isna().any()}
    if holes:
        raise ValueError(f"environment trace has steps it could not fill {holes} for cabin "
                         f"{cabin_start} / radiation {radiation_start}: pick another day")
    trace.attrs.update(cabin_source=CABIN_SOURCE, cabin_start=str(pd.Timestamp(cabin_start)),
                       radiation_source=rad_source,
                       radiation_start=str(pd.Timestamp(radiation_start)))
    return trace


def load_eda_daily_dose() -> pd.DataFrame:
    """EDA RR-1 daily dose (GCR / SAA / total in mGy/day; dose_rate in uGy/h). Check only.

    Whole days from 2014-09-28 (first full day after config.EDA_ISS_START); 32 days,
    mean 0.202 mGy/day = 8.42 uGy/h, consistent with DosTel2's time-weighted 8.8 uGy/h.
    """
    return load_source(DAILY_DOSE_SOURCE)


# ---------------------------------------------------------------- GOES (flare labels)

def load_goes_xrsb(start, end) -> pd.Series:
    """1-minute GOES-16 XRS-B flux [W/m2] in [start, end); xrsb_flag != 0 -> NaN, no fill."""
    import xarray as xr

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for day in pd.date_range(start.normalize(), end - pd.Timedelta("1ns"), freq="D"):
        path = next(GOES_DIR.glob(f"sci_xrsf-l2-avg1m_g16_d{day:%Y%m%d}_*.nc"), None)
        if path is None:
            raise FileNotFoundError(f"no GOES file for {day:%Y-%m-%d} in {GOES_DIR}")
        with xr.open_dataset(path) as ds:
            flux = ds["xrsb_flux"].where(ds["xrsb_flag"] == 0).to_series()
        parts.append(flux)
    series = pd.concat(parts).sort_index()
    return series[(series.index >= start) & (series.index < end)].rename("xrsb_flux")


def clip_goes(start, hours: float = 24) -> pd.Series:
    """GOES XRS-B flux indexed by minutes since `start` (0, 1, 2, ...); gaps stay NaN."""
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    grid = pd.date_range(start, end, freq="1min", inclusive="left")
    flux = load_goes_xrsb(start, end).reindex(grid)
    flux.index = pd.Index(np.arange(len(grid)), name="minute")
    return flux


def flare_class(flux_w_m2: float, official: bool = True) -> str:
    """Flare class from XRS-B flux; official=True first applies config.GOES_OFFICIAL_SCALE.

    The scale (0.635) is inferred in analysis A2 from the two issued classes, X9.3 and
    X8.2; for other flares it is an estimate. Public material quotes the issued class.
    """
    f = flux_w_m2 * (config.GOES_OFFICIAL_SCALE if official else 1.0)
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if f >= base:
            return f"{letter}{f / base:.1f}"
    return f"A{f / 1e-8:.1f}"
