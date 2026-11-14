"""Real heart-rate / HRV fluctuation segments from WESAD, sampled on demand.

WESAD has no heart-rate column. A one-off build step (`python -m spacefuzz.readers.physio_replay
--build`) reads each ~1 GB pickle, detects R peaks in the chest ECG (700 Hz),
and writes one HR / RMSSD value per 60 s window for every contiguous run of
labels 1 baseline, 2 stress, 3 amusement, 4 meditation (0 and 5-7 dropped):
- data/processed/wesad_hr_hrv.csv: the export (subject, wesad_label, segment, t_s,
  hr_bpm, rmssd_ms)
- data/processed/wesad_params.json: fitted per-subject / per-condition statistics
The app never loads the raw pickles: PhysioReplay.load() reads the export through
generic_reader + data/mappings/wearable_series.yaml, so any wearable source mapped
to that standard works the same way.

sample(condition, minutes) returns a real segment with hr_delta (bpm above the
subject's own baseline mean) and rmssd_ratio (relative to the baseline mean), to
overlay on the state model (plan 3.2). "relax" draws from baseline + meditation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

from spacefuzz import config
from spacefuzz.readers import generic_reader as gr

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "wesad"
EXPORT = ROOT / "data" / "processed" / "wesad_hr_hrv.csv"
PARAMS = ROOT / "data" / "processed" / "wesad_params.json"
MAPPING = ROOT / "data" / "mappings" / "wearable_series.yaml"


# ---------------------------------------------------------------- ECG -> HR / HRV

def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Sample indices of R peaks: band-pass the QRS band, fix polarity, adaptive height."""
    sos = butter(3, config.ECG_BANDPASS_HZ, btype="bandpass", fs=fs, output="sos")
    f = sosfiltfilt(sos, np.asarray(ecg, dtype=float).ravel())
    if -np.percentile(f, 0.5) > np.percentile(f, 99.5):   # inverted lead
        f = -f
    local_max = maximum_filter1d(f, size=2 * fs)
    peaks, _ = find_peaks(f, distance=int(config.ECG_MIN_RR_S * fs),
                          height=config.ECG_PEAK_REL_HEIGHT * local_max)
    return peaks


def clean_rr(rr_s: np.ndarray) -> np.ndarray:
    """NaN out RR intervals outside 40-180 bpm or > 20% off the local median."""
    rr = np.asarray(rr_s, dtype=float).copy()
    rr[(rr < config.ECG_MIN_RR_S) | (rr > config.ECG_MAX_RR_S)] = np.nan
    med = median_filter(np.nan_to_num(rr, nan=np.nanmedian(rr)), size=11, mode="nearest")
    rr[np.abs(rr - med) > config.ECG_RR_MAX_JUMP * med] = np.nan
    return rr


def hrv_windows(peaks: np.ndarray, fs: int, start: int, end: int) -> pd.DataFrame:
    """HR [bpm] and RMSSD [ms] per HRV_WINDOW_S window inside samples [start, end)."""
    win = config.HRV_WINDOW_S * fs
    p = peaks[(peaks >= start) & (peaks < end)]
    rows = []
    if len(p) > 2:
        rr = clean_rr(np.diff(p) / fs)
        beat_t = p[1:]                                    # RR interval ends at this beat
        for w0 in range(start, end - win + 1, win):
            sel = (beat_t >= w0) & (beat_t < w0 + win)
            r = rr[sel]
            good = r[~np.isnan(r)]
            if good.sum() < config.HRV_MIN_COVERAGE * config.HRV_WINDOW_S:
                continue
            pairs = np.diff(r)
            pairs = pairs[~np.isnan(pairs)]
            if len(pairs) < 2:
                continue
            rows.append({"t_s": (w0 - start) // fs,
                         "hr_bpm": 60.0 / good.mean(),
                         "rmssd_ms": float(np.sqrt(np.mean(pairs ** 2)) * 1000)})
    return pd.DataFrame(rows, columns=["t_s", "hr_bpm", "rmssd_ms"])


def label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, start, end) for each contiguous run of a kept label (1-4)."""
    labels = np.asarray(labels).ravel()
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)
            if int(labels[s]) in config.WESAD_LABELS]


# ---------------------------------------------------------------- build (one-off)

def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")   # Python 2 pickle


def process_subject(path: Path) -> pd.DataFrame:
    data = _load_pickle(path)
    fs = config.WESAD_CHEST_HZ
    ecg = data["signal"]["chest"]["ECG"].ravel()
    labels = data["label"].ravel()
    subject = str(data.get("subject", path.stem))
    peaks = detect_r_peaks(ecg, fs)
    frames = []
    for seg, (lab, s, e) in enumerate(label_runs(labels), 1):
        w = hrv_windows(peaks, fs, s, e)
        w.insert(0, "segment", seg)
        w.insert(0, "wesad_label", lab)
        w.insert(0, "subject", subject)
        frames.append(w)
    del data, ecg, labels
    return pd.concat(frames, ignore_index=True)


def fit_params(df: pd.DataFrame) -> dict:
    """Per subject x condition mean / std of HR and RMSSD, plus pooled condition stats.

    hr_dev_sd / rmssd_dev_cv are the spread around each segment's own mean, the same
    deviation the model's real-series pool uses (model.Fluctuation.from_wesad); hr_std /
    rmssd_std also contain the differences between segments and are kept for reference."""
    named = df.assign(condition=df["wesad_label"].map(config.WESAD_LABELS))
    seg = named.groupby(["subject", "segment"])[["hr_bpm", "rmssd_ms"]].transform("mean")
    named = named.assign(hr_dev=named["hr_bpm"] - seg["hr_bpm"],
                         rmssd_rel=named["rmssd_ms"] / seg["rmssd_ms"] - 1)
    stats = lambda g: {"hr_mean": round(g["hr_bpm"].mean(), 2), "hr_std": round(g["hr_bpm"].std(), 2),
                       "rmssd_mean": round(g["rmssd_ms"].mean(), 2),
                       "rmssd_std": round(g["rmssd_ms"].std(), 2),
                       "hr_dev_sd": round(g["hr_dev"].std(), 3),
                       "rmssd_dev_cv": round(g["rmssd_rel"].std(), 4), "minutes": int(len(g))}
    return {
        "source": "WESAD chest ECG, R-peak detection, 60 s windows (physio_replay.py)",
        "subjects": {s: {c: stats(g2) for c, g2 in g.groupby("condition")}
                     for s, g in named.groupby("subject")},
        "pooled": {c: stats(g) for c, g in named.groupby("condition")},
    }


def build(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("S*/S*.pkl"))
    if not paths:
        sys.exit(f"no WESAD pickles under {raw_dir}")
    frames = []
    for p in paths:
        print(f"processing {p.relative_to(ROOT)} ...", flush=True)
        frames.append(process_subject(p))
    df = pd.concat(frames, ignore_index=True)
    df["hr_bpm"] = df["hr_bpm"].round(2)
    df["rmssd_ms"] = df["rmssd_ms"].round(2)
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPORT, index=False)
    PARAMS.write_text(json.dumps(fit_params(df), indent=2) + "\n", encoding="utf-8")
    return df


# ---------------------------------------------------------------- runtime

class PhysioReplay:
    """Sample real HR / HRV segments from a wearable-standard frame."""

    def __init__(self, frame: pd.DataFrame):
        cols = gr.STANDARD_COLUMNS["wearable"]
        self.frame = frame[cols].dropna(subset=["hr", "rmssd"]).reset_index(drop=True)
        base = self.frame[self.frame["condition"] == "baseline"].groupby("subject")
        self.baseline = pd.DataFrame({"hr": base["hr"].mean(), "rmssd": base["rmssd"].mean()})

    @classmethod
    def load(cls, mapping=MAPPING, source: str = "wesad_export") -> "PhysioReplay":
        return cls(gr.read_source(mapping, source))

    def sample(self, condition: str, minutes: int, rng: np.random.Generator | None = None
               ) -> pd.DataFrame:
        """A random `minutes`-long run of consecutive rows from one real segment of `condition`
        (consecutive 60 s windows; a window dropped for poor ECG leaves a time gap)."""
        rng = rng or np.random.default_rng(config.RANDOM_SEED)
        conds = config.RELAX_CONDITIONS if condition == "relax" else (condition,)
        pool = self.frame[self.frame["condition"].isin(conds)]
        segs = [g.sort_values("minute") for _, g in pool.groupby(["subject", "segment_id"])
                if len(g) >= minutes]
        if not segs:
            raise ValueError(f"no {condition} segment with at least {minutes} minutes")
        seg = segs[rng.integers(len(segs))]
        i0 = int(rng.integers(len(seg) - minutes + 1))
        out = seg.iloc[i0:i0 + minutes].copy()
        out["minute"] = out["minute"] - out["minute"].iloc[0]
        subj = out["subject"].iloc[0]
        if subj in self.baseline.index:
            out["hr_delta"] = out["hr"] - self.baseline.at[subj, "hr"]
            out["rmssd_ratio"] = out["rmssd"] / self.baseline.at[subj, "rmssd"]
        else:
            out["hr_delta"] = np.nan
            out["rmssd_ratio"] = np.nan
        return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="WESAD HR/HRV segments")
    parser.add_argument("--build", action="store_true",
                        help="process data/raw/wesad pickles into data/processed (one-off)")
    parser.add_argument("--refit", action="store_true",
                        help="rewrite wesad_params.json from the local per-minute export")
    args = parser.parse_args()
    if args.refit:
        df = pd.read_csv(EXPORT)
        PARAMS.write_text(json.dumps(fit_params(df), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PARAMS.relative_to(ROOT)} from {EXPORT.relative_to(ROOT)}")
    elif args.build:
        df = build()
        print(f"wrote {EXPORT.relative_to(ROOT)} ({len(df)} rows) and "
              f"{PARAMS.relative_to(ROOT)}")
        print(df.groupby(["subject", "wesad_label"])["hr_bpm"].agg(["count", "mean"]).round(1))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
