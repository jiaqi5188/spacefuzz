"""Tests for physio_replay.py: ECG -> HR/HRV, label segmentation, and sampling."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacefuzz import config
from spacefuzz.readers import physio_replay as pr

FS = config.WESAD_CHEST_HZ
ROOT = Path(__file__).resolve().parents[1]


def synthetic_ecg(rr_s: np.ndarray, fs: int = FS, noise: float = 0.02, invert=False, seed=0):
    """Gaussian 'R waves' at the given RR intervals plus a little noise and baseline wander."""
    rng = np.random.default_rng(seed)
    beats = np.cumsum(np.concatenate([[0.5], rr_s]))
    n = int((beats[-1] + 1.0) * fs)
    t = np.arange(n) / fs
    ecg = 0.1 * np.sin(2 * np.pi * 0.3 * t) + noise * rng.standard_normal(n)
    for b in beats:
        i = int(b * fs)
        lo, hi = max(i - 20, 0), min(i + 20, n)
        ecg[lo:hi] += np.exp(-0.5 * ((np.arange(lo, hi) - i) / 4.0) ** 2)
    return (-ecg if invert else ecg), beats


# ---- R-peak detection and HR/HRV

@pytest.mark.parametrize("invert", [False, True])
def test_detects_r_peaks_at_known_times(invert):
    rr = np.full(120, 0.8)                       # 75 bpm for ~96 s
    ecg, beats = synthetic_ecg(rr, invert=invert)
    peaks = pr.detect_r_peaks(ecg, FS)
    assert len(peaks) == pytest.approx(len(beats), abs=1)
    assert np.median(np.diff(peaks)) / FS == pytest.approx(0.8, abs=0.01)


def test_hr_and_rmssd_per_window():
    rng = np.random.default_rng(1)
    rr = 0.75 + 0.03 * rng.standard_normal(300)  # ~80 bpm with beat-to-beat variability
    ecg, _ = synthetic_ecg(rr)
    peaks = pr.detect_r_peaks(ecg, FS)
    win = pr.hrv_windows(peaks, FS, start=0, end=len(ecg))
    assert list(win.columns) == ["t_s", "hr_bpm", "rmssd_ms"]
    assert win["hr_bpm"].mean() == pytest.approx(80, abs=2)
    expected_rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000
    assert win["rmssd_ms"].mean() == pytest.approx(expected_rmssd, rel=0.25)


def test_ectopic_beats_are_removed_from_rr():
    rr = np.full(100, 0.8)
    rr[50] = 0.4                                  # premature beat
    rr[51] = 1.2                                  # compensatory pause
    clean = pr.clean_rr(rr)
    assert np.isnan(clean[50]) and np.isnan(clean[51])
    assert np.nanstd(clean) < 1e-9


# ---- segmentation by label

def test_label_runs_keep_only_labels_1_to_4():
    labels = np.array([0] * 5 + [1] * 10 + [0] * 3 + [2] * 8 + [6] * 4 + [4] * 6)
    runs = pr.label_runs(labels)
    assert [(lab, s, e) for lab, s, e in runs] == [(1, 5, 15), (2, 18, 26), (4, 30, 36)]


# ---- runtime sampling on a small standard frame

def tiny_frame():
    rows = []
    for subj, base in (("S2", 70), ("S3", 80)):
        for seg, cond, offset, n in ((1, "baseline", 0, 20), (2, "stress", 15, 10),
                                     (3, "meditation", -3, 7)):
            for m in range(n):
                rows.append({"subject": subj, "condition": cond, "segment_id": seg,
                             "minute": float(m), "hr": base + offset + (m % 3),
                             "rmssd": 50.0 - offset})
    return pd.DataFrame(rows)


def test_sample_returns_requested_length_and_deltas_from_subject_baseline():
    rp = pr.PhysioReplay(tiny_frame())
    seg = rp.sample("stress", minutes=8, rng=np.random.default_rng(config.RANDOM_SEED))
    assert len(seg) == 8 and seg["minute"].tolist() == [float(m) for m in range(8)]
    assert seg["condition"].eq("stress").all()
    assert seg["hr_delta"].mean() == pytest.approx(15, abs=1.5)        # vs own baseline
    assert seg["rmssd_ratio"].iloc[0] == pytest.approx(35 / 50)


def test_relax_pool_and_reproducible_draws():
    rp = pr.PhysioReplay(tiny_frame())
    a = rp.sample("relax", minutes=5, rng=np.random.default_rng(3))
    b = rp.sample("relax", minutes=5, rng=np.random.default_rng(3))
    assert a["condition"].isin(config.RELAX_CONDITIONS).all()
    pd.testing.assert_frame_equal(a, b)


def test_sample_too_long_raises():
    with pytest.raises(ValueError, match="no stress segment"):
        pr.PhysioReplay(tiny_frame()).sample("stress", minutes=60)


# ---- the local WESAD export (git-ignored; built once from the 1 GB pickles)

EXPORT = ROOT / "data" / "processed" / "wesad_hr_hrv.csv"
PARAMS = ROOT / "data" / "processed" / "wesad_params.json"


@pytest.mark.skipif(not EXPORT.exists(), reason="run python -m spacefuzz.readers.physio_replay --build first")
def test_wesad_export_is_physiological_and_stress_raises_hr():
    rp = pr.PhysioReplay.load()            # goes through generic_reader + wearable_series.yaml
    df = rp.frame
    assert set(df["subject"]) == {"S2", "S3", "S4", "S5"}
    assert df["hr"].between(40, 180).all() and df["rmssd"].between(3, 300).all()
    by = df.groupby("condition")["hr"].mean()
    assert by["stress"] > by["baseline"]
    assert PARAMS.exists()
