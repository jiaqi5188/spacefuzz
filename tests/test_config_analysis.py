"""config.py data-derived constants must match notebooks/analysis.ipynb and name their source."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from spacefuzz import config

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "analysis_results.json"
FIGURES = ROOT / "notebooks" / "figures"

SOURCES = {
    "A1": ["EDA_ISS_START", "ENV_CO2_HIGH_PPM", "OP_B_CO2_AMPLIFY_RANGE", "OP_B_CO2_AMPLIFY_MAX",
           "ENV_HIGH_CO2_DAY", "ENV_TEMP_RANGE_C", "ENV_RH_RANGE_PCT", "OP_B_DAILY_PROB",
           "ENV_CO2_MAX_OBSERVED_PPM"],
    "A2": ["RAD_BASELINE_ISS_UGY_H", "RAD_BASELINE_MARS_UGY_H", "OP_D_SAA_PEAK_MULTIPLE",
           "OP_D_SPE_PEAK_MULTIPLE", "OP_D_SPE_DURATION_H", "OP_D_FLARE_TO_SPE_ONSET_H",
           "GOES_OFFICIAL_SCALE", "OP_D_DAILY_PROB"],
    "A3": ["CHAIN1_STRESS_HR_DELTA_BPM", "CHAIN1_STRESS_RMSSD_RATIO", "PHYSIO_HR_FLUCT_SD_BPM",
           "PHYSIO_RMSSD_FLUCT_CV", "BASELINE_HR_RANGE_BPM", "BASELINE_RMSSD_RANGE_MS"],
    "A6": ["PROBABILITY_SAMPLES"],
    "A4": ["BASELINE_GLUCOSE_MG_DL_RANGE", "BASELINE_HEMOGLOBIN_G_DL_RANGE", "I4_R1_PCT_CHANGE"],
}
ALL = [(a, n) for a, names in SOURCES.items() for n in names]


def normalise(v):
    return list(v) if isinstance(v, tuple) else v


@pytest.mark.parametrize("analysis, name", ALL)
def test_constant_matches_notebook_output(analysis, name):
    expected = json.loads(RESULTS.read_text())["constants"][name]
    assert normalise(getattr(config, name)) == expected


@pytest.mark.parametrize("analysis, name", ALL)
def test_constant_comment_names_its_analysis(analysis, name):
    """The assignment (and its continuation lines) must carry a comment citing the analysis."""
    lines = (ROOT / "spacefuzz" / "config.py").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if re.match(rf"{name}\s*=", l))
    block = [lines[i]]
    for l in lines[i + 1:]:
        if l.startswith((" ", "\t")) and l.strip():
            block.append(l)
        else:
            break
    comments = " ".join(l.split("#", 1)[1] for l in block if "#" in l)
    assert re.search(rf"\b{analysis}\b", comments), f"{name}: no '# ... {analysis} ...' comment"


def test_ranges_are_ordered():
    for name in ("OP_B_CO2_AMPLIFY_RANGE", "ENV_TEMP_RANGE_C", "ENV_RH_RANGE_PCT",
                 "BASELINE_HR_RANGE_BPM", "BASELINE_RMSSD_RANGE_MS",
                 "BASELINE_GLUCOSE_MG_DL_RANGE", "BASELINE_HEMOGLOBIN_G_DL_RANGE"):
        lo, hi = getattr(config, name)
        assert lo < hi, name


def test_four_figures_exist():
    for prefix in SOURCES:
        assert list(FIGURES.glob(f"{prefix}_*.png")), f"missing {prefix} figure"


def test_spe_profile_matches_peak_multiple():
    prof = pd.read_csv(ROOT / config.OP_D_SPE_PROFILE_PATH)
    assert prof["minute"].iloc[0] == 0 and prof["minute"].diff().dropna().eq(15).all()
    assert prof["excess_over_baseline"].max() + 1 == pytest.approx(config.OP_D_SPE_PEAK_MULTIPLE,
                                                                   abs=0.01)


def test_operator_b_distribution_keeps_the_observed_tail():
    """Sampling pool for operator B: real daily excursions, so 11-23% is common and the
    rare 31-40% days stay reachable (upper bound = observed maximum)."""
    ex = pd.read_csv(ROOT / config.OP_B_CO2_EXCURSIONS_PATH)["excursion"]
    lo, hi = config.OP_B_CO2_AMPLIFY_RANGE
    assert ex.max() == pytest.approx(config.OP_B_CO2_AMPLIFY_MAX)
    assert config.OP_B_CO2_AMPLIFY_MAX >= 0.30 > hi
    assert 0.5 <= ex.between(lo, hi).mean() <= 0.8          # most draws in the usual range
    assert (ex > 0.30).sum() >= 1                           # the tail is in the pool


def test_r45_glucose_artifacts_are_excluded_from_change_analysis():
    excluded = json.loads(RESULTS.read_text())["A4"]["excluded_points"]
    for subj in ("C002", "C003", "C004"):              # same delayed-processing samples:
        assert [subj, "R+45", "GLUCOSE"] in excluded   # glucose consumed by cells,
        assert [subj, "R+45", "POTASSIUM"] in excluded # potassium leaked from cells


def test_notebook_states_the_inspiration4_limitation_verbatim():
    nb = (ROOT / "notebooks" / "analysis.ipynb").read_text()
    assert "These data only decide what to watch and when to check, " \
           "not individual thresholds." in nb


def test_a1_excludes_the_stuck_co2_run_found_by_the_quality_gate():
    a1 = json.loads(RESULTS.read_text())["A1"]
    assert a1["co2_stuck_excluded"] == ["2014-09-21 04:24:00", "2014-09-27 13:25:00"]


def test_a1_transit_check_figure_exists():
    assert (FIGURES / "A1_check_rr1_transit.png").exists()
