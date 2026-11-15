"""NASA Data page (page_files/4_Data.py): how the real data enters the model.

Four blocks, each with a chart in the site palette, one line on what the data does in the
model, and the dataset name with its link (data/SOURCES.md). Charts are drawn only from files
committed to the repository (DATA_FILES), so a fresh clone without the data/raw downloads
shows the same page; the numbers that need data we may not redistribute or did not commit
(DosTel2, GOES, Inspiration4) come from the analysis results (notebooks/analysis_results.json).
Each block also folds the original notebook figure (notebooks/figures/A1-A4).
"""
import functools
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spacefuzz import config
from spacefuzz.core import state as st
from spacefuzz.readers import env_trace
from ui import charts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "notebooks/analysis_results.json"
WESAD_PARAMS = "data/processed/wesad_params.json"
EDA_FILE = "data/raw/eda/RR-1_telemetry.csv"
MARS_FILE = "data/raw/radlab/MSL-RAD_Mars_2017-09-04_to_09-18.csv"
DATA_FILES = (RESULTS, WESAD_PARAMS, EDA_FILE, MARS_FILE,
              "notebooks/figures/A1_cabin_co2_temp_humidity.png",
              "notebooks/figures/A2_radiation_dose_and_flares.png",
              "notebooks/figures/A3_wesad_stress_hr_hrv.png",
              "notebooks/figures/A4_inspiration4_clinical_changes.png")
PAL, SURF = charts.PAL, charts.SURF
INFO = SURF["info"]
FIG_H = 330


@functools.lru_cache(maxsize=1)
def results() -> dict:
    return json.loads((ROOT / RESULTS).read_text(encoding="utf-8"))


def _fig(rows: int = 1, cols: int = 1, titles=None, height: int = FIG_H) -> go.Figure:
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.09)
    charts._layout(fig, height)
    fig.update_layout(showlegend=False, margin=dict(l=50, r=16, t=36, b=36))
    return fig


# ---------------------------------------------------------------- 1. cabin air (EDA RR-1)

def _cabin_frame() -> pd.DataFrame:
    return env_trace.load_source(env_trace.CABIN_SOURCE)     # ISS columns, after transit


@functools.lru_cache(maxsize=1)
def _cabin_frame_cached() -> pd.DataFrame:
    return _cabin_frame()


def cabin_figure() -> go.Figure | None:
    """Daily pattern over the 34 ISS days: hourly median and P10-P90, the high-CO2 line."""
    try:
        df = _cabin_frame_cached()
    except (FileNotFoundError, OSError, KeyError, ValueError):
        return None
    hour = df["timestamp"].dt.hour
    fig = _fig(1, 3, ["CO2 [ppm]", "Temperature [°C]", "Humidity [%]"])
    for k, col in enumerate(("co2", "temp", "humidity"), 1):
        g = df.groupby(hour)[col]
        q = pd.DataFrame({"lo": g.quantile(0.1), "med": g.median(), "hi": g.quantile(0.9)})
        fig.add_trace(go.Scatter(x=q.index, y=q["hi"], line=dict(width=0), hoverinfo="skip"),
                      1, k)
        fig.add_trace(go.Scatter(x=q.index, y=q["lo"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(142,203,255,0.18)", hoverinfo="skip"), 1, k)
        fig.add_trace(go.Scatter(x=q.index, y=q["med"], line=dict(color=INFO, width=2),
                                 hovertemplate="%{x}:00 GMT: %{y:.1f}<extra>median</extra>"), 1, k)
        fig.update_xaxes(title_text="hour of day (GMT)", row=1, col=k)
    high = results()["A1"]["co2_high_ppm_p90"]
    fig.add_hline(y=high, line=dict(color=PAL["yellow"], dash="dash", width=1.5), row=1, col=1)
    fig.add_annotation(x=0.5, y=high, xref="x", yref="y", text=f"high CO2 {high:,} ppm (P90)",
                       showarrow=False, yshift=10, xanchor="left",
                       font=dict(color=PAL["yellow"], size=11))
    return fig


# ---------------------------------------------------------------- 2. radiation (RadLab, GOES)

def _mars_frame() -> pd.DataFrame:
    return env_trace.load_source(env_trace.RADIATION_SOURCES["mars"])


@functools.lru_cache(maxsize=1)
def _mars_frame_cached() -> pd.DataFrame:
    return _mars_frame()


def radiation_figure() -> go.Figure | None:
    """Mars surface dose rate (15-min means), the 2017-09 particle event and the flare peaks."""
    try:
        df = _mars_frame_cached()
    except (FileNotFoundError, OSError, KeyError, ValueError):
        return None
    a2 = results()["A2"]
    s = df.set_index("timestamp")["dose_rate"].resample("15min").mean()
    fig = _fig(height=FIG_H)
    fig.add_trace(go.Scatter(x=s.index, y=s.values, line=dict(color=INFO, width=1.4),
                             hovertemplate="%{x|%m-%d %H:%M} UTC: %{y:.1f} µGy/h<extra></extra>"))
    # after the trace: shapes on a subplot without data yet are dropped by plotly
    fig.add_vrect(x0=a2["spe_onset_utc"], x1=a2["spe_end_utc"], line_width=0,
                  fillcolor=PAL["red"], opacity=0.22, layer="below")
    fig.add_hline(y=a2["mars_baseline_ugy_h"], line=dict(color=SURF["muted"], dash="dot",
                                                         width=1))
    for day, f in a2["flares"].items():
        fig.add_vline(x=f["peak_utc"], line=dict(color=PAL["yellow"], dash="dash", width=1))
        fig.add_annotation(x=f["peak_utc"], y=1, yref="paper", showarrow=False, yshift=-8,
                           text=f"{f['official_class']} flare (GOES)", xanchor="left",
                           font=dict(color=PAL["yellow"], size=11))
    fig.add_annotation(x=a2["spe_end_utc"], y=0.62, yref="paper", xanchor="left",
                       xshift=6, showarrow=False, font=dict(color=PAL["red"], size=11),
                       text=f"particle event: peak {a2['spe_peak_multiple']:.2f}x, "
                            f"{a2['spe_duration_h']:.0f} h")
    fig.update_yaxes(title_text="dose rate [µGy/h]")
    return fig


# ---------------------------------------------------------------- 3. stress effect (WESAD)

def stress_figure() -> go.Figure:
    """Each subject's baseline and stress means (fitted parameters, committed)."""
    p = json.loads((ROOT / WESAD_PARAMS).read_text(encoding="utf-8"))["subjects"]
    fig = _fig(1, 2, ["Heart rate [bpm]", "RMSSD [ms]"])
    for k, (key, unit) in enumerate((("hr_mean", "bpm"), ("rmssd_mean", "ms")), 1):
        for subj, c in sorted(p.items()):
            ys = [c["baseline"][key], c["stress"][key]]
            fig.add_trace(go.Scatter(
                x=["baseline", "stress"], y=ys, mode="lines+markers+text",
                text=["", subj], textposition="middle right",
                line=dict(color=INFO, width=1.5), marker=dict(size=8, color=[INFO, PAL["red"]]),
                textfont=dict(color=SURF["muted"], size=10),
                hovertemplate=f"{subj} %{{x}}: %{{y:.1f}} {unit}<extra></extra>"), 1, k)
    return fig


# ---------------------------------------------------------------- 4. clinical baseline (I4)

def clinical_figure() -> go.Figure:
    """Pre-flight ranges of the Inspiration4 crew and SpaceFuzz's synthetic baseline."""
    rng = results()["A4"]["pre_flight_range"]
    base = st.make_baseline()
    fig = _fig(1, 2, ["Blood glucose [mg/dL]", "Hemoglobin [g/dL]"], height=230)
    marks = (("GLUCOSE", base.glucose, "synthetic baseline"),
             ("hemoglobin", sum(config.BASELINE_HEMOGLOBIN_G_DL_RANGE) / 2, "range centre"))
    for k, (key, ours, label) in enumerate(marks, 1):
        lo, hi, unit = rng[key]
        fig.add_trace(go.Bar(x=[hi - lo], base=[lo], y=["Inspiration4 pre-flight"],
                             orientation="h", width=0.45, marker=dict(color="rgba(142,203,255,0.35)",
                                                          line=dict(color=INFO, width=1)),
                             hovertemplate=f"{lo:g}-{hi:g} {unit}<extra></extra>"), 1, k)
        fig.add_trace(go.Scatter(x=[ours], y=["Inspiration4 pre-flight"], mode="markers+text",
                                 marker=dict(color=PAL["green"], size=12, symbol="diamond"),
                                 text=[label], textposition="top center",
                                 textfont=dict(color=PAL["green"], size=10),
                                 hovertemplate=f"{label}: %{{x:.0f}} {unit}<extra></extra>"),
                      1, k)
        fig.update_xaxes(range=[lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.25], row=1, col=k)
    fig.update_yaxes(showticklabels=False)
    return fig


# ---------------------------------------------------------------- text of the four blocks

def blocks() -> list[dict]:
    """Title, chart builder, the line on what it does in the model, dataset links, notes."""
    a1, a2, r = results()["A1"], results()["A2"], results()
    a3 = r["A3"]
    i4 = r["A4"]["pre_flight_range"]
    g, hb = i4["GLUCOSE"], i4["hemoglobin"]
    return [
        {"key": "cabin", "title": "Cabin air", "figure": cabin_figure,
         "model": f"Daily CO2 excursions (usual {100 * a1['daily_peak_excursion_p25_p50_p90'][0]:.0f}"
                  f"-{100 * a1['daily_peak_excursion_p25_p50_p90'][2]:.0f}%, at most "
                  f"{100 * a1['daily_peak_excursion_max']:.0f}%) drive operator B; CO2 above the "
                  f"high line of {a1['co2_high_ppm_p90']:,} ppm during precision work is rule 3. "
                  f"The cabin air of the Mars-base scenarios is ISS air, used as an analog.",
         "facts": f"{a1['days_used']} ISS days (2014), 15-minute steps; median and P10-P90 by "
                  f"hour. Temperature {a1['temp_c_p05_p95'][0]}-{a1['temp_c_p05_p95'][1]} °C, "
                  f"humidity {a1['humidity_pct_p05_p95'][0]:.0f}-"
                  f"{a1['humidity_pct_p05_p95'][1]:.0f}% (P5-P95).",
         "links": [("NASA OSDR Environmental Data App (EDA), RR-1 cabin telemetry",
                    "https://visualization.osdr.nasa.gov/eda/")],
         "figure_file": "notebooks/figures/A1_cabin_co2_temp_humidity.png"},
        {"key": "radiation", "title": "Radiation", "figure": radiation_figure,
         "model": f"The 2017-09 Mars-surface particle event (peak {a2['spe_peak_multiple']:.2f}x the "
                  f"{a2['mars_baseline_ugy_h']} µGy/h baseline, {a2['spe_duration_h']:.0f} h) is the "
                  f"shape operator D drops into the day, about "
                  f"{100 * a2['spe_daily_onset_prob']:.1f}% of days; an EVA during the warning is "
                  f"rule 4.",
         "facts": f"Flares (GOES, official classes): {a2['flares']['2017-09-06']['official_class']} "
                  f"on 09-06 and {a2['flares']['2017-09-10']['official_class']} on 09-10; the event "
                  f"started {a2['flare_peak_to_spe_onset_h']:.1f} h after the second flare. ISS "
                  f"(DosTel2): baseline {a2['iss_baseline_ugy_h_median']} µGy/h, South Atlantic "
                  f"Anomaly peaks {a2['iss_saa_peak_multiple_p99']:.1f}x (P99).",
         "links": [("NASA RadLab: MSL-RAD (Mars surface) and DLR DosTel2 (ISS)",
                    "https://visualization.osdr.nasa.gov/radlab/gui/data-overview/"),
                   ("NOAA GOES-16 XRS (flare times, supplementary)",
                    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
                    "goes16/l2/data/xrsf-l2-avg1m_science/2017/09/")],
         "figure_file": "notebooks/figures/A2_radiation_dose_and_flares.png"},
        {"key": "stress", "title": "Stress effect", "figure": stress_figure,
         "model": f"Acute stress adds +{a3['stress_hr_delta_bpm']} bpm to heart rate and multiplies "
                  f"RMSSD by {a3['stress_rmssd_ratio']}: the parameters of causal chain 1 (stress "
                  f"→ heart rate up, HRV down); the per-minute spread is the real fluctuation "
                  f"laid over every simulated day.",
         "facts": f"95% CI {a3['stress_hr_delta_95ci'][0]}-{a3['stress_hr_delta_95ci'][1]} bpm and "
                  f"{a3['stress_rmssd_ratio_95ci'][0]}-{a3['stress_rmssd_ratio_95ci'][1]} (4 subjects, "
                  f"chest ECG, subject fixed effects). WESAD is non-NASA supplementary data from a "
                  f"ground laboratory study (academic, non-commercial use); the repository holds "
                  f"only the fitted parameters.",
         "links": [("WESAD (Schmidt et al., ICMI 2018), UCI Machine Learning Repository",
                    "https://archive.ics.uci.edu/dataset/465/wesad")],
         "figure_file": "notebooks/figures/A3_wesad_stress_hr_hrv.png"},
        {"key": "clinical", "title": "Clinical baseline", "figure": clinical_figure,
         "model": f"Pre-flight ranges of the civilian Inspiration4 crew bound the synthetic "
                  f"personal baseline: blood glucose {g[0]:.0f}-{g[1]:.0f} mg/dL, hemoglobin "
                  f"{hb[0]:.1f}-{hb[1]:.1f} g/dL.",
         "facts": "4 people and a 3-day flight: these data decide what to watch and when to "
                  "check, not individual thresholds.",
         "links": [("NASA OSDR Inspiration4: complete blood count (OSD-569)",
                    "https://osdr.nasa.gov/bio/repo/data/studies/OSD-569"),
                   ("NASA OSDR Inspiration4: metabolic panel (OSD-575)",
                    "https://osdr.nasa.gov/bio/repo/data/studies/OSD-575")],
         "figure_file": "notebooks/figures/A4_inspiration4_clinical_changes.png"},
    ]
