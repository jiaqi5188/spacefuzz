"""Charts (plan 4.2 / 4.3, Plotly, dark): crash boundary heatmap, trajectory fan, trends.

- Crash boundary heatmap (for the video): rows = EVA extension, columns = sleep cut, cell =
  probability-track risk; colour bands follow the rating cut-offs (green < 1%, yellow 1-5%,
  red > 5%, darker to brighter red with risk) and every cell carries its value. The boundary
  the guided track found is drawn as rings (per EVA-extension bin, midway between the
  furthest surviving sleep cut and the closest crash beyond it); individual guided samples
  are not drawn. The deterministic minimum perturbation is marked.
- Trajectory fan: the kept probability-track runs as thin lines, crashed runs in red, as
  planned vs adjusted side by side on one y axis, with the illustrative threshold.
- Trends (left column): 14 days of synthetic example history + today's check-in, with the
  personal baseline and the illustrative threshold.

Both result charts are drawn once, when playback ends (data-testid="charts-done" marks them);
widgets live in st.fragment so switching a view never reruns the page or reloads the iframes.
"""
import dataclasses
import functools

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from spacefuzz import config
from spacefuzz.core import model as md
from spacefuzz.fuzzing import mutator as mu
from spacefuzz.core import state
import ui_config
from ui import checkin_store

PAL = ui_config.COLOR
SURF = ui_config.SURFACE
BG, GRID, INK, MUTED = SURF["panel"], SURF["grid"], SURF["ink"], SURF["muted"]
PLOT_CONFIG = {"displayModeBar": False}

# Colour per rating band: (low end, high end) of the band (palette in ui_config).
BAND = ui_config.BAND


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%" if p < 0.1 else f"{100 * p:.0f}%"


def _layout(fig, height: int) -> None:
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=INK, size=12), margin=dict(l=50, r=20, t=40, b=45),
                      legend=dict(orientation="h", y=-0.22, x=0, font=dict(color=MUTED)),
                      hoverlabel=dict(bgcolor=SURF["panel_2"], font_color=INK))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)


# ---------------------------------------------------------------- heatmap

def rating_colorscale() -> list:
    """Plotly colourscale on risk 0-1 with the rating cut-offs as hard band edges."""
    g, y = config.RISK_GREEN_BELOW, config.RISK_RED_ABOVE
    eps = 1e-6
    return [[0.0, BAND["green"][0]], [g - eps, BAND["green"][1]],
            [g, BAND["yellow"][0]], [y, BAND["yellow"][1]],
            [y + eps, BAND["red"][0]], [1.0, BAND["red"][1]]]


def guided_boundary(hist: dict, bin_h: float = 0.25) -> dict:
    """Per EVA-extension bin with surviving samples: midway between the furthest surviving
    sleep cut and the closest crash beyond it (the furthest survivor when nothing crashed
    beyond it). Bins where every guided sample crashed have no boundary point."""
    bins: dict = {}
    for i, d in enumerate(hist.get("discarded", [])):
        if d:
            continue
        y = round(round(hist["eva_extend_h"][i] / bin_h) * bin_h, 3)
        b = bins.setdefault(y, {"surv": [], "crash": []})
        b["crash" if hist["crashed"][i] else "surv"].append(hist["sleep_cut_h"][i])
    ys, xs = [], []
    for y in sorted(bins):
        b = bins[y]
        if not b["surv"]:
            continue                     # nothing survived here: no boundary inside the bin
        edge = max(b["surv"])
        beyond = [c for c in b["crash"] if c > edge]
        ys.append(y)
        xs.append(round((edge + min(beyond)) / 2 if beyond else edge, 3))
    return {"x": xs, "y": ys}


def heatmap_figure(res: dict, height: int = 400) -> go.Figure:
    fc = res["forecast"]
    hm = fc["heatmap"]
    cuts, exts = hm["sleep_cut_h"], hm["eva_extend_h"]
    z = [[hm["risk"][i][j] for i in range(len(cuts))] for j in range(len(exts))]
    text = [[_pct(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        x=cuts, y=exts, z=z, zmin=0, zmax=1, colorscale=rating_colorscale(), text=text,
        texttemplate="%{text}", textfont=dict(color=INK, size=12), xgap=2, ygap=2,
        name="risk (probability track)",
        hovertemplate="sleep cut %{x} h, EVA %{y:+} h<br>risk %{text}<extra></extra>",
        showscale=False))
    lo, hi = min(exts), max(exts)
    notes = []
    hist = (fc.get("guided_track") or {}).get("history")
    ring = None
    if hist:
        # Guided samples are not drawn as points: inside the grid they would sit on the cell
        # labels, and guided crashes stack other operators (e.g. A also raises the heat rate),
        # so they are not like-for-like with the cells. The boundary is drawn as rings.
        b = guided_boundary(hist)
        inside = [(x, y) for x, y in zip(b["x"], b["y"]) if lo - 1e-9 <= y <= hi + 1e-9]
        if inside:
            ring = inside
            fig.add_trace(go.Scatter(
                x=[x for x, _ in inside], y=[y for _, y in inside], mode="markers",
                name="guided-track boundary",
                marker=dict(symbol="circle-open", size=28, color=INK, line=dict(width=2.5)),
                hovertemplate="guided boundary: sleep cut %{x:.2f} h at EVA %{y:+} h"
                              "<extra></extra>"))
        survived_below = any(not c and not d and e < lo - 1e-9 for e, c, d in
                             zip(hist["eva_extend_h"], hist["crashed"], hist["discarded"]))
        if survived_below:
            notes.append("guided survivors: EVA as planned or shorter only")
    minp = fc.get("min_crash_perturbation")
    if isinstance(minp, dict) and minp.get("change") == "less sleep":
        sx = minp["minutes"] / 60
        fig.add_trace(go.Scatter(
            x=[sx], y=[0], mode="markers", name=minp["text"].split(" (")[0],
            marker=dict(symbol="star", size=14, color=PAL["yellow"],
                        line=dict(color=BG, width=1)),
            hovertemplate=minp["text"] + "<extra></extra>"))
        at0 = [x for x, y in (ring or []) if abs(y) < 1e-9]
        if at0:
            notes.insert(0, f"guided boundary {at0[0]:.2f} h · scan {sx:.2f} h "
                            f"({minp['minutes']} min): same edge")
    for k, text in enumerate(notes):
        fig.add_annotation(text=text, xref="paper", yref="y", x=0, y=lo - 0.45 - 0.3 * k,
                           xanchor="left", showarrow=False, font=dict(color=MUTED, size=10))
    _layout(fig, height)
    fig.update_layout(title=dict(text="Crash boundary: sleep cut x EVA extension",
                                 font=dict(size=14)), margin=dict(b=90))
    g, r = config.RISK_GREEN_BELOW, config.RISK_RED_ABOVE
    fig.add_annotation(text=f"cell = risk, {hm.get('samples_per_cell', 0):,} runs each: green < "
                            f"{g:.0%}, yellow {g:.0%}-{r:.0%}, red > {r:.0%} (illustrative)",
                       xref="paper", yref="paper", x=0, y=1.0, yanchor="bottom", xanchor="left",
                       showarrow=False, font=dict(color=MUTED, size=10))
    fig.update_xaxes(title_text="Sleep cut last night (h)", dtick=0.5)
    fig.update_yaxes(title_text="EVA extension (h)", tickvals=exts,
                     range=[lo - 0.25 - (0.3 * len(notes) + 0.2 if notes else 0), hi + 0.25])
    if len(exts) == 1 and not (hist and any(hist.get("eva_extend_h", []))):
        fig.update_yaxes(title_text="no EVA in this plan", showticklabels=False)
    return fig


# ---------------------------------------------------------------- trajectory fan

FAN_FIELDS = {"heart_rate": "Heart rate (bpm)", "cognitive_fatigue": "Cognitive fatigue (0-100)"}


def _threshold(field: str, baseline: state.Baseline) -> tuple[float, str]:
    if field == "heart_rate":
        v = config.CARDIO_HR_MAX_FRACTION * state.max_heart_rate(baseline)
        return v, f"{v:.0f} bpm = {config.CARDIO_HR_MAX_FRACTION:.0%} of max HR (illustrative)"
    v = config.COG_FATIGUE_ABOVE
    return v, f"fatigue {v:.0f} (illustrative)"


def _group(traces: dict, field: str, minutes: list, crashed: int) -> tuple[list, list]:
    xs, ys = [], []
    hours = [m / 60 for m in minutes]
    for run, c in zip(traces[field], traces["crashed"]):
        if c == crashed:
            xs += hours + [None]
            ys += list(run) + [None]
    return xs, ys


def fan_figure(res: dict, field: str, baseline: state.Baseline, height: int = 358) -> go.Figure:
    fc, adj = res["forecast"], res.get("adjust")
    prob = fc["probability_track"]
    panels = [("As planned", prob["traces"], prob["trajectory"], prob["trajectory_minutes"],
               fc["risk_probability"], fc["risk_rating"])]
    det = (adj or {}).get("details") or {}
    if det.get("traces_after"):
        pa = adj["plan_adjustment"]
        panels.append(("Adjusted", det["traces_after"], det["trajectory_after"],
                       det["trajectory_minutes_after"], pa["risk_after"], pa["rating_after"]))
    n = len(panels)
    shown = len(prob["traces"]["crashed"])
    titles = [f"{name} · {100 * p:.1f}% {r}<br><span style='font-size:11px'>"
              f"{sum(tr['crashed'])} of {len(tr['crashed'])} shown crashed</span>"
              for name, tr, _, _, p, r in panels]
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, subplot_titles=titles,
                        horizontal_spacing=0.04)
    thr, label = _threshold(field, baseline)
    for k, (_, tr, traj, minutes, _, _) in enumerate(panels, start=1):
        x, y = _group(tr, field, minutes, 0)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="no crash",
                                 line=dict(color=MUTED, width=1), opacity=0.5,
                                 showlegend=k == 1, hoverinfo="skip"), row=1, col=k)
        x, y = _group(tr, field, minutes, 1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="crashed simulation",
                                 line=dict(color=PAL["red"], width=1), opacity=0.4,
                                 showlegend=k == 1, hoverinfo="skip"), row=1, col=k)
        fig.add_trace(go.Scatter(x=[m / 60 for m in minutes], y=traj[field]["p50"],
                                 mode="lines", name="median", showlegend=k == 1,
                                 line=dict(color=INK, width=2),
                                 hovertemplate="median %{y:.0f}<extra></extra>"), row=1, col=k)
        fig.add_hline(y=thr, line=dict(color=PAL["yellow"], width=1.5, dash="dot"), row=1, col=k)
    fig.add_annotation(text=f"threshold: {label}", xref="paper", yref="y", x=0, y=thr,
                       xanchor="left", yanchor="bottom", showarrow=False,
                       font=dict(color=PAL["yellow"], size=11))
    _layout(fig, height)
    fig.update_layout(title=dict(text=f"{FAN_FIELDS[field]}: {shown} of {prob['samples']:,} "
                                      f"simulations each", font=dict(size=14), y=0.98),
                      margin=dict(t=85))
    ticks = list(range(6, 31, 3))
    fig.update_xaxes(tickvals=ticks, ticktext=[f"{t % 24:02d}:00" for t in ticks])
    return fig


# ---------------------------------------------------------------- trends

TRENDS = {"heart_rate": "Heart rate", "blood_pressure": "Blood pressure",
          "blood_glucose": "Glucose", "sleep": "Sleep", "reaction_time": "Reaction time"}
# Whole short words for a narrow column (menu open); the chart title keeps the full name.
TRENDS_SHORT = {"heart_rate": "Heart", "blood_pressure": "BP", "blood_glucose": "Glucose",
                "sleep": "Sleep", "reaction_time": "Reaction"}
TABS_SHORT_BELOW_PX = 395       # the full names need about 380 px


def trend_tab_css() -> str:
    """Tab labels follow the width of the tab row itself (CSS container query), so the five
    tabs always fit on one line: full names when there is room, short words otherwise."""
    short = "".join(f'.st-key-trend_field button:nth-of-type({i}) p::after {{ content:"{v}"; '
                    f'font-size:13px; }}\n' for i, v in enumerate(TRENDS_SHORT.values(), 1))
    return (".st-key-trend_field { container-type:inline-size; width:100%; }\n"
            f"@container (max-width: {TABS_SHORT_BELOW_PX}px) {{\n"
            ".st-key-trend_field button p { font-size:0; }\n" + short + "}")


@functools.lru_cache(maxsize=8)
def _today(name: str, checkin_json: str | None = None) -> tuple:
    sc = mu.Scenario.load(name)
    if checkin_json:
        sc = dataclasses.replace(sc, checkin=checkin_store.from_json(name, checkin_json))
    b = state.make_baseline()
    out = md.simulate(sc.plan, sc.trace, sc.checkin, b)
    return sc.checkin, b, float(out["heart_rate"].iloc[0]), float(out["reaction_time_ms"].iloc[0])


def trend_series(name: str, key: str, checkin_json: str | None = None) -> dict:
    """14 days of synthetic example history + today (day 0) from the current check-in."""
    c, b, hr_now, rt_model = _today(name, checkin_json)
    h = b.history
    days = list(h.index) + [0]
    s = {"key": key, "label": TRENDS[key], "days": days}
    if key == "heart_rate":
        lim = b.resting_hr + config.ACT_NOW_HR_ABOVE_BASELINE_BPM
        s.update(values=list(h["resting_hr"]) + [round(hr_now, 1)], unit="bpm",
                 baseline=b.resting_hr, threshold=lim,
                 threshold_label=f"act at {lim:.0f} bpm (illustrative)")
    elif key == "blood_pressure":
        s.update(values=list(h["sbp"]) + [c.systolic_bp], values2=list(h["dbp"]) + [c.diastolic_bp],
                 unit="mmHg", baseline=b.sbp, baseline2=b.dbp,
                 threshold=config.ACT_NOW_BP_RANGE["sbp"][1],
                 threshold2=config.ACT_NOW_BP_RANGE["dbp"][1],
                 threshold_label=f"upper limits {config.ACT_NOW_BP_RANGE['sbp'][1]:.0f} / "
                                 f"{config.ACT_NOW_BP_RANGE['dbp'][1]:.0f} mmHg (illustrative)")
    elif key == "blood_glucose":
        s.update(values=list(h["glucose"]) + [c.blood_glucose], unit="mg/dL", baseline=b.glucose,
                 threshold=config.GLUCOSE_LOW_MG_DL,
                 threshold_label=f"low below {config.GLUCOSE_LOW_MG_DL:.0f} mg/dL (illustrative)")
    elif key == "sleep":
        nights = list(c.sleep_last_3_nights_h)          # the check-in owns the last 3 nights
        past = list(h["sleep_h"])
        past = past[:len(past) - (len(nights) - 1)] if len(nights) > 1 else past
        s.update(values=past + nights, unit="h",
                 baseline=b.sleep_h, threshold=config.ACT_NOW_SLEEP_SHORT_H,
                 threshold_label=f"short below {config.ACT_NOW_SLEEP_SHORT_H:.0f} h "
                                 f"(illustrative)")
    else:
        today = c.reaction_time_ms if c.reaction_time_ms is not None else rt_model
        s.update(values=list(h["reaction_ms"]) + [today], unit="ms", baseline=b.reaction_ms)
    s["values"] = [round(float(v), 1) for v in s["values"]]
    if "values2" in s:
        s["values2"] = [round(float(v), 1) for v in s["values2"]]
    return s


def trend_figure(s: dict, height: int = 230, today_status: str = "green") -> go.Figure:
    fig = go.Figure()
    series = [("values", "baseline", "threshold", "")]
    if "values2" in s:
        series = [("values", "baseline", "threshold", "systolic "),
                  ("values2", "baseline2", "threshold2", "diastolic ")]
    for vk, bk, tk, prefix in series:
        fig.add_trace(go.Scatter(
            x=s["days"][:-1], y=s[vk][:-1], mode="lines+markers",
            name="synthetic example history", showlegend=vk == "values",
            line=dict(color=MUTED, width=2), marker=dict(size=5),
            hovertemplate=f"day %{{x}}: {prefix}%{{y}} {s['unit']}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=s["days"][-1:], y=s[vk][-1:], mode="markers", name="today",
            showlegend=vk == "values",
            marker=dict(size=11, color=PAL.get(today_status, INK), line=dict(color=BG, width=2)),
            hovertemplate=f"today: {prefix}%{{y}} {s['unit']}<extra></extra>"))
        fig.add_hline(y=s[bk], line=dict(color=INK, width=1, dash="dash"),
                      annotation=dict(text=f"{prefix}baseline {s[bk]:.0f} {s['unit']}",
                                      font=dict(color=MUTED, size=10), xanchor="left", x=0))
        if s.get(tk) is not None:
            fig.add_hline(y=s[tk], line=dict(color=PAL["yellow"], width=1, dash="dot"))
    _layout(fig, height)
    fig.update_layout(margin=dict(l=40, r=10, t=30, b=55),
                      title=dict(text=f"{s['label']} ({s['unit']}): synthetic example history, "
                                      f"14 days + today",
                                 font=dict(size=12)),
                      legend=dict(orientation="h", y=-0.28, x=0, font=dict(color=MUTED)))
    ticks = list(range(s["days"][0], 1, 2))
    fig.update_xaxes(tickvals=ticks, ticktext=[("today" if t == 0 else str(t)) for t in ticks])
    if s.get("threshold_label"):
        fig.add_annotation(text=s["threshold_label"], xref="paper", yref="paper", x=1, y=1.02,
                           xanchor="right", showarrow=False, font=dict(color=PAL["yellow"], size=10))
    return fig


# ---------------------------------------------------------------- Streamlit

@st.fragment
def show_trends(static: dict) -> None:
    """Left column: one indicator at a time (fragment: switching never reruns the page)."""
    st.html(f"<style>{trend_tab_css()}</style>")
    key = st.segmented_control("Trend", list(TRENDS), format_func=TRENDS.get,
                               default="sleep", key="trend_field",
                               label_visibility="collapsed") or "sleep"
    status = {r["key"]: r["status"] for r in static.get("indicators", [])}
    ind = {"heart_rate": "heart_rate", "blood_pressure": "blood_pressure",
           "blood_glucose": "blood_glucose", "sleep": "sleep"}.get(key)
    st.plotly_chart(trend_figure(trend_series(static["name"], key, static.get("checkin_json")),
                                 today_status=status.get(ind, "grey")),
                    theme=None, config=PLOT_CONFIG, key="trend_chart")
    st.html('<div data-testid="trends" style="display:none">trends</div>')


MISSING = ("Not in this cache: re-run the engine (python -m spacefuzz.fuzzing.adjuster --profile full) to get "
           "this chart.")


def _has_heatmap(res: dict) -> bool:
    return bool((res["forecast"].get("heatmap") or {}).get("risk"))


def _has_fan(res: dict) -> bool:
    prob = res["forecast"].get("probability_track") or {}
    return bool(prob.get("traces")) and bool(prob.get("trajectory"))


@st.fragment
def _result_charts(res: dict, stacked: bool = False) -> None:
    """Heatmap and fan side by side, or one above the other in a narrow column (wide layout)."""
    left, right = (st.container(), st.container()) if stacked else \
        st.columns([2, 3], gap="medium")
    with left:
        if _has_heatmap(res):
            st.plotly_chart(heatmap_figure(res), theme=None, config=PLOT_CONFIG,
                            key="heatmap_chart")
        else:
            st.caption(MISSING)
    with right:
        if _has_fan(res):
            field = st.segmented_control("Fan", list(FAN_FIELDS), format_func=FAN_FIELDS.get,
                                         default="heart_rate", key="fan_field",
                                         label_visibility="collapsed") or "heart_rate"
            st.plotly_chart(fan_figure(res, field, state.make_baseline()), theme=None,
                            config=PLOT_CONFIG, key="fan_chart")
        else:
            st.caption(MISSING)
    st.html('<div data-testid="charts-done" style="display:none">charts-done</div>')


def show(static: dict, res: dict | None, shown: int, frames: list,
         stacked: bool = False) -> None:
    """Heatmap + fan once playback has ended (drawn once per run: they hold widgets)."""
    if res is None or not frames or shown < len(frames) or \
            res["forecast"].get("risk_rating") == "UNKNOWN":
        st.html(f'<div data-testid="charts-pending" class="sf-pending sf-chart-slot'
                f'{" stacked" if stacked else ""}">Crash boundary map and trajectory fan appear '
                f'after the forecast.</div>')         # same height as the charts: no jump
        return
    _result_charts(res, stacked)
