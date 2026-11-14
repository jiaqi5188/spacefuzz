"""UI settings for app.py and ui/ (stage 4): profile choice, animation timing, colours.

Kept apart from config.py on purpose: config.py holds what the engine reads (thresholds,
mutation magnitudes, sampling tiers, seed) and is part of the engine cache key
(engine.SOURCES), so editing it re-runs the cached full-profile forecasts (~27 min).
Nothing here changes a forecast, so editing this file never invalidates cache/.
"""
import os

# Which sampling tier the app shows (tiers are defined in config.FUZZ_PROFILES). Demo,
# video and project page use the full profile; the FUZZ_PROFILE environment variable
# overrides it.
APP_FUZZ_PROFILE = "full"
# A value with no cached full run (an edited check-in, another midday value) is forecast
# with this quicker tier.
EDITED_PROFILE = "local"


def app_profile() -> str:
    return os.environ.get("FUZZ_PROFILE") or APP_FUZZ_PROFILE


# Window width [px] from which the full page uses four columns (fuzz log and charts get
# their own column); narrower windows keep the three-column layout.
WIDE_MIN_PX = 2400
# Dashboard column widths (st.columns): indicators | body map | forecast + plan board, and
# the fourth column (fuzz log, charts) of the wide layout.
COLUMNS = [4, 4, 5]
COLUMNS_WIDE = [4, 4, 5, 6]
# Role views (?view=... or their own menu entry): one screen each for the crew, the flight
# surgeon, the engineer; menu titles and icons.
VIEW_PAGES = {"crew": ("Crew screen (HUD)", "🧑‍🚀"),
              "surgeon": ("Flight surgeon screen", "🩺"),
              "engine": ("Engine screen (fuzz log)", "⚙️")}
VIEWS = tuple(VIEW_PAGES)

# The demo scenarios, in menu order, with their display names.
SCENARIOS = {
    "main_sleep_debt_eva": "Main: sleep debt + EVA",
    "radiation_spe": "Solar particle event warning",
    "backup_high_co2": "Backup: high CO2 + multitasking",
}


# Top bar. A fictional crew member, not a real astronaut (state it in the README); the
# personal baseline behind it is synthetic too.
CREW_MEMBER_NAME = "Cmdr. L. Tan"

# Playback pace in seconds: "fast" for everyday use and tests, "demo" for recording the
# video (values set once the video length is known). Pick with SPACEFUZZ_PACE=demo.
# SPACEFUZZ_ANIM_SCALE=0 still skips all delays (tests).
PACES = {
    # frame_s: pause between timeline frames; reveal_s: hold on a reveal;
    # board_s: plan-board slide + risk roll; hud_s: in-mission HUD replay of the task
    "fast": {"frame_s": 0.7, "reveal_s": 1.2, "board_s": 1.6, "hud_s": 4.0},
    "demo": {"frame_s": 2.0, "reveal_s": 3.0, "board_s": 3.0, "hud_s": 8.0},  # placeholder
}
PACE = os.environ.get("SPACEFUZZ_PACE", "fast")
if PACE not in PACES:
    raise ValueError(f"unknown SPACEFUZZ_PACE {PACE!r}; known: {sorted(PACES)}")
ANIM_SCALE = float(os.environ.get("SPACEFUZZ_ANIM_SCALE", "1"))
FRAME_S = PACES[PACE]["frame_s"]
REVEAL_S = PACES[PACE]["reveal_s"]
BOARD_S = PACES[PACE]["board_s"]
HUD_S = PACES[PACE]["hud_s"]

# Palette (plan 4.3): black background, neon green, highlight yellow, alert red, grey.
# The page background and the button colour (primaryColor = COLOR["green"]) are set in
# .streamlit/config.toml; tests/test_app.py keeps them in sync.
COLOR = {"green": "#40d317",   # neon green dimmed a little (user review 4.2)
          "yellow": "#ffd60a", "red": "#ff3b30", "grey": "#8b949e"}
RATING_COLOR = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "red",
                "UNKNOWN": "grey"}
# Surfaces and text (ui/styles.css --sf-<key>, "_" -> "-"); tests/test_styles.py keeps them equal.
SURFACE = {
    "bg": "#05070a", "panel": "#0b1017", "panel_2": "#0f141b",
    "line": "#1f2a38", "line_strong": "#2a3646", "grid": "#1a2330",
    "ink": "#e6edf3", "muted": "#8b949e",
    "info": "#8ecbff",          # fuzz log: new lowest-fitness lines
    "log": "#9fe6a0",           # fuzz log: body text
}
# Heatmap cell shades per rating band: (low end, high end); Plotly only.
BAND = {"green": ("#1f4d17", "#2f7a22"), "yellow": ("#6b5a06", "#a8900a"),
        "red": ("#4a1210", COLOR["red"])}

# Midday measurement prompts: list an indicator only if re-measuring it can move the risk by
# at least this many percentage points (within its measurement error).
MEASURE_MIN_RISK_PP = 1.0

# Reaction time test (ui/reaction.py). PVT-B: 3-minute test, 1-4 s random wait; a response
# >= 355 ms is a lapse (Jones et al., Sleep 2022, ISS). The demo runs a 1-minute version.
REACTION_TEST_S = {"demo (1 min)": 60, "PVT-B (3 min)": 180}
REACTION_ISI_S = (1.0, 4.0)
REACTION_LAPSE_MS = 355

# Midday check demo (manual 4.3): at 14:00, during the rest between the two EVA parts of the
# adjusted main plan, the crew confirms last night's sleep with the wristband. 5.9 h is an
# example input (the self-report said 6.2 h), shown as such in the UI.
MIDDAY_CHECK_MIN = 14 * 60
MIDDAY_EXAMPLE = {"main_sleep_debt_eva": {"sleep_last_night_h": 5.9}}

# Optional AI hint (ui/assist.py): when the quality gate rejects a check-in and an
# OPENAI_API_KEY is set (environment or .env), the check-in page offers a one-line
# explanation of what was likely mistyped. Advisory only: it never changes the gate
# decision or a number. Without a key the page works exactly as before.
ASSIST_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
ASSIST_TIMEOUT_S = 10          # one try; a slower answer falls back to the gate's message
ASSIST_MAX_TOKENS = 200
ASSIST_MAX_CHARS = 400         # longer answers are cut
