"""Project-wide constants: thresholds, mutation magnitudes, sampling settings, seeds.

Values marked "illustrative" are design choices, not data-derived; the rest come from the
data analysis (A1-A9, notebooks/analysis.ipynb) or the cited source. Each constant notes its
source.
"""

# ---- simulation grid
STEP_MINUTES = 15                       # one model step (plan 3.2)
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES  # 96
RANDOM_SEED = 147                       # fixed seed for every random draw (Mission Day 147)

# ---- environment trace (env_trace.py)
ENV_MAX_INTERP_STEPS = 2   # fill runs of up to 2 empty 15-min bins after resampling (a gap of
                           # up to about an hour between samples); longer gaps stay NaN and
                           # build_trace refuses a day with any left
                           # (covers DosTel2 2017-09-10 15:37-16:00 and MSL-RAD's 17 min cadence)
ENV_DEFAULT_CABIN_START = "2014-10-15 00:00"      # EDA RR-1 telemetry (2014-09-20..11-01);
                                                  # env_trace drops rows before EDA_ISS_START
ENV_DEFAULT_RADIATION_START = "2017-09-05 00:00"  # RadLab quiet day before the X9.3 flare
GOES_OFFICIAL_SCALE = 0.635  # A2: issued class / GOES-16 science peak flux, the same for X9.3
                             # (09-06) and X8.2 (09-10); flare_class() uses it as an estimate.
                             # Public material quotes the issued class (SOURCES.md section 6)
# Unit conversions (e.g. 1 mGy/day = 41.67 uGy/h) live in generic_reader._LINEAR.

# ---- WESAD heart rate / HRV extraction (physio_replay.py)
WESAD_CHEST_HZ = 700
WESAD_LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}  # 0, 5-7 dropped
RELAX_CONDITIONS = ("baseline", "meditation")  # "relax" pool for sampling; stress = label 2
ECG_BANDPASS_HZ = (5.0, 15.0)    # QRS energy band for R-peak detection
ECG_PEAK_REL_HEIGHT = 0.4        # peak must reach 40% of the local 2 s maximum
ECG_MIN_RR_S = 60 / 180          # physiological RR bounds: 40-180 bpm
ECG_MAX_RR_S = 60 / 40
ECG_RR_MAX_JUMP = 0.20           # drop RR deviating > 20% from the local median (ectopics, misses)
HRV_WINDOW_S = 60                # one HR / RMSSD value per 60 s window (ultra-short RMSSD)
HRV_MIN_COVERAGE = 0.5           # window needs >= 50% of its time covered by clean RR intervals


# =====================================================================================
# Data-derived constants: notebooks/analysis.ipynb (plan 2.5), numbers in
# notebooks/analysis_results.json. tests/test_config_analysis.py keeps them in sync.
# =====================================================================================

# ---- A1: EDA RR-1 ISS cabin air. Before EDA_ISS_START the "ISS" columns are not ISS cabin
# air: the mice were on the pad (09-20), in Dragon transit (CRS-4 launch 09-21 05:52 UTC,
# berthed 09-23 13:21 UTC) and awaiting transfer; CO2 is frozen at 2646 from 09-21 04:24.
EDA_ISS_START = "2014-09-27 13:30:00"    # A1: first reading after the frozen-CO2 run found by
                                         # quality_gate.stuck_mask; temp / RH normalise here too
ENV_CO2_HIGH_PPM = 3997                  # A1: P90 of 15-min CO2; above it = high-CO2 period
OP_B_CO2_AMPLIFY_RANGE = (0.112, 0.23)   # A1: usual range of operator B amplification,
                                         # P25-P90 of daily (max / median - 1); 62% of days
OP_B_CO2_AMPLIFY_MAX = 0.403             # A1: upper bound = observed maximum (3 real days
                                         # at 0.31-0.40); do not cap at the usual range
OP_B_CO2_EXCURSIONS_PATH = "data/processed/op_b_co2_excursions.csv"  # A1: the 34 daily
                                         # values; operator B samples from them by observed
                                         # frequency (mostly 11-23%, real tail to 40%)
OP_B_DAILY_PROB = 0.382                   # A1: share of days whose CO2 peak exceeds ENV_CO2_HIGH_PPM
                                         # (13 of 34 days); probability-track chance of operator B
ENV_CO2_MAX_OBSERVED_PPM = 4663          # A1: highest 15-min cabin CO2 recorded; above it the model
                                         # extrapolates (out-of-experience warning, plan 3.12)
ENV_HIGH_CO2_DAY = "2014-10-07"          # A1: highest daily mean CO2 (4219 ppm, max 4663).
                                         # Record only: the backup scenario replays the
                                         # default cabin day; its CO2 comes from the plan's
                                         # high_co2 event
ENV_TEMP_RANGE_C = (22.1, 22.9)          # A1: P5-P95 cabin temperature (from EDA_ISS_START)
ENV_RH_RANGE_PCT = (40.0, 49.0)          # A1: P5-P95 cabin relative humidity (from EDA_ISS_START)

# ---- A2: RadLab DosTel2 (ISS), MSL-RAD (Mars), GOES-16 XRS-B, 2017-09-04..18
RAD_BASELINE_ISS_UGY_H = 4.94            # A2: median 15-min DosTel2 dose rate (time-weighted
                                         # mean incl. SAA passes is 8.86)
RAD_BASELINE_MARS_UGY_H = 7.79           # A2: MSL-RAD median before the 2017-09-10 flare
OP_D_SAA_PEAK_MULTIPLE = 21.7            # A2: DosTel2 P99 / median (SAA passes; max 38.6x)
OP_D_SPE_PEAK_MULTIPLE = 2.98            # A2: MSL-RAD solar particle event peak / baseline
                                         # (event median 1.88x)
OP_D_SPE_DURATION_H = 47.0               # A2: MSL-RAD above 1.25x baseline, 09-11 05:00-09-13 04:00
OP_D_FLARE_TO_SPE_ONSET_H = 12.9         # A2: GOES X8.2 flare peak (09-10 16:06) to SPE onset
OP_D_DAILY_PROB = 0.017                  # A2: NOAA SWPC solar proton events 1976-2017, 261 events
                                         # (6.2 / year, 0-23): chance an event starts on a given day
OP_D_SPE_PROFILE_PATH = "data/processed/spe_profile_msl_rad.csv"  # A2: event shape per 15 min,
                                         # excess over baseline, added by operator D

# ---- A3: WESAD S2-S5 chest ECG, minute HR / RMSSD, subject fixed-effects regression
CHAIN1_STRESS_HR_DELTA_BPM = 21.7        # A3: stress vs relax HR effect (95% CI 19.3-24.0)
CHAIN1_STRESS_RMSSD_RATIO = 0.512        # A3: stress multiplies RMSSD (95% CI 0.476-0.551)
PHYSIO_HR_FLUCT_SD_BPM = {"relax": 2.6, "stress": 7.24}     # A3: SD of minute HR around
                                                            # each segment's mean
PHYSIO_RMSSD_FLUCT_CV = {"relax": 0.136, "stress": 0.211}   # A3: relative SD of minute
                                                            # RMSSD around segment mean
FLUCT_STRESS_POOL_LEVEL = 0.5      # model stress level above which the stress pool is drawn
FLUCT_RMSSD_REL_MIN = -0.9         # relative RMSSD deviation floor (fitted fallback pool)
HRV_READING_FLOOR_MS = 1.0         # an overlaid RMSSD reading never drops below this
BASELINE_HR_RANGE_BPM = (54.2, 72.7)     # A3: subjects' baseline-condition mean HR
BASELINE_RMSSD_RANGE_MS = (30.1, 101.4)  # A3: subjects' baseline-condition mean RMSSD

# ---- A4: Inspiration4 CMP + CBC, 4 crew (C003 L-92 platelet 12.9 excluded: recording error)
BASELINE_GLUCOSE_MG_DL_RANGE = (56.0, 100.0)     # A4: min-max of pre-flight glucose
BASELINE_HEMOGLOBIN_G_DL_RANGE = (11.5, 17.0)    # A4: min-max of pre-flight hemoglobin
I4_R1_PCT_CHANGE = {                              # A4: median % change at R+1 vs each
    "GLUCOSE": 19.1, "hemoglobin": -1.4,          # crew member's pre-flight mean
    "hematocrit": -1.6, "white_blood_cell_count": -14.9,
    "platelet_count": -2.3, "SODIUM": -1.5, "POTASSIUM": -5.5, "CREATININE": -3.2,
}


# ---- A6: engine convergence (probability track samples)
A6_HALF_WIDTH = {0.01: 0.004, 0.05: 0.01}   # illustrative criterion: 95% CI half-width needed at
                                            # the green/yellow (1%) and yellow/red (5%) lines
PROBABILITY_SAMPLES = 2500               # A6: smallest multiple of 500 meeting A6_HALF_WIDTH
                                         # (Wilson / normal bound; checked on 5 seeds)


# =====================================================================================
# State model (state.py, model.py). "illustrative" = literature-informed placeholder,
# not fitted; replace when better data exists. Analyses cited where they apply.
# =====================================================================================

# ---- state.py: synthetic 14-day personal baseline (fixed seed)
BASELINE_DAYS = 14
BASELINE_CENTER_QUANTILES = (0.25, 0.75)  # person-level centre drawn from the middle half of
                                          # each A3 / A4 range, so no baseline sits at an extreme
BASELINE_SLEEP_H = 7.5          # illustrative: lower end of the 7-9 h adult sleep need
BASELINE_SPO2_PCT = 98.0        # illustrative: healthy resting SpO2
BASELINE_CORE_TEMP_C = 36.8     # illustrative: resting core temperature
BASELINE_REACTION_MS = 270.0    # illustrative: rested mean reaction time, 3-min PVT-B
BASELINE_AGE_Y = 45             # illustrative: synthetic astronaut's age (typical for active crew)
HR_MAX_INTERCEPT_BPM = 220      # illustrative: age-predicted max HR = 220 - age (individual error
                                # about +-10 bpm); caps model and overlaid heart rate (state.max_heart_rate)
BASELINE_SBP_DBP = (118.0, 76.0)  # illustrative: synthetic BP (no public continuous data, plan 2.2)
BASELINE_DAILY_SD = {           # illustrative: day-to-day SD in the synthetic history
    "resting_hr": 2.0, "rmssd": 6.0, "spo2": 0.4, "core_temp": 0.1, "sleep_h": 0.5,
    "reaction_ms": 12.0, "glucose": 5.0, "sbp": 4.0, "dbp": 3.0,
}
FATIGUE_PER_SAMN_PERELLI_POINT = 6.0   # illustrative: check-in fatigue 1-7 -> model 0-36
                                       # (4 "a little tired" -> 18); tuned with the guard below

# ---- model.py: dynamics (first-order approach to a target each 15-min step)
MODEL_TAU_MIN = {               # illustrative time constants [min]
    "hr_rise": 10.0, "hr_recover": 30.0, "core_temp": 45.0, "spo2": 15.0,
    "stress": 20.0, "bp": 30.0, "glucose_return": 120.0,
}
# Per activity: HR rise [bpm], RMSSD multiplier, core temp rise [degC], SpO2 change [pct pt],
# BP rise [mmHg], fatigue rate [points/h], glucose drift [mg/dL per h] = net fall while
# counter-regulation (liver output) still works, i.e. glucose >= GLUCOSE_LOW_MG_DL. Chain 2, 6.
# EVA core +0.5 degC: an ordinary EVA stays below the 0.6 degC heat-strain line (HEAT_STRAIN_ONSET_C).
ACTIVITY_EFFECTS = {            # illustrative (EVA ~105-115 bpm sustained; operator A pushes to ~140)
    "eva":      {"hr": 45, "rmssd": 0.45, "temp": 0.5, "spo2": -1.0, "sbp": 15, "dbp": 5,
                 "fatigue": 4.0, "glucose_drift": 1.5},
    "treadmill": {"hr": 60, "rmssd": 0.35, "temp": 0.7, "spo2": -0.5, "sbp": 20, "dbp": 3,
                  "fatigue": 5.0, "glucose_drift": 3.0},
    "strength": {"hr": 35, "rmssd": 0.55, "temp": 0.4, "spo2": 0.0, "sbp": 18, "dbp": 8,
                 "fatigue": 4.0, "glucose_drift": 2.0},
    "high_load": {"hr": 8, "rmssd": 0.9, "temp": 0.0, "spo2": 0.0, "sbp": 5, "dbp": 3,
                  "fatigue": 5.0, "glucose_drift": 0.5},
    "work":     {"hr": 5, "rmssd": 0.95, "temp": 0.0, "spo2": 0.0, "sbp": 3, "dbp": 2,
                 "fatigue": 3.0, "glucose_drift": 0.3},
    "idle":     {"hr": 2, "rmssd": 1.0, "temp": 0.0, "spo2": 0.0, "sbp": 0, "dbp": 0,
                 "fatigue": 1.0, "glucose_drift": 0.0},   # unscheduled awake time
    "rest":     {"hr": 0, "rmssd": 1.0, "temp": 0.0, "spo2": 0.0, "sbp": 0, "dbp": 0,
                 "fatigue": -6.0, "glucose_drift": 0.0},
    "meal":     {"hr": 3, "rmssd": 1.0, "temp": 0.0, "spo2": 0.0, "sbp": 0, "dbp": 0,
                 "fatigue": -4.0, "glucose_drift": 0.0},
    "sleep":    {"hr": -8, "rmssd": 1.25, "temp": -0.3, "spo2": -0.5, "sbp": -10, "dbp": -6,
                 "fatigue": -12.0, "glucose_drift": 0.0},
}
# Heat storage (chain 2, NASA-STD-3001 V2 11011: impairment if core rises > 0.6 degC or pulse
# > 140 bpm). Stored heat raises core temperature; above the 0.6 degC line heart rate follows it
# (cardiovascular drift); at rest the heat dissipates (MODEL_TAU_MIN["core_temp"]) and HR falls.
HEAT_STRAIN_ONSET_C = 0.6       # V2 11011: core rise where heat strain begins
HEAT_MAX_CORE_RISE_C = 1.5      # illustrative: stored heat stops here (core ~38.3 degC)
HEAT_HR_PER_C = 30.0            # illustrative: bpm per degC of core rise above the onset; set by
                                # the target behaviours in tests/test_evaluator.py (no data to fit)
MEAL_GLUCOSE_RISE_MG_DL = 35.0  # illustrative: post-meal rise spread over the meal
GLUCOSE_LOW_MG_DL = 70.0        # ADA level-1 hypoglycaemia; below it counter-regulation fails
                                # during exertion and glucose keeps falling (chain 6)
GLUCOSE_LOW_FALL_PER_H = 4.0    # illustrative: extra fall per hour of exertion when below low
GLUCOSE_FLOOR_MG_DL = 40.0      # model floor (severe hypoglycaemia; the evaluator fires long before)
                                # Outside exertion, glucose returns to the check-in level
                                # (MODEL_TAU_MIN["glucose_return"]); meals add MEAL_GLUCOSE_RISE_MG_DL.
GLUCOSE_LOW_FATIGUE_PER_H = 3.0    # illustrative: chain 6, low glucose -> fatigue
STRESS_GLUCOSE_PER_H = 6.0      # illustrative: chain 6, cortisol raises glucose under stress
STRESS_SBP_DBP = (12.0, 6.0)    # illustrative: chain 1 BP rise at full stress
STRESS_TRIGGERS = {             # illustrative: chain 1 stress target per trigger (0-1)
    "high_load_task": 1.0, "spe_warning": 0.8, "dose_spike": 0.6,
}
DOSE_SPIKE_FACTOR = 1.5         # dose rate above 1.5x the trace median counts as an alarm (chain 1)
CO2_EFFECT_PER_1000_PPM = {"hr": 3.0, "spo2": -0.8, "fatigue": 2.0}  # illustrative: above
                                # ENV_CO2_HIGH_PPM (A1), per 1000 ppm of excess (chain 2)
# Chain 5: sleep debt = sum over the last 3 nights of max(0, need - slept).
SLEEP_DEBT_HR_PER_H = 0.8           # illustrative: resting HR +bpm per hour of debt
SLEEP_DEBT_RMSSD_PER_H = 0.03       # illustrative: resting RMSSD -3% per hour of debt
SLEEP_DEBT_RT_PER_H = 0.025         # illustrative: reaction time +2.5% per hour of debt
SLEEP_DEBT_FATIGUE_GAIN_PER_H = 0.04  # illustrative: fatigue builds 4% faster per hour of debt
SLEEP_DEBT_EXERTION_GAIN_PER_H = 0.03  # illustrative: exertion HR rise +3% per hour of debt
SLEEP_DEBT_RECOVERY_SLOWDOWN_PER_H = 0.08  # illustrative: HR recovery tau +8% per hour of debt
SLEEP_DEBT_REPAY_PER_SLEEP_H = 0.5  # illustrative: debt repaid per hour actually slept
RT_FATIGUE_MS_PER_POINT = 1.2   # illustrative: +1.2 ms reaction time per fatigue point
FATIGUE_MAX = 100.0
FATIGUE_RED_LINE = 80.0         # plan 3.5 cognitive rule: fatigue > 80 and SpO2 < 94% (illustrative)
# Design guard (tests/test_model.py): the unmutated main plan peaks 10-15 below the red line
# (66.3 with the values above); EVA fatigue, the Samn-Perelli mapping and the debt gain
# were tuned together for this, since all three are illustrative.
# Radiation: physical EVA dose accumulates without recovery; radiation_load is a 0-100
# strain index with a cap and slow recovery at rest (plan 3.2).
EVA_DOSE_MULTIPLIER = 1.5       # illustrative: less shielding outside the habitat
RAD_LOAD_PER_UGY = 0.2          # illustrative: load points per uGy above the trace's median
                                # dose rate (exposure beyond the habitat's usual level)
RAD_LOAD_RECOVERY_PER_H = 1.0   # illustrative: recovery at rest / sleep
RAD_LOAD_MAX = 100.0
SPO2_MIN_PCT, SPO2_MAX_PCT = 80.0, 100.0


# =====================================================================================
# Mutation operators A-F (mutator.py, plan 1.3 / 3.3). A pair (lo, hi) is a uniform
# distribution; durations are rounded to whole 15-min steps. Operator B's amplification
# and operator D's event shape come from the analyses (OP_B_*, OP_D_* above).
# =====================================================================================
MUT_OPS_PER_SAMPLE = (1, 2, 3)            # plan 3.3: each sample stacks 1-3 distinct operators
MUT_OPS_PER_SAMPLE_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)  # illustrative
MUT_OPERATOR_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0}  # illustrative
MUT_MAX_ATTEMPTS_FACTOR = 5               # generate(n) gives up after n x 5 draws
MUT_CHAIN_TOLERANCE = 1e-6                # numerical slack in the causal-chain direction checks
# A: EVA physical load (plan 1.3: HR to about 140 bpm, core temperature up, +-1-2 h), as heat storage
OP_A_EVA_HEAT_RATE_C_PER_H = (0.1, 0.3)  # illustrative: a harder EVA / poorer cooling stores heat:
                                         # core target +rate per hour of continuous EVA, so HR
                                         # rises late in a long EVA and resets with rest; set by
                                         # the target behaviours in tests/test_evaluator.py
OP_A_EVA_DURATION_DELTA_H = (1.0, 2.0)    # plan 1.3: EVA longer or shorter (random sign)
OP_A_LONGER_PROB = 0.5                    # chance the EVA gets longer rather than shorter
# B: cabin CO2 build-up (plan 1.3: CO2 +10-40%, continuous work +4 h, SpO2 down to 93%)
OP_B_WORK_TYPES = ("science_experiment", "maintenance")  # "continuous work" tasks
OP_B_WORK_EXTEND_H = (1.0, 4.0)           # illustrative: up to +4 h of continuous work
OP_B_SPO2_DROP_PCT = (2.0, 5.0)           # illustrative: extra SpO2 drop while CO2 is above
                                          # ENV_CO2_HIGH_PPM (98% -> about 93%)
# C: multi-task overload (plan 1.3: cognitive fatigue +20-40, insert docking / precision work)
OP_C_INSERT_TYPES = ("docking", "precision_manual")
OP_C_TASK_DURATION_H = (0.5, 1.5)         # illustrative
OP_C_FATIGUE_JUMP = (20.0, 40.0)          # plan 1.3: fatigue points added when the task starts
# D: solar particle event (plan 1.3: real event dose rate superimposed, fatigue rises slowly)
OP_D_SPE_THRESHOLD_EXCESS = 0.25          # same 1.25x baseline cut as OP_D_SPE_DURATION_H (A2);
                                          # the added spe_warning event covers this part
OP_D_FATIGUE_PER_H = (0.5, 2.0)           # illustrative: slow fatigue rise during the event
# E: isolation and sleep debt (plan 1.3: last night -0.5 to -3 h; 3 nights under 6 h)
OP_E_SLEEP_CUT_H = (0.5, 3.0)             # plan 1.3
OP_E_CHRONIC_PROB = 0.5                   # illustrative: share of E samples with 3 short nights
OP_E_CHRONIC_NIGHT_H = (4.0, 6.0)         # plan 1.3: "3 nights under 6 h"; never above the
                                          # night already recorded
OP_E_STRESS_SELF_DELTA = 1                # illustrative: self-rated stress +1 (max 5)
# F: sensor faults (plan 1.3: missing data, signal dropout, stuck values, spikes, offline)
OP_F_FIELDS = ("heart_rate", "hrv_rmssd", "spo2", "core_temp")  # wearable series (QG_VITAL_SERIES)
OP_F_KINDS = ("missing", "dropout", "stuck", "spike", "network_down")
OP_F_FAULTS_PER_SAMPLE = (1, 2)           # illustrative
OP_F_MISSING_FRACTION = (0.05, 0.6)       # illustrative: share of readings lost at random
OP_F_DROPOUT_MIN = (30, 180)              # illustrative: one field silent
OP_F_STUCK_MIN = (60, 240)                # illustrative: one field frozen (gate limits 60-180)
OP_F_SPIKES = (1, 5)                      # illustrative: number of spiked readings
OP_F_SPIKE_FACTOR = (2.0, 4.0)            # illustrative: spike = reading x factor
OP_F_NETWORK_DOWN_MIN = (60, 360)         # illustrative: every field silent (device offline)


# =====================================================================================
# Data quality gate (quality_gate.py, plan 3.4). Ranges are physiological plausibility
# limits (a value outside cannot be a real reading), not health thresholds.
# =====================================================================================
QG_PLAUSIBLE_RANGES = {          # illustrative plausibility limits
    "heart_rate": (25, 250), "hrv_rmssd": (2, 300), "spo2": (50, 100), "core_temp": (33, 43),
    "systolic_bp": (60, 260), "diastolic_bp": (30, 160), "blood_glucose": (20, 600),
    "reaction_time_ms": (100, 2000), "sleep_last_night_h": (0, 16),
    "sleep_last_3_nights_h": (0, 16),          # each night
    "fatigue_score": (1, 7), "mood": (1, 5), "stress": (1, 5),   # the scales themselves
    "co2": (300, 20000), "dose_rate": (0, 5000),
}
# Cabin temperature / humidity are not stuck-checked: at 0.1 degC / 1% resolution and with tight
# thermal control they plateau for 6-8 h on normal ISS days (EDA RR-1: up to 460 min).
QG_STUCK_MINUTES = {             # illustrative: identical consecutive readings this long = stuck
    "heart_rate": 60, "hrv_rmssd": 60, "spo2": 180,    # integer SpO2 can plateau for a while
    "core_temp": 120, "co2": 120, "dose_rate": 120,
}
QG_VITAL_SERIES = ("heart_rate", "hrv_rmssd", "spo2", "core_temp")
QG_CHECKIN_FIELDS = ("systolic_bp", "diastolic_bp", "blood_glucose", "sleep_last_night_h",
                     "sleep_last_3_nights_h", "fatigue_score", "mood", "stress",
                     "reaction_time_ms")        # the check-in values the model and rules read
QG_OPTIONAL_CHECKIN_FIELDS = ("reaction_time_ms",)   # may be left out (test not done): degraded
QG_MAX_MISSING_FRACTION = 0.5    # a series with more empty / filtered readings is unusable
QG_MIN_USABLE_FRACTION = 0.5     # fewer usable fields than this -> untrusted


# =====================================================================================
# Advice library, act-now part (advice.py, plan 3.11). Check-in triggers; illustrative.
# =====================================================================================
ACT_NOW_HR_ABOVE_BASELINE_BPM = 10.0  # illustrative: sustained resting HR this far above baseline
ACT_NOW_BP_RANGE = {"sbp": (90.0, 140.0), "dbp": (60.0, 90.0)}  # illustrative: outside -> recheck,
                                      # then ground (140/90 = common hypertension cut-off)
ACT_NOW_SLEEP_SHORT_H = 6.0           # same as temporal rule 1 (plan 3.5)
ACT_NOW_SLEEP_DEBT_H = 3.0            # illustrative: 3-night debt worth acting on even if last
                                      # night alone was >= 6 h (the main plan: 6.2 h, 6.9 h debt)
ACT_NOW_MOOD_LOW = 2                  # mood 1-5: at or below -> support
ACT_NOW_STRESS_HIGH = 4               # stress 1-5: at or above -> support
ACT_NOW_GLUCOSE_SEVERE_LOW = 54.0     # ADA level-2 hypoglycaemia (< 54 mg/dL; data/indicator_sources.md)
                                      # -> crew level, ground notified, not self-care
ACT_NOW_BP_SEVERE = {"sbp": 180.0, "dbp": 120.0}  # severe hypertension (AHA/ACC 2025 term):
                                      # recheck after ACT_NOW_BP_RECHECK_MIN with EVA and
                                      # high-load tasks on hold; still severe -> ground
ACT_NOW_BP_RECHECK_MIN = 5            # AHA: wait 5 minutes and measure again


# =====================================================================================
# Crash rules (evaluator.py, plan 3.5). Illustrative values, not clinical standards.
# =====================================================================================
# Composite rules, judged on the overlaid outputs (heart_rate, hrv_rmssd).
COG_FATIGUE_ABOVE = FATIGUE_RED_LINE      # cognitive: fatigue > 80 ...
COG_SPO2_BELOW_PCT = 94.0                 # ... and SpO2 < 94% at the same step (illustrative)
CARDIO_HR_MAX_FRACTION = 0.8              # illustrative: HR >= 0.8 x personal max HR (220 - age);
                                          # 140 bpm at 45, the EVA impairment pulse in NASA-STD-3001
                                          # V2 11011. Follows age, not resting HR (a fit crew member
                                          # with a low resting HR has no lower maximum)
CARDIO_HRV_RATIO = 0.5                    # illustrative: RMSSD <= 0.5 x baseline (A3 stress
                                          # effect is x0.51)
CARDIO_SUSTAIN_MIN = 30                   # illustrative: both must hold this long ("sustained")
CARDIO_BP_RANGE = ACT_NOW_BP_RANGE        # outside 90-140 / 60-90 mmHg (AHA/ACC, as act-now)
CARDIO_BP_SETTLE_MIN = 60                 # illustrative: BP judged only after this long at rest
                                          # (exertion raises BP normally; resting limits only)
METAB_PRE_EVA_GLUCOSE_MG_DL = GLUCOSE_LOW_MG_DL  # metabolic: glucose < 70 when an EVA starts
METAB_EVA_FALL_MG_DL = 20.0               # illustrative: ... or an unbroken fall of 20 mg/dL in EVA
RAD_EVA_DOSE_LIMIT_UGY = 150.0            # illustrative: EVA dose per day (all EVAs of the day,
                                          # both halves of a split EVA), about 2x a 6.5 h
                                          # Mars-surface EVA at the A2 baseline (7.79 x 1.5 x 6.5
                                          # = 76 uGy). Not a NASA limit: NASA's formal radiation
                                          # limits are career limits, a different quantity
# Temporal rules (one hand-written state machine each).
RULE1_MIN_SLEEP_H = 6.0                   # rule 1: after < 6 h sleep ...
RULE1_WINDOW_H = 12.0                     # ... no EVA may start within 12 h of waking
RULE1_NAP_MAX_H = 2.0                     # illustrative: a planned sleep up to this long is a nap,
                                          # not the night's sleep (it does not restart rule 1)
RULE2_GAP_H = 2.0                         # rule 2: no ARED / T2 training in the 2 h before an EVA
RULE3_MAX_WORK_H = 4.0                    # rule 3: after > 4 h continuous cabin work ...
RULE3_CO2_PPM = ENV_CO2_HIGH_PPM          # ... at CO2 >= this (A1 P90), no docking / precision work
# Fitness (plan 3.6): each rule contributes a distance in [0, 1]; 0 = violated.
RULE_DISTANCE_CAP_STEPS = 16              # temporal distance: steps / 16 (4 h of 15-min steps), max 1


# =====================================================================================
# Two-track engine (engine.py, plan 3.6 / 3.7). FUZZ_PROFILE (environment variable) picks
# the profile; default local. The probability track uses 2,500 samples in both (plan 1.3:
# about 2,500 are needed for the 1% tier; A6 convergence curve to confirm).
# =====================================================================================
FUZZ_PROFILES = {
    "local": {"probability_samples": PROBABILITY_SAMPLES, "heatmap_samples": 300, "guided_rounds": 2000,
              "adjust_samples": 500, "recheck_rounds": 300},
    "full": {"probability_samples": PROBABILITY_SAMPLES, "heatmap_samples": 2500, "guided_rounds": 10000,
             "adjust_samples": 2500, "recheck_rounds": 2000},
}
FUZZ_PROFILE_DEFAULT = "local"
RISK_GREEN_BELOW = 0.01            # plan 1.3: green < 1%; yellow 1%-5% (both ends included)
RISK_RED_ABOVE = 0.05              # plan 1.3: red > 5%
CRITICAL_TIME_TO_CRASH_MIN = 120   # plan 1.3: red and time to crash <= 2 h -> CRITICAL
TIME_TO_CRASH_FRACTION = 0.05      # plan 1.3: first time the cumulative crash share reaches 5%
# Probability track (plan 3.7, unbiased): each operator is drawn independently with the chance
# it happens on a given day, so a sample may have no operator at all (an ordinary day). The
# guided track keeps "1-3 stacked operators" (MUT_OPS_PER_SAMPLE) to search for boundaries.
PROB_OPERATOR_DAILY = {
    "A": 0.2,              # illustrative: an EVA runs hotter than planned (only on EVA days)
    "B": OP_B_DAILY_PROB,  # A1
    "C": 0.05,             # illustrative: an unplanned docking / precision task is added
    "D": OP_D_DAILY_PROB,  # A2 (NOAA SPE list)
    "E": 0.0,              # last night's sleep is measured at the check-in, not uncertain today;
                           # "what if less sleep" is the heatmap's sleep axis
    "F": 0.2,              # illustrative: a sensor fault during the day (tests the gate only)
}
GUIDED_EXPLORE_PROB = 0.2          # illustrative: chance of mutating a random seed, not the closest
GUIDED_POOL_MAX = 200              # illustrative: seed pool keeps the closest seeds
GUIDED_PLATEAU_ROUNDS = 300        # illustrative: rounds without a new crash type -> log a plateau
HEATMAP_SLEEP_CUT_H = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)   # heatmap rows (operator E range)
HEATMAP_EVA_EXTEND_H = (0.0, 0.5, 1.0, 1.5, 2.0)            # heatmap columns (operator A range)
MIN_PERTURBATION_MAX_H = 3.0       # deterministic scan for the smallest crashing change
ENGINE_TRAJECTORY_FIELDS = ("heart_rate", "cognitive_fatigue", "spo2", "core_temp",
                            "blood_glucose", "radiation_load")
ENGINE_TRACE_FIELDS = ("heart_rate", "cognitive_fatigue")  # individual traces kept for the UI
ENGINE_TRACE_SAMPLES = 300         # trajectory fan: first N probability-track samples (i.i.d.,
                                   # so an unbiased subset); size of the cache file, not accuracy
GUIDED_TIMING_EVERY = 500          # fuzz log: wall-clock checkpoint every N guided rounds
COVERAGE_BINS = {                  # indicator ranges reached (coverage report, plan 3.12)
    "heart_rate": (60, 80, 100, 120, 140, 160), "cognitive_fatigue": (20, 40, 60, 80),
    "spo2": (90, 92, 94, 96), "core_temp": (37.0, 37.4, 37.8, 38.2),
    "blood_glucose": (54, 70, 100, 140), "radiation_load": (10, 25, 50, 75),
}
ENGINE_CACHE_DIR = "cache"


# =====================================================================================
# Plan adjustment (adjuster.py, plan 3.9) and measurement prompts (advice.py, plan 3.11).
# Candidates are screened with "adjust_samples" probability-track samples (common random
# numbers), the chosen one is confirmed with the full probability track.
# =====================================================================================
ADJ_DELAY_MIN = (30, 60, 120)          # illustrative: move the EVA later
ADJ_SHORTEN_H = (1.0, 2.0)             # illustrative: shorten the EVA
ADJ_MIN_EVA_MIN = 60                 # a shortened EVA keeps at least this many minutes
ADJ_SPLIT_REST_MIN = 60                # illustrative: rest between the two halves of a split EVA
ADJ_REST_BEFORE_MIN = (30, 60)         # illustrative: rest before the EVA / a precision task
ADJ_NAP_BEFORE_MIN = (60, 90)          # illustrative: nap before the EVA (<= RULE1_NAP_MAX_H)
ADJ_MAX_COMBINED = 2                   # candidates combine up to 2 adjustments
ADJ_COST = {                           # illustrative "size of change" (smallest change first);
    "delay_per_h": 1.0,                # postponing the EVA to another day is by far the largest
    "shorten_per_h": 2.0,              # (removing work costs more than moving it)
    "split": 1.5, "rest_per_h": 1.0, "nap_per_h": 1.0, "move_training": 1.0,
    "postpone": 100.0,
}
ADJ_SLIP_MARGIN_MIN = 30               # guided re-test: a normal schedule slip of this much
                                       # (training later, EVA overrun, EVA into an SPE warning,
                                       # more high-CO2 work) must not make the plan crash
MEASUREMENT_ERROR = {                  # illustrative +- measurement error of check-in entries;
    "blood_glucose": 15.0,             # mg/dL      the entry whose error moves the risk most is
    "systolic_bp": 5.0,                # mmHg       the one worth measuring first at midday
    "diastolic_bp": 5.0,               # mmHg
    "sleep_last_night_h": 0.5,         # h (self-report vs wearable)
    "reaction_time_ms": 20.0,          # ms
}
ADJ_SAME_DAY_TOP = 5                   # same-day alternative (when postponing is chosen): extend the
                                       # best 5 two-change candidates with a third change


# =====================================================================================
# Out of model experience (engine.py, plan 3.12): a check-in or plan outside the ranges
# the model was built on is never rated green and is at least crew level.
# =====================================================================================
OOE_CHECKIN_RANGES = {
    "blood_glucose": BASELINE_GLUCOSE_MG_DL_RANGE,   # A4 pre-flight range (fasting values)
    "sleep_last_night_h": (3.0, 10.0),               # illustrative
    "systolic_bp": (90.0, 180.0),                    # illustrative (severe line, AHA/ACC)
    "diastolic_bp": (60.0, 120.0),                   # illustrative
    "reaction_time_ms": (150.0, 600.0),              # illustrative
}
OOE_SLEEP_DEBT_MAX_H = 12.0            # illustrative: 3-night debt the model was tuned for
OOE_EVA_MAX_H = 8.0                    # plan 1.3: EVAs last 6-8 h
