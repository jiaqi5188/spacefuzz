# Data Sources

All raw files live under `data/raw/<dataset>/`. Only two small public NASA files are
committed, because the app builds the environment trace from them on a fresh clone:
`data/raw/eda/RR-1_telemetry.csv` (EDA, NASA OSDR) and
`data/raw/radlab/MSL-RAD_Mars_2017-09-04_to_09-18.csv` (RadLab MSL-RAD). Everything else in
`data/raw/` stays local (WESAD in particular may not be redistributed). Processed artifacts
(environment scenario CSVs, WESAD fitted parameters, engine result cache) are committed so the
app runs without the other raw data.

## Summary

| # | Dataset | Folder | Provider | Downloaded | Used for |
|---|---------|--------|----------|------------|----------|
| 1 | Environmental Data App (EDA), RR-1 cabin telemetry and radiation | `eda/` | NASA OSDR | 2026-11-08 | Cabin CO2 / temperature / humidity scenarios; baseline radiation dose |
| 2 | WESAD wearable stress dataset | `wesad/` | University of Siegen | 2026-11-09 | Real heart-rate / HRV fluctuation segments (`physio_replay.py`) |
| 3 | RadLab DosTel2, ISS Columbus module | `radlab/` | NASA RadLab | 2026-11-09 | LEO dose-rate profile, including South Atlantic Anomaly passes |
| 4 | RadLab MSL-RAD, Mars surface (Curiosity) | `radlab/` | NASA RadLab | 2026-11-09 | Mars surface dose rate; 2017-09-11 solar particle event shape (operator D) |
| 5 | Inspiration4 clinical blood and urine panels (OSD-569, OSD-575, OSD-656: downloaded, not used in the app or analysis) | `inspiration4/` | NASA OSDR | 2026-11-09 | Clinical reference ranges and real pre/post-flight changes in civilian crew |
| 6 | GOES-16 XRS solar X-ray flux, 1-min averages | `goes/` | NOAA NCEI | 2026-11-09 | Flare timing to label and align the radiation scenarios |
| 7 | NASA Space Apps 2026 challenge resources (5 web pages, no dataset) | none | NASA, CSA | 2026-11-13 (read, nothing to download) | References only: hazards frame, risk-page links, ARCHeR background, CSA radiation note |

## 1. EDA (NASA OSDR Environmental Data App)

- Link: https://visualization.osdr.nasa.gov/eda/
- Files:
  - `RR-1_telemetry.csv`: 12,087 rows, 5-minute interval, CO2 / temperature / relative humidity.
  - `RR-1_radiation.csv`: 39 rows, one per day, dose in **mGy/day**.
- License: NASA open data.
- Notes:
  - Daily dose is spread evenly over the day when resampled to 15-minute steps (divide by 96).
  - Only the `*_ISS` columns are used; `*_Ground` is the ground control chamber.
  - **Readings before 2014-09-27 13:30 are not ISS cabin air and are excluded for all variables** (`config.EDA_ISS_START`; `env_trace` drops them, analysis A1 never uses them). The RR-1 mice flew on SpaceX CRS-4 (launch 2014-09-21 05:52 UTC, berthed to the ISS 2014-09-23 13:21 UTC; sources: NASA CRS-4 mission overview, Wikipedia "SpaceX CRS-4") and were then moved into the station. In the data: `CO2_ppm_ISS` is frozen at exactly 2646 ppm from 2014-09-21 04:24 to 09-27 13:25 (153 h, found by `quality_gate.stuck_mask`; normal days never stay unchanged for more than 75 min); temperature and humidity freeze at 26.8 degC / 70% for about a day after launch, then swing between 18.5 and 26.8 degC and up to 76% RH in transit, then sit warm and dry (down to 31% RH) after berthing. At 09-27 13:30 all three variables jump back to the ranges of the rest of the record (temperature SD 0.26 degC, RH SD 2.9 afterwards), which is taken as the installation time; that moment is inferred from the data, not from a mission log. The first day (09-20, the scrubbed launch attempt) is pad data. Figure: `notebooks/figures/A1_check_rr1_transit.png`.
  - The radiation table (`RR-1_radiation.csv`, from 2014-09-21) is cut the same way: whole days from 2014-09-28, the first full day after installation (32 days). Mean 0.202 mGy/day = **8.42 µGy/h** (GCR 0.131, SAA 0.072 mGy/day); with the pre-ISS days it was 0.204 / 8.50.
  - End of record checked (2014-10-20 to the last reading 2014-11-01 03:54): no step change around the Dragon CRS-4 release (2014-10-25 13:57 UTC) or afterwards; CO2 never stays unchanged for more than 55 min, temperature / humidity plateaus (up to 460 min) are as long as on mid-record days, so the end of the record is kept. Default cabin day is 2014-10-15 (`config.ENV_DEFAULT_CABIN_START`).

## 2. WESAD

- Links: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html · https://archive.ics.uci.edu/dataset/465/wesad
- Files: subjects S2, S3, S4, S5. Each folder keeps `S<n>.pkl` (synchronised chest + wrist signals with labels), `S<n>_quest.csv` and `S<n>_readme.txt` (age, height, weight, sex). `wesad_readme.pdf` documents the format.
- Removed from the project copy: `S<n>_E4_Data.zip` and `S<n>_respiban.txt` (raw duplicates of what the pickle already contains). Originals remain in the full download.
- License: academic, non-commercial use. Cite: Schmidt P., Reiss A., Duerichen R., Marberger C., Van Laerhoven K. *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018.
- Notes: S1 and S12 are absent from the official release (sensor malfunction); this is expected.
- Not redistributed: the public repository holds only `data/processed/wesad_params.json` (fitted
  summary statistics). The per-subject, per-minute series `data/processed/wesad_hr_hrv.csv` and
  the raw files stay local. To re-run the analysis or the real heart-rate / HRV fluctuation,
  download WESAD from the links above and run `python -m spacefuzz.readers.physio_replay --build`. Without it the
  heart-rate / HRV fluctuation is generated from the fitted parameters in `wesad_params.json`
  (model.Fluctuation.from_params); the cached demo results do not depend on either.

## 3–4. RadLab (DosTel2 and MSL-RAD)

- Link: NASA RadLab portal (https://visualization.osdr.nasa.gov/radlab/gui/data-overview/), export via "Export data: CSV".
- Window: 2017-09-04 to 2017-09-18 (UTC).
- Files:
  - `DosTel2_ISS_2017-09-04_to_09-18.csv`: ~100 s cadence, dropping to ~20 s inside SAA passes.
  - `MSL-RAD_Mars_2017-09-04_to_09-18.csv`: ~17 min cadence, ~9 min during the event.
- Columns: `timestamp, instrument_id, absorbed_dose_rate, instrument, module, spacecraft`. Dose rate is in **µGy/h** (1 mGy/day ≈ 41.7 µGy/h; EDA's mGy/day values are converted to µGy/h before the two are combined).
- License: NASA open data.
- Notes:
  - DosTel2 spikes up to ~480 µGy/h roughly every 90 minutes are **South Atlantic Anomaly passes**, not solar events. They inflate 15-minute means.
  - DosTel2 samples every ~20 s inside SAA passes vs ~100 s elsewhere (13% of rows). The plain row mean (21.7 µGy/h) is therefore biased high; the **time-weighted mean is 8.8 µGy/h**, in line with EDA RR-1's 8.42 µGy/h (0.202 mGy/day, ISS days only). `env_trace.clip` time-weights every 15-minute bin.
  - MSL-RAD shows the solar particle event clearly: baseline ~7–8 µGy/h, rise from ~2017-09-11 02:00, plateau ~20–23 µGy/h from 08:00 to 19:00, back to baseline by ~2017-09-13 04:00. Values from 2017-09-13 onward sit slightly lower (~6–7 µGy/h), consistent with a post-event Forbush decrease.
  - Gaps: DosTel2 2017-09-10 15:37–16:00; MSL-RAD 2017-09-10 08:50–15:34.
  - DosTel2 was chosen over DosTel1 because DosTel1 has gaps in this window.

## 5. Inspiration4 clinical panels (NASA OSDR)

- Links:
  - https://osdr.nasa.gov/bio/repo/data/studies/OSD-569 (Files → Complete Blood Count)
  - https://osdr.nasa.gov/bio/repo/data/studies/OSD-575 (Files → Comprehensive Metabolic Panel)
  - https://osdr.nasa.gov/bio/repo/data/studies/OSD-656 (Files → Immunoassay; downloaded, not used in the app or analysis)
- Files (original names kept so they can be traced back to OSDR):
  - `LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv`: 28 rows (4 crew × 7 timepoints), 20 analytes, each with value / `range_min` / `range_max`. Wide format.
  - `LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv`: 531 rows, 19 analytes (glucose, BUN, creatinine, electrolytes, liver panel). **Long format** (`ANALYTE, VALUE, RANGE_MIN, RANGE_MAX, UNITS, SUBJECT_ID, SEX, timepoint`); pivoted to wide format before joining with CBC.
  - `LSDS-64_Multiplex_urine.immune.AlamarPanel_TRANSFORMED.csv`: 22 urine samples × ~200 immune proteins, concentration (NPQ) and percent-normalised value.
  - `LSDS-64_Multiplex_200plex v1-Target Information.csv`: protein name lookup for the immune panel.
- Timepoints: L-92, L-44, L-3 (days before launch), R+1, R+45, R+82, R+194 (days after return). Crew IDs C001–C004.
- License: NASA open data. Cite: Sanders L.M. et al. *Inspiration4 data access through the NASA Open Science Data Repository.* npj Microgravity 10, 56 (2024). doi:10.1038/s41526-024-00393-5
- Notes:
  - CBC: hemoglobin columns are labelled `percent` but values (e.g. 14.7, upper limit 17.1) are **g/dL**. 21 missing values.
  - CBC: C003 platelet count at L-92 is 12.9 x10^3/uL while their other six timepoints are 185–396; treated as a recording error and excluded in analysis A4.
  - CMP: the R+45 samples of C002, C003 and C004 show the signature of delayed processing: glucose 36–44 mg/dL (far below all other visits; cells keep consuming it) together with raised potassium (C004 +46%, C003 +26%; cells leak it). Glucose and potassium from these three samples are excluded from baselines and from all pre/post-flight change figures and statements (analysis A4).
  - CMP: 78 empty cells, mostly one-sided reference ranges (e.g. eGFR has no upper limit), not missing measurements.
  - Immune panel: the third column header is missing (`Unnamed: 2`); it is `activin_a_percent_normalized_value`. C001 has no L-3 or R+1 sample. NPQ is a relative unit with **no reference range**, so this panel supports narrative and analysis only, not engine thresholds.

## 6. GOES-16 XRS (NOAA NCEI)

- Link: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/2017/09/
- Files: `sci_xrsf-l2-avg1m_g16_d20170904_v2-2-2.nc` … `d20170918_v2-2-2.nc` (15 daily NetCDF files, read with `xarray`), plus `GOES-R_XRS_L2_Data_ReadMe.pdf`.
- License: U.S. Government work, public domain.
- Notes:
  - Science-quality reprocessed data. GOES-16 was not yet the operational satellite in 2017; see the ReadMe caveats.
  - Flare class is read from the XRS-B band (0.1–0.8 nm). The window contains the X9.3 flare (2017-09-06) and the X8.2 flare (2017-09-10); the MSL-RAD event follows the latter.
  - Daily eclipse gaps (~56–82 min, flag=1) around the September equinox; left as gaps, not interpolated. Sporadic raw gaps of 3–6 min on 09-05, 09-10, 09-11, 09-12, 09-16, 09-18.
  - Flare class: the issued classes X9.3 (09-06) and X8.2 (09-10) are 0.635x the GOES-16 science peak flux (1.464e-3, 1.292e-3 W/m2); the common x0.7 rule would overstate them by ~10%. `config.GOES_OFFICIAL_SCALE = 0.635` (analysis A2) makes `flare_class()` reproduce both; for other flares it is an estimate. Project material quotes the issued class.
  - `xrsb_flag` bit meanings (from the variable's `flag_meanings` attribute): 1 = eclipse, 2 = bad_data, 4 = interpolated; values combine bits (e.g. 5 = eclipse + interpolated). The eclipse falls daily at about 05:10–06:31 UTC in this window.

## 6b. NOAA solar proton event list (operator D daily probability)

- Link: https://umbra.nascom.nasa.gov/SEP/ ("Solar Proton Events Affecting the Earth Environment", NOAA Space Environment Services Center, preliminary listing). The current listing it points to (ftp.swpc.noaa.gov/pub/indices/SPE.txt) could not be reached from our machine.
- File: `data/raw/noaa_spe/sep_list.html` (downloaded 2026-11-09).
- License: U.S. Government work, public domain.
- Notes:
  - Event = >10 MeV proton flux reaching 10 pfu at GOES (Earth orbit). The list covers 1976-2017, four full solar cycles: 261 events, 6.2 per year on average (0 to 23 per year; 5 years with none).
  - Analysis A2 turns this into the chance that an event starts on a given day, 261 / 15,341 days = 0.017 (`config.OP_D_DAILY_PROB`), averaged over the solar cycle; near solar maximum it is several times higher.
  - Earth-orbit events are a proxy for Mars: same Sun, but Mars has no global magnetic field and sits at a different distance, so the dose differs (stated in the UI).

## 7. NASA Space Apps 2026 official resources

Released on 2026-11-13. The challenge resource page lists five links; all five are
explanatory web pages. None of them, nor the pages two or three levels below them, offers a
downloadable dataset. SpaceFuzz therefore still takes all of its data from the public
sources in sections 1-6; the official resources are used as references (the risk pages
are the main links in `data/indicator_sources.md`).

| Resource | Link | How SpaceFuzz uses it |
| --- | --- | --- |
| NASA, 5 Hazards of Human Spaceflight | https://www.nasa.gov/hrp/hazards/ | Frame for what the scenarios cover: radiation, isolation and confinement, distance from Earth, hostile/closed environments (gravity fields are not modelled). The page notes that the hazards do not stand alone and can make each other worse. |
| NASA, Risks of Human Spaceflight | https://www.nasa.gov/hrp/risks/ | Entry page for the per-indicator risk links in `data/indicator_sources.md`. |
| NASA, Ongoing HRP Research | https://www.nasa.gov/hrp/ongoing-hrp-research/ | Background: ARCHeR (Artemis II) records crew activity and sleep with a wrist-worn actigraphy device (https://www.nasa.gov/reference/archer/). |
| CSA, Monitoring Radiation for Astronaut Health (2026-02-26) | https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp | Partner-agency reference for the radiation row: neutron radiation contributes about 30% of total radiation on the ISS. |
| CSA, Why Do We Conduct Science Experiments in Space? | https://www.asc-csa.gc.ca/eng/iss/science/why-do-we-conduct-science-experiments-in-space.asp | Deep-space missions bring longer communication delays and less ability to rely on Earth for emergency medical help (the premise of SpaceFuzz); space research on bones, heart, blood vessels and brain helps people on Earth with balance problems, osteoporosis, cardiovascular disorders and type 2 diabetes. |

**Why no real astronaut data.** Measurements from real astronauts (for example the
Spaceflight Standard Measures and Artemis II studies such as ARCHeR) are held in the NASA
Life Sciences Data Archive (LSDA). Access requires a formal request and approval, which can
include an ethics review, so they cannot be obtained during the hackathon. This is why
SpaceFuzz uses public OSDR data, the WESAD dataset and model simulation instead.
