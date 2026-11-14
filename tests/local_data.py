"""Which git-ignored data files are on this machine.

A fresh clone has data/raw/ too (two small CSVs are committed), so tests that read the
downloaded files check for those files, not for the folder.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WESAD_CSV = ROOT / "data" / "processed" / "wesad_hr_hrv.csv"

DOWNLOADED = [RAW / "radlab" / "DosTel2_ISS_2017-09-04_to_09-18.csv",
              RAW / "eda" / "RR-1_radiation.csv",
              RAW / "inspiration4" / "LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv"]
HAS_RAW = all(p.exists() for p in DOWNLOADED) and any((RAW / "goes").glob("*.nc"))
HAS_WESAD = WESAD_CSV.exists()
HAS_ALL = HAS_RAW and HAS_WESAD
