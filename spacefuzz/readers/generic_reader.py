"""Read any CSV / TSV / XLSX table into a SpaceFuzz standard format via a YAML mapping.

A mapping file names a standard ("env", "wearable" or "clinical") and one or more
source blocks; each block says which columns hold time, ids and fields, in which
format and units. env_trace.py and physio_replay.py read their own data through
the same mappings, so a new source (e.g. the 2026-11-13 official data) needs a
new block in data/mappings/, not new code.

Standards (output columns, in order):
- env:      timestamp, then any of co2 [ppm], temp [degC], humidity [percent],
            dose_rate [uGy/h], then `extra` columns. Pass to env_trace.clip().
- wearable: subject, condition, segment_id, minute, hr [bpm], rmssd [ms]
- clinical: source, subject, specimen, timepoint, phase, day_offset, date,
            analyte, value, unit, range_min, range_max   (long format)

Usage:  read_source("data/mappings/env_telemetry.yaml", "eda_rr1")
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

STANDARD_FIELDS = {
    "env": ["co2", "temp", "humidity", "dose_rate"],
    "wearable": ["hr", "rmssd"],
}
STANDARD_COLUMNS = {
    "wearable": ["subject", "condition", "segment_id", "minute", "hr", "rmssd"],
    "clinical": ["source", "subject", "specimen", "timepoint", "phase", "day_offset", "date",
                 "analyte", "value", "unit", "range_min", "range_max"],
}

# Unit conversions into each field's standard unit (physical constants, not tunable).
_LINEAR = {
    "co2": {"ppm": 1.0, "percent": 1e4},
    "humidity": {"percent": 1.0, "fraction": 100.0},
    "dose_rate": {"uGy/h": 1.0, "mGy/h": 1000.0, "mGy/day": 1000 / 24, "uGy/min": 60.0},
    "hr": {"bpm": 1.0, "hz": 60.0},
    "rmssd": {"ms": 1.0, "s": 1000.0},
}
_TEMP = {"degC": lambda x: x, "degF": lambda x: (x - 32) * 5 / 9, "K": lambda x: x - 273.15}
MISSION_DAY = re.compile(r"^\s*([LR])\s*([+-])\s*(\d+)\s*$", re.I)


# ---------------------------------------------------------------- basics

def convert(values: pd.Series, field: str, unit: str) -> pd.Series:
    """Convert a numeric series from `unit` into the standard unit of `field`."""
    values = pd.to_numeric(values, errors="coerce")
    if field == "temp" and unit in _TEMP:
        return _TEMP[unit](values)
    table = _LINEAR.get(field)
    if table is None:
        raise ValueError(f"no unit table for field {field!r}")
    if unit not in table:
        raise ValueError(f"unsupported unit {unit!r} for {field}; known: {sorted(table)}")
    return values * table[unit]


def load_mapping(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path: Path, opts: dict | None = None) -> pd.DataFrame:
    """Read CSV/TSV (delimiter sniffed, BOM-safe) or XLSX into a DataFrame."""
    opts = opts or {}
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=opts.get("sheet", 0), skiprows=opts.get("skiprows"))
    elif ext in (".csv", ".tsv", ".txt"):
        enc = opts.get("encoding", "utf-8-sig")
        sep = opts.get("sep")
        if sep is None:
            with open(path, encoding=enc, errors="replace") as f:
                head = f.read(8192)
            try:
                sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
            except csv.Error:
                sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, encoding=enc, skiprows=opts.get("skiprows"),
                         low_memory=False)
    else:
        raise ValueError(f"unsupported file type {ext} (CSV, TSV or XLSX)")
    return df.rename(columns=opts.get("rename", {}))


def _col(df: pd.DataFrame, name: str, where: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(f"{where}: column {name!r} not found; available: {list(df.columns)[:15]}")
    return df[name]


def parse_time(values: pd.Series, fmt: str) -> tuple[str, pd.Series]:
    """Return ("timestamp", datetimes) or ("minute", relative minutes)."""
    if fmt == "iso":
        # mixed or zoned ISO strings ("...Z", "+08:00"): convert to UTC, then drop the zone so
        # every timestamp compares with the naive window bounds used by env_trace.clip
        t = pd.to_datetime(values, errors="coerce", utc=True)
        return "timestamp", t.dt.tz_localize(None)
    if fmt == "epoch_s":
        return "timestamp", pd.to_datetime(pd.to_numeric(values), unit="s")
    if fmt == "epoch_ms":
        return "timestamp", pd.to_datetime(pd.to_numeric(values), unit="ms")
    if fmt == "seconds":
        return "minute", pd.to_numeric(values, errors="coerce") / 60
    if fmt == "minutes":
        return "minute", pd.to_numeric(values, errors="coerce").astype(float)
    return "timestamp", pd.to_datetime(values.astype(str).str.strip(), format=fmt, errors="coerce")


def parse_mission_day(labels: pd.Series, anchors: dict | None = None) -> pd.DataFrame:
    """'L-92' / 'R+1' -> phase, day_offset and (with anchors) calendar date."""
    parts = labels.astype(str).str.extract(MISSION_DAY)
    if parts.isna().any().any():
        bad = labels[parts.isna().any(axis=1)].unique()[:5]
        raise ValueError(f"unparseable mission-day labels: {list(bad)}")
    letter = parts[0].str.upper()
    offset = parts[2].astype(int) * np.where(parts[1] == "-", -1, 1)
    out = pd.DataFrame({"phase": np.where(letter == "L", "pre_launch", "post_return"),
                        "day_offset": offset.astype(int)}, index=labels.index)
    out["date"] = pd.NaT
    if anchors:
        base = letter.map({k.upper(): pd.Timestamp(v) for k, v in anchors.items()})
        out["date"] = base + pd.to_timedelta(out["day_offset"], unit="D")
    return out


def _apply_filters(df: pd.DataFrame, filters: list | None) -> pd.DataFrame:
    for f in filters or []:
        col = _col(df, f["column"], "filter")
        if "equals" in f:
            df = df[col == f["equals"]]
        elif "in" in f:
            df = df[col.isin(f["in"])]
        elif f.get("not_null"):
            df = df[col.notna()]
    return df


# ---------------------------------------------------------------- layouts

def _timeseries(df: pd.DataFrame, block: dict, standard: str) -> pd.DataFrame:
    df = _apply_filters(df, block.get("filters"))
    out = pd.DataFrame(index=df.index)
    for name, spec in (block.get("ids") or {}).items():
        values = _col(df, spec["column"], f"ids.{name}")
        if "value_map" in spec:
            mapped = values.map({k: v for k, v in spec["value_map"].items()})
            out = out[mapped.notna()]
            values = mapped[mapped.notna()]
        out[name] = values
    df = df.loc[out.index]

    t = block["time"]
    tname, tvals = parse_time(_col(df, t["column"], "time"), t["format"])
    out[tname] = tvals

    fields = block.get("fields") or {}
    allowed = STANDARD_FIELDS[standard]
    for name, spec in fields.items():
        if name not in allowed:
            raise ValueError(f"field {name!r} is not part of the {standard} standard {allowed}")
        out[name] = convert(_col(df, spec["column"], f"fields.{name}"), name, spec["unit"])
    for name, column in (block.get("extra") or {}).items():
        out[name] = _col(df, column, f"extra.{name}").values

    out = out[out[tname].notna()]
    if standard == "wearable":
        for name in STANDARD_COLUMNS["wearable"]:
            if name not in out.columns:
                out[name] = np.nan
        return out[STANDARD_COLUMNS["wearable"]].reset_index(drop=True)
    cols = [tname] + [f for f in allowed if f in out.columns] + list(block.get("extra") or {})
    return out[cols].sort_values(tname).reset_index(drop=True)


def _clinical(df: pd.DataFrame, block: dict, mapping: dict, source: str) -> pd.DataFrame:
    layout = block.get("layout", "long")
    if layout == "long":
        c = block["columns"]
        out = pd.DataFrame({
            "subject": _col(df, c["subject"], "columns.subject").astype(str),
            "specimen": df[c["specimen"]] if "specimen" in c else block.get("specimen"),
            "timepoint": _col(df, c["timepoint"], "columns.timepoint").astype(str),
            "analyte": _col(df, c["analyte"], "columns.analyte").astype(str),
            "value": pd.to_numeric(_col(df, c["value"], "columns.value"), errors="coerce"),
            "unit": df[c["unit"]] if "unit" in c else block.get("unit"),
            "range_min": pd.to_numeric(df[c["range_min"]], errors="coerce")
            if "range_min" in c else np.nan,
            "range_max": pd.to_numeric(df[c["range_max"]], errors="coerce")
            if "range_max" in c else np.nan,
        })
    elif layout == "wide":
        s, v = block["sample"], block["values"]
        ids = _col(df, s["column"], "sample.column").astype(str).str.extract(s["pattern"])
        if ids.isna().all(axis=None):
            raise ValueError(f"sample.pattern matched no {s['column']!r} values")
        vpat = re.compile(v["pattern"])
        frames = []
        for col in df.columns:
            m = vpat.match(str(col))
            if not m:
                continue
            g = m.groupdict()
            analyte, raw_unit = g["analyte"], g.get("unit")
            unit = block.get("unit_overrides", {}).get(analyte) or \
                block.get("unit_names", {}).get(raw_unit, raw_unit) or v.get("unit")
            part = pd.DataFrame({
                "subject": ids.get("subject"), "specimen": ids.get("specimen"),
                "timepoint": ids.get("timepoint"), "analyte": analyte,
                "value": pd.to_numeric(df[col], errors="coerce"), "unit": unit})
            for key in ("range_min", "range_max"):
                name = v[key].format(analyte=analyte, unit=raw_unit) if key in v else None
                part[key] = pd.to_numeric(df[name], errors="coerce") \
                    if name in df.columns else np.nan
            frames.append(part)
        if not frames:
            raise ValueError(f"values.pattern {v['pattern']!r} matched no columns")
        out = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"unknown clinical layout {layout!r}")

    out = out[out["value"].notna()].reset_index(drop=True)
    tp = parse_mission_day(out["timepoint"], (mapping.get("timepoints") or {}).get("anchors"))
    out = pd.concat([out, tp], axis=1)
    out["source"] = source
    return out[STANDARD_COLUMNS["clinical"]]


def read_source(mapping_path, source: str, data_path=None, base_dir: Path = ROOT) -> pd.DataFrame:
    """Read one source block of a mapping file into its standard format.

    `data_path` overrides the block's `file`; relative paths resolve against base_dir.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.is_absolute():
        mapping_path = base_dir / mapping_path
    mapping = load_mapping(mapping_path)
    standard = mapping["standard"]
    try:
        block = mapping["sources"][source]
    except KeyError:
        raise KeyError(f"source {source!r} not in {mapping_path.name}; "
                       f"available: {list(mapping['sources'])}") from None
    path = Path(data_path or block["file"])
    if not path.is_absolute():
        path = base_dir / path
    df = read_table(path, block.get("read"))
    if standard == "clinical":
        return _clinical(df, block, mapping, source)
    if standard in STANDARD_FIELDS:
        return _timeseries(df, block, standard)
    raise ValueError(f"unknown standard {standard!r}")
