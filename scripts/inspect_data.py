"""Inspect every data file under data/raw/ and write a Markdown report.

For each CSV, TSV, XLSX, PKL and NC file the report lists: path, rows x
columns, column names and dtypes, missing-value ratio, first rows,
describe(), and suspected time columns with their format.

Files are only read, never written. Note: .pkl files are unpickled, which can
execute code; only run this on data from trusted sources (e.g. WESAD).

Usage:  python scripts/inspect_data.py [--root data/raw] [--out data/inspect_report.md]
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "inspect_report.md"

SUPPORTED = {".csv", ".tsv", ".xlsx", ".pkl", ".nc"}
HEAD_ROWS = 5
WIDE_TABLE_COLS = 12        # transpose head() above this many columns
TIME_SAMPLE = 200           # values sampled when guessing a time format
TIME_PARSE_MIN_RATIO = 0.9  # share of sampled values that must parse
TIME_NAME_HINT = re.compile(r"time|date|timestamp|epoch|utc|gmt|^t$", re.I)
MISSION_DAY = re.compile(r"^[LR]\s*[+-]?\s*\d+$", re.I)  # Inspiration4 "L-92", "R+1"
MISSION_DAY_IN_TEXT = re.compile(r"(?<![A-Za-z0-9])([LR][+-]\d+)(?![0-9])")  # "C001_..._L-3_cbc"
EPOCH_S = (9e8, 4.1e9)       # 1998..2100 in seconds
EPOCH_MS = (9e11, 4.1e12)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 40)


# ---------------------------------------------------------------- time columns

@dataclass
class TimeColumn:
    column: str
    fmt: str
    detail: str = ""


def _span(ts: pd.Series) -> str:
    ts = ts.dropna()
    if ts.empty:
        return ""
    step = ts.sort_values().diff().dropna()
    median = f", median step {step.median()}" if not step.empty else ""
    return f"{ts.min()} → {ts.max()}{median}"


def _guess_string_format(values: pd.Series) -> str | None:
    sample = values.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(TIME_SAMPLE)
    if sample.empty or not sample.str.contains(r"\d[-/:T.]\d|\d{4}-\d", regex=True).any():
        return None
    fmt = pd.tseries.api.guess_datetime_format(sample.iloc[0])
    if not fmt:
        return None
    parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
    return fmt if parsed.notna().mean() >= TIME_PARSE_MIN_RATIO else None


def detect_time_columns(df: pd.DataFrame) -> list[TimeColumn]:
    found = []
    for col in df.columns:
        s = df[col]
        name = str(col)
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(TimeColumn(name, "datetime64", _span(s)))
        elif pd.api.types.is_timedelta64_dtype(s):
            found.append(TimeColumn(name, "timedelta64", f"{s.min()} → {s.max()}"))
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not TIME_NAME_HINT.search(name):
                continue
            v = s.dropna()
            if v.empty:
                continue
            lo, hi = float(v.min()), float(v.max())
            if EPOCH_S[0] <= lo and hi <= EPOCH_S[1]:
                found.append(TimeColumn(name, "epoch seconds", _span(pd.to_datetime(v, unit="s"))))
            elif EPOCH_MS[0] <= lo and hi <= EPOCH_MS[1]:
                found.append(TimeColumn(name, "epoch milliseconds",
                                        _span(pd.to_datetime(v, unit="ms"))))
            else:
                found.append(TimeColumn(name, "numeric offset (unit unknown)", f"{lo} → {hi}"))
        elif pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            text = s.dropna().astype(str).str.strip()
            text = text[text != ""]
            if text.empty:
                continue
            if text.head(TIME_SAMPLE).str.match(MISSION_DAY).mean() >= TIME_PARSE_MIN_RATIO:
                labels = sorted(text.unique(), key=_mission_day_key)
                found.append(TimeColumn(name, "mission-relative day label (L-n / R+n)",
                                        f"{len(labels)} labels: {', '.join(labels[:12])}"
                                        + (" …" if len(labels) > 12 else "")))
                continue
            embedded = text.str.extract(MISSION_DAY_IN_TEXT, expand=False)
            if embedded.notna().mean() >= TIME_PARSE_MIN_RATIO:
                labels = sorted(embedded.dropna().unique(), key=_mission_day_key)
                found.append(TimeColumn(name, "mission-relative day label embedded in text",
                                        f"e.g. {text.iloc[0]!r}; {len(labels)} labels: "
                                        + ", ".join(labels[:12])
                                        + (" …" if len(labels) > 12 else "")))
                continue
            fmt = _guess_string_format(text)
            if fmt:
                parsed = pd.to_datetime(text, format=fmt, errors="coerce")
                bad = parsed.isna().mean()
                extra = f"; {bad:.1%} unparseable" if bad else ""
                found.append(TimeColumn(name, fmt, _span(parsed) + extra))
    return found


def _mission_day_key(label: str) -> tuple[int, int]:
    m = re.match(r"^([LR])\s*([+-]?)\s*(\d+)$", label.strip(), re.I)
    if not m:
        return (2, 0)
    phase, sign, n = m.groups()
    return (0 if phase.upper() == "L" else 1, -int(n) if sign == "-" else int(n))


# ---------------------------------------------------------------- loaders

def _sniff_sep(path: Path, default: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        head = f.read(8192)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def _load_delimited(path: Path) -> pd.DataFrame:
    sep = _sniff_sep(path, "\t" if path.suffix.lower() == ".tsv" else ",")
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    # Files whose every line starts with "#" (WESAD *_quest.csv) have no header row.
    header = None if first.startswith("#") else "infer"
    return pd.read_csv(path, sep=sep, header=header, encoding="utf-8-sig",
                       low_memory=False)


def _array_frame(arr: np.ndarray) -> pd.DataFrame:
    if arr.ndim <= 1:
        return pd.DataFrame({"value": np.asarray(arr).ravel()})
    if arr.ndim == 2:
        return pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])
    return pd.DataFrame(arr.reshape(arr.shape[0], -1))


def _walk_pickle(obj, prefix: str, tables: list, notes: list) -> None:
    label = prefix or "(root)"
    if isinstance(obj, pd.DataFrame):
        tables.append((label, obj))
    elif isinstance(obj, pd.Series):
        tables.append((label, obj.to_frame()))
    elif isinstance(obj, np.ndarray) and obj.ndim >= 1 and obj.size > 1:
        tables.append((label, _array_frame(obj)))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _walk_pickle(value, f"{prefix}/{key}" if prefix else str(key), tables, notes)
    elif isinstance(obj, (list, tuple)) and obj and all(np.isscalar(x) for x in obj):
        tables.append((label, pd.DataFrame({"value": list(obj)})))
    else:
        text = repr(obj)
        notes.append(f"`{label}`: {type(obj).__name__} = {text[:120]}")


def _load_pickle(path: Path) -> tuple[list, list]:
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            # Python 2 pickles (e.g. WESAD) need latin1 to decode numpy buffers.
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    tables, notes = [], []
    _walk_pickle(obj, "", tables, notes)
    return tables, notes


def _load_netcdf(path: Path) -> tuple[list, list]:
    import xarray as xr

    tables, notes = [], []
    with xr.open_dataset(path) as ds:
        notes.append("dims: " + ", ".join(f"{k}={v}" for k, v in ds.sizes.items()))
        for key in ("title", "summary", "time_coverage_start", "time_coverage_end"):
            if key in ds.attrs:
                notes.append(f"attr `{key}`: {str(ds.attrs[key])[:200]}")
        units = [f"`{n}` [{v.attrs['units']}]" for n, v in ds.variables.items()
                 if "units" in v.attrs]
        if units:
            notes.append("units: " + ", ".join(units))
        multi = [n for n, v in ds.variables.items() if v.ndim > 1]
        scalars = [n for n, v in ds.variables.items() if v.ndim == 0]
        if multi:
            notes.append("multi-dimensional variables (not tabulated): "
                         + ", ".join(f"`{n}` {dict(ds[n].sizes)}" for n in multi))
        if scalars:
            notes.append("scalar variables: " + ", ".join(f"`{n}`" for n in scalars))
        for dim in ds.dims:
            names = [n for n, v in ds.variables.items() if v.dims == (dim,)]
            if not names:
                continue
            frame = pd.DataFrame({n: ds[n].values for n in names})
            tables.append((f"dim: {dim}", frame))
    return tables, notes


def load_tables(path: Path) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    """Read a file into (tables, notes). Raises on unreadable files."""
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        return [(path.name, _load_delimited(path))], []
    if ext == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None)
        return [(f"sheet: {name}", df) for name, df in sheets.items()], []
    if ext == ".pkl":
        return _load_pickle(path)
    if ext == ".nc":
        return _load_netcdf(path)
    raise ValueError(f"unsupported file type {ext}")


# ---------------------------------------------------------------- report

def _block(text: str) -> str:
    return f"```\n{text}\n```\n"


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def describe_table(name: str, df: pd.DataFrame) -> str:
    out = [f"#### {name}\n", f"{len(df):,} rows × {df.shape[1]} columns\n"]

    missing = df.isna().mean()
    out.append("| column | dtype | missing |\n| --- | --- | --- |")
    for col in df.columns:
        out.append(f"| {_md_cell(col)} | {df[col].dtype} | {missing[col]:.1%} |")
    out.append("")

    times = detect_time_columns(df)
    if times:
        out.append("Suspected time columns:\n")
        out += [f"- `{t.column}`: **{t.fmt}**" + (f" ({t.detail})" if t.detail else "")
                for t in times]
    else:
        out.append("Suspected time columns: none (time may be implicit in the row index; "
                   "see the dataset readme for the sampling rate).")
    out.append("")

    head = df.head(HEAD_ROWS)
    out.append(f"First {HEAD_ROWS} rows" + (" (transposed)" if df.shape[1] > WIDE_TABLE_COLS else "")
               + ":\n")
    out.append(_block((head.T if df.shape[1] > WIDE_TABLE_COLS else head).to_string()))

    try:
        desc = df.describe(include="all")
        out.append("describe() (transposed):\n")
        out.append(_block(desc.T.to_string()))
    except ValueError as exc:
        out.append(f"describe() unavailable: {exc}\n")
    return "\n".join(out)


def run(raw_dir: Path, out_path: Path) -> dict:
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    inspected = [p for p in files if p.suffix.lower() in SUPPORTED]
    skipped = [p for p in files if p.suffix.lower() not in SUPPORTED]

    sections, summary, failures = [], [], []
    for path in inspected:
        rel = path.relative_to(raw_dir).as_posix()
        print(f"reading {rel} ...", file=sys.stderr)
        try:
            tables, notes = load_tables(path)
        except Exception as exc:  # report and continue with the next file
            reason = f"{type(exc).__name__}: {exc}"
            failures.append((rel, reason, traceback.format_exc(limit=2)))
            summary.append(f"| {_md_cell(rel)} | ❌ unreadable | | |")
            continue
        size = path.stat().st_size / 1e6
        shapes = "; ".join(f"{n}: {df.shape[0]:,}×{df.shape[1]}" for n, df in tables[:4])
        if len(tables) > 4:
            shapes += f"; … ({len(tables)} tables)"
        tcols = sorted({t.column for _, df in tables for t in detect_time_columns(df)})
        summary.append(f"| {_md_cell(rel)} | {size:.1f} MB | {_md_cell(shapes)} | "
                       f"{_md_cell(', '.join(tcols)) or '—'} |")
        body = [f"### {rel}\n", f"Path: `{path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path}`"
                f" · {size:.1f} MB · {len(tables)} table(s)\n"]
        if notes:
            body.append("Notes:\n")
            body += [f"- {n}" for n in notes]
            body.append("")
        body += [describe_table(name, df) for name, df in tables]
        sections.append("\n".join(body))
        del tables

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = [
        "# Data inspection report\n",
        f"Generated {now} by `scripts/inspect_data.py` from `{raw_dir.name}/`. "
        f"{len(inspected)} files inspected, {len(failures)} unreadable, "
        f"{len(skipped)} skipped (unsupported type).\n",
        "## Summary\n",
        "| file | size | tables (rows×cols) | time columns |\n| --- | --- | --- | --- |",
        *summary, "",
        "## Unreadable files\n",
    ]
    if failures:
        for rel, reason, _ in failures:
            report.append(f"- `{rel}`: {reason}")
    else:
        report.append("None.")
    report += ["", "## Skipped files (not CSV/TSV/XLSX/PKL/NC)\n"]
    report += [f"- `{p.relative_to(raw_dir).as_posix()}`" for p in skipped] or ["None."]
    report += ["", "## Details\n", *sections]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"inspected": len(inspected), "failed": len(failures), "skipped": len(skipped)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    stats = run(args.root.resolve(), args.out.resolve())
    print(f"wrote {args.out}: {stats['inspected']} inspected, {stats['failed']} unreadable, "
          f"{stats['skipped']} skipped")


if __name__ == "__main__":
    main()
