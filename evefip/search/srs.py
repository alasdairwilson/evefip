"""NOAA Solar Region Summaries: daily region positions, and on-disk emergence detection."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evefip import config

log = logging.getLogger(__name__)

COLUMNS = ["date", "number", "id", "lat", "lon", "carr_lon", "area", "n_spots", "mag_type"]


def fetch_srs(start, end, path=None):
    """Download (or reuse cached) daily SRS files and return their paths."""
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    path = Path(path or config.CACHE_DIR / "srs")
    path.mkdir(parents=True, exist_ok=True)
    result = Fido.search(a.Time(start, end), a.Instrument.soon)
    files = Fido.fetch(result, path=str(path / "{file}"), progress=False)
    if files.errors:
        log.warning("%d SRS downloads failed", len(files.errors))
    return sorted(Path(f) for f in files)


def _date_from_filename(filename):
    # SRS locations are valid at 24:00 UT of the previous day, i.e. 00:00 UT of the issue date.
    match = re.search(r"(\d{8})SRS", Path(filename).name)
    return pd.Timestamp(match.group(1))


_REGION = re.compile(
    r"^(?P<number>\d{4,5})\s+(?P<ns>[NS])(?P<lat>\d{2})(?P<ew>[EW])(?P<lon>\d{2,3})\s+(?P<carr>\d{3})"
    r"(?:\s+(?P<area>\d{4})\s+\w{3}\s+\d+\s+(?P<nn>\d+)\s+(?P<mag>\S+))?"
)


def read_srs_file(filename):
    """
    Read sections I (spots) and IA (plages) of one SRS file into a DataFrame with `COLUMNS`.

    We parse the fixed text format ourselves: ``sunpy.io.special.srs.read_srs`` fails (Location
    column dtype conflict in ``vstack``) whenever a section reads "None", which drops roughly a
    quarter of the days in 2010-2014. Section II (regions due to return) is not on the disk.
    """
    date = _date_from_filename(filename)
    rows, section = [], None
    for line in Path(filename).read_text(errors="replace").splitlines():
        if line.startswith("IA."):
            section = "IA"
        elif line.startswith("II."):
            break
        elif line.startswith("I."):
            section = "I"
        elif section and (m := _REGION.match(line.strip())):
            number = int(m["number"])
            rows.append({
                "date": date,
                "number": number + 10000 if number < 10000 else number,
                "id": section,
                "lat": float(m["lat"]) * (1 if m["ns"] == "N" else -1),
                "lon": float(m["lon"]) * (1 if m["ew"] == "W" else -1),
                "carr_lon": float(m["carr"]),
                "area": float(m["area"]) if m["area"] else np.nan,
                "n_spots": float(m["nn"]) if m["nn"] else np.nan,
                "mag_type": m["mag"] or "",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def load_regions(start, end, cache=None):
    """Return a tidy table of every region on the disk each day, cached as parquet."""
    cache = Path(cache or config.CACHE_DIR / "srs_regions.parquet")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if cache.exists():
        df = pd.read_parquet(cache)
        if df["date"].min() <= start + pd.Timedelta(days=3) and df["date"].max() >= end - pd.Timedelta(days=3):
            return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    files = fetch_srs(start, end)
    # A day with no file would look like an empty disk to isolation_metrics, so flag it.
    missing = pd.date_range(start.floor("D"), end.floor("D")).difference([_date_from_filename(f) for f in files])
    if len(missing):
        log.warning("%d days have no SRS file, e.g. %s", len(missing), [str(d.date()) for d in missing[:5]])
    frames = [read_srs_file(f) for f in files]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True).sort_values(["date", "number"])
    df = df.reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def find_emergences(regions, east_limit=config.EAST_LIMB_LON, min_days=config.MIN_DAYS_ON_DISK,
                    rate=config.ROTATION_RATE):
    """
    Regions whose first SRS appearance is well inside the east limb, with their last SRS
    date and longitude (to tell whether they set behind the west limb or decayed on disk).

    A region first numbered at Stonyhurst longitude `lon0` reaches the west limb after
    roughly ``(90 - lon0) / rate`` days. We keep regions with ``lon0 > east_limit`` (not just
    rotated into view) and at least ``min_days`` left on the disk.
    """
    regions = regions.dropna(subset=["lon"]).sort_values("date")
    first = regions.groupby("number", as_index=False).first()
    first = first.rename(columns={"date": "first_date", "lat": "lat0", "lon": "lon0", "id": "first_id"})
    last = regions.groupby("number")[["date", "lon"]].last()
    first["last_date"] = first["number"].map(last["date"])
    first["last_lon"] = first["number"].map(last["lon"])
    first["days_to_west_limb"] = (90.0 - first["lon0"]) / rate
    keep = (first["lon0"] > east_limit) & (first["days_to_west_limb"] >= min_days)
    cols = ["number", "first_date", "first_id", "lat0", "lon0", "days_to_west_limb", "last_date", "last_lon"]
    return first.loc[keep, cols].sort_values("first_date").reset_index(drop=True)


def isolation_metrics(regions, number, t0, before=config.BACKGROUND_DAYS, after=config.TRACK_DAYS):
    """
    Count other regions on the disk around emergence time `t0`.

    Returns metrics over the background window ``[t0 - before, t0]`` and the tracking window
    ``[t0, t0 + after]``. Spot area is in millionths of a hemisphere; plages count with zero area.
    """
    t0 = pd.Timestamp(t0)
    others = regions[regions["number"] != number]
    out = {}
    for name, lo, hi in [("bg", t0 - pd.Timedelta(days=before), t0),
                         ("track", t0, t0 + pd.Timedelta(days=after))]:
        window = others[(others["date"] >= lo.floor("D")) & (others["date"] <= hi.ceil("D"))]
        days = pd.date_range(lo.floor("D"), hi.ceil("D"), freq="D")
        counts = window.groupby("date").size().reindex(days, fill_value=0)
        areas = window.groupby("date")["area"].sum(min_count=1).reindex(days).fillna(0)
        out[f"{name}_others_max"] = int(counts.max())
        out[f"{name}_others_mean"] = float(counts.mean())
        out[f"{name}_other_area_mean"] = float(areas.mean())
    return out
