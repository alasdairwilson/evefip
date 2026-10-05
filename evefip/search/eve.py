"""
SDO/EVE level 2B line irradiances, reduced to a mission-long hourly cache.

Level 2B (EVL_L2B) holds one-minute line irradiances, one file per day. The combined
``LINE_IRRADIANCE`` column is filled only when MEGS-B is observing, which from 2011 is just
~3 h/day plus 5 min/hour. We therefore read ``ChannelLinesData`` and take each line from a
single fixed channel, so cross-channel calibration offsets never cause jumps.
"""

import logging
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from astropy.io import fits

from evefip import config

log = logging.getLogger(__name__)

BASE_URL = "https://lasp.colorado.edu/eve/data_access/eve_data/products/level2b"
TAI_EPOCH = pd.Timestamp("1958-01-01")


def channel_for(wavelength_nm):
    """MEGS channel used for a line: A1 below 17 nm, A2 to 37 nm, B above."""
    if wavelength_nm < 17.0:
        return "MEGSA1"
    if wavelength_nm < 37.0:
        return "MEGSA2"
    return "MEGSB"


def _day_dir(date):
    date = pd.Timestamp(date)
    return f"{date.year}/{date.dayofyear:03d}"


def l2b_file(date, cache=None):
    """Local path of the EVL L2B file for `date`, downloading it if needed (None if absent)."""
    cache = Path(cache or config.CACHE_DIR / "eve" / "level2b")
    local_dir = cache / _day_dir(date)
    existing = sorted(local_dir.glob("EVL_L2B_*.fit.gz"))
    if existing:
        return existing[-1]
    url = f"{BASE_URL}/{_day_dir(date)}/"
    try:
        listing = urlopen(url, timeout=60).read().decode()
    except HTTPError as exc:
        if exc.code != 404:
            raise  # transient; don't record the day as having no data
        log.info("No EVE L2B data for %s", pd.Timestamp(date).date())
        return None
    names = sorted(set(re.findall(r'href="(EVL_L2B_[^"]+\.fit\.gz)"', listing)))
    if not names:
        return None
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / names[-1]  # highest version/revision
    tmp = target.with_suffix(".part")
    tmp.write_bytes(urlopen(url + names[-1], timeout=300).read())
    tmp.rename(target)
    return target


def read_l2b(filename, lines=config.SIGNAL_LINES):
    """
    One day of one-minute irradiances for `lines`, plus per-channel availability flags.

    Columns: one per entry in `lines` (W/m^2, NaN when missing), and ``MEGSA_ok`` /
    ``MEGSB_ok`` booleans for whether each channel returned any valid line that minute.
    """
    with fits.open(filename) as hdul:
        meta = hdul["ChannelLinesMeta"].data
        data = hdul["ChannelLinesData"].data
        names = [n.strip() for n in meta["NAME"]]
        waves = np.asarray(meta["WAVE_CENTER"], dtype=float)
        times = TAI_EPOCH + pd.to_timedelta(np.asarray(data["TAI"], dtype=float), unit="s")
        # Leap seconds: TAI - UTC was 34 s throughout 2009-2012 and 35 s from mid-2012.
        times = times - pd.Timedelta(seconds=35 if times[0] >= pd.Timestamp("2012-07-01") else 34)
        arrays = {}
        for ch in ("MEGSA1", "MEGSA2", "MEGSB"):
            x = np.array(data[f"{ch}_LINE_IRRADIANCE"], dtype=float)
            x[x <= 0] = np.nan
            arrays[ch] = x
    out = pd.DataFrame(index=pd.DatetimeIndex(times, name="time"))
    for label, (ion, wave) in lines.items():
        candidates = [i for i, n in enumerate(names) if n == ion]
        if not candidates:
            raise KeyError(f"{ion} not in EVL line list")
        i = min(candidates, key=lambda j: abs(waves[j] - wave))
        out[label] = arrays[channel_for(waves[i])][:, i]
    out["MEGSA_ok"] = np.isfinite(arrays["MEGSA2"]).any(axis=1)
    out["MEGSB_ok"] = np.isfinite(arrays["MEGSB"]).any(axis=1)
    return out


def coverage_metrics(h, t0, after=config.TRACK_DAYS):
    """
    How usable the tracking window is for hourly binning, from the hourly table.

    ``megsa_hours_frac`` is the fraction of hours with >= 30 min of MEGS-A, ``megsb_hours_frac``
    the fraction with >= 5 min of MEGS-B (enough for an hourly average), and
    ``megsb_min_per_day`` the mean number of MEGS-B minutes per day.
    """
    t0 = pd.Timestamp(t0)
    hours = pd.date_range(t0.ceil("h"), t0 + pd.Timedelta(days=after), freq="h", inclusive="left")
    counts = h.reindex(hours)[["n_megsa", "n_megsb"]].fillna(0) if len(h) else pd.DataFrame(
        0, index=hours, columns=["n_megsa", "n_megsb"])
    return {
        "megsa_hours_frac": float((counts["n_megsa"] >= 30).mean()),
        "megsb_hours_frac": float((counts["n_megsb"] >= 5).mean()),
        "megsb_min_per_day": float(counts["n_megsb"].sum() / after),
    }


def hourly(df, lines=config.SIGNAL_LINES):
    """
    Hourly mean irradiance of each line (NaN where no valid minutes), plus ``n_megsa`` /
    ``n_megsb``, the number of minutes each channel observed in that hour.
    """
    h = df[list(lines)].resample("h").mean()
    counts = df[["MEGSA_ok", "MEGSB_ok"]].resample("h").sum().astype(int)
    h["n_megsa"], h["n_megsb"] = counts["MEGSA_ok"], counts["MEGSB_ok"]
    return h


def _hourly_dir():
    return config.CACHE_DIR / "eve" / "hourly"


def _hourly_day(day, prune=False):
    """Reduce one day of L2B to hourly values, cached as one parquet per day (empty if no data)."""
    day = pd.Timestamp(day).floor("D")
    out = _hourly_dir() / f"{day:%Y%j}.parquet"
    if out.exists():
        return out
    empty = pd.DataFrame(columns=[*config.SIGNAL_LINES, "n_megsa", "n_megsb"],
                         index=pd.DatetimeIndex([], name="time"))
    try:
        f = l2b_file(day)
        h = hourly(read_l2b(f)) if f is not None else empty
    except Exception as exc:
        log.warning("EVE %s failed: %s", day.date(), exc)
        return None  # transient: retried on the next run
    out.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(out)
    if prune and f is not None:
        f.unlink()
    return out


def build_hourly_cache(start, end, workers=6, prune=False):
    """
    Download and reduce every L2B day in [start, end] to the hourly cache. Resumable: days
    already reduced (including those with no EVE data) are skipped.
    """
    from concurrent.futures import ThreadPoolExecutor

    days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D")
    todo = [d for d in days if not (_hourly_dir() / f"{d:%Y%j}.parquet").exists()]
    log.info("%d of %d days to reduce", len(todo), len(days))
    with ThreadPoolExecutor(workers) as pool:
        for i, result in enumerate(pool.map(lambda d: _hourly_day(d, prune), todo), 1):
            if i % 50 == 0:
                log.info("%d / %d days done", i, len(todo))
    failed = [d for d in todo if not (_hourly_dir() / f"{d:%Y%j}.parquet").exists()]
    if failed:
        log.warning("%d days failed, rerun to retry: %s", len(failed), [str(d.date()) for d in failed[:5]])


def load_hourly(start, end, download=True):
    """Hourly EVE line table for [start, end] from the cache (reducing missing days if allowed)."""
    frames = []
    for day in pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D"):
        f = _hourly_dir() / f"{day:%Y%j}.parquet"
        if not f.exists() and download:
            f = _hourly_day(day)
        if f is not None and f.exists():
            frames.append(pd.read_parquet(f))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=[*config.SIGNAL_LINES, "n_megsa", "n_megsb"])
    h = pd.concat(frames).sort_index()
    return h[(h.index >= pd.Timestamp(start)) & (h.index <= pd.Timestamp(end))]
