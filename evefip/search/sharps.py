"""HMI SHARP keywords from JSOC: magnetic flux histories and refined emergence times."""

import logging
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from evefip import config

log = logging.getLogger(__name__)

SERIES = "hmi.sharp_720s"
KEYS = ["HARPNUM", "T_REC", "NOAA_ARS", "USFLUX", "AREA_ACR", "LON_FWT", "LAT_FWT"]
HARP_MAP_URL = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"


def _jsoc_time(t):
    return pd.Timestamp(t).strftime("%Y.%m.%d_%H:%M:%S_TAI")


@lru_cache
def _client():
    import drms

    return drms.Client()


def _query(recset):
    import drms

    keys = _client().query(recset, key=",".join(KEYS))
    if len(keys) == 0:
        return pd.DataFrame(columns=["harpnum", "time", "noaa_ars", "usflux", "area", "lon", "lat"])
    df = pd.DataFrame({
        "harpnum": keys["HARPNUM"].astype(int),
        "time": drms.to_datetime(keys["T_REC"]),
        "noaa_ars": keys["NOAA_ARS"].astype(str),
        "usflux": pd.to_numeric(keys["USFLUX"], errors="coerce"),
        "area": pd.to_numeric(keys["AREA_ACR"], errors="coerce"),
        "lon": pd.to_numeric(keys["LON_FWT"], errors="coerce"),
        "lat": pd.to_numeric(keys["LAT_FWT"], errors="coerce"),
    })
    return df.sort_values(["harpnum", "time"]).reset_index(drop=True)


def harp_noaa_map(cache=None):
    """Return a DataFrame mapping each HARPNUM to its NOAA region number(s)."""
    cache = Path(cache or config.CACHE_DIR / "all_harps_with_noaa_ars.txt")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urlopen(HARP_MAP_URL, timeout=60).read())
    rows = []
    for line in cache.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2:
            rows += [(int(parts[0]), int(n)) for n in parts[1].split(",")]
    return pd.DataFrame(rows, columns=["harpnum", "noaa"])


def harps_for_noaa(noaa, mapping=None):
    mapping = harp_noaa_map() if mapping is None else mapping
    return sorted(mapping.loc[mapping["noaa"] == noaa, "harpnum"])


def harp_series(harpnum, start, end, step="1h"):
    """Keyword time series for one HARP."""
    return _query(f"{SERIES}[{harpnum}][{_jsoc_time(start)}-{_jsoc_time(end)}@{step}]")


def all_harps(start, end, step="6h"):
    """Keywords for every HARP on the disk between `start` and `end`."""
    return _query(f"{SERIES}[][{_jsoc_time(start)}-{_jsoc_time(end)}@{step}]")


def emergence_summary(series, flux_threshold=config.T0_FLUX, east_limit=config.EAST_LIMB_LON):
    """
    Describe how a HARP starts.

    ``t0`` is the first time the unsigned flux exceeds `flux_threshold`. ``emerged_on_disk`` is
    True when the HARP was first detected inside the east limb with less than 20% of the flux it
    reaches within five days, i.e. we see the flux grow rather than rotate into view.
    """
    # The first records of a HARP often carry zero flux and no position; skip them.
    series = series.dropna(subset=["usflux", "lon"])
    series = series[series["usflux"] > 0]
    if len(series) == 0:
        return {"harp_start": pd.NaT, "harp_lon0": np.nan, "t0_sharp": pd.NaT,
                "flux_start": np.nan, "flux_max_5d": np.nan, "emerged_on_disk": False}
    start = series.iloc[0]
    first5 = series[series["time"] <= start["time"] + pd.Timedelta(days=5)]
    flux_max = first5["usflux"].max()
    above = series[series["usflux"] >= flux_threshold]
    return {
        "harp_start": start["time"],
        "harp_lon0": start["lon"],
        "t0_sharp": above["time"].iloc[0] if len(above) else pd.NaT,
        "flux_start": start["usflux"],
        "flux_max_5d": flux_max,
        "emerged_on_disk": bool(start["lon"] > east_limit and start["usflux"] < 0.2 * flux_max),
    }


def other_flux_fraction(all_keys, target_harps, t0, flux_ref, before=config.BACKGROUND_DAYS,
                        after=config.TRACK_DAYS):
    """
    Mean summed |B| flux of all *other* HARPs, as a fraction of the target's peak flux.

    Computed separately over the background and tracking windows. Small values mean the target
    dominates the magnetic (and probably the EUV) variability of the disk.
    """
    t0 = pd.Timestamp(t0)
    others = all_keys[~all_keys["harpnum"].isin(target_harps)]
    out = {}
    for name, lo, hi in [("bg", t0 - pd.Timedelta(days=before), t0),
                         ("track", t0, t0 + pd.Timedelta(days=after))]:
        w = others[(others["time"] >= lo) & (others["time"] <= hi)]
        total = w.groupby("time")["usflux"].sum()
        out[f"{name}_other_flux_frac"] = float(total.mean() / flux_ref) if len(total) and flux_ref else np.nan
    return out
