"""GOES flares (via HEK / SWPC event list) around a candidate emergence."""

import logging

import numpy as np
import pandas as pd

from evefip import config

log = logging.getLogger(__name__)

CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def class_to_flux(goes_class):
    """'M2.3' -> 2.3e-5 W/m^2 (NaN if unparseable)."""
    try:
        return CLASS_FLUX[goes_class[0].upper()] * float(goes_class[1:])
    except (KeyError, ValueError, IndexError, TypeError):
        return np.nan


def flux_to_class(flux):
    if not np.isfinite(flux):
        return ""
    for letter in "XMCBA":
        if flux >= CLASS_FLUX[letter]:
            return f"{letter}{flux / CLASS_FLUX[letter]:.1f}"
    return "A0.0"


def flares(start, end):
    """SWPC GOES flares between `start` and `end` as a DataFrame."""
    from sunpy.net import attrs as a
    from sunpy.net import hek

    result = hek.HEKClient().search(
        a.Time(pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()),
        a.hek.EventType("FL"), a.hek.FRM.Name == "SWPC",
    )
    if len(result) == 0:
        return pd.DataFrame(columns=["start", "peak", "end", "goes_class", "flux", "noaa"])
    df = pd.DataFrame({
        "start": pd.to_datetime([str(t) for t in result["event_starttime"]]),
        "peak": pd.to_datetime([str(t) for t in result["event_peaktime"]]),
        "end": pd.to_datetime([str(t) for t in result["event_endtime"]]),
        "goes_class": [str(c) for c in result["fl_goescls"]],
        "noaa": np.asarray(result["ar_noaanum"], dtype=float),
    })
    df["flux"] = df["goes_class"].map(class_to_flux)
    return df.drop_duplicates(["peak", "goes_class"]).sort_values("peak").reset_index(drop=True)


def flare_metrics(flare_table, number, t0, before=config.BACKGROUND_DAYS, after=config.TRACK_DAYS):
    """Flare counts (>= C1) and peak class in the background and tracking windows."""
    t0 = pd.Timestamp(t0)
    out = {}
    for name, lo, hi in [("bg", t0 - pd.Timedelta(days=before), t0),
                         ("track", t0, t0 + pd.Timedelta(days=after))]:
        w = flare_table[(flare_table["peak"] >= lo) & (flare_table["peak"] <= hi)]
        big = w[w["flux"] >= 1e-6]
        out[f"{name}_n_flares_C"] = int(len(big))
        out[f"{name}_n_flares_C_target"] = int((big["noaa"] == number).sum())
        out[f"{name}_max_class"] = flux_to_class(w["flux"].max()) if len(w) else ""
    return out
