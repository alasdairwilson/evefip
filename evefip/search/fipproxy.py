"""
A cheap FIP-bias proxy from line ratios, and its stability during quiet periods.

The real measurement is a full-disk DEM fit with a free low-FIP/high-FIP abundance ratio
(phase 2). Before that, a high-FIP/low-FIP ratio of lines formed at the *same* temperature tracks
the FIP bias with little DEM sensitivity. S XIV 44.6 / Fe XVI 33.5 (log T 6.44 / 6.43) is the
cleanest pair; S is intermediate-FIP (10.4 eV) and usually behaves as high-FIP in the corona.
A FIP enhancement *lowers* the ratio.

No background subtraction: as with the f fit, the test is whether the proxy changes after
emergence by much more than it wanders over a few quiet days with no emergence.
"""

import numpy as np
import pandas as pd

from evefip import config
from evefip.search import background

DAY = pd.Timedelta(days=1)
PROXIES = {"S XIV/Fe XVI": ("S XIV 44.6", "Fe XVI 33.5"), "S XIV/Si XII": ("S XIV 44.6", "Si XII 49.9")}
MIN_MEGSB = 5  # minutes of MEGS-B in an hour for its hourly mean to count


def proxy_table(h, proxies=PROXIES):
    """Hourly proxy ratios (NaN where MEGS-B is too sparse)."""
    good = h["n_megsb"] >= MIN_MEGSB if "n_megsb" in h else pd.Series(True, index=h.index)
    return pd.DataFrame({name: (h[a] / h[b]).where(good) for name, (a, b) in proxies.items()}, index=h.index)


def half_change(series, start, end):
    """Second-half median / first-half median - 1 over ``[start, end]`` (NaN if sparse)."""
    mid = start + (end - start) / 2
    a = series.loc[start:mid].dropna()
    b = series.loc[mid:end].dropna()
    if len(a) < 6 or len(b) < 6:
        return np.nan
    return float(b.median() / a.median() - 1)


def null_windows(h, window_days=4.0, step="1D", lines=background.HOT_LINES, proxies=PROXIES):
    """
    Quiet windows across the whole table: ``[t, t + window_days]`` where the hot lines are flat
    by the same criteria as `background.flat_background`. For each window we record each
    proxy's half-to-half change, scatter and slope. Their spread is the noise floor any
    emergence has to beat.
    """
    p = proxy_table(h, proxies)
    rows = []
    for t in pd.date_range(h.index.min().ceil("D"), h.index.max() - window_days * DAY, freq=step):
        flat = background.flat_background(h, t, lines, max_days=window_days, min_days=window_days,
                                          forward=True, prefix="w")
        if flat["w_days"] == 0:
            continue
        row = {"start": t}
        for name in proxies:
            slope, scatter, _ = background.trend(p.loc[t:t + window_days * DAY, name], t)
            row[f"halfchange_{name}"] = half_change(p[name], t, t + window_days * DAY)
            row[f"slope_{name}"], row[f"scatter_{name}"] = slope, scatter
        rows.append(row)
    return pd.DataFrame(rows)


def null_summary(nulls, proxies=PROXIES):
    """Robust width (1.4826 x MAD) and 95th percentile of |half change|, per proxy."""
    out = {}
    for name in proxies:
        x = nulls[f"halfchange_{name}"].dropna() if len(nulls) else pd.Series(dtype=float)
        out[f"null_n_{name}"] = int(len(x))
        out[f"null_sigma_{name}"] = float(1.4826 * (x - x.median()).abs().median()) if len(x) else np.nan
        out[f"null_p95_{name}"] = float(x.abs().quantile(0.95)) if len(x) else np.nan
    return out


def emergence_change(h, t0, bg_start, track=config.TRACK_DAYS, late_start=2.0, null=None, proxies=PROXIES):
    """
    Proxy change from the pre-emergence background ``[bg_start, t0]`` to the late window
    ``[t0 + late_start, t0 + track]``: ``fipchange_<proxy>`` (late/pre - 1; negative = more
    FIP-enhanced) and, given `null_summary` output, ``fipsig_<proxy>`` = change / null sigma.
    """
    p = proxy_table(h, proxies)
    t0 = pd.Timestamp(t0)
    out = {}
    for name in proxies:
        pre = p.loc[pd.Timestamp(bg_start):t0, name].dropna()
        late = p.loc[t0 + late_start * DAY:t0 + track * DAY, name].dropna()
        change = float(late.median() / pre.median() - 1) if len(pre) >= 6 and len(late) >= 6 else np.nan
        out[f"fipchange_{name}"] = change
        sigma = null.get(f"null_sigma_{name}", np.nan) if null else np.nan
        out[f"fipsig_{name}"] = change / sigma if np.isfinite(change) and sigma and np.isfinite(sigma) else np.nan
    return out
