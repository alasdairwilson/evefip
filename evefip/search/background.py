"""
Is the full-disk EVE background stable, and does the new region produce a visible rise?

Before we can extract an AR DEM from Sun-as-a-star spectra, the hot lines have to be flat
before emergence (no other region decaying, leaving or brightening). Then they have to rise
by more than any pre-existing trend could explain, and track the new region's flux.
All functions take the hourly line table from `evefip.search.eve.load_hourly`.
"""

import numpy as np
import pandas as pd

from evefip import config

DAY = pd.Timedelta(days=1)
HOT_LINES = ("Fe XVI 33.5", "Fe XV 28.4")


def trend(series, t0):
    """
    Linear fit to `series` against time in days from `t0`.

    Returns ``(slope, scatter, baseline)``: the slope as a fraction of the median per day, the
    residual scatter (1.4826 x MAD) as a fraction of the median, and the median itself.
    """
    s = series.dropna()
    if len(s) < 6:
        return np.nan, np.nan, np.nan
    x = np.asarray((s.index - t0) / DAY, dtype=float)
    y = s.to_numpy(dtype=float)
    p = np.polyfit(x, y, 1)
    base = np.median(y)
    scatter = 1.4826 * np.median(np.abs(y - np.polyval(p, x)))
    return p[0] / base, scatter / base, base


def flat_background(h, t0, lines=HOT_LINES, max_days=4.0, min_days=1.0, step_days=0.25,
                    max_slope=0.05, max_change=0.06, max_scatter=0.04, min_coverage=0.5,
                    forward=False, prefix="bg"):
    """
    The longest window ``[t0 - d, t0]`` (``min_days <= d <= max_days``) where every line in
    `lines` is flat: ``|slope| < max_slope`` per day, total drift ``|slope| x d < max_change``
    (a per-day cut alone lets a long window absorb a short steep fall), and
    ``scatter < max_scatter``, with at least `min_coverage` of hours present.

    With ``forward=True`` the windows are ``[t0, t0 + d]`` instead (the background after the
    region has set behind the west limb).

    Returns ``<prefix>_start``, ``<prefix>_end``, ``<prefix>_days`` (0 if no window passes) and
    ``<prefix>_slope_<line>`` / ``<prefix>_scatter_<line>``. These come from the chosen window,
    or from the ``min_days`` window when nothing passes, so failures show why they failed.
    """
    t0 = pd.Timestamp(t0)
    result = None
    for d in np.arange(max_days, min_days - 1e-9, -step_days):
        start, end = (t0, t0 + d * DAY) if forward else (t0 - d * DAY, t0)
        w = h.loc[start:end - pd.Timedelta(minutes=1)]
        out = {f"{prefix}_start": start, f"{prefix}_end": end, f"{prefix}_days": float(d)}
        ok = True
        for line in lines:
            slope, scatter, _ = trend(w[line], t0) if line in w else (np.nan, np.nan, np.nan)
            out[f"{prefix}_slope_{line}"], out[f"{prefix}_scatter_{line}"] = slope, scatter
            covered = w[line].notna().sum() / (d * 24) if line in w else 0
            ok &= bool(np.isfinite(slope) and abs(slope) < max_slope and abs(slope) * d < max_change
                       and scatter < max_scatter and covered >= min_coverage)
        if ok:
            return out
        result = out
    result[f"{prefix}_days"] = 0.0
    return result


RATIOS = {"S XIV/Si XII": ("S XIV 44.6", "Si XII 49.9"), "Fe XVI/Fe XV": ("Fe XVI 33.5", "Fe XV 28.4")}


def bracket_metrics(h, pre_start, pre_end, post_start, post_end, lines=tuple(config.SIGNAL_LINES),
                    ratios=RATIOS):
    """
    Compare the background before emergence with the background after the region has set.

    ``bgchange_<line>`` is post/pre - 1 of the median irradiance: how much the rest of the Sun
    drifted during the transit. ``ratiochange_<ratio>`` does the same for line ratios. The
    S XIV/Si XII pair (high/low FIP, both ~2.5 MK) tests whether the background's *composition*
    stayed put, which background subtraction assumes.
    """
    pre = h.loc[pd.Timestamp(pre_start):pd.Timestamp(pre_end)].median()
    post = h.loc[pd.Timestamp(post_start):pd.Timestamp(post_end)].median()
    out = {f"bgchange_{line}": float(post[line] / pre[line] - 1) if line in pre else np.nan for line in lines}
    for name, (a, b) in ratios.items():
        out[f"ratiochange_{name}"] = float((post[a] / post[b]) / (pre[a] / pre[b]) - 1)
    return out


def set_time(last_date, last_lon, rate=config.ROTATION_RATE, min_lon=60.0):
    """
    When the region crosses the west limb, extrapolated from its last SRS position. NaT if it was
    last seen east of `min_lon`, i.e. it decayed on the disk rather than setting.
    """
    if not np.isfinite(last_lon) or last_lon < min_lon:
        return pd.NaT
    return pd.Timestamp(last_date) + (90.0 - last_lon) / rate * DAY


def rise_metrics(h, t0, bg_start, lines=tuple(config.SIGNAL_LINES), late_start=2.0,
                 after=config.TRACK_DAYS):
    """
    Rise of each line above the background, allowing for the background's own trend.

    Baseline = median over ``[bg_start, t0]``. Noise = residual scatter about the background
    trend. The drift allowance is ``|slope| x baseline x dt``, where dt runs from t0 to the middle
    of the late window ``[t0 + late_start, t0 + after]``. Returns ``rise_sig_<line> = rise /
    hypot(noise, drift)`` and ``frac_<line> = rise / baseline``.
    """
    t0 = pd.Timestamp(t0)
    bg = h.loc[pd.Timestamp(bg_start):t0 - pd.Timedelta(minutes=1)]
    late = h.loc[t0 + late_start * DAY:t0 + after * DAY]
    dt = (late_start + after) / 2
    out = {}
    for line in lines:
        slope, scatter, base = trend(bg[line], t0) if line in bg else (np.nan, np.nan, np.nan)
        l = late[line].dropna() if line in late else pd.Series(dtype=float)
        if not np.isfinite(base) or len(l) < 6:
            out[f"rise_sig_{line}"] = out[f"frac_{line}"] = np.nan
            continue
        rise = l.median() - base
        err = np.hypot(scatter * base, abs(slope) * base * dt)
        out[f"rise_sig_{line}"] = float(rise / err) if err > 0 else np.nan
        out[f"frac_{line}"] = float(rise / base)
    return out


def _on_index(flux, index):
    """Interpolate a time-indexed flux series onto `index` (NaN outside its range)."""
    flux = flux.dropna().groupby(level=0).sum().sort_index()
    if len(flux) < 2:
        return pd.Series(np.nan, index=index)
    both = flux.reindex(flux.index.union(index)).interpolate(method="time", limit_area="inside")
    return both.reindex(index)


def attribution(h, t0, target_flux, other_flux, line="Fe XVI 33.5", days=5.0, lags_h=(0, 12, 24, 36, 48)):
    """
    Spearman correlation of the line's hourly irradiance over ``[t0, t0 + days]`` with the target
    region's unsigned flux, allowing the EUV to lag the flux by each of `lags_h` hours. Coronal
    loops fill and heat after the flux emerges: 11072's flux plateaus about a day before its
    Fe XVI does, giving rho 0.59 at zero lag and 0.84 at 24 h.

    Returns the best ``rho_target``, its ``rho_lag_h``, and ``rho_others`` (all other regions'
    summed flux) at that same lag. Both fluxes are time-indexed Series.
    """
    t0 = pd.Timestamp(t0)
    y = h.loc[t0:t0 + days * DAY, line].dropna() if line in h else pd.Series(dtype=float)

    def rho(flux, lag):
        x = _on_index(flux.shift(freq=pd.Timedelta(hours=lag)), y.index) if len(y) else pd.Series(dtype=float)
        ok = x.notna() & y.notna()
        # Spearman = Pearson on ranks (pandas' method="spearman" needs scipy)
        return float(y[ok].rank().corr(x[ok].rank())) if ok.sum() >= 12 else np.nan

    best = {"rho_target": np.nan, "rho_lag_h": np.nan, "rho_others": np.nan}
    for lag in lags_h:
        r = rho(target_flux, lag)
        if np.isfinite(r) and not (r <= best["rho_target"]):
            best = {"rho_target": r, "rho_lag_h": lag, "rho_others": rho(other_flux, lag)}
    return best


def find_eve_rises(h, lines=HOT_LINES, step="6h", min_bg_days=1.5, look_days=3.0, min_sig=5.0,
                   merge_days=1.0):
    """
    Scan the hourly table for onsets: a flat background of at least `min_bg_days` followed by a
    rise with ``rise_sig > min_sig`` in every line over the next `look_days`. The late window
    starts 1 day after onset. Consecutive flags within `merge_days` merge into one event. The
    onset is the *last* flag in the group: earlier flags fire once the late window reaches the
    rise, whereas the last one is where the flat background runs right up to it.
    """
    times = pd.date_range(h.index.min().ceil("D") + 4 * DAY, h.index.max() - look_days * DAY, freq=step)
    flags = []
    for t in times:
        bg = flat_background(h, t, lines, max_days=min_bg_days, min_days=min_bg_days)
        if bg["bg_days"] == 0:
            continue
        rise = rise_metrics(h, t, bg["bg_start"], lines, late_start=1.0, after=look_days)
        sig = [rise[f"rise_sig_{line}"] for line in lines]
        if all(np.isfinite(s) and s > min_sig for s in sig):
            flags.append({"onset": t, "rise_sig": min(sig),
                          **{f"frac_{line}": rise[f"frac_{line}"] for line in lines}})
    if not flags:
        return pd.DataFrame(columns=["onset", "rise_sig", "n_flags"])
    f = pd.DataFrame(flags)
    group = (f["onset"].diff() > merge_days * DAY).cumsum()
    agg = {"onset": "last", "rise_sig": "max", **{c: "max" for c in f.columns if c.startswith("frac_")}}
    events = f.groupby(group).agg(agg)
    events["n_flags"] = f.groupby(group).size()
    return events.reset_index(drop=True)


def match_emergences(rises, emergences, window_days=2.0):
    """Attach the nearest stage-1 emergence (by ``t0_srs``) within ±`window_days` of each onset."""
    rises = rises.copy()
    t = pd.to_datetime(emergences["t0_srs"])
    nums, dts = [], []
    for onset in rises["onset"]:
        d = (t - onset).abs()
        i = d.idxmin() if len(d) else None
        if i is not None and d[i] <= window_days * DAY:
            nums.append(int(emergences.loc[i, "number"]))
            dts.append((t[i] - onset) / DAY)
        else:
            nums.append(pd.NA)
            dts.append(np.nan)
    rises["noaa"] = pd.array(nums, dtype="Int64")
    rises["t0_srs_minus_onset_days"] = dts
    return rises
