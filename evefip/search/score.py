"""
Combine SRS, EVE, SHARP and GOES metrics into a ranked candidate table.

Stage 1 covers every SRS emergence and runs offline once the hourly EVE cache exists. It adds
SRS isolation, the longest flat EVE background before ``t_ref``, and the trend-aware EVE rise
after it. ``t_ref`` is two days before the region's first SRS entry, because numbering lags
emergence, and a background that runs into the region's own early brightening looks like a
trend. Events with a flat background (``bg_days >= 1``) and a significant hot-line rise pass
to stage 2. Stage 2 needs JSOC and HEK: it refines t0 from SHARPs, adds flares, and recomputes
the EVE metrics at the refined t0, plus whether the EVE rise tracks this region's flux.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evefip import config
from evefip.search import background, eve, fipproxy, goes, sharps, srs

log = logging.getLogger(__name__)

DAY = pd.Timedelta(days=1)
MIN_BG_DAYS = 1.0
MIN_RISE_SIG = 3.0


def eve_metrics(h, t0, t_set=pd.NaT, null=None):
    """
    EVE metrics at `t0` from the hourly table: the pre-emergence flat background, the
    trend-aware rise, and coverage. If the region sets behind the west limb at `t_set`, the rise
    and coverage windows stop there. We also look for a flat post-setting background starting
    `config.LIMB_LOOP_DAYS` later and, if both backgrounds are flat (``bracketed``), compare them.
    The FIP-proxy change after emergence is compared with `null` (`fipproxy.null_summary`).
    """
    t0 = pd.Timestamp(t0)
    track = config.TRACK_DAYS
    if pd.notna(t_set):
        track = min(track, max((pd.Timestamp(t_set) - t0) / DAY, 2.5))
    out = background.flat_background(h, t0)
    out.update(background.rise_metrics(h, t0, out["bg_start"], after=track))
    out.update(eve.coverage_metrics(h, t0, after=track))
    out.update(fipproxy.emergence_change(h, t0, out["bg_start"], track=track, null=null))
    out["track_days"] = track
    out["t_set"] = t_set
    out["bracketed"] = False
    if pd.notna(t_set):
        post = background.flat_background(h, pd.Timestamp(t_set) + config.LIMB_LOOP_DAYS * DAY,
                                          forward=True, prefix="post")
        out.update(post)
        if out["bg_days"] > 0 and post["post_days"] > 0:
            out["bracketed"] = True
            out.update(background.bracket_metrics(h, out["bg_start"], t0, post["post_start"], post["post_end"]))
    out["bg_max_slope"] = max(abs(out[f"bg_slope_{line}"]) for line in background.HOT_LINES)
    out["hot_rise_sig"] = np.nanmax([out[f"rise_sig_{line}"] for line in background.HOT_LINES] + [-np.inf])
    return out


def stage1(regions, h, start=config.EVE_START, end=config.MEGSA_END, null=None):
    """Every on-disk SRS emergence in the EVE window with SRS and EVE metrics at ``t_ref``."""
    em = srs.find_emergences(regions)
    # SRS positions are valid at 00 UT, so the region was first seen some time the day before.
    em["t0_srs"] = em["first_date"] - DAY
    em["t_ref"] = em["first_date"] - 2 * DAY
    lo = pd.Timestamp(start) + MIN_BG_DAYS * DAY
    hi = pd.Timestamp(end) - config.TRACK_DAYS * DAY
    em = em[(em["t0_srs"] >= lo) & (em["t0_srs"] <= hi)].reset_index(drop=True)
    rows = []
    for r in em.itertuples():
        row = srs.isolation_metrics(regions, r.number, r.t_ref)
        row.update(eve_metrics(h, r.t_ref, background.set_time(r.last_date, r.last_lon), null))
        rows.append(row)
    em = pd.concat([em, pd.DataFrame(rows)], axis=1)
    em["passes_eve"] = (em["bg_days"] >= MIN_BG_DAYS) & (em["hot_rise_sig"] > MIN_RISE_SIG)
    return em.sort_values(["passes_eve", "hot_rise_sig"], ascending=False).reset_index(drop=True)


def composite_score(row):
    """
    Product of six factors in [0, 1], higher is better:

    * isolation: ``exp(-other_flux_frac)`` from SHARPs (falls back to SRS region counts);
    * quiet: ``exp(-n_C_flares / 10)`` over background + tracking windows;
    * emergence: 1 if SHARPs show the flux growing on disk, 0.1 otherwise (a region rotating
      over the east limb also shows a "flux rise", from projection);
    * coverage: MEGS-A hourly coverage x (0.5 + 0.5 x MEGS-B hourly coverage);
    * background: ``min(1, bg_days / 2) x exp(-max|bg_slope| / 0.05)``, so it is 0 with no flat
      background;
    * signal: ``clip(hot_rise_sig / 5, 0, 1) x max(rho_target, 0)``, i.e. the EVE rise must be
      significant beyond the background trend *and* track the region's flux (0.5 stands in for
      rho when there is no SHARP).
    """
    fracs = [row.get("bg_other_flux_frac", np.nan), row.get("track_other_flux_frac", np.nan)]
    fracs = [f for f in fracs if np.isfinite(f)]
    if fracs:
        isolation = np.exp(-np.mean(fracs))
    else:
        isolation = 1.0 / (1.0 + row["bg_others_mean"] + row["track_others_mean"])
    quiet = np.exp(-(row.get("bg_n_flares_C", 0) + row.get("track_n_flares_C", 0)) / 10.0)
    emergence = 1.0 if row.get("emerged_on_disk", False) else 0.1
    coverage = row.get("megsa_hours_frac", 0.0) * (0.5 + 0.5 * row.get("megsb_hours_frac", 0.0))
    slope = row.get("bg_max_slope", np.nan)
    bg = min(1.0, row.get("bg_days", 0.0) / 2.0) * (np.exp(-slope / 0.05) if np.isfinite(slope) else 0.0)
    sig = row.get("hot_rise_sig", np.nan)
    rho = row.get("rho_target", np.nan)
    signal = (np.clip(sig / 5.0, 0, 1) if np.isfinite(sig) else 0.0) * (max(rho, 0.0) if np.isfinite(rho) else 0.5)
    return float(isolation * quiet * emergence * coverage * bg * signal)


def stage2(candidate, regions, h, harp_map, outdir=None, null=None):
    """
    Full metrics for one stage-1 candidate (a row of the stage-1 table, as a dict).

    If `outdir` is given, the per-candidate time series used for plotting are saved there.
    """
    row = dict(candidate)
    number = int(row["number"])
    t0 = pd.Timestamp(row["t0_srs"])
    win = (t0 - 5 * DAY, t0 + (config.TRACK_DAYS + 1) * DAY)

    harps = sharps.harps_for_noaa(number, harp_map)
    row["harpnum"] = ",".join(map(str, harps))
    target = pd.concat([sharps.harp_series(hp, *win) for hp in harps]) if harps else pd.DataFrame()
    if len(target):
        target = target.sort_values("time")
        row.update(sharps.emergence_summary(target[target["harpnum"] == harps[0]]))
    else:
        row.update(sharps.emergence_summary(pd.DataFrame(columns=["usflux", "lon"])))
    # The background must end before the region contributes, so use the HARP's first detection
    # when SHARPs show the flux growing on disk, else the T0_FLUX crossing, else t_ref.
    t0 = pd.Timestamp(row["t_ref"])
    for key, use in [("harp_start", row["emerged_on_disk"]), ("t0_sharp", True)]:
        if use and pd.notna(row[key]) and abs(row[key] - pd.Timestamp(row["t0_srs"])) < 4 * DAY:
            t0 = pd.Timestamp(row[key])
            break
    row["t0"] = t0

    row.update(eve_metrics(h, t0, row["t_set"], null))
    before = max(row["bg_days"], MIN_BG_DAYS)
    row.update(srs.isolation_metrics(regions, number, t0, before=before))

    others = sharps.all_harps(t0 - before * DAY, t0 + config.TRACK_DAYS * DAY, step="3h")
    others = others[~others["harpnum"].isin(harps)]
    if len(target):
        track = target[(target["time"] >= t0) & (target["time"] <= t0 + config.TRACK_DAYS * DAY)]
        target_flux = target.groupby("time")["usflux"].sum()
        row["target_flux_max"] = float(track.groupby("time")["usflux"].sum().max())
        row.update(sharps.other_flux_fraction(others, harps, t0, row["target_flux_max"], before=before))
        row.update(background.attribution(h, t0, target_flux, others.groupby("time")["usflux"].sum()))
    else:
        row["rho_target"] = row["rho_others"] = np.nan

    flare_table = goes.flares(t0 - before * DAY, t0 + config.TRACK_DAYS * DAY)
    row.update(goes.flare_metrics(flare_table, number, t0, before=before))

    if null:
        row.update(null)
    row["score"] = composite_score(row)
    row["strictly_isolated"] = bool(row["bg_others_max"] == 0 and row["track_others_max"] == 0)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        target.to_parquet(outdir / "target_sharp.parquet")
        others.to_parquet(outdir / "all_sharps.parquet")
        flare_table.to_parquet(outdir / "flares.parquet")
        end = t0 + config.TRACK_DAYS * DAY
        if row.get("post_days", 0) > 0:
            end = max(end, pd.Timestamp(row["post_end"]))
        h.loc[t0 - 5 * DAY:end].to_parquet(outdir / "eve_hourly.parquet")
    return row
