"""Per-candidate summary figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evefip import config
from evefip.search import background, fipproxy


def _windows(ax, t0, bg_start, flat):
    # Grey = the flat background chosen; hatched = the shortest window tried, when none was flat.
    kw = {"color": "0.85"} if flat else {"facecolor": "none", "edgecolor": "C3", "hatch": "//"}
    ax.axvspan(bg_start, t0, lw=0, zorder=0, **kw)
    ax.axvline(t0, color="k", lw=0.8, ls="--")


def summary_plot(row, regions, datadir, filename):
    """
    Five stacked panels: SHARP flux, other regions, GOES flares, EVE line changes, and the FIP
    proxy ratios (6 h medians, relative to the background; down = more FIP-enhanced).

    Red dashed = west-limb setting time; a second grey band = flat post-setting background.
    The EVE panel shows each line relative to its background median. The background trend of
    the hot lines is drawn solid over the background window and dashed beyond it (the drift
    that `rise_metrics` allows for).
    """
    datadir = Path(datadir)
    t0 = pd.Timestamp(row["t0"])
    bg_start = pd.Timestamp(row["bg_start"])
    flat = row["bg_days"] > 0
    lo = t0 - pd.Timedelta(days=5)
    hi = t0 + pd.Timedelta(days=config.TRACK_DAYS)
    post = row.get("post_days", 0) > 0
    if post:
        hi = max(hi, pd.Timestamp(row["post_end"]))
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True, constrained_layout=True)

    ax = axes[0]
    target = pd.read_parquet(datadir / "target_sharp.parquet")
    others = pd.read_parquet(datadir / "all_sharps.parquet")
    if len(target):
        for h, g in target.groupby("harpnum"):
            ax.plot(g["time"], g["usflux"], label=f"HARP {h} (target)")
    if len(others):
        tgt = set(target["harpnum"]) if len(target) else set()
        tot = others[~others["harpnum"].isin(tgt)].groupby("time")["usflux"].sum()
        ax.plot(tot.index, tot.values, color="0.5", label="all other HARPs")
    ax.axhline(config.T0_FLUX, color="C3", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("Unsigned flux [Mx]")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    w = regions[(regions["date"] >= lo.floor("D")) & (regions["date"] <= hi.ceil("D"))
                & (regions["number"] != row["number"])]
    days = pd.date_range(lo.floor("D"), hi.ceil("D"), freq="D")
    ax.step(days, w.groupby("date").size().reindex(days, fill_value=0), where="mid", label="other regions")
    ax2 = ax.twinx()
    ax2.step(days, w.groupby("date")["area"].sum().reindex(days, fill_value=0), where="mid",
             color="C1", label="other spot area")
    ax.set_ylabel("N other regions")
    ax2.set_ylabel("Area [MSH]", color="C1")

    ax = axes[2]
    flares = pd.read_parquet(datadir / "flares.parquet")
    if len(flares):
        target_fl = flares["noaa"] == row["number"]
        ax.vlines(flares["peak"], 1e-8, flares["flux"], color=np.where(target_fl, "C3", "0.4"))
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-3)
    for f in [1e-6, 1e-5, 1e-4]:
        ax.axhline(f, color="0.8", lw=0.5)
    ax.set_ylabel("GOES peak [W m$^{-2}$]\n(red = target)")

    ax = axes[3]
    eve_file = datadir / "eve_hourly.parquet"
    if eve_file.exists():
        h = pd.read_parquet(eve_file)[list(config.SIGNAL_LINES)]
        bgw = h.loc[bg_start:t0 - pd.Timedelta(minutes=1)]
        mid = bg_start + (t0 - bg_start) / 2
        for i, col in enumerate(h.columns):
            base = bgw[col].median()
            ax.plot(h.index, h[col] / base - 1, lw=0.8, color=f"C{i}", label=col)
            if col in background.HOT_LINES:
                slope, _, _ = background.trend(bgw[col], t0)
                for a, b, ls in [(bg_start, t0, "-"), (t0, hi, "--")]:
                    # zero at the background midpoint, where the median sits for a linear trend
                    x = np.array([(a - mid) / pd.Timedelta(days=1), (b - mid) / pd.Timedelta(days=1)])
                    ax.plot([a, b], slope * x, color=f"C{i}", lw=1.5, ls=ls, alpha=0.6)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.set_ylabel("EVE change vs\nbackground median")

    ax = axes[4]
    if eve_file.exists():
        p = fipproxy.proxy_table(pd.read_parquet(eve_file))
        for i, col in enumerate(p.columns):
            base = p.loc[bg_start:t0, col].median()
            r = (p[col] / base - 1).resample("6h").median()
            ax.plot(r.index, r.values, marker=".", lw=0.8, color=f"C{i}", label=col)
            sigma = row.get(f"null_sigma_{col}", np.nan)
            if np.isfinite(sigma) and i == 0:
                ax.axhspan(-2 * sigma, 2 * sigma, color=f"C{i}", alpha=0.12, lw=0,
                           label=f"quiet-Sun null ±2σ ({col})")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7, loc="lower left")
    ax.set_ylabel("FIP proxy change\n(down = FIP-enhanced)")

    for ax in axes:
        _windows(ax, t0, bg_start, flat)
        if pd.notna(row.get("t_set", pd.NaT)):
            ax.axvline(pd.Timestamp(row["t_set"]), color="C3", lw=0.8, ls="--")
        if post:
            ax.axvspan(pd.Timestamp(row["post_start"]), pd.Timestamp(row["post_end"]), color="0.85", lw=0, zorder=0)
    axes[-1].set_xlim(lo, hi)
    fig.suptitle(f"NOAA {row['number']}  HARP {row.get('harpnum', '')}  t0={t0:%Y-%m-%d %H:%M}  "
                 f"lon0={row['lon0']:.0f}  score={row.get('score', np.nan):.2f}\n"
                 f"flat bg {row['bg_days']:.2f} d  max slope {row['bg_max_slope']:+.3f}/d  "
                 f"hot rise_sig {row['hot_rise_sig']:.1f}  rho_target {row.get('rho_target', np.nan):.2f}  "
                 f"rho_others {row.get('rho_others', np.nan):.2f}"
                 + (f"\nbracketed: post bg {row['post_days']:.2f} d, Fe XVI change "
                    f"{row['bgchange_Fe XVI 33.5']:+.3f}, S XIV/Si XII change "
                    f"{row['ratiochange_S XIV/Si XII']:+.3f}" if row.get("bracketed") else ""))
    fig.savefig(filename, dpi=110)
    plt.close(fig)
