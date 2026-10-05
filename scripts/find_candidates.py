"""
Find isolated, on-disk active-region emergences with a flat EVE background and a visible rise.

    python scripts/build_eve_cache.py            # once: hourly EVE line cache
    python scripts/find_candidates.py --max-detail 60

Writes to ``<out>/``:

* ``stage1_emergences.csv``: every SRS emergence, with SRS + EVE metrics at ``t_ref``;
* ``fip_null_windows.csv``: FIP-proxy stability in every quiet 4-day window (the noise floor);
* ``eve_rises.csv``: flat-then-rising onsets found in EVE alone, matched to emergences;
* ``candidates.csv``: full metrics for stage-1 events passing the EVE cut, ranked by score;
* ``<NOAA>/summary.png``: one figure per stage-2 candidate.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from evefip import config
from evefip.search import background, eve, fipproxy, plot, score, sharps, srs

log = logging.getLogger("find_candidates")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=str(config.EVE_START.date()))
    p.add_argument("--end", default=str(config.MEGSA_END.date()))
    p.add_argument("--max-detail", type=int, default=60, help="cap on stage-2 candidates")
    p.add_argument("--out", default="results")
    p.add_argument("--skip-rises", action="store_true", help="skip the EVE-only onset scan")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pad = pd.Timedelta(days=config.TRACK_DAYS + 5)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    regions = srs.load_regions(start - pad, end + pad)
    h = eve.load_hourly(start - pd.Timedelta(days=5), end + pad, download=False)
    log.info("%d region-days; %d EVE hours", len(regions), len(h))

    # Noise floor of the FIP proxy: how much it wanders over quiet 4-day windows.
    nulls = fipproxy.null_windows(h.loc[start:end])
    nulls.to_csv(out / "fip_null_windows.csv", index=False)
    null = fipproxy.null_summary(nulls)
    log.info("FIP-proxy null: %s", {k: round(v, 4) for k, v in null.items()})

    s1 = score.stage1(regions, h, start, end, null=null)
    s1.to_csv(out / "stage1_emergences.csv", index=False)
    passing = s1[s1["passes_eve"]]
    log.info("%d emergences, %d pass the EVE background/rise cut", len(s1), len(passing))

    if not args.skip_rises:
        rises = background.match_emergences(background.find_eve_rises(h), s1)
        rises.to_csv(out / "eve_rises.csv", index=False)
        log.info("%d EVE onsets, %d matched to an emergence", len(rises), rises["noaa"].notna().sum())

    harp_map = sharps.harp_noaa_map()
    rows = []
    for cand in passing.head(args.max_detail).to_dict("records"):
        log.info("Stage 2: NOAA %d (t0_srs %s)", cand["number"], cand["t0_srs"].date())
        cdir = out / str(cand["number"])
        try:
            row = score.stage2(cand, regions, h, harp_map, outdir=cdir, null=null)
            plot.summary_plot(row, regions, cdir, cdir / "summary.png")
        except Exception:
            log.exception("NOAA %d failed", cand["number"])
            continue
        rows.append(row)
    if rows:
        pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(out / "candidates.csv", index=False)
        log.info("Wrote %s", out / "candidates.csv")


if __name__ == "__main__":
    main()
