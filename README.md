# evefip

Track how the FIP bias of a newly emerging active region evolves, using SDO/EVE Sun-as-a-star spectra.

The plan:

1. Find a near-empty disk where a region emerges on the east half, so it can be watched for as long as possible. It is possible we won't get many empty disks so maybe just the emergent active region is dominant.
2. Take the pre-emergence EVE spectrum as the background.
3. Fit the background-subtracted DEM in time bins with an abundance ratio free parameter:
   low-FIP (Fe, Si, Mg, etc) relative to high-FIP (O, Ne, S, Ar?).

## Phase 1: candidate search

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python scripts/build_eve_cache.py        # once: ~3.7 GB of L2B -> hourly line cache
.venv/bin/python scripts/find_candidates.py        # writes results/
.venv/bin/python -m pytest
```

The key requirement is that the full-disk hot lines (Fe XVI 33.5, Fe XV 28.4) are **flat before
emergence and then rise because of the new region**. If we can't see the rise, there is no AR DEM
to extract.

- **Stage 1** (all SRS emergences, offline):
  - Takes every region first seen more than 15° inside the east limb with at least 7 days left on
    the disk.
  - Finds the longest flat EVE background ending at `t0_srs` (up to 4 d). Flat means
    |slope| < 5%/day, total drift < 6% and scatter < 4%; see `background.flat_background`.
  - Measures the rise after t0 against both the noise and the drift that the background trend
    would produce (`background.rise_metrics`).
  - Events with at least 1 d of flat background and a hot-line `rise_sig > 3` pass.
- **Stage 2** (passing events) adds:
  - HMI SHARP flux, which refines t0 and gives the other regions' flux;
  - GOES flares from HEK;
  - the EVE metrics recomputed at the refined t0;
  - `rho_target`, the rank correlation of Fe XVI with this region's flux, as a check that the
    rise is really this region's.
- `eve_rises.csv` lists flat-then-rising onsets found in EVE alone, matched to SRS emergences
  within ±2 d.

Outputs:
- `results/stage1_emergences.csv`
- `results/eve_rises.csv`
- `results/candidates.csv`, ranked by `score` (see `evefip.search.score.composite_score`)
- `results/<NOAA>/summary.png`. Grey shading = chosen flat background; red hatching = no flat
  window; the thick lines show the background trend, dashed where extrapolated.

Downloads are cached in `data/` (override with `EVEFIP_CACHE`).

## Data notes

- MEGS-A ends on 2014-05-26.
- MEGS-B carries most high-FIP lines (S XIV 44.6, Ne VII 46.5, O V 63.0, Ne VIII 77.0). Its observing
  regimes:

  | Period | MEGS-B |
  |---|---|
  | 2010-05-01 → 05-18 | continuous |
  | 2010-05-19 → ~08-31 | ~15 min/day (effectively off) |
  | 2010-09 → 2012-11 | 3 h block + 5 min every hour |
  | 2012-12 → 2013-11 | 5 min every hour |
  | 2013-12 → 2014-05 | 3 h block only |

- In L2B, `LINE_IRRADIANCE` is filled only while MEGS-B observes, so we read per-channel
  `ChannelLinesData` and take each line from one fixed channel.
- `sunpy.io.special.srs.read_srs` fails on SRS files where a section reads "None". That affects
  about 25% of days in 2010–2014, so `evefip.search.srs` parses the SRS text itself.
