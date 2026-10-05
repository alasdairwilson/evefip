"""Search window, thresholds and cache locations."""

import os
from pathlib import Path

import pandas as pd

# EVE science data start, and the MEGS-A CCD power anomaly that ended MEGS-A.
EVE_START = pd.Timestamp("2010-05-01")
MEGSA_END = pd.Timestamp("2014-05-26")

CACHE_DIR = Path(os.environ.get("EVEFIP_CACHE", Path(__file__).resolve().parents[1] / "data"))

# Mean synodic rotation rate of sunspot latitudes (deg/day).
ROTATION_RATE = 13.2

# Emergence: first sighting this far inside the east limb (Stonyhurst lon > -LIMB_MARGIN),
# so the region was not simply rotated into view.
EAST_LIMB_LON = -75.0
# ...and with at least this many days on disk before it reaches the west limb. Short is fine if
# the FIP bias builds up quickly; western emergers also set soon, giving a second background.
MIN_DAYS_ON_DISK = 3.0
# Hot loops stay visible for about a day after their footpoints cross the west limb.
LIMB_LOOP_DAYS = 1.0

# Background window before emergence, and tracking window after it (days).
BACKGROUND_DAYS = 2.0
TRACK_DAYS = 10.0

# SHARP unsigned flux at which we call the region "emerged" (Mx).
T0_FLUX = 1e21

# Representative EVE lines for lightcurves / signal checks, keyed by short label.
# Values are (ion, wavelength nm) and are matched to the nearest EVL line.
SIGNAL_LINES = {
    "Fe XVI 33.5": ("Fe XVI", 33.54),
    "Fe XV 28.4": ("Fe XV", 28.42),
    "Mg IX 36.8": ("Mg IX", 36.81),
    "Si XII 49.9": ("Si XII", 49.94),
    "S XIV 44.6": ("S XIV", 44.57),
    "Ne VIII 77.0": ("Ne VIII", 77.04),
    "O V 63.0": ("O V", 62.97),
}
