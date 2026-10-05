import numpy as np
import pandas as pd

from evefip.search import eve, goes, sharps
from evefip.search.score import composite_score

BASE = {"bg_others_mean": 0, "track_others_mean": 0, "bg_other_flux_frac": 0.1,
        "track_other_flux_frac": 0.1, "bg_n_flares_C": 0, "track_n_flares_C": 0,
        "emerged_on_disk": True, "megsa_hours_frac": 1.0, "megsb_hours_frac": 1.0,
        "bg_days": 3.0, "bg_max_slope": 0.01, "hot_rise_sig": 8.0, "rho_target": 0.9}


def test_score_monotonic():
    s0 = composite_score(BASE)
    assert 0 < s0 <= 1
    for key, worse in [("track_other_flux_frac", 2.0), ("track_n_flares_C", 5),
                       ("emerged_on_disk", False), ("megsa_hours_frac", 0.5),
                       ("megsb_hours_frac", 0.2), ("bg_days", 1.0), ("bg_max_slope", 0.04),
                       ("hot_rise_sig", 2.0), ("rho_target", 0.3)]:
        assert composite_score({**BASE, key: worse}) < s0, key


def test_score_falls_back_to_srs_counts():
    row = {**BASE, "bg_other_flux_frac": np.nan, "track_other_flux_frac": np.nan}
    crowded = {**row, "track_others_mean": 3}
    assert composite_score(crowded) < composite_score(row)


def test_score_zero_without_flat_background():
    assert composite_score({**BASE, "bg_days": 0.0}) == 0
    assert composite_score({**BASE, "rho_target": -0.5}) == 0


def test_goes_class_roundtrip():
    assert np.isclose(goes.class_to_flux("M2.3"), 2.3e-5)
    assert goes.flux_to_class(2.3e-5) == "M2.3"
    assert np.isnan(goes.class_to_flux(""))


def test_channel_for():
    assert eve.channel_for(13.1) == "MEGSA1"
    assert eve.channel_for(17.1) == "MEGSA2"
    assert eve.channel_for(36.8) == "MEGSA2"
    assert eve.channel_for(77.0) == "MEGSB"


def test_coverage_from_hourly():
    t0 = pd.Timestamp("2010-06-03")
    idx = pd.date_range(t0 - pd.Timedelta(days=2), t0 + pd.Timedelta(days=10), freq="h")
    h = pd.DataFrame({"n_megsa": 60, "n_megsb": 5}, index=idx)
    cov = eve.coverage_metrics(h, t0)
    assert cov["megsa_hours_frac"] == 1.0 and cov["megsb_hours_frac"] == 1.0
    assert np.isclose(cov["megsb_min_per_day"], 5 * 24)
    # missing hours count as no coverage
    cov = eve.coverage_metrics(h[h.index < t0 + pd.Timedelta(days=5)], t0)
    assert np.isclose(cov["megsa_hours_frac"], 0.5)


def test_emergence_summary():
    t = pd.date_range("2010-06-01", periods=48, freq="h")
    flux = np.r_[np.full(10, 1e20), np.geomspace(1e20, 5e21, 38)]
    s = pd.DataFrame({"time": t, "usflux": flux, "lon": np.linspace(-40, -15, 48)})
    out = sharps.emergence_summary(s)
    assert out["emerged_on_disk"]
    assert out["harp_start"] == t[0]
    assert out["t0_sharp"] == t[np.argmax(flux >= 1e21)]
    rotated = s.assign(usflux=5e21, lon=-80.0)
    assert not sharps.emergence_summary(rotated)["emerged_on_disk"]
