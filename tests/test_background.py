import numpy as np
import pandas as pd

from evefip.search import background

T0 = pd.Timestamp("2010-06-10")
LINES = ("A", "B")


def _hourly(profile, noise=0.005, seed=0):
    """Hourly table for lines A and B, value = profile(days from T0) x (1 + noise)."""
    idx = pd.date_range(T0 - pd.Timedelta(days=6), T0 + pd.Timedelta(days=10), freq="h")
    x = np.asarray((idx - T0) / pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: profile(x) * (1 + noise * rng.standard_normal(len(x))) for k in LINES}, index=idx)


def test_flat_background_takes_longest_window():
    h = _hourly(lambda x: np.ones_like(x))
    bg = background.flat_background(h, T0, LINES)
    assert bg["bg_days"] == 4.0 and bg["bg_start"] == T0 - pd.Timedelta(days=4)
    assert abs(bg["bg_slope_A"]) < 0.01


def test_falling_background_fails_and_reports_slope():
    h = _hourly(lambda x: np.exp(-0.15 * x))  # ~ -15 %/day
    bg = background.flat_background(h, T0, LINES)
    assert bg["bg_days"] == 0
    assert bg["bg_slope_A"] < -0.05


def test_falling_then_flat_picks_flat_tail():
    # The 11076 shape: falling until T0 - 1.5 d, flat afterwards.
    h = _hourly(lambda x: np.where(x < -1.5, 1 + 0.15 * (-1.5 - x), 1.0))
    bg = background.flat_background(h, T0, LINES)
    assert 1.0 <= bg["bg_days"] <= 2.0


def test_rise_beyond_trend_is_significant_but_trend_continuation_is_not():
    step = _hourly(lambda x: np.where(x < 0, 1.0, 1.3))
    bg = background.flat_background(step, T0, LINES)
    r = background.rise_metrics(step, T0, bg["bg_start"], LINES)
    assert r["rise_sig_A"] > 10 and np.isclose(r["frac_A"], 0.3, atol=0.02)
    # A steady 4 %/day rise passes as "flat", but continuing it is no real rise.
    ramp = _hourly(lambda x: 1 + 0.04 * x)
    bg = background.flat_background(ramp, T0, LINES)
    assert bg["bg_days"] > 0
    r = background.rise_metrics(ramp, T0, bg["bg_start"], LINES)
    assert r["rise_sig_A"] < 1.5


def test_attribution_tracks_target():
    h = _hourly(lambda x: 1 + np.clip(x, 0, None) * 0.1)
    idx = pd.date_range(T0, T0 + pd.Timedelta(days=5), freq="3h")
    days = np.asarray((idx - T0) / pd.Timedelta(days=1))
    target = pd.Series(1e21 * (1 + days), index=idx)
    others = pd.Series(1e22 * (1 + np.sin(3 * days)), index=idx)
    out = background.attribution(h, T0, target, others, line="A")
    assert out["rho_target"] > 0.9
    assert abs(out["rho_others"]) < 0.5


def test_find_eve_rises_and_match():
    h = _hourly(lambda x: np.where(x < 0, 1.0, 1 + 0.2 * np.clip(x, 0, 1)))
    rises = background.find_eve_rises(h, LINES)
    assert len(rises) == 1
    assert abs((rises.loc[0, "onset"] - T0) / pd.Timedelta(days=1)) <= 1.0
    em = pd.DataFrame({"number": [11111, 12222], "t0_srs": [T0 + pd.Timedelta(hours=12), T0 + pd.Timedelta(days=9)]})
    matched = background.match_emergences(rises, em)
    assert matched.loc[0, "noaa"] == 11111


def test_post_setting_background_and_bracketing():
    # Region adds 30% between T0 and T0 + 3 d (it sets then); the rest of the Sun is flat
    # but S XIV/Si XII-like ratio in the background shifts by 10% if `shift`.
    def table(shift):
        h = _hourly(lambda x: np.where((x >= 0) & (x < 3), 1.3, 1.0))
        h["S XIV 44.6"] = h["A"] * np.where(h.index > T0 + pd.Timedelta(days=3), 1 + shift, 1.0)
        h["Si XII 49.9"] = h["B"]
        h["Fe XVI 33.5"], h["Fe XV 28.4"] = h["A"], h["B"]
        return h
    h = table(0.0)
    post = background.flat_background(h, T0 + pd.Timedelta(days=4), LINES, forward=True, prefix="post")
    assert post["post_days"] == 4.0 and post["post_start"] == T0 + pd.Timedelta(days=4)
    b = background.bracket_metrics(h, T0 - pd.Timedelta(days=2), T0, post["post_start"], post["post_end"],
                                   lines=LINES)
    assert abs(b["bgchange_A"]) < 0.01 and abs(b["ratiochange_S XIV/Si XII"]) < 0.01
    b = background.bracket_metrics(table(0.1), T0 - pd.Timedelta(days=2), T0, post["post_start"],
                                   post["post_end"], lines=LINES)
    assert np.isclose(b["ratiochange_S XIV/Si XII"], 0.1, atol=0.01)


def test_set_time():
    t = background.set_time(pd.Timestamp("2010-06-10"), 77.0)
    assert abs(t - pd.Timestamp("2010-06-11")) < pd.Timedelta(minutes=30)
    assert pd.isna(background.set_time(pd.Timestamp("2010-06-10"), 20.0))  # decayed on disk


def test_attribution_finds_lag():
    # EUV follows the flux one day late; flux saturates after 1.5 d.
    idx = pd.date_range(T0 - pd.Timedelta(days=2), T0 + pd.Timedelta(days=6), freq="3h")
    days = np.asarray((idx - T0) / pd.Timedelta(days=1))
    flux = pd.Series(1e21 * (1 + np.clip(days, 0, 1.5)), index=idx)
    h = _hourly(lambda x: 1 + 0.1 * np.clip(x - 1, 0, 1.5))
    out = background.attribution(h, T0, flux, pd.Series(1e21 * (2 + np.sin(5 * days)), index=idx), line="A")
    assert out["rho_lag_h"] == 24 and out["rho_target"] > 0.9
