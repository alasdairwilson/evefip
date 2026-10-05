from pathlib import Path

import numpy as np
import pandas as pd

from evefip.search import srs

DATA = Path(__file__).parent / "data"


def test_read_srs_file():
    df = srs.read_srs_file(DATA / "20100601SRS.txt")
    assert list(df.columns) == srs.COLUMNS
    assert set(df["number"]) == {11073, 11074, 11075, 11076}
    assert (df["date"] == pd.Timestamp("2010-06-01")).all()
    r = df.set_index("number")
    # N12E09 -> east is negative longitude
    assert r.loc[11073, "lon"] == -9 and r.loc[11073, "lat"] == 12
    assert r.loc[11075, "lon"] == 32
    # plage (IA) rows have no spot area
    assert r.loc[11074, "id"] == "IA" and np.isnan(r.loc[11074, "area"])


def _track(number, start, lon0, days, lat=10.0):
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"date": dates, "number": number, "id": "I", "lat": lat,
                         "lon": lon0 + 13.2 * np.arange(days), "carr_lon": 0.0, "area": 50.0,
                         "n_spots": 3.0, "mag_type": "Beta"})


def test_find_emergences():
    regions = pd.concat([
        _track(1, "2010-06-01", -80, 12),   # rotated in at the east limb: rejected
        _track(2, "2010-06-03", -40, 9),    # emerged on disk: kept
        _track(3, "2010-06-05", 60, 3),     # emerged on disk, too few days left: rejected
    ])
    em = srs.find_emergences(regions)
    assert em["number"].tolist() == [2]
    assert em.loc[0, "lon0"] == -40
    assert np.isclose(em.loc[0, "days_to_west_limb"], 130 / 13.2)
    assert em.loc[0, "last_date"] == pd.Timestamp("2010-06-11")
    assert np.isclose(em.loc[0, "last_lon"], -40 + 8 * 13.2)


def test_isolation_metrics():
    regions = pd.concat([_track(2, "2010-06-03", -40, 9), _track(9, "2010-06-01", -30, 4)])
    m = srs.isolation_metrics(regions, 2, pd.Timestamp("2010-06-03"), before=2, after=5)
    # region 9 is on disk 1-4 June: all of the background window, part of the tracking window
    assert m["bg_others_max"] == 1 and m["bg_others_mean"] == 1
    assert m["track_others_max"] == 1 and 0 < m["track_others_mean"] < 1
    alone = srs.isolation_metrics(regions, 2, pd.Timestamp("2010-06-08"), before=2, after=2)
    assert alone["bg_others_max"] == 0 and alone["track_others_max"] == 0


def test_read_srs_file_with_empty_sections():
    # Sections IA and II read "None"; sunpy's read_srs fails on this layout.
    df = srs.read_srs_file(DATA / "20100929SRS.txt")
    assert df["number"].tolist() == [11108, 11109, 11110]
    assert (df["id"] == "I").all()
    assert df.set_index("number").loc[11108, "lon"] == 83
