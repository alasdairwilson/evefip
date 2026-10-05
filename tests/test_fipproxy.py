import numpy as np
import pandas as pd

from evefip.search import fipproxy

T0 = pd.Timestamp("2010-06-10")


def _table(fip_drop=0.0, megsb_every=1, seed=0):
    """Hot lines flat; S XIV/Fe XVI drops by `fip_drop` linearly over the 4 days after T0."""
    idx = pd.date_range(T0 - pd.Timedelta(days=8), T0 + pd.Timedelta(days=10), freq="h")
    x = np.asarray((idx - T0) / pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    n = lambda: 1 + 0.01 * rng.standard_normal(len(x))
    fe16 = 1.0 * n()
    s14 = 0.05 * (1 - fip_drop * np.clip(x / 4, 0, 1)) * n()
    return pd.DataFrame({"Fe XVI 33.5": fe16, "Fe XV 28.4": 2.0 * n(), "Si XII 49.9": 0.5 * n(),
                         "S XIV 44.6": s14, "n_megsa": 60,
                         "n_megsb": np.where(np.arange(len(x)) % megsb_every == 0, 60, 0)}, index=idx)


def test_proxy_masks_sparse_megsb():
    p = fipproxy.proxy_table(_table(megsb_every=3))
    assert np.isclose(p["S XIV/Fe XVI"].notna().mean(), 1 / 3, atol=0.01)


def test_null_and_emergence_significance():
    quiet = _table()
    nulls = fipproxy.null_windows(quiet.loc[:T0])
    assert len(nulls) > 0
    null = fipproxy.null_summary(nulls)
    assert null["null_sigma_S XIV/Fe XVI"] < 0.01
    out = fipproxy.emergence_change(_table(fip_drop=0.3), T0, T0 - pd.Timedelta(days=3), null=null)
    assert np.isclose(out["fipchange_S XIV/Fe XVI"], -0.3, atol=0.05)
    assert out["fipsig_S XIV/Fe XVI"] < -20
    flat = fipproxy.emergence_change(quiet, T0, T0 - pd.Timedelta(days=3), null=null)
    assert abs(flat["fipsig_S XIV/Fe XVI"]) < 3
