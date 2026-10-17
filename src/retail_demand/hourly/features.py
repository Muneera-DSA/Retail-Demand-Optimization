"""Week-ahead features. A capacity plan for next week is drawn up at the start of the week,
so every feature for hour t uses only data recorded up to t - 168 hours. Nothing more
recent is allowed: experiments.leakage() shows how much a shorter lag would flatter the
model if it were (wrongly) used.

All features are computed on the complete hourly grid, where unrecorded hours are NaN, so a
shift of k rows is exactly k hours."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import config

CALENDAR = ["hour", "dow", "how", "weekend", "month", "doy_sin", "doy_cos", "trend_weeks"]
HISTORY = [f"lag{k}" for k in config.HOURLY_LAGS] + ["lag8736", "how_mean_4w", "how_n_4w", "how_mean_all", "level_week", "gap_share_week"]
FEATURES = CALENDAR + HISTORY


def build(grid: pd.DataFrame, min_lag: int = config.HOURLY_MIN_LAG, extra_lags: tuple[int, ...] = ()) -> pd.DataFrame:
    g = grid.sort_values("ts").reset_index(drop=True)
    assert (g.ts.diff().dropna() == pd.Timedelta(hours=1)).all(), "grid must be hourly and complete"
    s = g.demand
    f = pd.DataFrame({"ts": g.ts, "date": g.date, "status": g.status, "demand": s})
    f["hour"], f["dow"], f["how"] = g.hour, g.dow, g.how
    f["weekend"] = (g.dow >= 5).astype(int)
    f["month"] = g.ts.dt.month
    doy = g.ts.dt.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["trend_weeks"] = (g.ts - g.ts.min()).dt.total_seconds() / (3600 * 168)
    for k in config.HOURLY_LAGS:
        assert k >= min_lag
        f[f"lag{k}"] = s.shift(k)
    f["lag8736"] = s.shift(8736)                                    # same hour, 52 weeks earlier
    lags = np.column_stack([s.shift(k) for k in config.HOURLY_LAGS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # all four lags missing -> NaN
        f["how_mean_4w"] = np.nanmean(lags, axis=1)
    f["how_n_4w"] = np.isfinite(lags).sum(axis=1)
    # long-run mean of the same hour of week, using only weeks that ended >= 1 week before t
    prev = s.groupby(g.how).shift(1)                                # same hour of week, 168 h earlier
    cnt = prev.notna().astype(float).groupby(g.how).cumsum()
    tot = prev.fillna(0).groupby(g.how).cumsum()
    f["how_mean_all"] = (tot / cnt.replace(0, np.nan)).to_numpy()
    past = s.shift(min_lag)                                         # the week that ended one week ago
    f["level_week"] = past.rolling(168, min_periods=24).mean()
    f["gap_share_week"] = past.isna().astype(float).rolling(168, min_periods=1).mean()
    for k in extra_lags:                                            # only for the leakage demonstration
        f[f"lag{k}"] = s.shift(k)
    return f


def profile(train: pd.DataFrame) -> pd.Series:
    """Mean demand by hour of week over the training rows (the 'static profile' baseline)."""
    return train.groupby("how").demand.mean()
