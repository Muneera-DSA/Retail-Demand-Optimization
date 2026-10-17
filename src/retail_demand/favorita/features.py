"""Origin-based features for 16-day replenishment forecasts.

At a forecast origin o (an order date), the model predicts each of the next 16 days for
every series. A row is (series, origin o, horizon h); its target is sales on day o + h.
Features use only:
  - sales and transactions up to and including day o;
  - information planned in advance for day o + h: promotions, holidays, calendar.
Oil is taken at the origin (its future price is not known).

Means are on log1p(sales), which is the scale the model is trained on (RMSLE objective).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import config
from .data import Panel

SERIES_STATS = ["lmean_7", "lmean_14", "lmean_28", "lmean_56", "lmean_112", "llast", "lstd_28", "zero_share_28",
                "days_since_sale", "promo_lmean_28", "tx_lmean_28", "days_open"]
TARGET_COLS = ["h", "dow", "dom", "month", "payday", "hol_national", "hol_regional", "hol_local", "workday", "event", "quake",
               "promo_target", "promo_vs_recent", "dow_lmean_4", "dow_lmean_8", "dow_ratio"]
STATIC = ["store_nbr", "family_code", "type_code", "cluster"]
ORIGIN = ["loil", "oil_change_28"]
FEATURES = STATIC + SERIES_STATS + ORIGIN + TARGET_COLS


def _nanmean(a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def static_codes(p: Panel) -> pd.DataFrame:
    s = p.series
    return pd.DataFrame({"store_nbr": s.store_nbr.to_numpy(float),
                         "family_code": pd.Categorical(s.family).codes.astype(float),
                         "type_code": pd.Categorical(s["type"]).codes.astype(float),
                         "cluster": s.cluster.to_numpy(float)})


def origin_rows(p: Panel, o: int, horizon: int | None = None, with_target: bool = True) -> pd.DataFrame:
    """All rows (series x horizon) for forecast origin index o."""
    horizon = horizon or config.FAV_HORIZON
    n = len(p.series)
    ls = np.log1p(p.sales[:, : o + 1])
    ls = np.where(p.is_open[:, : o + 1], ls, np.nan)                  # before opening: not demand
    stats = {}
    for w in (7, 14, 28, 56, 112):
        stats[f"lmean_{w}"] = _nanmean(ls[:, -w:])
    stats["llast"] = ls[:, -1]
    stats["lstd_28"] = _nanstd(ls[:, -28:])
    stats["zero_share_28"] = _nanmean((ls[:, -28:] == 0).astype(float) + np.where(np.isnan(ls[:, -28:]), np.nan, 0))
    sold = np.where(np.nan_to_num(ls) > 0, np.arange(o + 1)[None, :], -1).max(axis=1)
    stats["days_since_sale"] = np.where(sold >= 0, o - sold, 9999).clip(max=365).astype(float)
    stats["promo_lmean_28"] = np.log1p(p.promo[:, o - 27: o + 1]).mean(axis=1)
    stats["tx_lmean_28"] = _nanmean(np.log1p(p.transactions[:, o - 27: o + 1]))
    stats["days_open"] = p.is_open[:, : o + 1].sum(axis=1).astype(float)
    base = pd.concat([static_codes(p), pd.DataFrame(stats)], axis=1)
    base["loil"] = np.log(p.oil[o])
    base["oil_change_28"] = np.log(p.oil[o]) - np.log(p.oil[max(o - 28, 0)])
    base["dead"] = stats["days_since_sale"] >= config.FAV_DEAD_DAYS
    rows = []
    for h in range(1, horizon + 1):
        t = o + h
        if t >= len(p.dates):
            break
        r = base.copy()
        dt = p.dates[t]
        r["series"] = np.arange(n)
        r["origin"], r["target_idx"], r["date"] = o, t, dt
        r["h"], r["dow"], r["dom"], r["month"] = h, dt.dayofweek, dt.day, dt.month
        r["payday"] = float(dt.day == 15 or dt.is_month_end)
        r["hol_national"], r["workday"], r["event"], r["quake"] = p.hol_national[t], p.workday[t], p.event[t], p.quake[t]
        r["hol_regional"], r["hol_local"] = p.hol_regional[:, t], p.hol_local[:, t]
        r["promo_target"] = np.log1p(p.promo[:, t])
        r["promo_vs_recent"] = r["promo_target"] - r["promo_lmean_28"]
        k0 = int(np.ceil(h / 7))                                      # same weekday, most recent one known at o
        same = np.column_stack([ls[:, t - 7 * k] if t - 7 * k <= o else np.full(n, np.nan) for k in range(k0, k0 + 8)])
        r["dow_lmean_4"], r["dow_lmean_8"] = _nanmean(same[:, :4]), _nanmean(same)
        r["dow_ratio"] = r["dow_lmean_8"] - _nanmean(ls[:, -56:])
        raw_same = np.expm1(same)
        r["seasonal_naive"] = raw_same[:, 0]                            # same weekday, last known week
        r["base_dow_mean4"] = _nanmean(raw_same[:, :4])                 # same weekday, mean of last 4 known weeks
        r["base_mean14"] = _nanmean(np.expm1(ls[:, -14:]))              # mean of the last 14 days
        r["base_mean28"] = _nanmean(np.expm1(ls[:, -28:]))
        r["open_target"] = p.is_open[:, t]
        if with_target:
            r["y"] = p.sales[:, t]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def make(p: Panel, origins: list[int], with_target: bool = True) -> pd.DataFrame:
    df = pd.concat([origin_rows(p, o, with_target=with_target) for o in origins], ignore_index=True)
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    return df


def training_origins(p: Panel, before_idx: int, start: str | None = None, step: int | None = None) -> list[int]:
    """Origins whose 16 target days all fall strictly before `before_idx`."""
    start, step = start or config.FAV_TRAIN_FROM, step or config.FAV_ORIGIN_STEP
    first = int(p.dates.get_loc(pd.Timestamp(start)))
    last = before_idx - config.FAV_HORIZON - 1
    return list(range(last, first - 1, -step))[::-1]


def trainable(df: pd.DataFrame) -> pd.Series:
    """Rows usable as training targets: store open, day observed, series not dead at origin."""
    return df.open_target & df.y.notna() & ~df.dead
