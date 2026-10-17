"""Order-date features for weekly ordering (Part A).

At an order date o the model predicts each of the next 7 days for all 500 product x outlet
series, using sales up to o and the week's planned selling price. Intermittent-demand
estimates (Croston, SBA, TSB) are computed recursively up to o and used both as baselines and
as features."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import config
from .data import Panel

STATIC = ["product_code", "department_identifier", "category_code", "outlet", "state_code"]
HISTORY = ["mean_7", "mean_28", "mean_56", "mean_112", "std_28", "zero_share_28", "zero_share_112", "days_since_sale",
           "size_112", "interval_112", "croston", "sba", "tsb", "dow_mean_8"]
TARGET = ["h", "dow", "dom", "month", "woy_sin", "woy_cos", "price", "price_rel", "price_change", "price_rel_dept"]
FEATURES = STATIC + HISTORY + TARGET


def _nanmean(a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def intermittent_paths(sales: np.ndarray, alpha: float | None = None) -> dict[str, np.ndarray]:
    """One-step-ahead Croston, SBA and TSB forecasts available at the end of each day (n_series, n_dates).
    Croston (1972): smooth non-zero sizes z and intervals p; forecast z/p. SBA (Syntetos & Boylan 2005):
    Croston x (1 - alpha/2). TSB (Teunter, Syntetos & Babai 2011): smooth demand probability every period."""
    a = alpha or config.TCS_INTERMITTENT_ALPHA
    n, T = sales.shape
    cro, tsb = np.full((n, T), np.nan), np.full((n, T), np.nan)
    for i in range(n):
        y = np.nan_to_num(sales[i])
        nz = np.flatnonzero(y > 0)
        if len(nz) == 0:
            cro[i], tsb[i] = 0.0, 0.0
            continue
        z, p, q = y[nz[0]], float(nz[0] + 1), 1.0 / (nz[0] + 1)
        prob, size = q, z
        last = nz[0]
        for t in range(T):
            if t > nz[0]:
                if y[t] > 0:
                    z = z + a * (y[t] - z)
                    p = p + a * ((t - last) - p)
                    last = t
                    size = size + a * (y[t] - size)
                prob = prob + a * (float(y[t] > 0) - prob)
            if t >= nz[0]:
                cro[i, t] = z / p
                tsb[i, t] = prob * size
            else:
                cro[i, t] = tsb[i, t] = 0.0
    return {"croston": cro, "sba": cro * (1 - a / 2), "tsb": tsb}


def static_codes(p: Panel) -> pd.DataFrame:
    s = p.series
    return pd.DataFrame({"product_code": pd.Categorical(s.product_identifier).codes.astype(float),
                         "department_identifier": s.department_identifier.astype(float),
                         "category_code": pd.Categorical(s.category_of_product).codes.astype(float),
                         "outlet": s.outlet.astype(float), "state_code": pd.Categorical(s.state).codes.astype(float)})


def origin_rows(p: Panel, o: int, paths: dict, horizon: int | None = None) -> pd.DataFrame:
    horizon = horizon or config.TCS_HORIZON
    n = len(p.series)
    y = p.sales[:, : o + 1]
    st = {f"mean_{w}": _nanmean(y[:, -w:]) for w in (7, 28, 56, 112)}
    st["std_28"] = np.nanstd(y[:, -28:], axis=1)
    st["zero_share_28"] = _nanmean((y[:, -28:] == 0).astype(float))
    st["zero_share_112"] = _nanmean((y[:, -112:] == 0).astype(float))
    sold = np.where(np.nan_to_num(y) > 0, np.arange(o + 1)[None, :], -1).max(axis=1)
    st["days_since_sale"] = np.where(sold >= 0, o - sold, 999).clip(max=999).astype(float)
    w = y[:, -112:]
    nzc = (w > 0).sum(axis=1)
    st["size_112"] = np.where(nzc > 0, np.where(w > 0, w, 0).sum(axis=1) / np.maximum(nzc, 1), 0.0)
    st["interval_112"] = 112.0 / np.maximum(nzc, 1)
    for k in ("croston", "sba", "tsb"):
        st[k] = paths[k][:, o]
    base = pd.concat([static_codes(p), pd.DataFrame(st)], axis=1)
    dept_price = pd.Series(p.price[:, o]).groupby(p.series.department_identifier.to_numpy()).transform("median").to_numpy()
    med_price = np.nanmedian(p.price[:, : o + 1], axis=1)
    rows = []
    for h in range(1, horizon + 1):
        t = o + h
        if t >= len(p.dates):
            break
        r = base.copy()
        dt = p.dates[t]
        r["series"], r["origin"], r["target_idx"], r["date"], r["h"] = np.arange(n), o, t, dt, h
        r["dow"], r["dom"], r["month"] = dt.dayofweek, dt.day, dt.month
        woy = dt.isocalendar().week
        r["woy_sin"], r["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)
        r["price"] = p.price[:, t]
        r["price_rel"] = p.price[:, t] / med_price
        r["price_change"] = p.price[:, t] / p.price[:, max(t - 7, 0)] - 1
        r["price_rel_dept"] = p.price[:, t] / dept_price
        same = np.column_stack([p.sales[:, t - 7 * k] for k in range(int(np.ceil(h / 7)), int(np.ceil(h / 7)) + 8)])
        r["dow_mean_8"] = _nanmean(same)
        r["seasonal_naive"] = same[:, 0]
        r["y"] = p.sales[:, t]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def make(p: Panel, origins: list[int], paths: dict) -> pd.DataFrame:
    df = pd.concat([origin_rows(p, o, paths) for o in origins], ignore_index=True)
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    return df
