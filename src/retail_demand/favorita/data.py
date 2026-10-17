"""Favorita: load the Kaggle files, check them, and arrange them as series x date matrices.

The unit of forecasting is one store x product family per day (54 x 33 = 1,782 series).
Everything is aligned on a daily calendar from 2013-01-01 to 2017-08-31 (the last 16 days
are Kaggle's unlabelled test period; only their planned promotions and holidays are used).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

FILES = ("train", "stores", "oil", "holidays_events", "transactions", "test")


def load(folder: Path | None = None) -> dict[str, pd.DataFrame]:
    folder = folder or config.FAVORITA_DIR
    missing = [f for f in FILES if not (folder / f"{f}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Favorita files missing in {folder}: {missing}. See data/README.md for the Kaggle download.")
    out = {f: pd.read_csv(folder / f"{f}.csv") for f in FILES}
    for f in ("train", "oil", "holidays_events", "transactions", "test"):
        out[f]["date"] = pd.to_datetime(out[f]["date"])
    return out


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    series: pd.DataFrame          # one row per series: store_nbr, family, city, state, type, cluster
    sales: np.ndarray             # (n_series, n_dates) float32; NaN = not observed (closed day or future)
    promo: np.ndarray             # (n_series, n_dates) float32: items of the family on promotion (planned, known ahead)
    is_open: np.ndarray           # (n_stores_by_series, n_dates) bool: store has started trading
    hol_national: np.ndarray      # (n_dates,) national holiday actually celebrated that day
    hol_regional: np.ndarray      # (n_series, n_dates) regional holiday in the store's state
    hol_local: np.ndarray         # (n_series, n_dates) local holiday in the store's city
    workday: np.ndarray           # (n_dates,) compensating working day (normally a weekend)
    event: np.ndarray             # (n_dates,) national event (Black Friday, Mother's Day, World Cup, ...)
    quake: np.ndarray             # (n_dates,) days after the 16 April 2016 earthquake (relief period)
    oil: np.ndarray               # (n_dates,) WTI oil price, carried forward over missing days
    transactions: np.ndarray      # (n_series, n_dates) store transactions (NaN when missing)
    last_train_date: pd.Timestamp


def build_panel(d: dict[str, pd.DataFrame]) -> Panel:
    tr, te, st = d["train"], d["test"], d["stores"]
    dates = pd.date_range(tr.date.min(), te.date.max(), freq="D")
    series = (tr[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"])
              .merge(st, on="store_nbr", how="left").reset_index(drop=True))
    key = pd.MultiIndex.from_frame(series[["store_nbr", "family"]])
    di = {dt: i for i, dt in enumerate(dates)}

    def matrix(df, col):
        m = np.full((len(series), len(dates)), np.nan, dtype=np.float32)
        r = key.get_indexer(pd.MultiIndex.from_frame(df[["store_nbr", "family"]]))
        c = df.date.map(di).to_numpy()
        m[r, c] = df[col].to_numpy(np.float32)
        return m

    sales = matrix(tr, "sales")
    promo = np.nan_to_num(matrix(pd.concat([tr, te]), "onpromotion"), nan=0.0)
    first_sale = tr[tr.sales > 0].groupby("store_nbr").date.min()
    opened = series.store_nbr.map(first_sale).to_numpy()
    is_open = dates.to_numpy()[None, :] >= opened[:, None]

    ho = d["holidays_events"]
    celebrated = ho[~ho.transferred & (ho.type != "Work Day")]
    nat = celebrated[(celebrated.locale == "National") & (celebrated.type != "Event")]
    ev = celebrated[(celebrated.locale == "National") & (celebrated.type == "Event")]
    quake_rows = ev.description.str.contains("Terremoto", case=False)
    hol_national = dates.isin(nat.date).astype(np.float32)
    event = dates.isin(ev[~quake_rows].date).astype(np.float32)
    quake = dates.isin(ev[quake_rows].date).astype(np.float32)
    workday = dates.isin(ho[ho.type == "Work Day"].date).astype(np.float32)
    reg = celebrated[celebrated.locale == "Regional"]
    loc = celebrated[celebrated.locale == "Local"]
    hol_regional = np.zeros((len(series), len(dates)), np.float32)
    hol_local = np.zeros_like(hol_regional)
    for _, h in reg[reg.date.isin(dates)].iterrows():
        hol_regional[(series.state == h.locale_name).to_numpy(), di[h.date]] = 1
    for _, h in loc[loc.date.isin(dates)].iterrows():
        hol_local[(series.city == h.locale_name).to_numpy(), di[h.date]] = 1

    oil = d["oil"].set_index("date").dcoilwtico.reindex(dates).ffill().bfill().to_numpy(np.float32)
    tx = d["transactions"].pivot(index="store_nbr", columns="date", values="transactions").reindex(columns=dates)
    transactions = tx.reindex(series.store_nbr).to_numpy(np.float32)
    return Panel(dates, series, sales, promo, is_open, hol_national, hol_regional, hol_local, workday, event, quake,
                 oil, transactions, tr.date.max())


def quality(d: dict[str, pd.DataFrame], p: Panel) -> dict:
    tr, ho, oil, tx, st = d["train"], d["holidays_events"], d["oil"], d["transactions"], d["stores"]
    train_dates = pd.date_range(tr.date.min(), tr.date.max())
    missing_dates = sorted(set(train_dates) - set(tr.date))
    totals = tr.groupby(["store_nbr", "family"]).sales.sum()
    first_sale = tr[tr.sales > 0].groupby("store_nbr").date.min()
    daily = tr.groupby(["store_nbr", "date"]).sales.sum().unstack(0)
    closed_after_open = {int(s): int(((daily[s] == 0) & (daily.index > first_sale[s])).sum()) for s in daily.columns}
    sd = tr.groupby(["store_nbr", "date"]).sales.sum().reset_index()
    sd = sd.merge(tx, on=["store_nbr", "date"], how="left")
    t = p.dates <= p.last_train_date
    s_tr = p.sales[:, t]
    zero_share_by_family = tr.assign(z=tr.sales == 0).groupby("family").z.mean().sort_values(ascending=False)
    # accuracy: robust outliers per series (log scale, > 6 MAD from the series median, open days with sales)
    with np.errstate(all="ignore"):
        ls = np.log1p(np.where(s_tr > 0, s_tr, np.nan))
        med = np.nanmedian(ls, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(ls - med), axis=1, keepdims=True) * 1.4826
        z = (ls - med) / np.where(mad > 0, mad, np.nan)
    n_out = int(np.nansum(np.abs(z) > 6))
    return {
        "validity": {
            "train_rows": int(len(tr)), "test_rows": int(len(d["test"])),
            "negative_sales": int((tr.sales < 0).sum()), "negative_promotions": int((tr.onpromotion < 0).sum()),
            "fractional_sales_share": float((tr.sales % 1 != 0).mean()),
            "fractional_sales_families": sorted(tr[tr.sales % 1 != 0].family.unique().tolist()),
            "null_values": {k: int(v.isna().sum().sum()) for k, v in d.items()},
            "store_numbers_not_in_stores_file": int((~tr.store_nbr.isin(st.store_nbr)).sum()),
        },
        "uniqueness": {"duplicate_store_family_date": int(tr.duplicated(["date", "store_nbr", "family"]).sum()),
                       "duplicate_transactions": int(tx.duplicated(["date", "store_nbr"]).sum()),
                       "duplicate_holiday_date_place": int(ho.duplicated(["date", "locale_name"]).sum())},
        "completeness": {
            "first_date": str(tr.date.min().date()), "last_date": str(tr.date.max().date()),
            "series": int(len(totals)), "expected_series": int(st.store_nbr.nunique() * tr.family.nunique()),
            "dates_absent_from_file": [str(x.date()) for x in missing_dates],
            "stores_opening_after_start": {int(k): str(v.date()) for k, v in first_sale[first_sale > tr.date.min() + pd.Timedelta(days=14)].items()},
            "series_never_selling": int((totals == 0).sum()),
            "never_selling_by_family": tr.loc[tr.set_index(["store_nbr", "family"]).index.isin(totals[totals == 0].index)]
                                         .drop_duplicates(["store_nbr", "family"]).family.value_counts().to_dict(),
            "zero_sales_share": float((tr.sales == 0).mean()),
            "zero_share_by_family_top": zero_share_by_family.head(8).round(3).to_dict(),
            "promotions_first_recorded": str(tr[tr.onpromotion > 0].date.min().date()),
            "oil_missing_values": int(oil.dcoilwtico.isna().sum()), "oil_days": int(len(oil)),
            "oil_calendar_days_missing": int(len(pd.date_range(oil.date.min(), oil.date.max())) - len(oil)),
            "transactions_store_days_missing_with_sales": int(((sd.sales > 0) & sd.transactions.isna()).sum()),
        },
        "consistency": {
            "store_days_closed_after_opening": int(sum(closed_after_open.values())),
            "stores_with_most_closed_days": dict(sorted(closed_after_open.items(), key=lambda kv: -kv[1])[:5]),
            "sales_vs_transactions_correlation_store_days": float(sd[["sales", "transactions"]].corr().iloc[0, 1]),
            "transferred_holidays": int(ho.transferred.sum()), "transfer_rows": int((ho.type == "Transfer").sum()),
            "test_dates_follow_train": bool(d["test"].date.min() == tr.date.max() + pd.Timedelta(days=1)),
            "test_series_match_train": bool(set(zip(d["test"].store_nbr, d["test"].family)) == set(zip(tr.store_nbr, tr.family))),
        },
        "accuracy": {"robust_outliers_over_6_mad": n_out, "share_of_selling_days": float(n_out / np.isfinite(ls).sum()),
                     "largest_single_day": tr.nlargest(3, "sales")[["date", "store_nbr", "family", "sales"]].astype(str).to_dict("records"),
                     "policy": "outliers are kept; the model is trained on log1p(sales), which limits their pull"},
        "relevance": {"stores": int(st.store_nbr.nunique()), "cities": int(st.city.nunique()), "states": int(st.state.nunique()),
                      "store_types": st.type.value_counts().to_dict(), "families": int(tr.family.nunique()),
                      "years_of_history": float((tr.date.max() - tr.date.min()).days / 365.25)},
    }
