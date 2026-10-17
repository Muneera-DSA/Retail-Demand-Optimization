"""Part A data: 50 products x 10 outlets, daily unit sales Jan 2012 - Feb 2014, weekly prices.

Arranged as series x date matrices (500 series), like Part B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

FILES = ("train_data", "product_prices", "date_to_week_id_map")


def load(folder: Path | None = None) -> dict[str, pd.DataFrame]:
    folder = folder or config.TCS_DIR
    missing = [f for f in FILES if not (folder / f"{f}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"TCS files missing in {folder}: {missing} (see data/README.md)")
    d = {f: pd.read_csv(folder / f"{f}.csv") for f in FILES}
    d["train_data"]["date"] = pd.to_datetime(d["train_data"]["date"])
    d["date_to_week_id_map"]["date"] = pd.to_datetime(d["date_to_week_id_map"]["date"])
    return d


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    series: pd.DataFrame          # product_identifier, outlet, department_identifier, category_of_product, state
    sales: np.ndarray             # (n_series, n_dates)
    price: np.ndarray             # (n_series, n_dates): the week's selling price, known for the week ahead
    week_id: np.ndarray           # (n_dates,)


def build_panel(d: dict[str, pd.DataFrame]) -> Panel:
    tr, pr, wk = d["train_data"], d["product_prices"], d["date_to_week_id_map"]
    dates = pd.date_range(tr.date.min(), tr.date.max(), freq="D")
    series = (tr[["product_identifier", "outlet", "department_identifier", "category_of_product", "state"]]
              .drop_duplicates(["product_identifier", "outlet"]).sort_values(["outlet", "product_identifier"]).reset_index(drop=True))
    key = pd.MultiIndex.from_frame(series[["product_identifier", "outlet"]])
    r = key.get_indexer(pd.MultiIndex.from_frame(tr[["product_identifier", "outlet"]]))
    c = dates.get_indexer(tr.date)
    sales = np.full((len(series), len(dates)), np.nan, np.float32)
    sales[r, c] = tr.sales.to_numpy(np.float32)
    week = wk.set_index("date").week_id.reindex(dates).to_numpy()
    pw = pr.pivot_table(index=["product_identifier", "outlet"], columns="week_id", values="sell_price")
    pw = pw.reindex(key)
    price = pw.reindex(columns=week).to_numpy(np.float32)
    return Panel(dates, series, sales, price, week)


def classify(sales: np.ndarray) -> pd.DataFrame:
    """Syntetos-Boylan-Croston demand classes: ADI (average days between sales) and CV^2 of
    non-zero sizes; cut-offs 1.32 and 0.49."""
    out = []
    for row in sales:
        nz = row[row > 0]
        adi = len(row) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
        cls = ("smooth" if adi < 1.32 and cv2 < 0.49 else "erratic" if adi < 1.32 else
               "intermittent" if cv2 < 0.49 else "lumpy")
        out.append({"adi": adi, "cv2": cv2, "class": cls, "zero_share": float(np.mean(row == 0)), "mean": float(row.mean())})
    return pd.DataFrame(out)


def quality(d: dict[str, pd.DataFrame], p: Panel) -> dict:
    tr, pr, wk = d["train_data"], d["product_prices"], d["date_to_week_id_map"]
    grid = len(p.series) * len(p.dates)
    m = tr.merge(wk, on="date", how="left").merge(pr, on=["outlet", "product_identifier", "week_id"], how="left")
    sp = pr.sort_values("week_id").groupby(["outlet", "product_identifier"]).sell_price
    rel = pr.sell_price / pr.groupby(["outlet", "product_identifier"]).sell_price.transform("median")
    cls = classify(p.sales)
    zero_runs = []
    for row in p.sales:
        run = best = 0
        for v in row:
            run = run + 1 if v == 0 else 0
            best = max(best, run)
        zero_runs.append(best)
    zr = np.array(zero_runs)
    q = np.nanpercentile(p.sales, 99.9)
    return {
        "validity": {"rows": int(len(tr)), "negative_sales": int((tr.sales < 0).sum()), "non_integer_sales": int((tr.sales % 1 != 0).sum()),
                     "non_positive_prices": int((pr.sell_price <= 0).sum()), "price_min": float(pr.sell_price.min()),
                     "price_max": float(pr.sell_price.max()), "null_values": {k: int(v.isna().sum().sum()) for k, v in d.items()}},
        "uniqueness": {"duplicate_product_outlet_date": int(tr.duplicated(["date", "product_identifier", "outlet"]).sum()),
                       "duplicate_price_rows": int(pr.duplicated(["outlet", "product_identifier", "week_id"]).sum())},
        "completeness": {"series": int(len(p.series)), "days": int(len(p.dates)), "first_date": str(p.dates[0].date()),
                         "last_date": str(p.dates[-1].date()), "grid_cells": grid, "rows": int(len(tr)),
                         "sales_rows_without_price": int(m.sell_price.isna().sum()), "zero_sales_share": float((tr.sales == 0).mean())},
        "consistency": {"departments_per_product_max": int(tr.groupby("product_identifier").department_identifier.nunique().max()),
                        "categories_per_department_max": int(tr.groupby("department_identifier").category_of_product.nunique().max()),
                        "states_per_outlet_max": int(tr.groupby("outlet").state.nunique().max()),
                        "price_changes_share_of_weeks": float((sp.diff().fillna(0) != 0).mean()),
                        "outlets_by_state": {k: sorted(map(int, v)) for k, v in tr.groupby("state").outlet.unique().items()}},
        "accuracy": {"prices_below_half_series_median": int((rel < 0.5).sum()), "prices_above_double_series_median": int((rel > 2).sum()),
                     "sales_99_9th_percentile": float(q), "sales_max": float(np.nanmax(p.sales)),
                     "series_longest_zero_run_over_180_days": int((zr > 180).sum()), "longest_zero_run_days": int(zr.max())},
        "relevance": {"demand_classes": cls["class"].value_counts().to_dict(), "median_zero_share": float(cls.zero_share.median()),
                      "products_by_category": tr.groupby("category_of_product").product_identifier.nunique().to_dict()},
    }
