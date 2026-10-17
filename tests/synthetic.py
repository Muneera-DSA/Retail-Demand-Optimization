"""Small synthetic datasets in the exact formats of the real files (tests only)."""

import numpy as np
import pandas as pd


def hourly(days=420, seed=0, gaps=True):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2019-01-01", periods=days * 24, freq="h")
    how = ts.dayofweek * 24 + ts.hour
    lam = 40 + 30 * np.sin(np.pi * ts.hour / 24) ** 2 + 15 * (ts.dayofweek >= 5)
    demand = rng.poisson(lam).astype(int) + 1
    df = pd.DataFrame({"date": ts.strftime("%Y-%m-%d"), "hour": ts.hour, "demand": demand, "how": how})
    if gaps:
        night = df.hour.isin([3, 4]) & (rng.random(len(df)) < 0.5)
        outage = df.date.between("2019-09-10", "2019-09-14")
        df = df[~night & ~outage]
    return df[["date", "hour", "demand"]].reset_index(drop=True)


def favorita(days=500, stores=3, families=("GROCERY I", "BEVERAGES", "BOOKS"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    rows = []
    for s in range(1, stores + 1):
        for fi, fam in enumerate(families):
            base = 0 if fam == "BOOKS" else 50 * (fi + 1) * s
            promo = rng.poisson(2, days) * (rng.random(days) < 0.3)
            lam = base * (1 + 0.3 * (dates.dayofweek >= 5)) * (1 + 0.3 * (promo > 0))
            rows.append(pd.DataFrame({"date": dates, "store_nbr": s, "family": fam, "sales": rng.poisson(lam).astype(float), "onpromotion": promo}))
    tr = pd.concat(rows, ignore_index=True)
    tr = tr[tr.date != "2015-12-25"].reset_index(drop=True)
    tr.insert(0, "id", np.arange(len(tr)))
    test_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=16)
    te = pd.DataFrame([(d, s, f, 0) for d in test_dates for s in range(1, stores + 1) for f in families],
                      columns=["date", "store_nbr", "family", "onpromotion"])
    te.insert(0, "id", np.arange(len(te)) + len(tr))
    st = pd.DataFrame({"store_nbr": range(1, stores + 1), "city": ["Quito", "Guayaquil", "Quito"][:stores],
                       "state": ["Pichincha", "Guayas", "Pichincha"][:stores], "type": ["A", "B", "D"][:stores], "cluster": [1, 2, 3][:stores]})
    od = pd.date_range(dates[0], test_dates[-1], freq="B")
    oil = pd.DataFrame({"date": od, "dcoilwtico": 50 + np.cumsum(rng.normal(size=len(od)))})
    oil.loc[::40, "dcoilwtico"] = np.nan
    ho = pd.DataFrame({"date": pd.to_datetime(["2015-05-01", "2015-08-10", "2015-11-02", "2015-03-02"]),
                       "type": ["Holiday", "Holiday", "Holiday", "Holiday"], "locale": ["National", "National", "Local", "Regional"],
                       "locale_name": ["Ecuador", "Ecuador", "Quito", "Guayas"], "description": ["Dia del Trabajo", "Grito", "Fundacion", "Prov"],
                       "transferred": [False, False, False, False]})
    tx = tr.groupby(["date", "store_nbr"]).sales.sum().reset_index().assign(transactions=lambda d: (d.sales / 10 + 100).astype(int))
    return {"train": tr, "test": te, "stores": st, "oil": oil, "holidays_events": ho, "transactions": tx[["date", "store_nbr", "transactions"]]}


def tcs(days=600, outlets=(111, 222), seed=0):
    """TCS iON format: daily sales per product x outlet, weekly prices, a date -> week_id map."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=days, freq="D")
    week = 49 + (np.arange(days) + 6) // 7                    # 2012-01-01 is a Sunday: its own (partial) week
    products = [(74, 11, "others", 0.15, 1.5), (337, 11, "others", 0.4, 2.0), (423, 12, "fast_moving_consumer_goods", 0.8, 3.0),
                (432, 12, "fast_moving_consumer_goods", 0.05, 4.0), (500, 13, "drinks_and_food", 1.0, 6.0), (501, 13, "drinks_and_food", 0.6, 2.5)]
    states = {111: "Maharashtra", 222: "Telangana"}
    sales, prices = [], []
    for o in outlets:
        for pid, dep, cat, p_sale, size in products:
            weeks = np.unique(week)
            base = round(float(rng.uniform(1, 10)), 2)
            disc = rng.random(len(weeks)) < 0.15
            wp = np.where(disc, round(base * 0.8, 2), base)
            prices.append(pd.DataFrame({"outlet": o, "product_identifier": pid, "week_id": weeks, "sell_price": wp}))
            lift = np.where(disc[week - week.min()], 1.4, 1.0) * (1 + 0.3 * (dates.dayofweek >= 5))
            sold = rng.random(days) < np.clip(p_sale * lift, 0, 1)
            qty = np.where(sold, 1 + rng.poisson(size * lift - 1 if size > 1 else 0), 0)
            sales.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "product_identifier": pid, "department_identifier": dep,
                                       "category_of_product": cat, "outlet": o, "state": states[o], "sales": qty.astype(int)}))
    tr = pd.concat(sales, ignore_index=True).sort_values(["date", "outlet", "product_identifier"]).reset_index(drop=True)
    wk = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "week_id": week})
    return {"train_data": tr, "product_prices": pd.concat(prices, ignore_index=True), "date_to_week_id_map": wk}
