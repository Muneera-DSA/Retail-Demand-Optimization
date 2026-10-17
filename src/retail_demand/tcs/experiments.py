"""Part A experiments: 500 product x outlet series, intermittent daily demand -> weekly orders.

    python -m retail_demand.tcs.experiments          # all stages -> results/tcs/*.json

Validation: the last 8 weeks (3 Jan - 28 Feb 2014) are the test period, scored once with
weekly order dates; the model screen uses 4 expanding-window folds of 8 weeks inside the
development period. Primary point metric: RMSE. For intermittent counts the median forecast
is often zero, so MAE rewards forecasting nothing; stock decisions need the mean (or a
quantile), which RMSE targets. MAE, WAPE and bias are reported too.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config
from ..common import stats, treeshap
from . import data, features

STAGES = ["quality", "eda", "v1", "screen", "tune", "test", "decision", "explain", "hypotheses"]
BASELINES = {"Zero (forecast nothing)": None, "Seasonal naive (same weekday last week)": "seasonal_naive",
             "Moving average, 28 days": "mean_28", "Croston": "croston", "SBA (Syntetos-Boylan)": "sba", "TSB": "tsb"}
FIRST_ORIGIN = 112


def _num(scale=True):
    return make_pipeline(*([SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])))


HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0)


def candidates(seed=config.SEED) -> dict:
    num = [c for c in features.FEATURES if c not in ("product_code", "category_code", "state_code", "outlet", "department_identifier")]
    lin = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["product_code", "outlet", "department_identifier"]),
                             ("num", _num(), num)])
    tree = ColumnTransformer([("n", SimpleImputer(strategy="median"), features.FEATURES)])
    return {
        "Ridge": make_pipeline(lin, Ridge(alpha=10.0)),
        "Poisson GLM": make_pipeline(lin, PoissonRegressor(alpha=1e-3, max_iter=500)),
        "k-nearest neighbours (k=50)": make_pipeline(ColumnTransformer([("n", _num(), features.FEATURES)]), KNeighborsRegressor(n_neighbors=50)),
        "Random forest": make_pipeline(tree, RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=seed)),
        "Extra trees": make_pipeline(tree, ExtraTreesRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.7, n_jobs=-1, random_state=seed)),
        "Gradient boosting (Poisson)": HistGradientBoostingRegressor(loss="poisson", **HGB, random_state=seed),
        "Gradient boosting (squared error)": HistGradientBoostingRegressor(**HGB, random_state=seed),
        "Neural network (MLP 64-32)": make_pipeline(ColumnTransformer([("n", _num(), features.FEATURES)]),
                                                    MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=200, early_stopping=True, random_state=seed)),
    }


TUNE_GRIDS = {
    "Gradient boosting": [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=lv, min_samples_leaf=ml)
                          for lr, it in ((0.03, 500), (0.08, 200)) for lv in (15, 63) for ml in (50, 300)],
    "Random forest": [{"randomforestregressor__min_samples_leaf": m, "randomforestregressor__max_features": f} for m, f in ((5, 0.5), (50, 0.5), (20, 0.8))],
    "Extra trees": [{"extratreesregressor__min_samples_leaf": m, "extratreesregressor__max_features": f} for m, f in ((5, 0.7), (50, 0.7), (20, 0.9))],
    "Ridge": [{"ridge__alpha": a} for a in (1, 100)],
    "Poisson GLM": [{"poissonregressor__alpha": a} for a in (1e-4, 1e-2)],
}


def grid_for(name):
    return next((g for k, g in TUNE_GRIDS.items() if name.startswith(k)), [])


# ---------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, out: Path):
        self.out = out
        (out / "cache").mkdir(parents=True, exist_ok=True)
        self.d = data.load()
        self.p = data.build_panel(self.d)
        self.paths = features.intermittent_paths(self.p.sales)
        T = len(self.p.dates)
        H = config.TCS_HORIZON
        self.test_origins = [T - 1 - H * k for k in range(config.TCS_TEST_WEEKS, 0, -1)]
        self.test_start = self.test_origins[0] + 1
        self._frames = {}

    def frame(self, origins):
        key = (origins[0], origins[-1], len(origins))
        if key not in self._frames:
            self._frames[key] = features.make(self.p, origins, self.paths)
        return self._frames[key]

    def origins_before(self, idx):
        """Weekly origins whose 7 targets all fall before day index idx."""
        last = idx - config.TCS_HORIZON - 1
        return list(range(last, FIRST_ORIGIN - 1, -config.TCS_HORIZON))[::-1]

    def cv_folds(self):
        out = []
        for k in range(config.TCS_CV_FOLDS, 0, -1):
            start = self.test_start - k * config.TCS_CV_FOLD_WEEKS * 7
            vo = [start - 1 + 7 * w for w in range(config.TCS_CV_FOLD_WEEKS)]
            out.append((self.origins_before(start), vo))
        return out

    def save(self, name, obj):
        (self.out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_json))

    def load(self, name):
        return json.loads((self.out / f"{name}.json").read_text())


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def point(y, f) -> dict:
    return {"rmse": stats.rmse(y, f), "mae": stats.mae(y, f), "wape": stats.wape(y, f), "bias": stats.bias(y, f), "n": int(len(y))}


def predict(name, est, train, X):
    if name in BASELINES:
        col = BASELINES[name]
        return np.zeros(len(X)) if col is None else np.nan_to_num(X[col].to_numpy(float))
    m = clone(est).fit(train[features.FEATURES], train.y)
    return np.clip(m.predict(X[features.FEATURES]), 0, None)


def series_test(df, pa, pb):
    """Wilcoxon over series of the sum of squared errors (a vs b). Positive mean diff = a worse."""
    ea = pd.Series((df.y.to_numpy() - pa) ** 2).groupby(df.series.to_numpy()).sum()
    eb = pd.Series((df.y.to_numpy() - pb) ** 2).groupby(df.series.to_numpy()).sum()
    d = (ea - eb)
    d = d[d != 0]
    return {"p": float(sps.wilcoxon(d).pvalue), "series_where_a_worse": float((d > 0).mean())}


def series_bootstrap(df, fn, n_boot=1000, seed=config.SEED):
    """Bootstrap over series (each series keeps all its days)."""
    groups = df.series.to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        vals.append(fn(pick))
    return {"value": float(fn(np.arange(len(df)))), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


# ----------------------------------------------------------------------------- stages

def stage_quality(c: Ctx):
    c.save("quality", data.quality(c.d, c.p))


def stage_eda(c: Ctx):
    tr = c.d["train_data"]
    cls = data.classify(c.p.sales)
    cat_share = tr.groupby("category_of_product").sales.sum() / tr.sales.sum()
    dow = tr.groupby(tr.date.dt.dayofweek).sales.mean()
    # within-series price effect: days at a discount (price < 95% of the series median) vs other days
    rel = c.p.price / np.nanmedian(c.p.price, axis=1, keepdims=True)
    lift = []
    for i in range(len(c.p.series)):
        disc = rel[i] < 0.95
        if disc.sum() >= 14 and (~disc).sum() >= 14:
            lift.append(np.log1p(c.p.sales[i][disc].mean()) - np.log1p(c.p.sales[i][~disc].mean()))
    lift = np.array(lift)
    monthly = tr.set_index("date").groupby("category_of_product").sales.resample("ME").sum().unstack(0)
    c.save("eda", {
        "share_of_sales_by_category": cat_share.round(4).to_dict(), "share_of_sales_by_state": (tr.groupby("state").sales.sum() / tr.sales.sum()).round(4).to_dict(),
        "mean_daily_sales_by_weekday": {int(k): float(v) for k, v in dow.items()},
        "weekend_vs_weekday_ratio": float(dow[[5, 6]].mean() / dow[[0, 1, 2, 3, 4]].mean()),
        "demand_classes": cls["class"].value_counts().to_dict(),
        "demand_class_share_of_sales": (cls.groupby("class")["mean"].sum() / cls["mean"].sum()).round(4).to_dict(),
        "series_mean_daily_sales_quantiles": cls["mean"].quantile([0.1, 0.5, 0.9, 0.99]).round(3).to_dict(),
        "top_10pct_series_share_of_sales": float(np.sort(cls["mean"])[::-1][: len(cls) // 10].sum() / cls["mean"].sum()),
        "price_discount_effect_within_series": {"series_with_discount_days": int(len(lift)),
                                                "median_log_diff": float(np.median(lift)) if len(lift) else None,
                                                "share_positive": float((lift > 0).mean()) if len(lift) else None,
                                                "wilcoxon_p": float(sps.wilcoxon(lift).pvalue) if len(lift) > 5 else None},
        "monthly_sales_by_category": {str(k.date()): {kk: float(vv) for kk, vv in v.items()} for k, v in monthly.iterrows()},
    })


def stage_v1(c: Ctx):
    """Re-run the original notebook's design (per category: cube-root of sales and price, IDs and date parts as
    numeric features, extra trees with a random 80/20 split) and the same model with a time split."""
    tr = c.d["train_data"].merge(c.d["date_to_week_id_map"], on="date").merge(c.d["product_prices"], on=["outlet", "product_identifier", "week_id"])
    df = tr.assign(Month=tr.date.dt.month, Week=tr.date.dt.isocalendar().week.astype(int), Year=tr.date.dt.year, Day=tr.date.dt.day)
    df["sales_t"], df["price_t"] = df.sales ** (1 / 3), df.sell_price ** (1 / 3)
    df = pd.get_dummies(df, columns=["state"], dtype=float)
    feats = ["product_identifier", "department_identifier", "outlet", "price_t", "Month", "Week", "Year", "Day"] + [c_ for c_ in df if c_.startswith("state_")]
    test_from = c.p.dates[c.test_start]
    out = {}
    for cat, g in df.groupby("category_of_product"):
        X, y = g[feats], g.sales_t
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(Xtr, ytr)
        rand = {"r2_train": float(m.score(Xtr, ytr)), "r2_test": float(m.score(Xte, yte)),
                "mae_units": stats.mae(yte ** 3, np.clip(m.predict(Xte), 0, None) ** 3)}
        tmask = (g.date >= test_from).to_numpy()
        m2 = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(X[~tmask], y[~tmask])
        pt = np.clip(m2.predict(X[tmask]), 0, None)
        timed = {"r2_test": float(r2_score(y[tmask], pt)), "mae_units": stats.mae(y[tmask] ** 3, pt ** 3),
                 "rmse_units": stats.rmse(y[tmask] ** 3, pt ** 3)}
        out[cat] = {"rows": int(len(g)), "random_split": rand, "time_split_last_8_weeks": timed}
        print(f"[v1] {cat}: random R2 {rand['r2_test']:.3f} vs time {timed['r2_test']:.3f}")
    c.save("v1", {"design": "per category; cube root of sales and price; IDs and date parts as numeric features; ExtraTreesRegressor(100)",
                  "categories": out})


def stage_screen(c: Ctx):
    folds = c.cv_folds()
    rows, oof = [], {}
    cands = {**{b: None for b in BASELINES}, **candidates()}
    val_frames = [c.frame(vo) for _, vo in folds]
    train_frames = [c.frame(to) for to, _ in folds]
    for name, est in cands.items():
        t0 = time.time()
        preds = [predict(name, est, tr, va) for tr, va in zip(train_frames, val_frames)]
        oof[name] = np.concatenate(preds)
        per = [point(va.y.to_numpy(), p) for va, p in zip(val_frames, preds)]
        rows.append({"model": name, "baseline": name in BASELINES, "cv_rmse": float(np.mean([m["rmse"] for m in per])),
                     "cv_rmse_sd": float(np.std([m["rmse"] for m in per], ddof=1)), "cv_mae": float(np.mean([m["mae"] for m in per])),
                     "cv_wape": float(np.mean([m["wape"] for m in per])), "cv_bias": float(np.mean([m["bias"] for m in per])),
                     "fold_rmse": [m["rmse"] for m in per], "seconds": time.time() - t0})
        print(f"[screen] {name}: CV RMSE {rows[-1]['cv_rmse']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["cv_rmse"])
    allv = pd.concat(val_frames, ignore_index=True)
    best = rows[0]["model"]
    for r in rows[1:]:
        r["vs_best"] = series_test(allv, oof[r["model"]], oof[best])
    adj = stats.holm({r["model"]: r["vs_best"]["p"] for r in rows if "vs_best" in r})
    for r in rows:
        if "vs_best" in r:
            r["vs_best"]["p_holm"] = adj[r["model"]]
    c.save("screen", {"best": best, "folds": [[str(c.p.dates[vo[0] + 1].date()), str(c.p.dates[vo[-1] + 7].date())] for _, vo in folds],
                      "table": rows})


def stage_tune(c: Ctx, top: int = 3):
    sc = c.load("screen")
    rows = [r for r in sc["table"] if not r["baseline"]][:top]
    if not any(r["model"] == "Gradient boosting (Poisson)" for r in rows):          # always tuned: explained with TreeSHAP
        rows.append(next(r for r in sc["table"] if r["model"] == "Gradient boosting (Poisson)"))
    folds = c.cv_folds()
    vf, tf = [c.frame(vo) for _, vo in folds], [c.frame(to) for to, _ in folds]
    tuned = []
    for r in rows:
        name = r["model"]
        best = {"model": name, "params": {}, "cv_rmse": r["cv_rmse"]}
        for params in grid_for(name):
            est = clone(candidates()[name]).set_params(**params)
            score = float(np.mean([stats.rmse(va.y, predict(name, est, tr, va)) for tr, va in zip(tf, vf)]))
            print(f"[tune] {name} {params}: {score:.4f}", flush=True)
            if score < best["cv_rmse"]:
                best = {"model": name, "params": params, "cv_rmse": score}
        tuned.append(best)
    chosen = min(tuned, key=lambda t: t["cv_rmse"])
    c.save("tune", {"model": chosen["model"], "params": chosen["params"], "cv_rmse": chosen["cv_rmse"], "tuned": tuned,
                    "rule": "the 3 best model families (plus Poisson gradient boosting) are tuned on the same CV folds; lowest CV RMSE is chosen"})


def final(c: Ctx, name=None):
    t = c.load("tune")
    name = name or t["model"]
    params = next((x["params"] for x in t["tuned"] if x["model"] == name), {})
    return name, clone(candidates()[name]).set_params(**params)


def stage_test(c: Ctx):
    name, est = final(c)
    train = c.frame(c.origins_before(c.test_start))
    test = c.frame(c.test_origins)
    y = test.y.to_numpy()
    preds = {name: predict(name, est, train, test)}
    for b in BASELINES:
        preds[b] = predict(b, None, train, test)
    for other in ("Ridge", "Gradient boosting (Poisson)"):
        if other != name:
            preds[other] = predict(other, final(c, other)[1], train, test)
    table = {}
    for k, p in preds.items():
        m = point(y, p)
        if k != name:
            m["rmse_diff_vs_model"] = series_bootstrap(test, lambda i, p=p: stats.rmse(y[i], p[i]) - stats.rmse(y[i], preds[name][i]))
            m["series_test_vs_model"] = series_test(test, p, preds[name])
        table[k] = m
    cls = data.classify(c.p.sales[:, : c.test_start])["class"].to_numpy()
    seg = {}
    for k in np.unique(cls):
        mm = cls[test.series.to_numpy()] == k
        seg[k] = {"rows": int(mm.sum()), "share_of_sales": float(y[mm].sum() / y.sum()),
                  **{m: point(y[mm], preds[m][mm])["rmse"] for m in (name, "SBA (Syntetos-Boylan)", "Moving average, 28 days")}}
    pd.DataFrame({"date": test.date.astype(str), "origin": test.origin, "series": test.series, "y": y,
                  **{k: np.round(v, 4) for k, v in preds.items()}}).to_csv(c.out / "test_predictions.csv.gz", index=False)
    c.save("test", {"model": name, "window": [str(test.date.min().date()), str(test.date.max().date())],
                    "train_rows": int(len(train)), "table": table, "rmse_by_demand_class": seg})


def _weekly(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    g = pd.DataFrame({"series": df.series.to_numpy(), "origin": df.origin.to_numpy(), "y": df.y.to_numpy(), "f": pred})
    return g.groupby(["series", "origin"]).sum().reset_index()


def _weekly_rows(c: Ctx, origins: list[int]) -> pd.DataFrame:
    """One row per series and order date; target = demand over the next 7 days."""
    f = c.frame(origins)
    g = f[f.h == 1][["series", "origin"] + features.STATIC + features.HISTORY].copy()
    agg = f.groupby(["series", "origin"]).agg(y=("y", "sum"), price=("price", "mean"), price_rel=("price_rel", "mean"))
    return g.merge(agg.reset_index(), on=["series", "origin"])


YF = namedtuple("YF", "y f")                               # weekly demand and forecast as numpy arrays (fast bootstrap)
WEEKLY_FEATURES = features.STATIC + features.HISTORY + ["price", "price_rel"]


def _past_weekly_quantile(c: Ctx, w: pd.DataFrame, q: float = 0.9, weeks: int = 26) -> np.ndarray:
    """Per series and order date: the q-quantile of the last `weeks` 7-day sales totals up to the order date
    (a per-item safety-stock rule a planner can run in a spreadsheet)."""
    cs = np.concatenate([np.zeros((len(c.p.series), 1)), np.cumsum(np.nan_to_num(c.p.sales), axis=1)], axis=1)
    out = np.empty(len(w))
    for j, (s_, o) in enumerate(zip(w.series.to_numpy(), w.origin.to_numpy())):
        ends = o + 1 - 7 * np.arange(weeks)
        ends = ends[ends - 7 >= 0]
        out[j] = np.quantile(cs[s_, ends] - cs[s_, ends - 7], q)
    return out


RULE = "Rule of thumb: each item's 90th percentile of its last 26 weekly sales"
RULE_AVG = "Rule of thumb: 28-day average x 7"


def stage_decision(c: Ctx):
    """Weekly order-up-to level per series, target fill rate 95%. Each policy's single scale factor is set on the
    last CV fold (8 weeks before the test) to reach the target, then applied unchanged to the 8 test weeks."""
    target = config.TCS_FILL_TARGET
    name, est = final(c)
    cal_tr_o, cal_va_o = c.cv_folds()[-1]
    cal_tr, cal_va = c.frame(cal_tr_o), c.frame(cal_va_o)
    train, test = c.frame(c.origins_before(c.test_start)), c.frame(c.test_origins)
    pol_cal = {f"{name}: mean forecast x 7": predict(name, est, cal_tr, cal_va), "SBA (Syntetos-Boylan) x 7": predict("SBA (Syntetos-Boylan)", None, cal_tr, cal_va),
               "Rule of thumb: 28-day average x 7": predict("Moving average, 28 days", None, cal_tr, cal_va)}
    pol_test = {f"{name}: mean forecast x 7": predict(name, est, train, test), "SBA (Syntetos-Boylan) x 7": predict("SBA (Syntetos-Boylan)", None, train, test),
                "Rule of thumb: 28-day average x 7": predict("Moving average, 28 days", None, train, test)}
    weekly_cal, weekly_test = {k: _weekly(cal_va, v) for k, v in pol_cal.items()}, {k: _weekly(test, v) for k, v in pol_test.items()}
    # forecast trained on the decision itself: quantile gradient boosting of the 7-day total
    wtr_cal, wva_cal = _weekly_rows(c, cal_tr_o), _weekly_rows(c, cal_va_o)
    wtr, wte = _weekly_rows(c, c.origins_before(c.test_start)), _weekly_rows(c, c.test_origins)
    qname = "Quantile gradient boosting of the 7-day total (q = 0.9)"
    qm = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=config.SEED)
    weekly_cal[qname] = wva_cal[["series", "origin", "y"]].assign(f=clone(qm).fit(wtr_cal[WEEKLY_FEATURES], wtr_cal.y).predict(wva_cal[WEEKLY_FEATURES]).clip(0))
    weekly_test[qname] = wte[["series", "origin", "y"]].assign(f=clone(qm).fit(wtr[WEEKLY_FEATURES], wtr.y).predict(wte[WEEKLY_FEATURES]).clip(0))
    for wk, base in ((weekly_cal, weekly_cal[RULE_AVG]), (weekly_test, weekly_test[RULE_AVG])):
        b = base[["series", "origin", "y"]].copy()
        wk[RULE] = b.assign(f=_past_weekly_quantile(c, b))

    def fill(w, k, wt=1.0):
        y, f = (w.y, w.f) if isinstance(w, tuple) else (w.y.to_numpy(float), w.f.to_numpy(float))
        return 1 - (wt * np.clip(y - k * (f + 1e-9), 0, None)).sum() / (wt * y).sum()

    def factor(w, tgt, wt=1.0):
        lo, hi = 0.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if fill(w, mid, wt) >= tgt else (mid, hi)
        return hi

    res = {}
    for k in weekly_test:
        kc = factor(weekly_cal[k], target)
        w = weekly_test[k]
        S = kc * w.f
        res[k] = {"scale_factor_from_calibration": kc, "fill_rate": float(fill(w, kc)), "cycle_service_level": float(np.mean(w.y <= S)),
                  "stock_vs_demand": float(S.sum() / w.y.sum()), "excess_units": float(np.clip(S - w.y, 0, None).sum()),
                  "short_units": float(np.clip(w.y - S, 0, None).sum()), "wape_of_weekly_totals": stats.wape(w.y, w.f),
                  "stock_vs_demand_at_95pct_fill_ex_post": float(factor(w, target) * w.f.sum() / w.y.sum())}
    for ref, key in ((RULE, "stock_saving_vs_rule_ex_post"), (RULE_AVG, "stock_saving_vs_28day_rule_ex_post")):
        for r in res.values():
            r[key] = 1 - r["stock_vs_demand_at_95pct_fill_ex_post"] / res[ref]["stock_vs_demand_at_95pct_fill_ex_post"]
    # uncertainty of the quantile model's saving vs each rule: bootstrap over series (integer weights per series)
    sw = {k: weekly_test[k].sort_values(["series", "origin"]).reset_index(drop=True) for k in (qname, RULE, RULE_AVG)}
    assert all((sw[k].series.to_numpy() == sw[qname].series.to_numpy()).all() for k in sw)
    arr = {k: YF(w.y.to_numpy(float), w.f.to_numpy(float)) for k, w in sw.items()}
    series = np.unique(sw[qname].series)
    pos = np.searchsorted(series, sw[qname].series.to_numpy())
    rng = np.random.default_rng(config.SEED)
    boot = {RULE: [], RULE_AVG: []}
    for _ in range(500):
        wt = np.bincount(rng.integers(0, len(series), len(series)), minlength=len(series))[pos]
        sq = factor(arr[qname], target, wt) * (wt * arr[qname].f).sum()
        for ref in boot:
            boot[ref].append(1 - sq / (factor(arr[ref], target, wt) * (wt * arr[ref].f).sum()))
    for ref, key in ((RULE, "stock_saving_vs_rule"), (RULE_AVG, "stock_saving_vs_28day_rule")):
        b = np.array(boot[ref])
        res[qname][f"{key}_ex_post_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        res[qname][f"{key}_bootstrap_p"] = float(max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1 / len(b)))
    wq = sw[qname]
    c.save("decision", {"fill_target": target, "weeks": len(c.test_origins), "series": int(wq.series.nunique()), "reference_rule": RULE, "policies": res})


def stage_explain(c: Ctx, n_rows: int = 300):
    name = "Gradient boosting (Poisson)"
    _, est = final(c, name)
    train, test = c.frame(c.origins_before(c.test_start)), c.frame(c.test_origins)
    m = clone(est).fit(train[features.FEATURES], train.y)
    rows = test.sample(min(n_rows, len(test)), random_state=config.SEED)
    X = rows[features.FEATURES].to_numpy(float)
    t0 = time.time()
    phi, base = treeshap.shap_values(m, X, progress=50)
    imp = pd.Series(np.abs(phi).mean(0), index=features.FEATURES).sort_values(ascending=False)
    pi = features.FEATURES.index("price_rel")
    np.savez_compressed(c.out / "shap_values.npz", phi=phi, X=X, base=base, features=np.array(features.FEATURES))
    c.save("explain", {"model": name + " (tuned)", "n_rows": int(len(X)), "seconds": time.time() - t0, "base_value": base,
                       "scale": "log of expected units (Poisson link)",
                       "additivity_max_error": float(np.max(np.abs(base + phi.sum(1) - m._raw_predict(X).ravel()))),
                       "mean_abs_shap": imp.round(4).to_dict(),
                       "price_rel_shap_correlation": float(np.corrcoef(X[:, pi], phi[:, pi])[0, 1]) if np.std(X[:, pi]) > 0 else None})


def stage_hypotheses(c: Ctx):
    t, eda, dec = c.load("test"), c.load("eda"), c.load("decision")
    name = t["model"]
    hyp = []
    for hid, b in (("H1", "Seasonal naive (same weekday last week)"), ("H2", "Moving average, 28 days"), ("H3", "SBA (Syntetos-Boylan)"),
                   ("H4", "Croston")):
        r = t["table"][b]
        d = r["rmse_diff_vs_model"]
        st = r["series_test_vs_model"]
        # direction from the same test as the p-value (Wilcoxon over series); the aggregate RMSE difference is reported alongside
        hyp.append({"id": hid, "hypothesis": f"The model has lower error than {b} on most series",
                    "evidence": (f"{b} worse on {st['series_where_a_worse']:.0%} of series; RMSE {t['table'][name]['rmse']:.3f} vs {r['rmse']:.3f} "
                                 f"(difference {d['value']:+.3f}, CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f})"),
                    "p": st["p"], "direction_ok": st["series_where_a_worse"] > 0.5})
    pe = eda["price_discount_effect_within_series"]
    if pe["wilcoxon_p"] is not None:
        hyp.append({"id": "H5", "hypothesis": "Sales are higher when an item is discounted (price < 95% of its usual price)",
                    "evidence": f"median log difference {pe['median_log_diff']:+.3f} over {pe['series_with_discount_days']} series", "p": pe["wilcoxon_p"],
                    "direction_ok": pe["median_log_diff"] > 0})
    q = dec["policies"]["Quantile gradient boosting of the 7-day total (q = 0.9)"]
    ci = q["stock_saving_vs_rule_ex_post_ci"]
    hyp.append({"id": "H6", "hypothesis": "Ordering from the quantile model needs less stock than the per-item 90th-percentile rule for a 95% fill rate",
                "evidence": f"saving {q['stock_saving_vs_rule_ex_post']:+.1%} (series bootstrap CI {ci[0]:+.1%} to {ci[1]:+.1%})",
                "p": q["stock_saving_vs_rule_bootstrap_p"], "direction_ok": q["stock_saving_vs_rule_ex_post"] > 0,
                "note": "two-sided series-bootstrap p-value (500 resamples; floor 1/500)"})
    adj = stats.holm({h["id"]: h["p"] for h in hyp})
    for h in hyp:
        h["p_holm"] = adj.get(h["id"], float("nan"))
        h["verdict"] = ("not testable" if not np.isfinite(h["p_holm"]) else "supported" if h["p_holm"] < 0.05 and h["direction_ok"] else
                        "rejected (opposite direction)" if h["p_holm"] < 0.05 else "not supported")
    c.save("hypotheses", hyp)


def run(stages=STAGES, out: Path | None = None):
    c = Ctx(out or config.TCS_RESULTS)
    for s in stages:
        t0 = time.time()
        globals()[f"stage_{s}"](c)
        print(f"[{s}] done in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    run(a.stages, a.out)


if __name__ == "__main__":
    main()
