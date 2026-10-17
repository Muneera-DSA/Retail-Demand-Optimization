"""Part B experiments: Favorita, 1,782 store x family series, 16-day replenishment.

    python -m retail_demand.favorita.experiments              # all stages -> results/favorita/*.json

Validation design (no future information in any training row):
  - test window: the last 16 days of the file (2017-07-31 .. 2017-08-15), touched once;
  - 4 validation windows: the four 16-day windows before it, used for the screen, tuning and
    calibrating the inventory policy. Screen models are trained on origins whose targets all
    fall before the earliest validation window and are not retrained between windows
    (a model refreshed monthly, as in practice).
Primary metric: WAPE (units, the scale of stock decisions). Also: RMSLE (Kaggle's metric),
MAE, bias. Series that sold nothing in the last 56 days are forecast as zero by every method.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config
from ..common import stats, treeshap
from . import data, features

STAGES = ["quality", "eda", "screen", "tune", "test", "decision", "leakage", "explain", "hypotheses"]
BASELINES = {"Seasonal naive (same weekday, last week)": "seasonal_naive",
             "Same-weekday mean, last 4 weeks": "base_dow_mean4",
             "Mean of the last 14 days": "base_mean14"}
SUBSAMPLE = 300_000          # rows for the slower learners (forests, neural network)


class LogTarget:
    """Wrap an estimator so it is trained on log1p(y) and predicts in units."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.est.predict(X))


def _num(scale=True):
    steps = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    return make_pipeline(*steps)


HGB_POISSON = dict(loss="poisson", max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100)


def candidates(seed=config.SEED) -> dict:
    num = [c for c in features.FEATURES if c not in ("family_code", "type_code", "store_nbr", "cluster")]
    lin = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["family_code", "type_code"]), ("num", _num(), num)])
    trees = ColumnTransformer([("n", SimpleImputer(strategy="median"), features.FEATURES)])
    return {
        "Ridge (log scale)": (LogTarget(make_pipeline(lin, Ridge(alpha=10.0))), None),
        "Random forest (log scale)": (LogTarget(make_pipeline(trees, RandomForestRegressor(
            n_estimators=150, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=seed))), SUBSAMPLE),
        "Extra trees (log scale)": (LogTarget(make_pipeline(trees, ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=20, max_features=0.7, n_jobs=-1, random_state=seed))), SUBSAMPLE),
        "Neural network (MLP 128-64, log scale)": (LogTarget(make_pipeline(ColumnTransformer([("n", _num(), features.FEATURES)]),
                                                                           MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=60,
                                                                                        early_stopping=True, random_state=seed))), SUBSAMPLE),
        "Gradient boosting (log scale, squared error)": (LogTarget(HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100, random_state=seed)), None),
        "Gradient boosting (Poisson, units)": (HistGradientBoostingRegressor(**HGB_POISSON, random_state=seed), None),
    }


TUNE_GRIDS = {
    "Gradient boosting": [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=lv, min_samples_leaf=ml, l2_regularization=1.0)
                          for lr, it in ((0.05, 600), (0.1, 300)) for lv in (63, 255) for ml in (50, 200)],
    "Random forest": [{"min_samples_leaf": m, "max_features": f} for m, f in ((10, 0.5), (40, 0.5), (20, 0.3), (20, 0.8))],
    "Extra trees": [{"min_samples_leaf": m, "max_features": f} for m, f in ((10, 0.7), (40, 0.7), (20, 0.9))],
    "Neural network": [{"alpha": a} for a in (1e-5, 1e-3)],
}


def _inner(est):
    """The innermost estimator (inside LogTarget and Pipeline)."""
    inner = est.est if isinstance(est, LogTarget) else est
    return inner.steps[-1][1] if hasattr(inner, "steps") else inner


def with_params(est, params: dict):
    e = LogTarget(clone(est.est)) if isinstance(est, LogTarget) else clone(est)
    if params:
        _inner(e).set_params(**params)
    return e


def grid_for(name: str) -> list[dict]:
    return next((g for k, g in TUNE_GRIDS.items() if name.startswith(k)), [])


# -------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, out: Path):
        self.out = out
        (out / "cache").mkdir(parents=True, exist_ok=True)
        self.d = data.load()
        self.p = data.build_panel(self.d)
        self.test0 = int(self.p.dates.get_loc(pd.Timestamp(config.FAV_TEST_START)))
        H = config.FAV_HORIZON
        self.valid_starts = [self.test0 - H * k for k in range(config.FAV_N_VALID_WINDOWS, 0, -1)]

    def save(self, name, obj):
        (self.out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_json))

    def load(self, name):
        return json.loads((self.out / f"{name}.json").read_text())

    def frame(self, name, origins):
        f = self.out / "cache" / f"{name}.pkl"
        if f.exists():
            return pd.read_pickle(f)
        df = features.make(self.p, origins)
        df.to_pickle(f)
        return df


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def rmsle(y, f):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.clip(f, 0, None))) ** 2)))


def scored(df: pd.DataFrame) -> np.ndarray:
    """Rows that count in evaluation: store open and the day observed."""
    return (df.open_target & df.y.notna()).to_numpy()


def fit_predict(est, sub, train: pd.DataFrame, X: pd.DataFrame, seed=config.SEED):
    tr = train[features.trainable(train)]
    if sub and len(tr) > sub:
        tr = tr.sample(sub, random_state=seed)
    m = clone(est) if not isinstance(est, LogTarget) else LogTarget(clone(est.est))
    m.fit(tr[features.FEATURES], tr.y.to_numpy())
    pred = np.clip(m.predict(X[features.FEATURES]), 0, None)
    pred[X.dead.to_numpy()] = 0.0
    return pred, m


def baseline_pred(col: str, X: pd.DataFrame) -> np.ndarray:
    pred = np.nan_to_num(X[col].to_numpy(float), nan=0.0)
    pred[X.dead.to_numpy()] = 0.0
    return pred


def metrics(y, f) -> dict:
    return {"wape": stats.wape(y, f), "rmsle": rmsle(y, f), "mae": stats.mae(y, f), "bias": stats.bias(y, f), "n": int(len(y))}


def series_abs_error(df, pred):
    ok = scored(df)
    return pd.Series(np.abs(df.y.to_numpy()[ok] - pred[ok])).groupby(df.series.to_numpy()[ok]).sum()


def store_bootstrap_wape_diff(df, pa, pb, n_boot=1000, seed=config.SEED):
    """WAPE(a) - WAPE(b), resampling whole stores (errors are correlated within a store)."""
    ok = scored(df)
    g = pd.DataFrame({"store": df.store_nbr.to_numpy()[ok], "y": df.y.to_numpy()[ok],
                      "ea": np.abs(df.y.to_numpy()[ok] - pa[ok]), "eb": np.abs(df.y.to_numpy()[ok] - pb[ok])}).groupby("store").sum()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(g))
    vals = []
    for _ in range(n_boot):
        s = g.iloc[rng.choice(idx, len(idx))]
        vals.append((s.ea.sum() - s.eb.sum()) / s.y.sum())
    point = (g.ea.sum() - g.eb.sum()) / g.y.sum()
    return {"value": float(point), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def paired_series_test(df, pa, pb):
    """Wilcoxon signed-rank over series of total absolute error (a vs b). Positive median = a worse."""
    ea, eb = series_abs_error(df, pa), series_abs_error(df, pb)
    d = (ea - eb)
    d = d[d != 0]
    w = sps.wilcoxon(d)
    return {"p": float(w.pvalue), "series_where_a_worse": float((d > 0).mean()), "n_series": int(len(d))}


# ---------------------------------------------------------------------------- stages

def stage_quality(c: Ctx):
    c.save("quality", data.quality(c.d, c.p))


def stage_eda(c: Ctx):
    tr = c.d["train"].merge(c.d["stores"], on="store_nbr")
    tr = tr[tr.date >= "2014-04-01"]
    daily = tr.groupby("date").sales.sum()
    fam = tr.groupby("family").sales.sum().sort_values(ascending=False)
    p = c.p
    # within-series promotion association (2016-01 .. last train day), log scale
    t0, t1 = int(p.dates.get_loc(pd.Timestamp("2016-01-01"))), int(p.dates.get_loc(p.last_train_date)) + 1
    ls, pr, op = np.log1p(p.sales[:, t0:t1]), p.promo[:, t0:t1] > 0, p.is_open[:, t0:t1]
    lift = []
    for i in range(len(p.series)):
        v = op[i] & np.isfinite(ls[i])
        a, b = ls[i][v & pr[i]], ls[i][v & ~pr[i]]
        if len(a) >= 20 and len(b) >= 20:
            lift.append(a.mean() - b.mean())
    lift = np.array(lift)
    # paydays (15th and month end) vs other days of the same weekday, within series
    dts = p.dates[t0:t1]
    pay = (dts.day == 15) | dts.is_month_end
    pay_diff = np.nanmean(np.where(pay[None, :] & op, ls, np.nan), axis=1) - np.nanmean(np.where(~pay[None, :] & op, ls, np.nan), axis=1)
    hol = p.hol_national[t0:t1] > 0
    hol_diff = np.nanmean(np.where(hol[None, :] & op, ls, np.nan), axis=1) - np.nanmean(np.where(~hol[None, :] & op, ls, np.nan), axis=1)
    dow = tr.groupby(tr.date.dt.dayofweek).sales.sum() / tr.groupby(tr.date.dt.dayofweek).date.nunique()
    c.save("eda", {
        "total_sales_by_year": {int(k): float(v) for k, v in tr.groupby(tr.date.dt.year).sales.sum().items()},
        "daily_total_series": {str(k.date()): round(float(v), 1) for k, v in daily.items()},
        "share_of_sales_top_families": (fam / fam.sum()).head(10).round(4).to_dict(),
        "families_for_80pct_of_sales": int((fam.cumsum() / fam.sum() < 0.8).sum() + 1),
        "mean_daily_sales_by_weekday": {int(k): float(v) for k, v in dow.items()},
        "sales_by_store_type_share": (tr.groupby("type").sales.sum() / tr.sales.sum()).round(3).to_dict(),
        "promotion_lift_within_series": {"series": int(len(lift)), "median_log_diff": float(np.median(lift)),
                                         "median_ratio": float(np.expm1(np.median(lift)) + 1), "share_positive": float((lift > 0).mean()),
                                         "wilcoxon_p": float(sps.wilcoxon(lift).pvalue)},
        "payday_effect_within_series": {"median_log_diff": float(np.nanmedian(pay_diff)), "share_positive": float(np.nanmean(pay_diff > 0)),
                                        "wilcoxon_p": float(sps.wilcoxon(pay_diff[np.isfinite(pay_diff)]).pvalue)},
        "national_holiday_effect_within_series": {"median_log_diff": float(np.nanmedian(hol_diff)), "share_positive": float(np.nanmean(hol_diff > 0)),
                                                  "wilcoxon_p": float(sps.wilcoxon(hol_diff[np.isfinite(hol_diff)]).pvalue)},
        "oil_vs_monthly_sales_correlation": float(pd.concat([c.d["oil"].set_index("date").dcoilwtico.resample("ME").mean(),
                                                             daily.resample("ME").sum()], axis=1).dropna().corr().iloc[0, 1]),
    })


def _valid_frames(c: Ctx):
    vf = [c.frame(f"valid_{w}", [w - 1]) for w in c.valid_starts]
    train = c.frame("train_screen", features.training_origins(c.p, c.valid_starts[0]))
    return train, vf


def stage_screen(c: Ctx):
    train, vf = _valid_frames(c)
    rows, preds = [], {}
    for name, col in BASELINES.items():
        preds[name] = [baseline_pred(col, v) for v in vf]
    for name, (est, sub) in candidates().items():
        t0 = time.time()
        fitted = None
        out = []
        for v in vf:
            if fitted is None:
                p0, fitted = fit_predict(est, sub, train, v)
            else:
                p0 = np.clip(fitted.predict(v[features.FEATURES]), 0, None)
                p0[v.dead.to_numpy()] = 0.0
            out.append(p0)
        preds[name] = out
        print(f"[screen] {name}: {time.time() - t0:.0f}s")
    for name, ps in preds.items():
        per = [metrics(v.y.to_numpy()[scored(v)], p[scored(v)]) for v, p in zip(vf, ps)]
        rows.append({"model": name, "baseline": name in BASELINES, "wape": float(np.mean([m["wape"] for m in per])),
                     "wape_by_window": [m["wape"] for m in per], "rmsle": float(np.mean([m["rmsle"] for m in per])),
                     "bias": float(np.mean([m["bias"] for m in per]))})
    rows.sort(key=lambda r: r["wape"])
    best = rows[0]["model"]
    allv = pd.concat(vf, ignore_index=True)
    pb = np.concatenate(preds[best])
    for r in rows:
        if r["model"] != best:
            r["vs_best"] = paired_series_test(allv, np.concatenate(preds[r["model"]]), pb)
    adj = stats.holm({r["model"]: r["vs_best"]["p"] for r in rows if "vs_best" in r})
    for r in rows:
        if "vs_best" in r:
            r["vs_best"]["p_holm"] = adj[r["model"]]
    with open(c.out / "cache" / "screen_preds.pkl", "wb") as fh:
        pickle.dump(preds, fh)
    c.save("screen", {"best": best, "train_rows": int(features.trainable(train).sum()),
                      "valid_windows": [[str(c.p.dates[w].date()), str(c.p.dates[w + 15].date())] for w in c.valid_starts], "table": rows})


def _valid_wape(model, vf):
    ws, preds = [], []
    for v in vf:
        pv = np.clip(model.predict(v[features.FEATURES]), 0, None)
        pv[v.dead.to_numpy()] = 0
        preds.append(pv)
        ws.append(stats.wape(v.y.to_numpy()[scored(v)], pv[scored(v)]))
    return float(np.mean(ws)), preds


def stage_tune(c: Ctx, top: int = 3):
    """Tune the top `top` model families of the screen on the same validation windows; keep the best."""
    sc = c.load("screen")
    rows = [r for r in sc["table"] if not r["baseline"]][:top]
    hgb = "Gradient boosting (log scale, squared error)"      # always tuned: it is the model explained with exact TreeSHAP
    if not any(r["model"] == hgb for r in rows):
        rows.append(next(r for r in sc["table"] if r["model"] == hgb))
    train, vf = _valid_frames(c)
    tuned = []
    for r in rows:
        name = r["model"]
        est, sub = candidates()[name]
        best = {"model": name, "params": {}, "wape": r["wape"]}
        for params in grid_for(name):
            t0 = time.time()
            _, m = fit_predict(with_params(est, params), sub, train, vf[0])
            w, _ = _valid_wape(m, vf)
            print(f"[tune] {name} {params}: {w:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if w < best["wape"]:
                best = {"model": name, "params": params, "wape": w}
        tuned.append(best)
    chosen = min(tuned, key=lambda t: t["wape"])
    est, sub = candidates()[chosen["model"]]
    _, m = fit_predict(with_params(est, chosen["params"]), sub, train, vf[0])
    _, vpred = _valid_wape(m, vf)
    with open(c.out / "cache" / "valid_final_preds.pkl", "wb") as fh:
        pickle.dump(vpred, fh)
    c.save("tune", {"model": chosen["model"], "params": chosen["params"], "wape": chosen["wape"], "tuned": tuned,
                    "rule": ("the 3 best model families of the screen, plus gradient boosting (always tuned, for TreeSHAP), are tuned on the "
                             "same 4 validation windows; the lowest mean WAPE is chosen")})


def final_estimator(c: Ctx):
    t = c.load("tune")
    est, sub = candidates()[t["model"]]
    return t["model"], with_params(est, t["params"]), sub


def stage_test(c: Ctx):
    name, est, sub = final_estimator(c)
    train = c.frame("train_final", features.training_origins(c.p, c.test0))
    test = c.frame("test", [c.test0 - 1])
    pred, model = fit_predict(est, sub, train, test)
    with open(c.out / "cache" / "final_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    ok = scored(test)
    y = test.y.to_numpy()
    preds = {name: pred, **{b: baseline_pred(col, test) for b, col in BASELINES.items()}}
    ridge = "Ridge (log scale)"
    if ridge != name:
        preds[ridge], _ = fit_predict(*candidates()[ridge], train, test)
    table = {}
    for k, p in preds.items():
        m = metrics(y[ok], p[ok])
        if k != name:
            m["wape_diff_vs_model_store_bootstrap"] = store_bootstrap_wape_diff(test, p, pred)
            m["paired_series_test_vs_model"] = paired_series_test(test, p, pred)
        table[k] = m
    fam = test.assign(pred=pred, err=np.abs(y - pred), fam=c.p.series.family.to_numpy()[test.series])[ok]
    by_fam = fam.groupby("fam").apply(lambda g: pd.Series({"wape": g.err.sum() / max(g.y.sum(), 1e-9), "sales": g.y.sum()}), include_groups=False)
    by_h = test.assign(err=np.abs(y - pred))[ok].groupby("h").apply(lambda g: g.err.sum() / g.y.sum(), include_groups=False)
    zero_share = test.zero_share_28.to_numpy()
    seg = {}
    for lab, m in (("intermittent (>50% zero days)", zero_share > 0.5), ("regular", zero_share <= 0.5)):
        mm = ok & m
        seg[lab] = {"rows": int(mm.sum()), "share_of_sales": float(y[mm].sum() / y[ok].sum()), **metrics(y[mm], pred[mm])}
    pd.DataFrame({"date": test.date.astype(str), "series": test.series, "store_nbr": test.store_nbr.astype(int),
                  "family": c.p.series.family.to_numpy()[test.series], "y": y, **{k: np.round(v, 3) for k, v in preds.items()}})[ok].to_csv(
        c.out / "test_predictions.csv.gz", index=False)
    c.save("test", {"model": name, "window": [str(test.date.min().date()), str(test.date.max().date())],
                    "train_rows": int(features.trainable(train).sum()), "table": table,
                    "wape_by_family": by_fam.sort_values("sales", ascending=False).round(4).to_dict("index"),
                    "wape_by_horizon": {int(k): float(v) for k, v in by_h.items()}, "segments": seg})


def _cycle_totals(df, pred, days):
    """Per-series demand and forecast totals over the first `days` days after the order date (active series only)."""
    m = scored(df) & (df.h.to_numpy() <= days) & ~df.dead.to_numpy()
    g = pd.DataFrame({"series": df.series.to_numpy()[m], "y": df.y.to_numpy()[m], "f": pred[m]}).groupby("series").sum()
    return g


def _policy_inputs(c: Ctx):
    """Row-level forecasts for every policy, on the 4 validation windows and on the test window."""
    _, vf = _valid_frames(c)
    with open(c.out / "cache" / "screen_preds.pkl", "rb") as fh:
        screen = pickle.load(fh)
    with open(c.out / "cache" / "valid_final_preds.pkl", "rb") as fh:
        final_valid = pickle.load(fh)
    name = c.load("test")["model"]
    test = c.frame("test", [c.test0 - 1])
    tp = pd.read_csv(c.out / "test_predictions.csv.gz")
    ok = scored(test)

    def full(col):
        a = np.zeros(len(test))
        a[ok] = tp[col].to_numpy()
        return a

    pol = {
        f"{name} (final model)": (final_valid, full(name)),
        "Ridge (log scale)": (screen["Ridge (log scale)"], full("Ridge (log scale)")),
        "Same-weekday mean, last 4 weeks": (screen["Same-weekday mean, last 4 weeks"], full("Same-weekday mean, last 4 weeks")),
        "Rule of thumb: 28-day average": ([baseline_pred("base_mean28", v) for v in vf], baseline_pred("base_mean28", test)),
    }
    return vf, test, pol


def stage_decision(c: Ctx):
    """Order-up-to level S per series for an ordering cycle of 7 or 16 days, targeting a 95% cycle service level.
    Each policy: S = forecast total x a family buffer, the 95th percentile of (actual + 1) / (forecast + 1) over the 4
    validation windows. Buffers are fixed before the test window, which is then scored once."""
    sl = config.FAV_SERVICE_LEVEL
    fam = c.p.series.family.to_numpy()
    vf, test, pol = _policy_inputs(c)
    out = {}
    for days in (7, config.FAV_HORIZON):
        res = {}
        for pname, (vpreds, tpred) in pol.items():
            cal = pd.concat([_cycle_totals(v, pv, days) for v, pv in zip(vf, vpreds)])
            buf = ((cal.y + 1) / (cal.f + 1)).groupby(fam[cal.index]).quantile(sl)
            tt = _cycle_totals(test, tpred, days)
            S = (tt.f + 1) * pd.Series(fam[tt.index], index=tt.index).map(buf) - 1
            short = np.clip(tt.y - S, 0, None)

            def stock_at(target, S=S, tt=tt):
                lo, hi = 0.2, 5.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if np.mean(tt.y <= S.clip(lower=0) * mid) >= target else (mid, hi)
                return float((S.clip(lower=0) * hi).sum())

            def stock_at_fill(target, S=S, tt=tt):
                lo, hi = 0.2, 5.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    fill = 1 - np.clip(tt.y - S.clip(lower=0) * mid, 0, None).sum() / tt.y.sum()
                    lo, hi = (lo, mid) if fill >= target else (mid, hi)
                return float((S.clip(lower=0) * hi).sum())

            res[pname] = {"forecast_wape_of_totals": stats.wape(tt.y, tt.f), "cycle_service_level": float(np.mean(tt.y <= S)),
                          "stock_for_99pct_fill_ex_post": stock_at_fill(0.99), "stock_for_995pct_fill_ex_post": stock_at_fill(0.995),
                          "fill_rate": float(1 - short.sum() / tt.y.sum()), "stock_vs_demand": float(S.sum() / tt.y.sum()),
                          "stock_units": float(S.sum()), "short_units": float(short.sum()),
                          "stock_for_95pct_service_ex_post": stock_at(0.95), "stock_for_90pct_service_ex_post": stock_at(0.90),
                          "series": int(len(tt))}
        rule = res["Rule of thumb: 28-day average"]
        for r in res.values():
            r["stock_saving_vs_rule_at_95pct"] = 1 - r["stock_for_95pct_service_ex_post"] / rule["stock_for_95pct_service_ex_post"]
            r["stock_saving_vs_rule_at_99pct_fill"] = 1 - r["stock_for_99pct_fill_ex_post"] / rule["stock_for_99pct_fill_ex_post"]
            r["stock_saving_vs_rule_at_995pct_fill"] = 1 - r["stock_for_995pct_fill_ex_post"] / rule["stock_for_995pct_fill_ex_post"]
        out[f"{days}-day cycle"] = res
    c.save("decision", {"service_target": sl, "test_demand_units_16d": float(test.y[scored(test)].sum()), "cycles": out})


def stage_leakage(c: Ctx):
    """What would happen with information not available at the order date."""
    name, est, sub = final_estimator(c)
    train = c.frame("train_final", features.training_origins(c.p, c.test0))
    test = c.frame("test", [c.test0 - 1])
    ok = scored(test)
    y = test.y.to_numpy()[ok]
    out = {}
    pred, _ = fit_predict(est, sub, train, test)
    out["correct (information at the order date + planned promotions/holidays)"] = metrics(y, pred[ok])
    tx = c.p.transactions
    for df in (train, test):
        df["tx_target"] = np.log1p(tx[df.series.to_numpy(), df.target_idx.to_numpy()])
    cols = features.FEATURES + ["tx_target"]
    tr = train[features.trainable(train)]
    m = clone(est).fit(tr[cols], tr.y) if not isinstance(est, LogTarget) else LogTarget(clone(est.est)).fit(tr[cols], tr.y.to_numpy())
    pl = np.clip(m.predict(test[cols]), 0, None)
    pl[test.dead.to_numpy()] = 0
    out["leaky: adds the target day's store transactions (only known after the day)"] = metrics(y, pl[ok])
    c.save("leakage", out)


def hgb_explainer(c: Ctx):
    t = next(x for x in c.load("tune")["tuned"] if x["model"] == "Gradient boosting (log scale, squared error)")
    est, sub = candidates()[t["model"]]
    return t, with_params(est, t["params"]), sub


def stage_explain(c: Ctx, n_rows: int = 80, repeats: int = 3):
    """(1) Permutation importance of the final model on the test window (any model);
    (2) exact TreeSHAP on the tuned gradient-boosting model (log scale)."""
    name, est, sub = final_estimator(c)
    train = c.frame("train_final", features.training_origins(c.p, c.test0))
    test = c.frame("test", [c.test0 - 1])
    ok = scored(test) & ~test.dead.to_numpy()
    with open(c.out / "cache" / "final_model.pkl", "rb") as fh:
        final = pickle.load(fh)
    X, y = test[ok][features.FEATURES].copy(), test.y.to_numpy()[ok]
    base_w = stats.wape(y, np.clip(final.predict(X), 0, None))
    rng = np.random.default_rng(config.SEED)
    perm = {}
    for f in features.FEATURES:
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            inc.append(stats.wape(y, np.clip(final.predict(Xp), 0, None)) - base_w)
        perm[f] = float(np.mean(inc))
    perm = dict(sorted(perm.items(), key=lambda kv: -kv[1]))
    out = {"final_model": name, "permutation_importance_wape_increase": perm, "base_wape": base_w}
    t, hest, hsub = hgb_explainer(c)
    _, hmodel = fit_predict(hest, hsub, train, test)
    h = hmodel.est
    rows = test[ok].sample(min(n_rows, int(ok.sum())), random_state=config.SEED)
    Xs = rows[features.FEATURES].to_numpy(float)
    t0 = time.time()
    phi, base = treeshap.shap_values(h, Xs, progress=10)
    imp = pd.Series(np.abs(phi).mean(0), index=features.FEATURES).sort_values(ascending=False)
    pi = features.FEATURES.index("promo_target")
    np.savez_compressed(c.out / "shap_values.npz", phi=phi, X=Xs, base=base, features=np.array(features.FEATURES))
    top_perm, top_shap = list(perm)[:8], list(imp.index[:8])
    out.update({"shap_model": t["model"] + " (tuned)", "shap_rows": int(len(rows)), "shap_seconds": time.time() - t0, "shap_base_value": base,
                "shap_scale": "log1p(units)", "additivity_max_error": float(np.max(np.abs(base + phi.sum(1) - h._raw_predict(Xs).ravel()))),
                "mean_abs_shap": imp.round(4).to_dict(),
                "top8_overlap_permutation_vs_shap": sorted(set(top_perm) & set(top_shap)),
                "promo_shap_promoted_vs_not": {"promoted": float(phi[Xs[:, pi] > 0, pi].mean()) if (Xs[:, pi] > 0).any() else None,
                                               "not_promoted": float(phi[Xs[:, pi] == 0, pi].mean())}})
    c.save("explain", out)


def stage_hypotheses(c: Ctx):
    t, eda = c.load("test"), c.load("eda")
    name = t["model"]
    hyp = []
    for hid, b in (("H1", "Seasonal naive (same weekday, last week)"), ("H2", "Same-weekday mean, last 4 weeks"),
                   ("H3", "Mean of the last 14 days"), ("H4", "Ridge (log scale)")):
        if b not in t["table"]:
            continue
        r = t["table"][b]
        diff = r["wape_diff_vs_model_store_bootstrap"]
        st = r["paired_series_test_vs_model"]
        # The p-value is a Wilcoxon test over series, so the direction is read from the same test (share of series where the
        # baseline is worse). The aggregate WAPE difference is reported alongside; the two can disagree (large series dominate WAPE).
        hyp.append({"id": hid, "hypothesis": f"The model has lower error than {b} on most series",
                    "evidence": (f"{b} worse on {st['series_where_a_worse']:.0%} of series; aggregate WAPE {t['table'][name]['wape']:.3f} vs {r['wape']:.3f} "
                                 f"(difference {diff['value']:+.3f}, CI {diff['ci'][0]:+.3f} to {diff['ci'][1]:+.3f})"),
                    "p": st["p"], "direction_ok": st["series_where_a_worse"] > 0.5})
    pl = eda["promotion_lift_within_series"]
    hyp.append({"id": "H5", "hypothesis": "Sales are higher on promotion days (within series; association, not causal)",
                "evidence": f"median ratio {pl['median_ratio']:.2f}, {pl['share_positive']:.0%} of {pl['series']} series higher",
                "p": pl["wilcoxon_p"], "direction_ok": pl["median_log_diff"] > 0})
    pd_ = eda["payday_effect_within_series"]
    hyp.append({"id": "H6", "hypothesis": "Sales differ on paydays (15th and month end)",
                "evidence": f"median log difference {pd_['median_log_diff']:+.3f}, {pd_['share_positive']:.0%} of series higher",
                "p": pd_["wilcoxon_p"], "direction_ok": True})
    nh = eda["national_holiday_effect_within_series"]
    hyp.append({"id": "H7", "hypothesis": "Sales differ on national holidays",
                "evidence": f"median log difference {nh['median_log_diff']:+.3f}, {nh['share_positive']:.0%} of series higher",
                "p": nh["wilcoxon_p"], "direction_ok": True})
    adj = stats.holm({h["id"]: h["p"] for h in hyp})
    for h in hyp:
        h["p_holm"] = adj.get(h["id"], float("nan"))
        h["verdict"] = ("not testable" if not np.isfinite(h["p_holm"]) else
                        "supported" if h["p_holm"] < 0.05 and h["direction_ok"] else
                        "rejected (opposite direction)" if h["p_holm"] < 0.05 else "not supported")
    c.save("hypotheses", hyp)


def run(stages=STAGES, out: Path | None = None):
    c = Ctx(out or config.FAVORITA_RESULTS)
    for s in stages:
        t0 = time.time()
        globals()[f"stage_{s}"](c)
        print(f"[{s}] done in {time.time() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    run(args.stages, args.out)


if __name__ == "__main__":
    main()
