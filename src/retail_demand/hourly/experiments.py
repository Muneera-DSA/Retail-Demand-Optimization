"""Part C experiments: hourly demand -> week-ahead capacity plan.

    python -m retail_demand.hourly.experiments            # all stages, writes results/hourly/*.json

Stages
  quality      data-quality checks (quality.py)
  eda          seasonality, level, autocorrelation, shift
  screen       baselines + 11 model families, expanding-window CV inside the development period
  select       pre-specified rule: lowest CV MAE; small tuning grid if gradient boosting wins
  test         one evaluation on the held-out test period (Jul 2020 - Feb 2021)
  horizon      the same model a day ahead and an hour ahead: how much is predictable at all?
  aggregation  error per hour vs per 4-hour shift vs per day
  intervals    80% prediction intervals (quantile boosting, then conformal calibration)
  decision     capacity plan at the cost-optimal quantile vs plans built from baselines
  leakage      what shorter lags or a random split would (wrongly) claim
  missing      sensitivity: treat unrecorded hours as zero demand
  explain      exact TreeSHAP for the final model
  hypotheses   Holm-corrected tests
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from .. import config
from ..common import stats, treeshap
from . import data, features, models, quality

STAGES = ["quality", "eda", "screen", "select", "test", "horizon", "aggregation", "intervals", "decision",
          "leakage", "missing", "explain", "hypotheses"]
H_WEEK = 168


# --------------------------------------------------------------------------- setup

class Ctx:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.raw = data.load_raw()
        self.grid = data.to_grid(self.raw)
        self.F = features.build(self.grid)
        self.split = data.split(self.grid).to_numpy()
        usable = (self.F.status == "observed").to_numpy() & (self.F.ts >= self.F.ts.min() + pd.Timedelta(hours=H_WEEK)).to_numpy()
        self.dev = self.F[usable & (self.split == "dev")].reset_index(drop=True)
        self.test = self.F[usable & (self.split == "test")].reset_index(drop=True)
        self.prof = features.profile(self.dev)

    def save(self, name, obj):
        (self.out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_json))

    def load(self, name):
        return json.loads((self.out / f"{name}.json").read_text())


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def cv_folds(dev: pd.DataFrame, n_folds=None, weeks=None):
    """Expanding window: each validation block is `weeks` weeks; training is everything
    before it. The last block ends where the development period ends."""
    n_folds, weeks = n_folds or config.HOURLY_CV_FOLDS, weeks or config.HOURLY_CV_FOLD_WEEKS
    end = dev.ts.max() + pd.Timedelta(hours=1)
    out = []
    for k in range(n_folds, 0, -1):
        v0 = end - pd.Timedelta(weeks=weeks * k)
        v1 = v0 + pd.Timedelta(weeks=weeks)
        tr = np.flatnonzero((dev.ts < v0).to_numpy())
        va = np.flatnonzero(((dev.ts >= v0) & (dev.ts < v1)).to_numpy())
        out.append((tr, va))
    return out


def predict_any(name, est, train, X, prof=None):
    """Fit-and-predict for a candidate (estimator) or a baseline (name in BASELINES)."""
    if name in models.BASELINES:
        return models.baseline(name, X, features.profile(train) if prof is None else prof)
    m = clone(est).fit(train[features.FEATURES], train.demand)
    return np.clip(m.predict(X[features.FEATURES]), 0, None)


# -------------------------------------------------------------------------- stages

def stage_quality(c: Ctx):
    c.save("quality", quality.run(c.raw, c.grid))


def stage_eda(c: Ctx):
    obs = c.grid[c.grid.status == "observed"].copy()
    daily = obs.groupby("date").demand.agg(["sum", "mean", "size"])
    full = c.grid.set_index("ts").demand
    heat = obs.groupby(["dow", "hour"]).demand.mean().unstack().round(1)
    wk = obs.assign(weekend=obs.dow >= 5).groupby("date").agg(mean=("demand", "mean"), weekend=("weekend", "first"))
    from scipy import stats as sps
    mw = sps.mannwhitneyu(wk[wk.weekend]["mean"], wk[~wk.weekend]["mean"])
    c.save("eda", {
        "summary": obs.demand.describe().round(2).to_dict(),
        "skew": float(obs.demand.skew()), "variance_to_mean": float(obs.demand.var() / obs.demand.mean()),
        "by_hour": obs.groupby("hour").demand.mean().round(1).to_dict(),
        "by_dow": obs.groupby("dow").demand.mean().round(1).to_dict(),
        "by_month": obs.groupby(obs.ts.dt.month).demand.mean().round(1).to_dict(),
        "by_quarter": {str(k): v for k, v in obs.groupby(obs.ts.dt.to_period("Q")).demand.mean().round(1).items()},
        "heatmap_dow_hour": heat.to_dict(),
        "daily_mean_series": {str(k.date()): round(v, 2) for k, v in daily["mean"].items()},
        "autocorrelation_hourly": {k: float(full.autocorr(k)) for k in (1, 2, 3, 24, 48, 168, 336)},
        "autocorrelation_daily_mean": {k: float(daily["mean"].reindex(pd.date_range(daily.index.min(), daily.index.max())).autocorr(k))
                                       for k in (1, 7, 14, 28)},
        "weekend_vs_weekday": {"weekend_daily_mean": float(wk[wk.weekend]["mean"].mean()),
                               "weekday_daily_mean": float(wk[~wk.weekend]["mean"].mean()),
                               "ratio": float(wk[wk.weekend]["mean"].mean() / wk[~wk.weekend]["mean"].mean()),
                               "mannwhitney_p": float(mw.pvalue),
                               "cliffs_delta": stats.cliffs_delta(wk[wk.weekend]["mean"], wk[~wk.weekend]["mean"])},
        "share_of_variance_explained_by_hour_of_week": _r2_profile(obs),
    })


def _r2_profile(obs):
    fit = obs.groupby("how").demand.transform("mean")
    return float(1 - ((obs.demand - fit) ** 2).sum() / ((obs.demand - obs.demand.mean()) ** 2).sum())


def stage_screen(c: Ctx):
    folds = cv_folds(c.dev)
    cands = {**{b: None for b in models.BASELINES}, **models.candidates()}
    rows, oof = [], {}
    for name, est in cands.items():
        t0 = time.time()
        fold_mae, preds = [], np.full(len(c.dev), np.nan)
        for tr, va in folds:
            train, val = c.dev.iloc[tr], c.dev.iloc[va]
            p = predict_any(name, est, train, val)
            preds[va] = p
            fold_mae.append(stats.mae(val.demand, p))
        oof[name] = preds
        va_all = np.concatenate([va for _, va in folds])
        rows.append({"model": name, "baseline": name in models.BASELINES, "cv_mae": float(np.mean(fold_mae)),
                     "cv_mae_sd": float(np.std(fold_mae, ddof=1)), "fold_mae": fold_mae,
                     "cv_wape": stats.wape(c.dev.demand.iloc[va_all], preds[va_all]),
                     "cv_rmse": stats.rmse(c.dev.demand.iloc[va_all], preds[va_all]), "seconds": time.time() - t0})
        print(f"[screen] {name}: CV MAE {np.mean(fold_mae):.2f}")
    va_all = np.concatenate([va for _, va in folds])
    best = min(rows, key=lambda r: r["cv_mae"])["model"]
    y = c.dev.demand.to_numpy()[va_all]
    for r in rows:
        if r["model"] == best:
            continue
        dm = stats.diebold_mariano(y - oof[r["model"]][va_all], y - oof[best][va_all], h=H_WEEK)
        r["dm_vs_best"] = dm
    adj = stats.holm({r["model"]: r["dm_vs_best"]["p"] for r in rows if "dm_vs_best" in r})
    for r in rows:
        if r["model"] in adj:
            r["dm_vs_best"]["p_holm"] = adj[r["model"]]
    rows.sort(key=lambda r: r["cv_mae"])
    c.save("screen", {"best": best, "folds": [{"train_hours": int(len(tr)), "val_hours": int(len(va)),
                                                "val_start": str(c.dev.ts.iloc[va].min()), "val_end": str(c.dev.ts.iloc[va].max())}
                                               for tr, va in folds], "table": rows})


def stage_select(c: Ctx, top: int = 3):
    """Tune the `top` model families of the screen on the same CV folds; pick the lowest tuned CV MAE.
    If no model beats the best baseline by more than one CV standard error, that is reported."""
    sc = c.load("screen")
    rows = [r for r in sc["table"] if not r["baseline"]]
    folds = cv_folds(c.dev)
    tuned = []
    to_tune = [r for r in rows[:top]]
    if not any(r["model"] == "Gradient boosting (Poisson)" for r in to_tune):     # always tuned: used for quantiles and SHAP
        to_tune.append(next(r for r in rows if r["model"] == "Gradient boosting (Poisson)"))
    for r in to_tune:
        name = r["model"]
        best = {"model": name, "params": {}, "cv_mae": r["cv_mae"], "fold_mae": r["fold_mae"]}
        for params in models.tune_grid(name):
            e = clone(models.candidates()[name]).set_params(**params)
            maes = [stats.mae(c.dev.demand.iloc[va], predict_any(name, e, c.dev.iloc[tr], c.dev.iloc[va])) for tr, va in folds]
            print(f"[select] {name} {params}: {np.mean(maes):.3f}")
            if np.mean(maes) < best["cv_mae"]:
                best = {"model": name, "params": params, "cv_mae": float(np.mean(maes)), "fold_mae": maes}
        tuned.append(best)
    chosen = min(tuned, key=lambda t: t["cv_mae"])
    base = min((r for r in sc["table"] if r["baseline"]), key=lambda r: r["cv_mae"])
    se = float(np.std(np.array(chosen["fold_mae"]) - np.array(base["fold_mae"]), ddof=1) / np.sqrt(len(folds)))
    gain = base["cv_mae"] - chosen["cv_mae"]
    c.save("select", {"model": chosen["model"], "params": chosen["params"], "cv_mae": chosen["cv_mae"], "tuned": tuned,
                      "best_baseline": base["model"], "best_baseline_cv_mae": base["cv_mae"],
                      "cv_gain_vs_best_baseline": gain, "cv_gain_se": se,
                      "recommended_point_forecast": chosen["model"] if gain > se else base["model"],
                      "rule": ("the 3 best model families (plus Poisson gradient boosting) are tuned on the same CV folds; the lowest "
                               "tuned CV MAE is the candidate. One-standard-error rule: if it beats the best baseline by less than one "
                               "standard error of the fold differences, the simpler baseline is recommended for deployment")})


def final_estimator(c: Ctx):
    sel = c.load("select")
    return sel["model"], clone(models.candidates()[sel["model"]]).set_params(**sel["params"])


def hgb_reference(c: Ctx):
    """Gradient boosting with its tuned settings (for intervals, horizons and SHAP even when a
    linear model is the point forecaster)."""
    sel = c.load("select")
    t = next(t for t in sel["tuned"] if t["model"] == "Gradient boosting (Poisson)")
    return clone(models.candidates()[t["model"]]).set_params(**t["params"])


def stage_test(c: Ctx):
    name, est = final_estimator(c)
    model = est.fit(c.dev[features.FEATURES], c.dev.demand)
    y = c.test.demand.to_numpy()
    preds = {name: np.clip(model.predict(c.test[features.FEATURES]), 0, None)}
    for b in models.BASELINES:
        preds[b] = models.baseline(b, c.test, c.prof)
    ridge = "Ridge (hour-of-week dummies + history)"
    if ridge != name:
        preds[ridge] = predict_any(ridge, models.candidates()[ridge], c.dev, c.test)
    hgb = "Gradient boosting (Poisson), tuned"
    preds[hgb] = np.clip(hgb_reference(c).fit(c.dev[features.FEATURES], c.dev.demand).predict(c.test[features.FEATURES]), 0, None)
    scale = stats.mae(c.dev.demand, models.baseline(models.BASELINES[0], c.dev, c.prof))    # in-sample seasonal-naive MAE
    table = {}
    for k, p in preds.items():
        m = stats.point_metrics(y, p, scale)
        m["mae_ci"] = stats.block_bootstrap(lambda i, p=p: stats.mae(y[i], p[i]), len(y), H_WEEK, 1000, config.SEED)["ci"]
        if k != name:
            m["dm_vs_model"] = stats.diebold_mariano(y - p, y - preds[name], h=H_WEEK)
            m["mae_diff_vs_model_ci"] = stats.block_bootstrap(lambda i, p=p: stats.mae(y[i], p[i]) - stats.mae(y[i], preds[name][i]),
                                                              len(y), H_WEEK, 1000, config.SEED)
        table[k] = m
    err = c.test.assign(pred=preds[name], abs_err=np.abs(y - preds[name]))
    pd.DataFrame({"ts": c.test.ts.astype(str), "demand": y, **{k: np.round(v, 2) for k, v in preds.items()}}).to_csv(
        c.out / "test_predictions.csv.gz", index=False)
    resid = pd.Series(y - preds[name], index=c.test.ts).reindex(pd.date_range(c.test.ts.min(), c.test.ts.max(), freq="h"))
    c.save("test", {"model": name, "test_start": str(c.test.ts.min()), "test_end": str(c.test.ts.max()), "n": int(len(y)),
                    "mase_scale_in_sample_seasonal_naive_mae": scale, "table": table,
                    "mae_by_hour": err.groupby("hour").abs_err.mean().round(2).to_dict(),
                    "mae_by_month": {str(k): v for k, v in err.groupby(err.ts.dt.to_period("M")).abs_err.mean().round(2).items()},
                    "actual_by_month": {str(k): v for k, v in err.groupby(err.ts.dt.to_period("M")).demand.mean().round(2).items()},
                    "residual_autocorrelation": {k: float(resid.autocorr(k)) for k in (1, 24, 168)}})


def stage_horizon(c: Ctx):
    """Same model family with the information a day ahead / an hour ahead would allow."""
    est = hgb_reference(c)
    F = features.build(c.grid, extra_lags=(1, 2, 3, 24, 48, 72, 96, 120, 144))
    s = F.demand
    F["level_24h_from_1"] = s.shift(1).rolling(24, min_periods=6).mean()
    F["level_24h_from_24"] = s.shift(24).rolling(24, min_periods=6).mean()
    usable = (F.status == "observed").to_numpy() & (F.ts >= F.ts.min() + pd.Timedelta(hours=H_WEEK)).to_numpy()
    dev, test = F[usable & (c.split == "dev")], F[usable & (c.split == "test")]
    sets = {"week ahead (168 h)": [], "day ahead (24 h)": ["lag24", "lag48", "lag72", "lag96", "lag120", "lag144", "level_24h_from_24"],
            "hour ahead (1 h)": ["lag1", "lag2", "lag3", "lag24", "lag48", "level_24h_from_1"]}
    out, preds = {}, {}
    for h, extra in sets.items():
        cols = features.FEATURES + extra
        m = clone(est).fit(dev[cols], dev.demand)
        preds[h] = np.clip(m.predict(test[cols]), 0, None)
        out[h] = stats.point_metrics(test.demand, preds[h])
    y = test.demand.to_numpy()
    out["day_vs_week_dm"] = stats.diebold_mariano(y - preds["week ahead (168 h)"], y - preds["day ahead (24 h)"], h=24)
    c.save("horizon", out)


def stage_aggregation(c: Ctx):
    """Planning is done per shift; errors partly cancel when hours are added up."""
    t = c.load("test")
    p = pd.read_csv(c.out / "test_predictions.csv.gz", parse_dates=["ts"])
    name = t["model"]
    out = {}
    for label, key in (("hour", p.ts.dt.floor("h")), ("4-hour shift", p.ts.dt.floor("4h")), ("day", p.ts.dt.floor("D"))):
        g = p.groupby(key)[["demand", name, models.BASELINES[0], models.BASELINES[2]]].sum()
        out[label] = {k: stats.point_metrics(g.demand, g[k]) for k in (name, models.BASELINES[0], models.BASELINES[2])}
    c.save("aggregation", out)


def _quantile_fit(c: Ctx, alpha, train):
    """Quantile gradient boosting with the tree settings tuned for the Poisson reference model."""
    keep = ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")
    params = {k: v for k, v in hgb_reference(c).get_params().items() if k in keep}
    return models.quantile_model(alpha, params).fit(train[features.FEATURES], train.demand)


def stage_intervals(c: Ctx):
    lo_a, hi_a = config.HOURLY_INTERVAL
    nominal = hi_a - lo_a
    y = c.test.demand.to_numpy()
    lo = _quantile_fit(c, lo_a, c.dev).predict(c.test[features.FEATURES])
    hi = _quantile_fit(c, hi_a, c.dev).predict(c.test[features.FEATURES])
    raw_cov = stats.coverage_test(y, lo, hi, nominal)
    # conformalised quantile regression (Romano et al. 2019): calibrate on the last 8 dev weeks
    cut = c.dev.ts.max() - pd.Timedelta(weeks=8)
    fit_part, cal = c.dev[c.dev.ts <= cut], c.dev[c.dev.ts > cut]
    lo_c = _quantile_fit(c, lo_a, fit_part).predict(cal[features.FEATURES])
    hi_c = _quantile_fit(c, hi_a, fit_part).predict(cal[features.FEATURES])
    scores = np.maximum(lo_c - cal.demand.to_numpy(), cal.demand.to_numpy() - hi_c)
    n = len(scores)
    qhat = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * nominal) / n)))
    conf = stats.coverage_test(y, lo - qhat, hi + qhat, nominal)
    by_hour = pd.DataFrame({"hour": c.test.hour, "in": (y >= lo - qhat) & (y <= hi + qhat)}).groupby("hour")["in"].mean().round(3).to_dict()
    crit = config.HOURLY_UNDER_COST / (config.HOURLY_UNDER_COST + config.HOURLY_OVER_COST)
    q75 = _quantile_fit(c, crit, c.dev).predict(c.test[features.FEATURES])
    pd.DataFrame({"ts": c.test.ts.astype(str), "lo": lo.round(2), "hi": hi.round(2), "q75": q75.round(2)}).to_csv(c.out / "test_intervals.csv.gz", index=False)
    c.save("intervals", {"nominal": nominal, "quantile_boosting": raw_cov, "conformal": {**conf, "adjustment": qhat},
                         "conformal_coverage_by_hour": by_hour, "mean_width_raw": float(np.mean(hi - lo)),
                         "mean_width_conformal": float(np.mean(hi - lo + 2 * qhat))})


def _plan_cost(y, plan, cu=config.HOURLY_UNDER_COST, co=config.HOURLY_OVER_COST):
    under = np.clip(y - plan, 0, None)
    over = np.clip(plan - y, 0, None)
    return {"cost": float(cu * under.sum() + co * over.sum()), "service_level": float(np.mean(y <= plan)),
            "fill_rate": float(1 - under.sum() / y.sum()), "idle_capacity_share": float(over.sum() / plan.sum()),
            "unmet_units": float(under.sum()), "idle_units": float(over.sum()), "planned_units": float(plan.sum())}


def stage_decision(c: Ctx):
    """Capacity plan per hour. With under-capacity costing 3x idle capacity, the cost-minimising
    plan is the 75th percentile of demand (newsvendor critical ratio 3/(3+1))."""
    cu, co = config.HOURLY_UNDER_COST, config.HOURLY_OVER_COST
    q = cu / (cu + co)
    y = c.test.demand.to_numpy()
    plans = {"Model: 75th-percentile forecast": _quantile_fit(c, q, c.dev).predict(c.test[features.FEATURES])}
    # baselines get a fair chance: a multiplicative buffer tuned on development data for the same cost
    for b in models.BASELINES[:3]:
        base_dev = models.baseline(b, c.dev, c.prof)
        grid = np.linspace(0.8, 2.0, 121)
        k = grid[np.argmin([_plan_cost(c.dev.demand.to_numpy(), base_dev * g)["cost"] for g in grid])]
        plans[f"{b} x {k:.2f} (buffer tuned on dev)"] = models.baseline(b, c.test, c.prof) * k
    q_prof = c.dev.groupby("how").demand.quantile(q)
    plans["Empirical 75th percentile of each hour of week (dev)"] = c.test.how.map(q_prof).to_numpy(float)
    name, est = final_estimator(c)
    med = np.clip(est.fit(c.dev[features.FEATURES], c.dev.demand).predict(c.test[features.FEATURES]), 0, None)
    plans[f"{name}: point forecast, no buffer"] = med
    res = {k: _plan_cost(y, p) for k, p in plans.items()}
    ref = "Model: 75th-percentile forecast"
    for k, p in plans.items():
        if k == ref:
            continue
        res[k]["cost_saving_of_model_vs_this"] = stats.block_bootstrap(
            lambda i, p=p: 1 - (_plan_cost(y[i], plans[ref][i])["cost"] / _plan_cost(y[i], p[i])["cost"]), len(y), H_WEEK, 500, config.SEED)
    c.save("decision", {"critical_ratio": q, "under_cost": cu, "over_cost": co, "plans": res})


def stage_leakage(c: Ctx):
    """Gradient boosting (it uses every column it is given) with correct and leaky designs."""
    est = hgb_reference(c)
    y = c.test.demand.to_numpy()
    out = {}
    m = clone(est).fit(c.dev[features.FEATURES], c.dev.demand)
    out["correct: week-ahead features, time split"] = stats.mae(y, m.predict(c.test[features.FEATURES]))
    F = features.build(c.grid, extra_lags=(1, 24))
    usable = (F.status == "observed").to_numpy() & (F.ts >= F.ts.min() + pd.Timedelta(hours=H_WEEK)).to_numpy()
    dev, test = F[usable & (c.split == "dev")], F[usable & (c.split == "test")]
    cols = features.FEATURES + ["lag1", "lag24"]
    m2 = clone(est).fit(dev[cols], dev.demand)
    out["leaky: adds last hour and same hour yesterday (not known a week ahead)"] = stats.mae(test.demand, m2.predict(test[cols]))
    # random split: shuffled 5-fold CV on all data mixes future and past
    from sklearn.model_selection import KFold
    Fu = F[usable].reset_index(drop=True)
    maes = []
    for tr, va in KFold(5, shuffle=True, random_state=config.SEED).split(Fu):
        mm = clone(est).fit(Fu.iloc[tr][cols], Fu.iloc[tr].demand)
        maes.append(stats.mae(Fu.iloc[va].demand, mm.predict(Fu.iloc[va][cols])))
    out["leaky: random 5-fold split with the short lags"] = float(np.mean(maes))
    c.save("leakage", {k: float(v) for k, v in out.items()})


def stage_missing(c: Ctx):
    """Sensitivity: if unrecorded hours (not the outage) were really zero demand."""
    name, est = final_estimator(c)
    g0 = c.grid.copy()
    g0.loc[g0.status == "gap", "demand"] = 0.0
    g0.loc[g0.status == "gap", "status"] = "observed"
    F0 = features.build(g0)
    usable = (F0.status == "observed").to_numpy() & (F0.ts >= F0.ts.min() + pd.Timedelta(hours=H_WEEK)).to_numpy()
    dev0, test0 = F0[usable & (c.split == "dev")], F0[usable & (c.split == "test")]
    m0 = clone(est).fit(dev0[features.FEATURES], dev0.demand)
    p0 = np.clip(m0.predict(test0[features.FEATURES]), 0, None)
    rec = (c.grid.status.to_numpy()[usable & (c.split == "test")] == "observed")
    base = clone(est).fit(c.dev[features.FEATURES], c.dev.demand)
    p_base = np.clip(base.predict(c.test[features.FEATURES]), 0, None)
    c.save("missing", {
        "test_mae_recorded_hours_main": stats.mae(c.test.demand, p_base),
        "test_mae_recorded_hours_if_gaps_zero": stats.mae(test0.demand[rec], p0[rec]),
        "mean_forecast_recorded_hours_main": float(p_base.mean()), "mean_forecast_recorded_hours_if_gaps_zero": float(p0[rec].mean()),
        "test_mae_all_non_outage_hours_if_gaps_zero": stats.mae(test0.demand, p0),
        "share_test_hours_unrecorded": float(1 - rec.mean()),
    })


def stage_explain(c: Ctx, n_rows: int = 300):
    name, est = "Gradient boosting (Poisson), tuned", hgb_reference(c)
    model = est.fit(c.dev[features.FEATURES], c.dev.demand)
    rows = c.test.sample(min(n_rows, len(c.test)), random_state=config.SEED)
    X = rows[features.FEATURES].to_numpy(float)
    t0 = time.time()
    phi, base = treeshap.shap_values(model, X)
    raw = model._raw_predict(X).ravel()
    imp = pd.Series(np.abs(phi).mean(0), index=features.FEATURES).sort_values(ascending=False)
    hour_effect = pd.DataFrame({"hour": rows.hour.to_numpy(), "phi": phi[:, features.FEATURES.index("hour")]}).groupby("hour").phi.mean()
    np.savez_compressed(c.out / "shap_values.npz", phi=phi, X=X, base=base, features=np.array(features.FEATURES))
    c.save("explain", {"model": name, "n_rows": int(len(X)), "base_value": base, "seconds": time.time() - t0,
                       "additivity_max_error": float(np.max(np.abs(base + phi.sum(1) - raw))),
                       "mean_abs_shap": imp.round(3).to_dict(), "mean_shap_of_hour_by_hour": hour_effect.round(2).to_dict()})


def stage_hypotheses(c: Ctx):
    t, eda, q, iv, hz = c.load("test"), c.load("eda"), c.load("quality"), c.load("intervals"), c.load("horizon")
    name = t["model"]
    hyp = []
    for hid, b, text in (("H1", models.BASELINES[0], "The model beats the seasonal-naive forecast (same hour last week)"),
                         ("H2", models.BASELINES[1], "The model beats the 4-week hour-of-week mean"),
                         ("H3", models.BASELINES[2], "The model beats the static hour-of-week profile")):
        r = t["table"][b]
        hyp.append({"id": hid, "hypothesis": text, "evidence": f"test MAE {t['table'][name]['mae']:.2f} vs {r['mae']:.2f}",
                    "p": r["dm_vs_model"]["p"], "direction_ok": r["dm_vs_model"]["mean_loss_diff"] > 0})
    ridge = "Ridge (hour-of-week dummies + history)"
    if ridge in t["table"]:
        r = t["table"][ridge]
        hyp.append({"id": "H4", "hypothesis": "A non-linear model beats a linear model with the same features",
                    "evidence": f"test MAE {t['table'][name]['mae']:.2f} vs {r['mae']:.2f}", "p": r["dm_vs_model"]["p"],
                    "direction_ok": r["dm_vs_model"]["mean_loss_diff"] > 0})
    w = eda["weekend_vs_weekday"]
    hyp.append({"id": "H5", "hypothesis": "Weekend demand differs from weekday demand",
                "evidence": f"daily mean ratio {w['ratio']:.2f}, Cliff's delta {w['cliffs_delta']:+.2f}",
                "p": w["mannwhitney_p"], "direction_ok": True})
    m = q["missingness"]
    hyp.append({"id": "H6", "hypothesis": "Unrecorded hours are not missing at random (they fall on quieter days)",
                "evidence": f"afternoon demand {m['peak_mean_days_with_night_gap']:.0f} vs {m['peak_mean_days_without']:.0f}",
                "p": m["mannwhitney_p"], "direction_ok": m["cliffs_delta"] < 0})
    cv = iv["quantile_boosting"]
    hyp.append({"id": "H7", "hypothesis": "The 80% prediction intervals miss their nominal coverage (two-sided binomial)",
                "evidence": f"coverage {cv['coverage']:.1%}", "p": cv["p_binomial"], "direction_ok": True})
    d = hz["day_vs_week_dm"]
    hyp.append({"id": "H8", "hypothesis": "A day-ahead update improves on the week-ahead plan",
                "evidence": f"MAE {hz['day ahead (24 h)']['mae']:.2f} vs {hz['week ahead (168 h)']['mae']:.2f}",
                "p": d["p"], "direction_ok": d["mean_loss_diff"] > 0})
    adj = stats.holm({h["id"]: h["p"] for h in hyp})
    for h in hyp:
        h["p_holm"] = adj[h["id"]]
        h["verdict"] = ("supported" if h["p_holm"] < 0.05 and h["direction_ok"] else
                        "rejected (opposite direction)" if h["p_holm"] < 0.05 else "not supported")
    c.save("hypotheses", hyp)


def run(stages=STAGES, out: Path | None = None):
    c = Ctx(out or config.HOURLY_RESULTS)
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
