"""Classical time-series benchmark: ARIMA, SARIMA, Holt-Winters (ETS) and Prophet.

    python -m retail_demand.classical --part hourly favorita      # -> results/<part>/classical.json
    python -m retail_demand.classical --report                    # -> reports/classical.md + figure

The question: do the statistical forecasting models a planner would reach for first match the
machine-learning models of Parts B and C on the same held-out data, under the same information rules?

Part C (hourly, week-ahead). Every forecast for hour t may use data up to t - 168 h only, the rule the
ML features follow. For ARIMA/SARIMA and Holt-Winters this is applied exactly: parameters are
estimated once on the training data, the model is then run forward over the whole series with those
parameters fixed, and the forecast for t is the 168-step forecast from the state at t - 168. Prophet
has no state to run forward, so it is refitted every week on data recorded at least 168 h before
every hour of the week it forecasts (a slightly stricter rule than the others get).
  - ARIMA: ARIMA(p,d,q) errors around weekly Fourier terms (dynamic harmonic regression, K = 10;
    harmonics 7, 14 and 21 are the daily cycle). The usual way to handle a 168-hour season.
  - SARIMA: the same plus a seasonal ARMA at lag 24 (daily).
  - Holt-Winters: additive seasonality with period 168; unrecorded hours are filled with the
    training hour-of-week profile, because the recursion cannot skip them.
  - Prophet: daily, weekly and yearly seasonality.
Two specifications per family are compared on the last 8 development weeks (the validation weeks of
the ML screen); the better one is refitted on all development data and evaluated once on the test
period (Jul 2020 - Feb 2021, the same 4,525 recorded hours as the ML models).

Part B (Favorita, 16-day replenishment). One model per store x family series, fitted on log1p(sales)
from August 2015 (or the store's opening) up to the order date and forecasting the next 16 days.
Series with no sales in the last 56 days are forecast as zero, as for every other method.
  - ARIMA(p,1,q) (no seasonal terms: the textbook benchmark, expected to miss the weekly cycle)
  - SARIMA(p,d,q)(0,1,1)7
  - SARIMAX: SARIMA plus the two inputs known in advance (promotions, national holidays), compared
    with the SARIMA specifications on validation as a third candidate of the SARIMA family
  - Holt-Winters: additive weekly seasonality, with or without a damped trend
  - Prophet: weekly + yearly seasonality, national holidays, promotions as a regressor
The specification of each family is chosen on the last validation window (2017-07-15 .. 07-30) over
a fixed random sample of 300 active series; the chosen one is then fitted on every series for the test
window (2017-07-31 .. 08-15), scored on exactly the rows the ML models were scored on.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .common import stats

H_WEEK = 168
K_WEEK = 10
BUFFER_GRID = np.linspace(0.8, 2.0, 121)

HOURLY_SPECS = {
    "ARIMA": {"ARIMA(2,0,1) + weekly Fourier": dict(order=(2, 0, 1)),
              "ARIMA(2,1,1) + weekly Fourier": dict(order=(2, 1, 1))},
    "SARIMA": {"SARIMA(1,0,1)(1,0,1)24 + weekly Fourier": dict(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)),
               "SARIMA(2,0,0)(2,0,0)24 + weekly Fourier": dict(order=(2, 0, 0), seasonal_order=(2, 0, 0, 24))},
    "Holt-Winters": {"Holt-Winters, additive season 168, no trend": dict(trend=None),
                     "Holt-Winters, additive season 168, damped trend": dict(trend="add", damped_trend=True)},
    "Prophet": {"Prophet, additive seasonality": dict(seasonality_mode="additive"),
                "Prophet, multiplicative seasonality": dict(seasonality_mode="multiplicative")},
}

FAV_SPECS = {
    "ARIMA": {"ARIMA(1,1,1)": dict(order=(1, 1, 1)), "ARIMA(2,1,2)": dict(order=(2, 1, 2))},
    "SARIMA": {"SARIMA(1,0,1)(0,1,1)7": dict(order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)),
               "SARIMA(1,1,1)(0,1,1)7": dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)),
               "SARIMAX(1,0,1)(0,1,1)7 + promotions, national holidays": dict(order=(1, 0, 1), seasonal_order=(0, 1, 1, 7), exog=True)},
    "Holt-Winters": {"Holt-Winters (ETS A,N,A), weekly": dict(trend=None),
                     "Holt-Winters (ETS A,Ad,A), weekly": dict(trend="add", damped_trend=True)},
    "Prophet": {"Prophet, weekly + yearly + holidays + promotions": dict(yearly=True),
                "Prophet, weekly + holidays + promotions": dict(yearly=False)},
}
FAV_SAMPLE = 300
FAV_MIN_HISTORY = 56


def _quiet():
    warnings.filterwarnings("ignore")
    for name in ("cmdstanpy", "prophet", "prophet.plot"):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).disabled = True


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


# ============================================================================ Part C: hourly

def fourier(how: np.ndarray, K: int = K_WEEK) -> np.ndarray:
    """sin/cos pairs of the hour of week (period 168) plus a constant column."""
    a = 2 * np.pi * np.outer(np.asarray(how, float), np.arange(1, K + 1)) / H_WEEK
    return np.column_stack([np.ones(len(a)), np.sin(a), np.cos(a)])


def sarimax_week_ahead(y: np.ndarray, X: np.ndarray, spec: dict, fit_end: int, H: int = H_WEEK):
    """H-step forecasts for every t, each from the filtered state at t - H.

    Parameters are estimated on y[:fit_end] only. The model is then filtered over the whole series
    with the parameters fixed; the filtered state at s uses y[:s+1] only, so the forecast for
    t = s + H uses nothing recorded after t - H. Missing hours (NaN) are skipped by the Kalman filter.
    Regression on X (Fourier terms + constant) enters the observation equation, so
    yhat[s+H] = X[s+H] @ beta + Z T^H a[s|s]."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    from statsmodels.tsa.statespace import kalman_filter as kf

    res = SARIMAX(y[:fit_end], exog=X[:fit_end], trend="n", **spec).fit(disp=False, maxiter=200, low_memory=True)
    mod = SARIMAX(y, exog=X, trend="n", **spec)
    # keep only the filtered states: the per-hour covariance matrices would need several GB for long seasonal models
    keep_states = (kf.MEMORY_NO_FORECAST | kf.MEMORY_NO_PREDICTED | kf.MEMORY_NO_FILTERED_COV | kf.MEMORY_NO_LIKELIHOOD
                   | kf.MEMORY_NO_GAIN | kf.MEMORY_NO_SMOOTHING | kf.MEMORY_NO_STD_FORECAST)
    mod.ssm.set_conserve_memory(keep_states)
    mod.update(res.params)
    filt = mod.ssm.filter()

    class _Full:                                                        # the attributes used below
        params, param_names, model, filter_results = res.params, res.param_names, mod, filt
    full = _Full
    ssm = full.model.ssm
    Z, T = ssm["design"][:, :, 0] if ssm["design"].ndim == 3 else ssm["design"], ssm["transition"]
    T = T[:, :, 0] if T.ndim == 3 else T
    Z = Z if Z.ndim == 2 else Z[None, :]
    M = Z @ np.linalg.matrix_power(T, H)                                # (1, k_states)
    a = full.filter_results.filtered_state                             # (k_states, n)
    beta = full.params[[f"x{i + 1}" for i in range(X.shape[1])]].to_numpy() if hasattr(full.params, "index") else \
        full.params[:X.shape[1]]
    out = np.full(len(y), np.nan)
    out[H:] = X[H:] @ beta + (M @ a[:, :-H]).ravel()
    return out, {"params": {k: float(v) for k, v in pd.Series(full.params, index=full.param_names).items()},
                 "aic": float(res.aic), "converged": bool(res.mle_retvals.get("converged", True))}


def holt_winters_week_ahead(yfill: np.ndarray, spec: dict, fit_end: int, H: int = H_WEEK, m: int = H_WEEK):
    """Holt-Winters with parameters estimated on yfill[:fit_end], then run over the whole series with
    them fixed; the forecast for t is the H-step forecast from the states at t - H."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    kw = dict(seasonal="add", seasonal_periods=m, **spec)
    fit = ExponentialSmoothing(yfill[:fit_end], initialization_method="estimated", **kw).fit()
    p = fit.params
    known = dict(initialization_method="known", initial_level=p["initial_level"], initial_seasonal=p["initial_seasons"])
    if spec.get("trend"):
        known["initial_trend"] = p["initial_trend"]
    fixed = {"smoothing_level": p["smoothing_level"], "smoothing_seasonal": p["smoothing_seasonal"], "optimized": False}
    if spec.get("trend"):
        fixed["smoothing_trend"] = p["smoothing_trend"]
        if spec.get("damped_trend"):
            fixed["damping_trend"] = p["damping_trend"]
    full = ExponentialSmoothing(yfill, **kw, **known).fit(**fixed)
    level, season = np.asarray(full.level), np.asarray(full.season)
    trend_part = 0.0
    if spec.get("trend"):
        phi = p["damping_trend"] if spec.get("damped_trend") else 1.0
        trend_part = np.asarray(full.trend) * (sum(phi ** i for i in range(1, H + 1)))
    k = (H - 1) // m + 1
    fc = level + trend_part                                             # forecast made at s for s + H
    out = np.full(len(yfill), np.nan)
    s = np.arange(len(yfill) - H)
    out[H:] = fc[s] + season[s + H - m * k]
    return out, {"params": {k_: float(v) for k_, v in p.items() if v is not None and np.ndim(v) == 0 and np.isfinite(v)}, "sse": float(fit.sse)}


def prophet_weekly_refit(ts: pd.Series, y: np.ndarray, origins: list[pd.Timestamp], spec: dict, H: int = H_WEEK):
    """Prophet refitted at each weekly origin on data recorded at least H hours before every hour of
    the week it forecasts (data up to origin - H)."""
    from prophet import Prophet

    _quiet()
    df = pd.DataFrame({"ds": ts.to_numpy(), "y": y})
    out = pd.Series(np.nan, index=pd.DatetimeIndex(ts))
    for o in origins:
        hist = df[(df.ds <= o - pd.Timedelta(hours=H)) & df.y.notna()]
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, uncertainty_samples=0, **spec)
        m.fit(hist)
        fut = pd.DataFrame({"ds": pd.date_range(o, periods=H, freq="h")})
        f = m.predict(fut)
        idx = out.index.intersection(fut.ds)
        out.loc[idx] = f.set_index("ds").yhat.reindex(idx).to_numpy()
    return out.to_numpy(), {"refits": len(origins)}


def _week_origins(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    o, out = start, []
    while o <= end:
        out.append(o)
        o += pd.Timedelta(hours=H_WEEK)
    return out


def _plan_cost(y, plan, cu=config.HOURLY_UNDER_COST, co=config.HOURLY_OVER_COST):
    return float(cu * np.clip(y - plan, 0, None).sum() + co * np.clip(plan - y, 0, None).sum())


def run_hourly(out_dir: Path | None = None, families=tuple(HOURLY_SPECS)) -> dict:
    from .hourly import experiments

    _quiet()
    out_dir = out_dir or config.HOURLY_RESULTS
    c = experiments.Ctx(out_dir)
    grid = c.grid.sort_values("ts").reset_index(drop=True)
    ts, y = grid.ts, grid.demand.to_numpy(float)
    pos = pd.Series(np.arange(len(grid)), index=pd.DatetimeIndex(ts))
    test_start = pd.Timestamp(config.HOURLY_TEST_START)
    val_start = test_start - pd.Timedelta(weeks=8)
    i_val, i_test = int(pos[val_start]), int(pos[test_start])
    X = fourier(grid.how.to_numpy())

    dev_rows = c.dev[c.dev.ts >= val_start]
    i_dev_rows, i_test_rows = pos[dev_rows.ts].to_numpy(), pos[c.test.ts].to_numpy()
    y_val, y_test = dev_rows.demand.to_numpy(), c.test.demand.to_numpy()

    def filled(fit_end):
        train = grid.iloc[:fit_end]
        prof = train[train.status == "observed"].groupby("how").demand.mean()
        return np.where(np.isnan(y), grid.how.map(prof).to_numpy(float), y)

    def forecast(family, spec, fit_end, origins):
        if family in ("ARIMA", "SARIMA"):
            return sarimax_week_ahead(y, X, spec, fit_end)
        if family == "Holt-Winters":
            return holt_winters_week_ahead(filled(fit_end), spec, fit_end)
        return prophet_weekly_refit(ts, y, origins, spec)

    val_origins = _week_origins(val_start, test_start - pd.Timedelta(hours=1))
    test_origins = _week_origins(test_start, c.test.ts.max())
    res = {"protocol": {"validation": [str(val_start), str(test_start - pd.Timedelta(hours=1))], "validation_hours": int(len(y_val)),
                        "test": [str(c.test.ts.min()), str(c.test.ts.max())], "test_hours": int(len(y_test)),
                        "information_rule": "every forecast for hour t uses data recorded up to t - 168 h only",
                        "fourier_harmonics_week": K_WEEK}, "families": {}}
    test_preds, val_preds = {}, {}
    for fam in families:
        rows = {}
        for label, spec in HOURLY_SPECS[fam].items():
            t0 = time.time()
            f, info = forecast(fam, spec, i_val, val_origins)
            p = np.clip(f[i_dev_rows], 0, None)
            rows[label] = {"validation_mae": stats.mae(y_val, p), "seconds": time.time() - t0, **{k: v for k, v in info.items() if k != "params"}}
            val_preds[label] = p
            print(f"[hourly] {label}: validation MAE {rows[label]['validation_mae']:.2f} ({time.time() - t0:.0f}s)")
        chosen = min(rows, key=lambda k: rows[k]["validation_mae"])
        t0 = time.time()
        f, info = forecast(fam, HOURLY_SPECS[fam][chosen], i_test, test_origins)
        test_preds[chosen] = np.clip(f[i_test_rows], 0, None)
        k = float(BUFFER_GRID[np.argmin([_plan_cost(y_val, val_preds[chosen] * g) for g in BUFFER_GRID])])
        res["families"][fam] = {"specs": rows, "chosen": chosen, "buffer_tuned_on_validation": k, "fit": info,
                                "test_seconds": time.time() - t0}
        print(f"[hourly] {fam}: chose {chosen}; test MAE {stats.mae(y_test, test_preds[chosen]):.2f}")

    ml = pd.read_csv(out_dir / "test_predictions.csv.gz", parse_dates=["ts"]).set_index("ts").reindex(c.test.ts)
    iv = pd.read_csv(out_dir / "test_intervals.csv.gz", parse_dates=["ts"]).set_index("ts").reindex(c.test.ts)
    model = json.loads((out_dir / "test.json").read_text())["model"]
    profile = "Static hour-of-week profile"
    ref = {model: ml[model].to_numpy(), profile: ml[profile].to_numpy()}
    scale = json.loads((out_dir / "test.json").read_text())["mase_scale_in_sample_seasonal_naive_mae"]
    table = {}
    for name, p in {**ref, **test_preds}.items():
        m = stats.point_metrics(y_test, p, scale)
        m["mae_ci"] = stats.block_bootstrap(lambda i, p=p: stats.mae(y_test[i], p[i]), len(y_test), H_WEEK, 1000, config.SEED)["ci"]
        if name != model:
            m["dm_vs_ml"] = stats.diebold_mariano(y_test - p, y_test - ref[model], h=H_WEEK)
            m["mae_diff_vs_ml_ci"] = stats.block_bootstrap(lambda i, p=p: stats.mae(y_test[i], p[i]) - stats.mae(y_test[i], ref[model][i]),
                                                           len(y_test), H_WEEK, 1000, config.SEED)
        if name not in ref:
            m["dm_vs_profile"] = stats.diebold_mariano(y_test - p, y_test - ref[profile], h=H_WEEK)
        table[name] = m
    adj = stats.holm({k: v["dm_vs_ml"]["p"] for k, v in table.items() if k in test_preds})
    for k in adj:
        table[k]["dm_vs_ml"]["p_holm"] = adj[k]

    q75 = iv["q75"].to_numpy()
    cost_ref = _plan_cost(y_test, q75)
    decision = {"Quantile gradient boosting, 75th percentile (Part C model)": {"relative_cost": 1.0}}
    for fam, r in res["families"].items():
        plan = test_preds[r["chosen"]] * r["buffer_tuned_on_validation"]
        decision[f"{r['chosen']} x {r['buffer_tuned_on_validation']:.2f}"] = {
            "relative_cost": _plan_cost(y_test, plan) / cost_ref, "service_level": float(np.mean(y_test <= plan)),
            "saving_of_ml_plan_vs_this": stats.block_bootstrap(lambda i, plan=plan: 1 - _plan_cost(y_test[i], q75[i]) / _plan_cost(y_test[i], plan[i]),
                                                               len(y_test), H_WEEK, 500, config.SEED)}
    res.update({"ml_model": model, "table": table, "decision": decision})
    pd.DataFrame({"ts": c.test.ts.astype(str), "demand": y_test, **{k: np.round(v, 2) for k, v in test_preds.items()}}).to_csv(
        out_dir / "classical_test_predictions.csv.gz", index=False)
    (out_dir / "classical.json").write_text(json.dumps(res, indent=2, default=_json))
    return res


# ============================================================================ Part B: Favorita

_FAV = {}          # per-process state for the worker pool (set before forking)


def _fav_one(args):
    """Forecast one series for the 16 days after `origin` (an index into the panel dates). Returns
    (series, forecast in units or None if the model could not be fitted)."""
    family, spec, s, origin = args
    _quiet()
    p, H = _FAV["panel"], config.FAV_HORIZON
    start = max(_FAV["train_from"], int(np.argmax(p.is_open[s])) if p.is_open[s].any() else origin)
    hist = p.sales[s, start:origin + 1].astype(float)
    if np.isfinite(hist).sum() < FAV_MIN_HISTORY:
        return s, None
    yl = np.log1p(hist)
    fut = slice(origin + 1, origin + 1 + H)
    try:
        if family == "Holt-Winters":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            yi = pd.Series(yl).interpolate(limit_direction="both").to_numpy()
            f = ExponentialSmoothing(yi, seasonal="add", seasonal_periods=7, initialization_method="estimated", **spec).fit().forecast(H)
        elif family in ("ARIMA", "SARIMA"):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            spec = dict(spec)
            use_exog = spec.pop("exog", False)
            d = spec["order"][1] + (spec.get("seasonal_order", (0, 0, 0, 0))[1])
            exog = exog_f = None
            if use_exog:
                exog = np.column_stack([np.log1p(p.promo[s, start:origin + 1]), p.hol_national[start:origin + 1]])
                exog_f = np.column_stack([np.log1p(p.promo[s, fut]), p.hol_national[fut]])
            r = SARIMAX(yl, exog=exog, trend="n" if d else "c", **spec).fit(disp=False, maxiter=100)
            f = r.forecast(H, exog=exog_f)
        else:
            from prophet import Prophet
            dates = p.dates[start:origin + 1]
            df = pd.DataFrame({"ds": dates, "y": yl, "promo": np.log1p(p.promo[s, start:origin + 1])}).dropna()
            hol = pd.DataFrame({"holiday": "national", "ds": p.dates[p.hol_national > 0]})
            m = Prophet(weekly_seasonality=True, yearly_seasonality=bool(spec["yearly"]) and len(df) >= 365,
                        daily_seasonality=False, holidays=hol, uncertainty_samples=0)
            m.add_regressor("promo")
            m.fit(df)
            f = m.predict(pd.DataFrame({"ds": p.dates[fut], "promo": np.log1p(p.promo[s, fut])})).yhat.to_numpy()
        f = np.asarray(f, float)
        if not np.all(np.isfinite(f)):
            return s, None
        return s, np.clip(np.expm1(np.clip(f, -1, 20)), 0, None)
    except Exception:                                                   # noqa: BLE001  (counted and reported)
        return s, None


def _fav_window(frame: pd.DataFrame, family: str, spec: dict, series: np.ndarray, origin: int, workers: int):
    """Forecasts for one window as an array aligned with `frame` (rows of other series = NaN)."""
    from multiprocessing import get_context

    jobs = [(family, spec, int(s), origin) for s in series]
    with get_context("fork").Pool(workers) as pool:
        got = dict(pool.map(_fav_one, jobs, chunksize=8))
    out = np.full(len(frame), np.nan)
    ser, h = frame.series.to_numpy(), frame.h.to_numpy().astype(int)
    fb = 0
    fallback = np.nan_to_num(frame.base_dow_mean4.to_numpy(float))
    for s in series:
        m = ser == s
        f = got.get(int(s))
        if f is None:
            out[m] = fallback[m]
            fb += 1
        else:
            out[m] = f[h[m] - 1]
    out[frame.dead.to_numpy()] = 0.0
    return out, fb


def run_favorita(out_dir: Path | None = None, families=("Holt-Winters", "Prophet", "ARIMA", "SARIMA"), workers: int = 2) -> dict:
    from .favorita import experiments as fx

    _quiet()
    out_dir = out_dir or config.FAVORITA_RESULTS
    c = fx.Ctx(out_dir)
    p = c.p
    _FAV.update(panel=p, train_from=int(p.dates.get_loc(pd.Timestamp(config.FAV_TRAIN_FROM))))
    vstart = c.valid_starts[-1]
    vf = c.frame(f"valid_{vstart}", [vstart - 1])
    test = c.frame("test", [c.test0 - 1])
    active_v = np.unique(vf.series[~vf.dead.to_numpy()])
    sample = np.sort(np.random.default_rng(config.SEED).choice(active_v, size=min(FAV_SAMPLE, len(active_v)), replace=False))
    vmask = np.isin(vf.series.to_numpy(), sample) & fx.scored(vf)
    all_series = np.arange(len(p.series))

    res = {"protocol": {"validation_window": [str(vf.date.min().date()), str(vf.date.max().date())], "validation_series_sample": int(len(sample)),
                        "test_window": [str(test.date.min().date()), str(test.date.max().date())], "series": int(len(all_series)),
                        "history_from": config.FAV_TRAIN_FROM, "scale": "log1p(sales), back-transformed with expm1",
                        "fallback": "series with < 56 days of history, or a failed fit: same-weekday mean of the last 4 weeks"},
           "families": {}}
    preds = {}
    y_v = vf.y.to_numpy()
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for fam in families:
        cf = cache / f"classical_{fam}.npz"
        if cf.exists():                                                  # resume: a family already run is not re-fitted
            z = np.load(cf, allow_pickle=True)
            res["families"][fam] = json.loads(str(z["meta"]))
            preds[res["families"][fam]["chosen"]] = z["pred"]
            print(f"[favorita] {fam}: loaded from cache")
            continue
        specs = FAV_SPECS[fam]
        rows = {}
        for label, spec in specs.items():
            t0 = time.time()
            f, fb = _fav_window(vf, fam, spec, sample, vstart - 1, workers)
            rows[label] = {"validation_wape_sample": stats.wape(y_v[vmask], f[vmask]), "fallbacks": fb, "seconds": time.time() - t0}
            print(f"[favorita] {label}: validation WAPE {rows[label]['validation_wape_sample']:.4f} ({time.time() - t0:.0f}s, {fb} fallbacks)")
        chosen = min(rows, key=lambda k: rows[k]["validation_wape_sample"])
        t0 = time.time()
        f, fb = _fav_window(test, fam, specs[chosen], all_series, c.test0 - 1, workers)
        preds[chosen] = f
        res["families"][fam] = {"specs": rows, "chosen": chosen, "test_fallbacks": fb, "test_seconds": time.time() - t0}
        np.savez_compressed(cf, pred=f, meta=json.dumps(res["families"][fam], default=_json))
        ok = fx.scored(test)
        print(f"[favorita] {fam}: chose {chosen}; test WAPE {stats.wape(test.y.to_numpy()[ok], f[ok]):.4f} ({time.time() - t0:.0f}s)")

    ok = fx.scored(test)
    y = test.y.to_numpy()
    tp = pd.read_csv(out_dir / "test_predictions.csv.gz")
    model = c.load("test")["model"]
    ref_cols = [model, "Ridge (log scale)", "Same-weekday mean, last 4 weeks", "Seasonal naive (same weekday, last week)"]
    ref = {}
    for col in ref_cols:
        a = np.zeros(len(test))
        a[ok] = tp[col].to_numpy()
        ref[col] = a
    table = {}
    for name, f in {**ref, **preds}.items():
        m = fx.metrics(y[ok], f[ok])
        if name != model:
            m["wape_diff_vs_ml_store_bootstrap"] = fx.store_bootstrap_wape_diff(test, f, ref[model])
            m["paired_series_test_vs_ml"] = fx.paired_series_test(test, f, ref[model])
        table[name] = m
    adj = stats.holm({k: v["paired_series_test_vs_ml"]["p"] for k, v in table.items() if k in preds})
    for k in adj:
        table[k]["paired_series_test_vs_ml"]["p_holm"] = adj[k]
    res.update({"ml_model": model, "table": table})
    pd.DataFrame({"date": test.date.astype(str), "series": test.series, "y": y, **{k: np.round(v, 3) for k, v in preds.items()}})[ok].to_csv(
        out_dir / "classical_test_predictions.csv.gz", index=False)
    (out_dir / "classical.json").write_text(json.dumps(res, indent=2, default=_json))
    return res


# ============================================================================ report

def report() -> Path:
    from .common import plot
    import matplotlib.pyplot as plt

    h = json.loads((config.HOURLY_RESULTS / "classical.json").read_text())
    f = json.loads((config.FAVORITA_RESULTS / "classical.json").read_text())
    lines = ["# Classical time-series benchmark: ARIMA, SARIMA, Holt-Winters, Prophet", "",
             "Generated by `python -m retail_demand.classical --report` from `results/hourly/classical.json` and "
             "`results/favorita/classical.json`. Protocol: see the docstring of `src/retail_demand/classical.py`.", ""]

    lines += ["## Part C: hourly demand, week ahead", "", "| Forecast (test period) | MAE (95% CI) | WAPE | MAE difference vs "
              f"{h['ml_model']} (95% CI) | Diebold-Mariano p (Holm) |", "|---|---|---|---|---|"]
    for name, m in sorted(h["table"].items(), key=lambda kv: kv[1]["mae"]):
        diff = m.get("mae_diff_vs_ml_ci")
        dm = m.get("dm_vs_ml", {})
        lines.append(f"| {name} | {m['mae']:.1f} ({m['mae_ci'][0]:.1f}-{m['mae_ci'][1]:.1f}) | {m['wape']:.1%} | "
                     + (f"{diff['value']:+.1f} ({diff['ci'][0]:+.1f} to {diff['ci'][1]:+.1f})" if diff else "reference") + " | "
                     + (f"{dm.get('p_holm', dm.get('p')):.2g}" if dm else "") + " |")
    lines += ["", "Specification chosen on the 8 validation weeks:", ""]
    for fam, r in h["families"].items():
        lines.append(f"- **{fam}**: " + "; ".join(f"{k}: MAE {v['validation_mae']:.2f}" for k, v in r["specs"].items()) + f" → {r['chosen']}")
    lines += ["", "| Capacity plan (75th-percentile cost rule) | Cost relative to the ML quantile plan | ML plan's saving (95% CI) |", "|---|---|---|"]
    for name, d in h["decision"].items():
        s = d.get("saving_of_ml_plan_vs_this")
        lines.append(f"| {name} | {d['relative_cost']:.2f} | " + (f"{s['value']:.1%} ({s['ci'][0]:.1%} to {s['ci'][1]:.1%})" if s else "reference") + " |")

    lines += ["", "## Part B: Favorita, 16-day replenishment (1,782 series)", "",
              f"| Forecast (test window) | WAPE | RMSLE | WAPE difference vs {f['ml_model']} (store bootstrap 95% CI) | Series where it is worse | Wilcoxon p (Holm) |",
              "|---|---|---|---|---|---|"]
    for name, m in sorted(f["table"].items(), key=lambda kv: kv[1]["wape"]):
        d = m.get("wape_diff_vs_ml_store_bootstrap")
        pt = m.get("paired_series_test_vs_ml")
        lines.append(f"| {name} | {m['wape']:.1%} | {m['rmsle']:.3f} | "
                     + (f"{d['value'] * 100:+.1f} pts ({d['ci'][0] * 100:+.1f} to {d['ci'][1] * 100:+.1f})" if d else "reference") + " | "
                     + (f"{pt['series_where_a_worse']:.0%}" if pt else "") + " | "
                     + (f"{pt.get('p_holm', pt['p']):.2g}" if pt else "") + " |")
    lines += ["", f"Specification chosen on the validation window ({f['protocol']['validation_series_sample']} random active series):", ""]
    for fam, r in f["families"].items():
        lines.append(f"- **{fam}**: " + "; ".join(f"{k}: WAPE {v['validation_wape_sample']:.1%}" for k, v in r["specs"].items())
                     + f" → {r['chosen']} (test fallbacks: {r['test_fallbacks']})")
    out = config.REPORTS_DIR / "classical.md"
    out.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, tab, key, title, fmt in ((axes[0], h["table"], "mae", "Part C, hourly: test MAE", "{:.1f}"),
                                     (axes[1], f["table"], "wape", "Part B, Favorita: test WAPE", "{:.1%}")):
        items = sorted(tab.items(), key=lambda kv: -kv[1][key])
        ml = h["ml_model"] if tab is h["table"] else f["ml_model"]
        cols = [plot.ORANGE if k == ml else (plot.BLUE if any(t in k for t in ("ARIMA", "Holt", "Prophet")) else plot.GREY) for k, _ in items]
        plot.barh(ax, [k if len(k) < 46 else k[:44] + "…" for k, _ in items], [v[key] for _, v in items], cols, fmt=fmt)
        plot.style(ax, title + " (orange = ML model, blue = classical, grey = baselines)" if tab is h["table"] else title)
        ax.set_xlim(0, max(v[key] for _, v in items) * 1.15)
        if key == "wape":
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plot.save(fig, config.REPORTS_DIR / "figures" / "classical_benchmark.png")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="*", default=[], choices=["hourly", "favorita"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if "hourly" in a.part:
        run_hourly()
    if "favorita" in a.part:
        run_favorita(workers=a.workers)
    if a.report:
        print(report())


if __name__ == "__main__":
    main()
