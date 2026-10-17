"""Part C report: reports/hourly.md and reports/figures/hourly_*.png from results/hourly/*.json.

    python -m retail_demand.hourly.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd

from .. import config
from ..common.plot import AQUA, BLUE, GREY, LIGHT, ORANGE, barh, plt, save, style

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


LABELS = {"how_mean_all": "long-run mean, same hour of week", "hour": "hour of day", "doy_sin": "season (sine)", "doy_cos": "season (cosine)",
          "month": "month", "trend_weeks": "trend", "lag168": "same hour, 1 week ago", "lag336": "same hour, 2 weeks ago",
          "lag504": "same hour, 3 weeks ago", "lag672": "same hour, 4 weeks ago", "lag8736": "same hour, 52 weeks ago", "how": "hour of week",
          "how_mean_4w": "mean of same hour, last 4 weeks", "how_n_4w": "weeks recorded (of last 4)", "level_week": "level of the week before",
          "gap_share_week": "unrecorded share, week before", "dow": "day of week", "weekend": "weekend"}


def _load(res: Path, name):
    return json.loads((res / f"{name}.json").read_text())


def figures(res: Path, fig: Path):
    eda, q, sc, t, dec, ex = (_load(res, n) for n in ("eda", "quality", "screen", "test", "decision", "explain"))
    # 1. hour-of-week heat map
    heat = pd.DataFrame(eda["heatmap_dow_hour"]).T
    heat.index = heat.index.astype(int)
    heat = heat.sort_index()[sorted(heat.columns, key=int)]
    f, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(heat.to_numpy(float), aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), DOW)
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    f.colorbar(im, ax=ax, fraction=0.025, label="mean demand")
    style(ax, "Mean demand by day and hour (recorded hours)", "hour of day")
    save(f, fig / "hourly_heatmap.png")
    # 2. daily mean over time with outage and split
    d = pd.Series(eda["daily_mean_series"])
    d.index = pd.to_datetime(d.index)
    d = d.reindex(pd.date_range(d.index.min(), d.index.max()))
    f, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(d.index, d.values, color=BLUE, lw=0.8)
    ax.plot(d.rolling(28, min_periods=7, center=True).mean(), color=ORANGE, lw=1.6, label="28-day mean")
    for r in q["completeness"]["missing_day_runs"]:
        ax.axvspan(pd.Timestamp(r["start"]), pd.Timestamp(r["end"]) + pd.Timedelta(days=1), color=LIGHT, label=None)
    ax.axvline(pd.Timestamp(config.HOURLY_TEST_START), color=GREY, ls="--", lw=1)
    ax.text(pd.Timestamp(config.HOURLY_TEST_START), ax.get_ylim()[1] * 0.97, " test period", color=GREY, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "Daily mean demand (grey band: recording outage)", None, "demand per hour")
    save(f, fig / "hourly_daily.png")
    # 3. missing share by hour
    m = pd.Series(q["completeness"]["missing_share_by_hour_on_recorded_days"]).sort_index(key=lambda s: s.astype(int))
    f, ax = plt.subplots(figsize=(8, 2.4))
    ax.bar(m.index.astype(int), m.values, color=ORANGE)
    style(ax, "Share of hours with no record, on days that have records", "hour of day", "share missing")
    save(f, fig / "hourly_missing_by_hour.png")
    # 4. model screen
    rows = sc["table"][::-1]
    f, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1))
    barh(ax, [r["model"] for r in rows], [r["cv_mae"] for r in rows], [GREY if r["baseline"] else BLUE for r in rows],
         errors=[r["cv_mae_sd"] for r in rows])
    ax.set_xlim(25, max(r["cv_mae"] for r in rows) * 1.12)
    style(ax, "Model screen: cross-validated MAE, week ahead (grey = baselines)", "MAE (demand units per hour)")
    save(f, fig / "hourly_screen.png")
    # 5. two test weeks with interval
    p = pd.read_csv(res / "test_predictions.csv.gz", parse_dates=["ts"])
    iv = pd.read_csv(res / "test_intervals.csv.gz", parse_dates=["ts"])
    p = p.merge(iv, on="ts")
    start = p.ts.min() + pd.Timedelta(weeks=4)
    w = p[(p.ts >= start) & (p.ts < start + pd.Timedelta(weeks=2))]
    full = pd.DataFrame({"ts": pd.date_range(w.ts.min(), w.ts.max(), freq="h")}).merge(w, on="ts", how="left")
    f, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(full.ts, full.lo, full.hi, color=BLUE, alpha=0.15, label="80% interval")
    ax.plot(full.ts, full[t["model"]], color=BLUE, lw=1.2, label=f"forecast ({t['model']})")
    ax.plot(full.ts, full.demand, color="#0b0b0b", lw=0.8, label="actual")
    ax.step(full.ts, full.q75, where="mid", color=ORANGE, lw=0.9, label="capacity plan (75th percentile)")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_ylim(0, full[["demand", "hi"]].max().max() * 1.15)
    style(ax, "Two test weeks, forecast made a week ahead (gaps: hours with no record)", None, "demand per hour")
    save(f, fig / "hourly_test_weeks.png")
    # 6. decision
    plans = dec["plans"]
    names = list(plans)[::-1]
    f, ax = plt.subplots(figsize=(7.5, 0.4 * len(names) + 1))
    base = plans["Model: 75th-percentile forecast"]["cost"]
    barh(ax, [n.replace(" (buffer tuned on dev)", "") for n in names], [plans[n]["cost"] / base for n in names],
         [AQUA if n.startswith("Model: 75th") else GREY for n in names], fmt="{:.2f}")
    style(ax, "Cost of each capacity plan on the test period (model plan = 1.00)", "relative cost (unmet demand x3 + idle capacity x1)")
    save(f, fig / "hourly_decision.png")
    # 7. SHAP
    if "mean_abs_shap" in ex:
        imp = pd.Series(ex["mean_abs_shap"]).head(12)[::-1]
        f, ax = plt.subplots(figsize=(6.5, 3.6))
        barh(ax, [LABELS.get(k, k) for k in imp.index], list(imp.values), [BLUE] * len(imp), fmt="{:.3f}")
        style(ax, "Mean |SHAP| (exact TreeSHAP, log scale), 300 test hours", "mean absolute contribution")
        save(f, fig / "hourly_shap.png")


def _f(v, d=2):
    return f"{v:.{d}f}"


def markdown(res: Path) -> str:
    q, eda, sc, sel, t, hz, ag, iv, dec, lk, ms, ex, hyp = (_load(res, n) for n in (
        "quality", "eda", "screen", "select", "test", "horizon", "aggregation", "intervals", "decision", "leakage", "missing",
        "explain", "hypotheses"))
    c, v, mi, rel = q["completeness"], q["validity"], q["missingness"], q["relevance"]
    L = ["# Part C results: hourly demand → week-ahead capacity plan",
         "", "Generated by `python -m retail_demand.hourly.report` from `results/hourly/*.json`. Do not edit by hand.", "",
         "## Data checks", "",
         "| Dimension | Check | Result |", "|---|---|---|",
         f"| Validity | rows; hour in 0-23; demand integer, non-negative | {v['rows']:,} rows; {v['hour_outside_0_23']} invalid hours; "
         f"{v['non_integer_demand']} non-integer, {v['negative_demand']} negative; **{v['zero_demand']} zeros** (min {v['demand_min']:.0f}, max {v['demand_max']:.0f}) |",
         f"| Uniqueness | duplicate date-hours | {q['uniqueness']['duplicate_date_hour']} |",
         f"| Completeness | hours recorded on the full hourly grid | {c['observed_hours']:,} of {c['grid_hours']:,} ({c['observed_share']:.1%}) |",
         "| Completeness | whole days missing | " + "; ".join(f"{r['start']} to {r['end']} ({r['days']} d)" for r in c["missing_day_runs"]) + " |",
         f"| Completeness | missing hours on recorded days | 3-5 am: {max(c['missing_share_by_hour_on_recorded_days'][str(h)] for h in (3, 4, 5)):.0%}; "
         f"10 am-6 pm: ≤ {max(c['missing_share_by_hour_on_recorded_days'][str(h)] for h in range(10, 19)):.0%} |",
         f"| Accuracy | values > {q['accuracy']['threshold_mad']:.0f} robust SDs from their hour-of-week median | {q['accuracy']['robust_outliers']} "
         f"({q['accuracy']['robust_outlier_share']:.2%}), kept |",
         f"| Consistency | file in time order; DST duplicates | {q['consistency']['file_sorted_by_time']}; {q['consistency']['dst_duplicate_hours']} |",
         f"| Relevance | test vs development level; hour-of-week profile correlation | {rel['level_change_test_vs_dev']:+.1%}; "
         f"{rel['hour_of_week_profile_correlation_dev_vs_test']:.2f} |", "",
         "**Missing hours are not random.** On days with a missing night hour, afternoon demand averages "
         f"{mi['peak_mean_days_with_night_gap']:.0f} against {mi['peak_mean_days_without']:.0f} on other days "
         f"(Mann-Whitney p = {mi['mannwhitney_p']:.1g}, Cliff's δ = {mi['cliffs_delta']:+.2f}). With no zeros anywhere in the file, the gaps "
         "are most likely hours with no activity (zero or closed). Models are fitted and evaluated on recorded hours; treating the gaps as "
         f"zero instead changes test MAE on recorded hours from {ms['test_mae_recorded_hours_main']:.2f} to {ms['test_mae_recorded_hours_if_gaps_zero']:.2f}. "
         "The 30-day block of missing days is a recording outage and is excluded.", "",
         "## Exploratory analysis", "",
         f"- Hour-of-week explains **{eda['share_of_variance_explained_by_hour_of_week']:.0%}** of the variance of hourly demand.",
         f"- Weekends are busier: daily mean ratio {eda['weekend_vs_weekday']['ratio']:.2f} (Cliff's δ {eda['weekend_vs_weekday']['cliffs_delta']:+.2f}).",
         f"- Hourly demand is over-dispersed (variance/mean {eda['variance_to_mean']:.0f}) and only weakly autocorrelated "
         f"(lag 1 h: {eda['autocorrelation_hourly']['1']:.2f}; lag 1 week: {eda['autocorrelation_hourly']['168']:.2f}).",
         f"- Spring 2020 (COVID) daily demand was {rel['covid_spring_2020_vs_2019_daily_ratio']:.2f}x spring 2019.", "",
         "## Model screen (expanding-window CV, 5 folds × 8 weeks, development period only)", "",
         "| Model | CV MAE (SD) | CV WAPE | DM p vs best (Holm) |", "|---|---|---|---|"]
    for r in sc["table"]:
        p = r.get("dm_vs_best", {}).get("p_holm")
        L.append(f"| {r['model']}{' (baseline)' if r['baseline'] else ''} | {r['cv_mae']:.2f} ({r['cv_mae_sd']:.2f}) | {r['cv_wape']:.3f} | "
                 f"{'' if p is None else f'{p:.2g}'} |")
    L += ["", "Tuned on the same folds: " + "; ".join(f"{x['model']} {x['cv_mae']:.2f}" for x in sel["tuned"]) + ".",
          f"Best model **{sel['model']}** beats the best baseline ({sel['best_baseline']}) by {sel['cv_gain_vs_best_baseline']:.2f} MAE, "
          f"one standard error = {sel['cv_gain_se']:.2f}. Recommended point forecast under the one-standard-error rule: **{sel['recommended_point_forecast']}**.", "",
          f"## Test period ({t['test_start'][:10]} to {t['test_end'][:10]}, {t['n']:,} recorded hours, touched once)", "",
          "| Forecast | MAE (95% CI) | RMSE | WAPE | MASE | Bias | MAE difference vs model (95% CI) | DM p |", "|---|---|---|---|---|---|---|---|"]
    for k, m in t["table"].items():
        dci = m.get("mae_diff_vs_model_ci")
        L.append(f"| {k} | {m['mae']:.2f} ({m['mae_ci'][0]:.2f}-{m['mae_ci'][1]:.2f}) | {m['rmse']:.2f} | {m['wape']:.3f} | {m['mase']:.3f} | {m['bias']:+.1%} | "
                 + (f"{dci['value']:+.2f} ({dci['ci'][0]:+.2f} to {dci['ci'][1]:+.2f}) | {m['dm_vs_model']['p']:.2g} |" if dci else "reference | |"))
    L += ["", "**Error by aggregation level** (WAPE, same forecasts summed):", "", "| Level | " + " | ".join(ag["hour"]) + " |",
          "|---|" + "---|" * len(ag["hour"])]
    for lvl, d in ag.items():
        L.append(f"| {lvl} | " + " | ".join(f"{d[k]['wape']:.1%}" for k in d) + " |")
    L += ["", "**Horizon** (gradient boosting with the information each horizon allows):", "", "| Horizon | MAE | WAPE |", "|---|---|---|"]
    for h in ("week ahead (168 h)", "day ahead (24 h)", "hour ahead (1 h)"):
        L.append(f"| {h} | {hz[h]['mae']:.2f} | {hz[h]['wape']:.1%} |")
    qb, cf = iv["quantile_boosting"], iv["conformal"]
    L += ["", "## Uncertainty and the capacity decision", "",
          f"80% intervals (quantile gradient boosting): coverage **{qb['coverage']:.1%}** (binomial p = {qb['p_binomial']:.2g}), "
          f"{qb['below']:.1%} below / {qb['above']:.1%} above, mean width {iv['mean_width_raw']:.0f}. Conformal calibration on the last 8 "
          f"development weeks widened them to {iv['mean_width_conformal']:.0f} and over-covered ({cf['coverage']:.1%}), so the plain quantile "
          "intervals are used.", "",
          f"Capacity plan: unmet demand costs {dec['under_cost']:.0f}× idle capacity, so the cost-minimising plan is the "
          f"{dec['critical_ratio']:.0%} quantile. Baseline plans get a buffer tuned on development data for the same cost.", "",
          "| Plan | Relative cost | Hours fully covered | Demand served | Idle share of capacity | Model's saving vs this plan (95% CI) |", "|---|---|---|---|---|---|"]
    base = dec["plans"]["Model: 75th-percentile forecast"]["cost"]
    for k, m in dec["plans"].items():
        s = m.get("cost_saving_of_model_vs_this")
        L.append(f"| {k} | {m['cost'] / base:.2f} | {m['service_level']:.1%} | {m['fill_rate']:.1%} | {m['idle_capacity_share']:.1%} | "
                 + (f"{s['value']:.1%} ({s['ci'][0]:.1%} to {s['ci'][1]:.1%}) |" if s else "reference |"))
    L += ["", "## Leakage checks (test MAE)", "", "| Design | MAE |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in lk.items()]
    if "mean_abs_shap" in ex:
        L += ["", "## Explanation (exact TreeSHAP, gradient boosting)", "",
              f"{ex['n_rows']} test hours; additivity error {ex['additivity_max_error']:.1e}. Mean |SHAP| (log scale): "
              + ", ".join(f"{k} {v:.3f}" for k, v in list(ex["mean_abs_shap"].items())[:6]) + "."]
    L += ["", "## Hypothesis tests (Holm-corrected)", "", "| # | Hypothesis | Evidence | p (Holm) | Verdict |", "|---|---|---|---|---|"]
    for h in hyp:
        L.append(f"| {h['id']} | {h['hypothesis']} | {h['evidence']} | {h['p_holm']:.2g} | {h['verdict']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=config.HOURLY_RESULTS)
    ap.add_argument("--reports", type=Path, default=config.REPORTS_DIR)
    a = ap.parse_args(argv)
    figures(a.results, a.reports / "figures")
    (a.reports / "hourly.md").write_text(markdown(a.results))


if __name__ == "__main__":
    main()
