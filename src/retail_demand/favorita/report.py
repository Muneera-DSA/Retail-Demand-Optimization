"""Part B report: reports/favorita.md and reports/figures/favorita_*.png from results/favorita/*.json.

    python -m retail_demand.favorita.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import config
from ..common.plot import AQUA, BLUE, GREY, ORANGE, barh, plt, save, style


LABELS = {"dow_lmean_8": "same weekday, last 8 weeks", "dow_lmean_4": "same weekday, last 4 weeks", "lmean_7": "last 7 days",
          "lmean_14": "last 14 days", "lmean_28": "last 28 days", "lmean_56": "last 56 days", "lmean_112": "last 112 days",
          "llast": "sales on the order date", "promo_target": "items on promotion that day", "promo_vs_recent": "promotion vs recent level",
          "days_open": "days since store opened", "dow": "day of week", "dom": "day of month", "hol_national": "national holiday",
          "days_since_sale": "days since last sale", "cluster": "store cluster", "zero_share_28": "share of zero days, last 28",
          "lstd_28": "volatility, last 28 days", "tx_lmean_28": "store transactions, last 28 days", "family_code": "product family",
          "store_nbr": "store", "h": "days ahead", "month": "month", "payday": "payday", "loil": "oil price", "type_code": "store type"}


def _lab(k):
    return LABELS.get(k, k)


def _load(res: Path, name):
    return json.loads((res / f"{name}.json").read_text())


def figures(res: Path, fig: Path):
    eda, sc, t, dec, ex = (_load(res, n) for n in ("eda", "screen", "test", "decision", "explain"))
    # daily total sales
    d = pd.Series(eda["daily_total_series"])
    d.index = pd.to_datetime(d.index)
    f, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.plot(d.index, d.values / 1e3, color=BLUE, lw=0.6)
    ax.plot(d.rolling(28, center=True).mean() / 1e3, color=ORANGE, lw=1.6, label="28-day mean")
    ax.axvspan(pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-16"), color=GREY, alpha=0.2)
    ax.text(pd.Timestamp("2016-04-20"), ax.get_ylim()[1] * 0.95, "earthquake\nrelief", fontsize=7, color=GREY, va="top")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, "Chain-wide daily sales (all 54 stores, 33 families), from April 2014", None, "units (thousands)")
    save(f, fig / "favorita_daily.png")
    # screen
    rows = sc["table"][::-1]
    f, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 1))
    barh(ax, [r["model"] for r in rows], [r["wape"] * 100 for r in rows], [GREY if r["baseline"] else BLUE for r in rows], fmt="{:.1f}%")
    style(ax, "Model screen: mean WAPE over 4 validation windows (grey = baselines)", "WAPE (%)")
    save(f, fig / "favorita_screen.png")
    # WAPE by family (top 12 by sales)
    fam = pd.DataFrame(t["wape_by_family"]).T.sort_values("sales", ascending=False).head(12)[::-1]
    f, ax = plt.subplots(figsize=(6.5, 4))
    barh(ax, list(fam.index), list(fam.wape * 100), [BLUE] * len(fam), fmt="{:.1f}%")
    style(ax, "Test WAPE by family (12 largest by sales)", "WAPE (%)")
    save(f, fig / "favorita_family.png")
    # horizon
    h = pd.Series(t["wape_by_horizon"])
    h.index = h.index.astype(int)
    f, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(h.index, h.values * 100, "o-", color=BLUE)
    style(ax, "Test WAPE by days ahead", "days after the order date", "WAPE (%)")
    save(f, fig / "favorita_horizon.png")
    # decision: stock needed for 99% fill, by policy and cycle
    cyc = dec["cycles"]
    f, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharex=True)
    for ax, (cname, res) in zip(axes, cyc.items()):
        names = list(res)[::-1]
        vals = [res[n]["stock_for_99pct_fill_ex_post"] / res["Rule of thumb: 28-day average"]["stock_for_99pct_fill_ex_post"] for n in names]
        barh(ax, [n.replace(" (final model)", "") for n in names], vals, [AQUA if n.startswith("Rule") else BLUE for n in names], fmt="{:.3f}")
        ax.set_xlim(0.9, max(vals) * 1.06)
        style(ax, f"{cname}: stock for 99% fill (rule = 1)", "relative stock")
    save(f, fig / "favorita_decision.png")
    # importance
    if "mean_abs_shap" in ex:
        f, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        imp = pd.Series(ex["mean_abs_shap"]).head(12)[::-1]
        barh(axes[0], [_lab(k) for k in imp.index], list(imp.values), [BLUE] * len(imp), fmt="{:.3f}")
        style(axes[0], f"Mean |SHAP|, {ex['shap_model']}", "log1p(units)")
        pi = pd.Series(ex["permutation_importance_wape_increase"]).head(12)[::-1] * 100
        barh(axes[1], [_lab(k) for k in pi.index], list(pi.values), [ORANGE] * len(pi), fmt="{:.1f}")
        style(axes[1], f"Permutation importance, {ex['final_model']}", "WAPE increase (points)")
        save(f, fig / "favorita_importance.png")


def markdown(res: Path) -> str:
    q, eda, sc, tu, t, dec, lk, ex, hyp = (_load(res, n) for n in ("quality", "eda", "screen", "tune", "test", "decision", "leakage",
                                                                  "explain", "hypotheses"))
    v, c, cs = q["validity"], q["completeness"], q["consistency"]
    L = ["# Part B results: Favorita, 1,782 store × family series → 16-day replenishment", "",
         "Generated by `python -m retail_demand.favorita.report` from `results/favorita/*.json`. Do not edit by hand.", "",
         "## Data checks", "", "| Dimension | Check | Result |", "|---|---|---|",
         f"| Validity | negative sales / promotions; fractional sales | {v['negative_sales']} / {v['negative_promotions']}; {v['fractional_sales_share']:.1%} of rows, "
         f"all in weighed families ({', '.join(f.title() for f in v['fractional_sales_families'][:4])}, ...) |",
         f"| Uniqueness | duplicate store-family-days | {q['uniqueness']['duplicate_store_family_date']} |",
         f"| Completeness | dates absent from the file | {', '.join(c['dates_absent_from_file'])} (stores closed on 25 December) |",
         f"| Completeness | stores opening after the start | {len(c['stores_opening_after_start'])} stores, the last on {max(c['stores_opening_after_start'].values())}; "
         "earlier zeros are not demand and are masked |",
         f"| Completeness | series that never sell | {c['series_never_selling']} of {c['series']} ("
         + ", ".join(f"{k.title()} {n}" for k, n in c["never_selling_by_family"].items()) + ") |",
         f"| Completeness | promotions recorded from | {c['promotions_first_recorded']} (all zero before: training starts after) |",
         f"| Completeness | oil price | {c['oil_missing_values']} missing values and no weekend rows; carried forward |",
         f"| Consistency | store-days closed after opening | {cs['store_days_closed_after_opening']} |",
         f"| Consistency | sales vs transactions (store-days) | correlation {cs['sales_vs_transactions_correlation_store_days']:.2f}; "
         f"{c['transactions_store_days_missing_with_sales']} store-days with sales but no transaction count |",
         f"| Accuracy | days > 6 robust SDs from the series median (log scale) | {q['accuracy']['robust_outliers_over_6_mad']:,} "
         f"({q['accuracy']['share_of_selling_days']:.2%}), kept; the largest is {q['accuracy']['largest_single_day'][0]['sales']} units "
         f"({q['accuracy']['largest_single_day'][0]['family'].title()}, store {q['accuracy']['largest_single_day'][0]['store_nbr']}, "
         f"{q['accuracy']['largest_single_day'][0]['date'][:10]}, earthquake relief) |",
         f"| Relevance | coverage | {q['relevance']['stores']} stores, {q['relevance']['cities']} cities, {q['relevance']['families']} families, "
         f"{q['relevance']['years_of_history']:.1f} years |", "",
         "## Exploratory analysis", "",
         f"- {eda['families_for_80pct_of_sales']} of 33 families make 80% of unit sales: "
         + ", ".join(f"{k.title()} {v:.0%}" for k, v in list(eda["share_of_sales_top_families"].items())[:4]) + ".",
         f"- Weekend sales are {eda['mean_daily_sales_by_weekday']['6'] / eda['mean_daily_sales_by_weekday']['3']:.2f}× Thursday's.",
         f"- Within a series, promotion days sell {eda['promotion_lift_within_series']['median_ratio']:.2f}× other days (median over "
         f"{eda['promotion_lift_within_series']['series']} series; {eda['promotion_lift_within_series']['share_positive']:.0%} higher). "
         "This is an association: promotions are planned for good days.",
         f"- Paydays (15th and month end): {eda['payday_effect_within_series']['median_log_diff']:+.1%} (log scale); national holidays: "
         f"{eda['national_holiday_effect_within_series']['median_log_diff']:+.1%}.",
         f"- Oil price vs monthly sales: correlation {eda['oil_vs_monthly_sales_correlation']:.2f} (both trend; not causal).", "",
         "## Model screen (4 validation windows of 16 days, models trained before the first)", "",
         "| Model | Mean WAPE | WAPE by window | RMSLE | Bias | p vs best (Wilcoxon over series, Holm) |", "|---|---|---|---|---|---|"]
    for r in sc["table"]:
        p = r.get("vs_best", {}).get("p_holm")
        L.append(f"| {r['model']}{' (baseline)' if r['baseline'] else ''} | {r['wape']:.1%} | {' / '.join(f'{w:.1%}' for w in r['wape_by_window'])} | "
                 f"{r['rmsle']:.3f} | {r['bias']:+.1%} | {'' if p is None else f'{p:.2g}'} |")
    L += ["", "Tuned on the same windows: " + "; ".join(f"{x['model']} {x['wape']:.1%}" for x in tu["tuned"]) + f". Chosen: **{tu['model']}**.", "",
          f"## Test window ({t['window'][0]} to {t['window'][1]}, touched once; model retrained on {t['train_rows']:,} rows)", "",
          "| Forecast | WAPE | RMSLE | MAE | Bias | WAPE difference vs model (store bootstrap 95% CI) | Series where it is worse |", "|---|---|---|---|---|---|---|"]
    for k, m in t["table"].items():
        dci, ps = m.get("wape_diff_vs_model_store_bootstrap"), m.get("paired_series_test_vs_model")
        L.append(f"| {k} | {m['wape']:.1%} | {m['rmsle']:.3f} | {m['mae']:.1f} | {m['bias']:+.1%} | "
                 + (f"{dci['value']:+.1%} ({dci['ci'][0]:+.1%} to {dci['ci'][1]:+.1%}) | {ps['series_where_a_worse']:.0%} |" if dci else "reference | |"))
    seg = t["segments"]
    L += ["", "By demand pattern: " + "; ".join(f"{k}: WAPE {v['wape']:.1%} ({v['share_of_sales']:.1%} of sales)" for k, v in seg.items()) + ".", "",
          "## Inventory decision: how much to stock per series for the next order cycle", "",
          f"Order-up-to level S = forecast total over the cycle × a family buffer: the {dec['service_target']:.0%} quantile of "
          "(actual + 1)/(forecast + 1) over the 4 validation windows. Buffers are fixed before the test window. The ex-post columns "
          "rescale each policy until it reaches the stated service on the test window, to compare stock at equal service.", ""]
    for cname, res in dec["cycles"].items():
        L += [f"**{cname}**", "", "| Policy | WAPE of cycle totals | Cycle service level | Fill rate | Stock / demand | Stock for 99% fill vs rule | "
              "Stock for 95% cycle service vs rule |", "|---|---|---|---|---|---|---|"]
        for k, m in res.items():
            L.append(f"| {k} | {m['forecast_wape_of_totals']:.1%} | {m['cycle_service_level']:.1%} | {m['fill_rate']:.2%} | {m['stock_vs_demand']:.3f} | "
                     f"{-m['stock_saving_vs_rule_at_99pct_fill']:+.1%} | {-m['stock_saving_vs_rule_at_95pct']:+.1%} |")
        L.append("")
    L += [
        "## Leakage check", "", "| Design | WAPE | RMSLE |", "|---|---|---|"] + [f"| {k} | {m['wape']:.1%} | {m['rmsle']:.3f} |" for k, m in lk.items()]
    if "mean_abs_shap" in ex:
        L += ["", "## Explanations", "",
              f"- Permutation importance of {ex['final_model']} (WAPE increase on the test window): "
              + ", ".join(f"{k} {v * 100:+.1f}" for k, v in list(ex["permutation_importance_wape_increase"].items())[:6]) + " points.",
              f"- Exact TreeSHAP for {ex['shap_model']} ({ex['shap_rows']} test rows, additivity error {ex['additivity_max_error']:.1e}): "
              + ", ".join(f"{k} {v:.3f}" for k, v in list(ex["mean_abs_shap"].items())[:6]) + ".",
              f"- Shared among the top 8 of both: {', '.join(ex['top8_overlap_permutation_vs_shap'])}."]
    L += ["", "## Hypothesis tests (Holm-corrected)", "", "| # | Hypothesis | Evidence | p (Holm) | Verdict |", "|---|---|---|---|---|"]
    for h in hyp:
        L.append(f"| {h['id']} | {h['hypothesis']} | {h['evidence']} | {h['p_holm']:.2g} | {h['verdict']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=config.FAVORITA_RESULTS)
    ap.add_argument("--reports", type=Path, default=config.REPORTS_DIR)
    a = ap.parse_args(argv)
    figures(a.results, a.reports / "figures")
    (a.reports / "favorita.md").write_text(markdown(a.results))


if __name__ == "__main__":
    main()
