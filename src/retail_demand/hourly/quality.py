"""Data-quality checks for the hourly file, grouped by the dimension they test:
validity, uniqueness, completeness, accuracy, consistency, relevance; plus a test of the
missing-data mechanism, which decides how missing hours are treated."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .. import config
from ..common.stats import cliffs_delta


def run(raw: pd.DataFrame, grid: pd.DataFrame, test_start: str | None = None) -> dict:
    test_start = test_start or config.HOURLY_TEST_START
    out: dict = {}
    # ---------------------------------------------------------------- validity
    dates = pd.to_datetime(raw["date"], errors="coerce")
    out["validity"] = {
        "rows": int(len(raw)), "columns": list(raw.columns),
        "unparseable_dates": int(dates.isna().sum()),
        "hour_outside_0_23": int((~raw["hour"].between(0, 23)).sum()),
        "non_integer_demand": int((raw["demand"] != np.round(raw["demand"])).sum()),
        "negative_demand": int((raw["demand"] < 0).sum()),
        "zero_demand": int((raw["demand"] == 0).sum()),
        "null_values": int(raw.isna().sum().sum()),
        "demand_min": float(raw["demand"].min()), "demand_max": float(raw["demand"].max()),
    }
    # -------------------------------------------------------------- uniqueness
    out["uniqueness"] = {"duplicate_date_hour": int(raw.duplicated(["date", "hour"]).sum()),
                         "duplicate_rows": int(raw.duplicated().sum())}
    # ------------------------------------------------------------ completeness
    days = grid.drop_duplicates("date")
    miss_days = days[days.day_missing]
    runs = []
    if len(miss_days):
        r = (miss_days.date.diff() != pd.Timedelta(days=1)).cumsum()
        for _, g in miss_days.groupby(r):
            runs.append({"start": str(g.date.min().date()), "end": str(g.date.max().date()), "days": int(len(g))})
    on_days = grid[~grid.day_missing]
    out["completeness"] = {
        "first_hour": str(grid.ts.min()), "last_hour": str(grid.ts.max()),
        "grid_hours": int(len(grid)), "observed_hours": int((grid.status == "observed").sum()),
        "observed_share": float((grid.status == "observed").mean()),
        "outage_hours": int((grid.status == "outage").sum()), "gap_hours": int((grid.status == "gap").sum()),
        "missing_day_runs": runs,
        "missing_share_by_hour_on_recorded_days": on_days.groupby("hour").demand.apply(lambda s: float(s.isna().mean())).round(3).to_dict(),
        "gap_run_lengths": _gap_runs(on_days),
        "observed_share_dev_vs_test": {k: float((g.status == "observed").mean()) for k, g in
                                       grid.groupby(np.where(grid.ts < pd.Timestamp(test_start), "dev", "test"))},
    }
    # ---------------------------------------------------------------- accuracy
    obs = grid[grid.status == "observed"].copy()
    dev = obs[obs.ts < pd.Timestamp(test_start)]
    med = dev.groupby("how").demand.median()
    mad = dev.groupby("how").demand.apply(lambda s: float(np.median(np.abs(s - s.median())))) * 1.4826
    obs["robust_z"] = (obs.demand - obs.how.map(med)) / obs.how.map(mad).clip(lower=1)
    flagged = obs[obs.robust_z.abs() > config.HOURLY_OUTLIER_MAD]
    nb = grid.set_index("ts").demand
    spikes = obs[(obs.demand > 3 * obs.ts.map(lambda t: nb.get(t - pd.Timedelta(hours=1), np.nan)))
                 & (obs.demand > 3 * obs.ts.map(lambda t: nb.get(t + pd.Timedelta(hours=1), np.nan)))]
    out["accuracy"] = {
        "robust_outliers": int(len(flagged)), "robust_outlier_share": float(len(flagged) / len(obs)),
        "threshold_mad": config.HOURLY_OUTLIER_MAD,
        "largest_flagged": (flagged.nlargest(10, "robust_z")[["ts", "demand", "robust_z"]]
                            .assign(ts=lambda d: d.ts.astype(str)).round(1).to_dict("records")),
        "isolated_spikes_3x_both_neighbours": int(len(spikes)),
        "policy": ("flagged values are kept: no evidence they are errors rather than real peaks; "
                   "models are compared on MAE, which limits their influence"),
    }
    # ------------------------------------------------------------- consistency
    raw_ts = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["hour"], unit="h")
    per_day = obs.groupby("date").size()
    out["consistency"] = {
        "file_sorted_by_time": bool(raw_ts.is_monotonic_increasing),
        "hours_per_recorded_day": {"min": int(per_day.min()), "median": float(per_day.median()), "max": int(per_day.max())},
        "days_with_fewer_than_12_hours": int((per_day < 12).sum()),
        "dst_duplicate_hours": int(raw.duplicated(["date", "hour"]).sum()),
    }
    # ------------------------------------------------- missing-data mechanism
    out["missingness"] = missingness(grid)
    # --------------------------------------------------------------- relevance
    out["relevance"] = relevance(grid, test_start)
    return out


def _gap_runs(on_days: pd.DataFrame) -> dict:
    lengths = []
    for _, g in on_days.groupby("date"):
        run = 0
        for v in g.demand.isna().values:
            if v:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return {int(k): int(v) for k, v in pd.Series(lengths).value_counts().sort_index().items()}


def missingness(grid: pd.DataFrame) -> dict:
    """Are missing hours missing at random? If they were random recording failures, days
    with a missing night hour would be as busy as other days. Compare the afternoon peak
    (13:00-16:59) on days with and without a missing night hour (00:00-06:59)."""
    on = grid[~grid.day_missing]
    night_missing = on[on.hour <= 6].groupby("date").demand.apply(lambda s: s.isna().any())
    peak = on[on.hour.between(13, 16)].groupby("date").demand.mean()
    a, b = peak[night_missing[night_missing].index].dropna(), peak[night_missing[~night_missing].index].dropna()
    mw = sps.mannwhitneyu(a, b, alternative="two-sided")
    # demand in the hour just before a gap, relative to that hour-of-week's typical level
    g = grid.sort_values("ts").reset_index(drop=True)
    typical = g[g.status == "observed"].groupby("how").demand.median()
    before_gap = g[(g.status == "observed") & (g.status.shift(-1) == "gap")]
    ratio = (before_gap.demand / before_gap.how.map(typical)).median()
    return {
        "days_with_night_gap": int(night_missing.sum()), "days_without": int((~night_missing).sum()),
        "peak_mean_days_with_night_gap": float(a.mean()), "peak_mean_days_without": float(b.mean()),
        "mannwhitney_p": float(mw.pvalue), "cliffs_delta": cliffs_delta(a, b),
        "median_ratio_hour_before_gap_to_typical": float(ratio),
        "conclusion": ("missing hours are informative: they occur on quieter days and after quieter hours, "
                       "consistent with hours with no activity recorded (likely zero or closed), not random loss"),
    }


def relevance(grid: pd.DataFrame, test_start: str) -> dict:
    obs = grid[grid.status == "observed"].copy()
    ts0 = pd.Timestamp(test_start)
    dev, test = obs[obs.ts < ts0], obs[obs.ts >= ts0]
    prof_dev, prof_test = dev.groupby("how").demand.mean(), test.groupby("how").demand.mean()
    daily = obs.groupby("date").demand.sum()
    covid = (daily["2020-03-16":"2020-06-30"].mean(), daily["2019-03-16":"2019-06-30"].mean())
    return {
        "period_days": int(grid.date.nunique()),
        "dev_hours": int(len(dev)), "test_hours": int(len(test)),
        "dev_mean": float(dev.demand.mean()), "test_mean": float(test.demand.mean()),
        "level_change_test_vs_dev": float(test.demand.mean() / dev.demand.mean() - 1),
        "hour_of_week_profile_correlation_dev_vs_test": float(np.corrcoef(prof_dev, prof_test.reindex(prof_dev.index))[0, 1]),
        "covid_spring_2020_vs_2019_daily_ratio": float(covid[0] / covid[1]),
        "note": "origin and units of the series are not documented; no holiday calendar can be attached without knowing the location",
    }
