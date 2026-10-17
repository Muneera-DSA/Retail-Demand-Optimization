"""Load the hourly demand file and put it on a complete hourly grid.

The raw file has one row per (date, hour) that has a recorded demand value. Hours that
are absent stay absent in the raw file; on the grid they become NaN and are classified:
  - outage: part of a run of >= HOURLY_OUTAGE_MIN_DAYS whole days with no record at all
  - gap: any other missing hour (mostly at night; see quality.py for why these are
    informative rather than random)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def load_raw(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.HOURLY_CSV)


def to_grid(raw: pd.DataFrame, outage_min_days: int | None = None) -> pd.DataFrame:
    """Complete hourly grid from the first to the last recorded day, with status flags."""
    outage_min_days = outage_min_days or config.HOURLY_OUTAGE_MIN_DAYS
    d = raw.copy()
    d["ts"] = pd.to_datetime(d["date"]) + pd.to_timedelta(d["hour"], unit="h")
    start, end = d.ts.min().normalize(), d.ts.max().normalize() + pd.Timedelta(hours=23)
    grid = pd.DataFrame({"ts": pd.date_range(start, end, freq="h")})
    grid = grid.merge(d[["ts", "demand"]], on="ts", how="left")
    grid["demand"] = grid["demand"].astype(float)
    grid["date"] = grid.ts.dt.normalize()
    day_has = grid.groupby("date").demand.transform(lambda s: s.notna().any())
    # runs of whole missing days
    days = grid.drop_duplicates("date")[["date"]].assign(missing=~day_has.groupby(grid.date).first().values)
    run_id = (days.missing != days.missing.shift()).cumsum()
    run_len = days.groupby(run_id).missing.transform("size")
    days["outage"] = days.missing & (run_len >= outage_min_days)
    grid = grid.merge(days[["date", "outage", "missing"]].rename(columns={"missing": "day_missing"}), on="date")
    grid["status"] = np.where(grid.demand.notna(), "observed", np.where(grid.outage, "outage", "gap"))
    grid["hour"] = grid.ts.dt.hour
    grid["dow"] = grid.ts.dt.dayofweek
    grid["how"] = grid.dow * 24 + grid.hour          # hour of week, 0 = Monday 00:00
    return grid.drop(columns=["outage"])


def split(grid: pd.DataFrame, test_start: str | None = None) -> pd.Series:
    """'dev' before test_start, 'test' from it."""
    test_start = test_start or config.HOURLY_TEST_START
    return pd.Series(np.where(grid.ts < pd.Timestamp(test_start), "dev", "test"), index=grid.index)
