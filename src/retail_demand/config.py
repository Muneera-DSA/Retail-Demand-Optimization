"""Central configuration. Every analytical choice a reviewer may question lives here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
SEED = 42

# --------------------------------------------------------------------------------------
# Part C: hourly demand -> week-ahead capacity planning
# --------------------------------------------------------------------------------------
HOURLY_CSV = DATA_DIR / "hourly" / "hourly_demand.csv"
HOURLY_RESULTS = RESULTS_DIR / "hourly"

# Week-ahead protocol: a plan for next week is made at the start of the week, so every
# feature must be known at least one week (168 hours) before the hour it predicts.
HOURLY_MIN_LAG = 168
HOURLY_LAGS = (168, 336, 504, 672)          # same hour, 1-4 weeks earlier

# Chronological split. Development data is used for EDA-driven choices, the model screen
# and tuning; the test period is touched once, for the final comparison.
HOURLY_TEST_START = "2020-07-01"

# A run of at least this many consecutive missing days is treated as a recording outage
# (a month of zero demand in peak season is not credible), not as zero demand.
HOURLY_OUTAGE_MIN_DAYS = 3

# Model screen: expanding-window CV inside the development period, each fold 8 weeks.
HOURLY_CV_FOLDS = 5
HOURLY_CV_FOLD_WEEKS = 8

# Accuracy check: a value further than this many robust SDs (MAD-based) from the median of
# its hour-of-week is flagged for review (flagged, not deleted).
HOURLY_OUTLIER_MAD = 5.0

# Prediction interval (capacity planning) and the planning quantile used in the decision.
HOURLY_INTERVAL = (0.10, 0.90)
# Staffing cost ratio: an hour of unmet demand is assumed to cost 3x an hour of idle
# capacity, so the cost-minimising plan is the 3/(3+1) = 75th percentile of demand.
HOURLY_UNDER_COST = 3.0
HOURLY_OVER_COST = 1.0

# --------------------------------------------------------------------------------------
# Part B: Favorita (Corporacion Favorita, Ecuador) -> 16-day replenishment
# --------------------------------------------------------------------------------------
FAVORITA_DIR = DATA_DIR / "favorita"        # Kaggle files, not redistributed (see data/README.md)
FAVORITA_RESULTS = RESULTS_DIR / "favorita"
FAV_HORIZON = 16                             # days ahead, the replenishment cycle (as in the Kaggle task)
FAV_TEST_START = "2017-07-31"                # final test window: the last 16 days of the file
FAV_N_VALID_WINDOWS = 4                      # earlier 16-day windows for the model screen and tuning
FAV_TRAIN_FROM = "2015-08-01"                # first forecast origin used for training (promotions recorded from 2014-04)
FAV_ORIGIN_STEP = 14                         # days between training origins
FAV_DEAD_DAYS = 56                           # a series with no sales in the last 56 days is forecast as zero
FAV_SERVICE_LEVEL = 0.95                     # target cycle service level for the order-up-to decision

# --------------------------------------------------------------------------------------
# Part A: TCS iON project data (50 products x 10 outlets) -> weekly ordering
# --------------------------------------------------------------------------------------
TCS_DIR = DATA_DIR / "tcs"
TCS_RESULTS = RESULTS_DIR / "tcs"
TCS_HORIZON = 7                     # weekly ordering: forecast the next 7 days at each order date
TCS_TEST_WEEKS = 8                  # the last 8 weeks (Jan-Feb 2014) are the test period, scored once
TCS_CV_FOLDS = 4                    # expanding-window CV inside the development period
TCS_CV_FOLD_WEEKS = 8
TCS_FILL_TARGET = 0.95              # target fill rate for the weekly order-up-to decision
TCS_INTERMITTENT_ALPHA = 0.1        # smoothing constant for Croston / SBA / TSB
