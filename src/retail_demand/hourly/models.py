"""Baselines and the candidate models for the screen.

Baselines are the forecasts a planner could make without a model; a model is only worth
deploying if it beats them on data it has not seen:
  - seasonal naive: the same hour last week (falling back to 2, 3, 4 weeks ago)
  - 4-week hour-of-week mean: average of the same hour over the last four weeks
  - static profile: average of each hour-of-week over all training data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .. import config
from .features import FEATURES, HISTORY

BASELINES = ["Seasonal naive (same hour last week)", "Hour-of-week mean, last 4 weeks", "Static hour-of-week profile"]


def baseline(name: str, X: pd.DataFrame, prof: pd.Series) -> np.ndarray:
    fallback = X.how.map(prof).to_numpy(float)
    if name == BASELINES[0]:
        out = X["lag168"].to_numpy(float)
        for k in config.HOURLY_LAGS[1:]:
            out = np.where(np.isnan(out), X[f"lag{k}"].to_numpy(float), out)
    elif name == BASELINES[1]:
        out = X["how_mean_4w"].to_numpy(float)
    elif name == BASELINES[2]:
        out = fallback.copy()
    else:
        raise ValueError(name)
    return np.where(np.isnan(out), fallback, out)


def _numeric_imputed(scale=True):
    steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps)


def _linear_design(scale=True):
    """Hour-of-week as 168 dummies (the seasonal profile) + history features."""
    return ColumnTransformer([
        ("how", OneHotEncoder(handle_unknown="ignore"), ["how"]),
        ("num", _numeric_imputed(scale), [c for c in FEATURES if c not in ("how", "hour", "dow")]),
    ])


# a conservative starting point (many noisy hourly observations); tuned later on the same folds
HGB_DEFAULT = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, early_stopping=False)

# small tuning grids for the families that reach the top of the screen
TUNE_GRIDS = {
    "Ridge": [{"ridge__alpha": a} for a in (0.1, 1, 10, 100, 1000)],
    "Lasso": [{"lasso__alpha": a} for a in (0.01, 0.05, 0.2, 0.5)],
    "Poisson GLM": [{"poissonregressor__alpha": a} for a in (1e-5, 1e-4, 1e-3, 1e-2)],
    "Gradient boosting": [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=lv, min_samples_leaf=ml, l2_regularization=l2)
                          for lr, it in ((0.03, 300), (0.05, 150)) for lv in (7, 15) for ml in (50, 200) for l2 in (1.0, 10.0)],
    "Random forest": [{"randomforestregressor__min_samples_leaf": m, "randomforestregressor__max_features": f}
                      for m in (5, 20, 50) for f in (0.3, 0.6)],
    "Extra trees": [{"extratreesregressor__min_samples_leaf": m, "extratreesregressor__max_features": f}
                    for m in (5, 20, 50) for f in (0.5, 0.9)],
    "k-nearest": [{"kneighborsregressor__n_neighbors": k} for k in (25, 50, 100, 200)],
}


def tune_grid(name: str) -> list[dict]:
    for prefix, grid in TUNE_GRIDS.items():
        if name.startswith(prefix):
            return grid
    return []


def candidates(seed: int = config.SEED) -> dict:
    """Name -> unfitted estimator taking the FEATURES frame."""
    trees = dict(random_state=seed)
    return {
        "Ridge (hour-of-week dummies + history)": make_pipeline(_linear_design(), Ridge(alpha=1.0)),
        "Lasso (hour-of-week dummies + history)": make_pipeline(_linear_design(), Lasso(alpha=0.05, max_iter=5000)),
        "Poisson GLM (hour-of-week dummies + history)": make_pipeline(_linear_design(), PoissonRegressor(alpha=1e-4, max_iter=1000)),
        "k-nearest neighbours (k=25)": make_pipeline(ColumnTransformer([("n", _numeric_imputed(), FEATURES)]), KNeighborsRegressor(n_neighbors=25)),
        "Decision tree (depth 8)": make_pipeline(ColumnTransformer([("n", SimpleImputer(strategy="median", add_indicator=True), FEATURES)]),
                                                DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, **trees)),
        "Random forest": make_pipeline(ColumnTransformer([("n", SimpleImputer(strategy="median", add_indicator=True), FEATURES)]),
                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, **trees)),
        "Extra trees": make_pipeline(ColumnTransformer([("n", SimpleImputer(strategy="median", add_indicator=True), FEATURES)]),
                                     ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7, n_jobs=1, **trees)),
        "Gradient boosting (squared error)": HistGradientBoostingRegressor(**HGB_DEFAULT, **trees),
        "Gradient boosting (Poisson)": HistGradientBoostingRegressor(loss="poisson", **HGB_DEFAULT, **trees),
        "Gradient boosting (absolute error)": HistGradientBoostingRegressor(loss="absolute_error", **HGB_DEFAULT, **trees),
        "Neural network (MLP 64-32)": make_pipeline(ColumnTransformer([("n", _numeric_imputed(), FEATURES)]),
                                                    MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400,
                                                                 early_stopping=True, random_state=seed)),
    }


def is_hgb(model) -> bool:
    return isinstance(model, HistGradientBoostingRegressor)


def quantile_model(alpha: float, params: dict | None = None, seed: int = config.SEED) -> HistGradientBoostingRegressor:
    p = dict(HGB_DEFAULT, random_state=seed)
    p.update(params or {})
    p.pop("loss", None)
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, **p)


HISTORY_FEATURES = HISTORY
