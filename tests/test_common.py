import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from retail_demand.common import stats, treeshap


def _data(n=400, p=5, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2 * X[:, 0] + X[:, 1] * X[:, 2] + np.where(X[:, 3] > 0, 1.5, 0) + rng.normal(scale=0.3, size=n)
    if missing:
        X[rng.random((n, p)) < 0.1] = np.nan
    return X, y


@pytest.mark.parametrize("missing", [False, True])
def test_treeshap_matches_brute_force_shapley_on_a_tree(missing):
    X, y = _data(missing=missing)
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(np.nan_to_num(X) if not missing else X, y)
    t, _ = treeshap.trees_from_model(tree)
    for x in X[:8]:
        np.testing.assert_allclose(treeshap.tree_shap_one(t[0], x, X.shape[1]), treeshap.brute_force_shap(t[0], x, X.shape[1]), atol=1e-9)


@pytest.mark.parametrize("model", [
    HistGradientBoostingRegressor(max_iter=40, max_leaf_nodes=8, random_state=0),
    HistGradientBoostingRegressor(loss="poisson", max_iter=40, max_leaf_nodes=8, random_state=0),
    RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0),
])
def test_treeshap_is_additive_and_tree_walk_matches_model(model):
    X, y = _data(missing=isinstance(model, HistGradientBoostingRegressor))
    y = np.abs(y) + 0.1 if getattr(model, "loss", "") == "poisson" else y
    model.fit(X, y)
    raw = model._raw_predict(X[:30]).ravel() if hasattr(model, "_raw_predict") else model.predict(X[:30])
    np.testing.assert_allclose(treeshap.raw_predict(model, X[:30]), raw, rtol=1e-6, atol=1e-6)
    phi, base = treeshap.shap_values(model, X[:30])
    np.testing.assert_allclose(base + phi.sum(1), raw, rtol=1e-6, atol=1e-6)


def test_treeshap_matches_brute_force_for_hgb_trees():
    X, y = _data(missing=True)
    m = HistGradientBoostingRegressor(max_iter=5, max_leaf_nodes=6, random_state=0).fit(X, y)
    trees, _ = treeshap.trees_from_model(m)
    for t in trees:
        for x in X[:5]:
            np.testing.assert_allclose(treeshap.tree_shap_one(t, x, X.shape[1]), treeshap.brute_force_shap(t, x, X.shape[1]), atol=1e-9)


def test_diebold_mariano_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=3000)
    e_good, e_bad = noise, noise * 1.5
    assert stats.diebold_mariano(e_bad, e_good, h=1)["p"] < 1e-6
    e_same = rng.normal(size=3000)
    assert stats.diebold_mariano(noise, e_same, h=1)["p"] > 0.01


def test_holm_adjustment():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_block_bootstrap_ci_contains_point_and_widens_with_dependence():
    rng = np.random.default_rng(2)
    y = np.repeat(rng.normal(size=60), 50)                     # strongly dependent in blocks of 50
    r1 = stats.block_bootstrap(lambda i: y[i].mean(), len(y), 1, 300)
    r50 = stats.block_bootstrap(lambda i: y[i].mean(), len(y), 50, 300)
    assert r50["ci"][0] <= r50["value"] <= r50["ci"][1]
    assert (r50["ci"][1] - r50["ci"][0]) > 2 * (r1["ci"][1] - r1["ci"][0])


def test_metrics():
    y, f = np.array([10.0, 0, 5]), np.array([8.0, 1, 5])
    assert stats.wape(y, f) == pytest.approx(3 / 15)
    assert stats.mae(y, f) == pytest.approx(1.0)
    assert stats.bias(y, f) == pytest.approx((14 / 3 - 5) / 5)
    cov = stats.coverage_test(np.arange(100), np.arange(100) - 1, np.arange(100) + 1, 0.8)
    assert cov["coverage"] == 1.0
